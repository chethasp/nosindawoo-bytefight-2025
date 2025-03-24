from game import player_board
from game.enums import Action
from collections.abc import Callable
import random
from collections import deque
import numpy as np

class PlayerController:
    def __init__(self, time_left: Callable):
        self.time_left = time_left
        self.board_width = None
        self.board_height = None
        self.portal_mask = None
        self.wall_mask = None
        self.future_apples_deque = None
        self.distance_map = None
        self.pairwise_distances_precomputed = False
        self.max_depth = 4

    def bid(self, board: player_board.PlayerBoard, time_left: Callable) -> int:
        self.board_width = board.get_dim_x()
        self.board_height = board.get_dim_y()
        self.portal_mask = board.get_portal_mask(descriptive=True)
        self.wall_mask = board.get_wall_mask()
        future_apples = board.get_future_apples()
        if future_apples.size > 0:
            indices = np.argsort(future_apples[:, 0])
            future_apples = future_apples[indices]
        self.future_apples_deque = deque(future_apples)
        return 0

    def play(self, board: player_board.PlayerBoard, time_left: Callable) -> list[Action]:
        if not self.pairwise_distances_precomputed:
            self.precompute_pairwise_distances(board)

        self.head_location = tuple(board.get_head_location())

        score, best_move = self.minimax_alphabeta(board, self.max_depth, True)

        if best_move is None:
            valid_moves = [move for move in board.get_possible_directions() if board.is_valid_move(move)] 
            return [valid_moves[0]] if valid_moves else [Action.FF]
        return [best_move]

    def find_closest_apple(self, head: tuple, apples: np.ndarray) -> tuple:
        if apples.size == 0:
            return None
        min_dist = float('inf')
        closest_apple = None
        for ax, ay in apples:
            dist = self.distance_map[head][ay][ax]
            if dist is not None and dist < min_dist:
                min_dist = dist
                closest_apple = (ax, ay)
        return closest_apple
    
    def evaluate(self, board: player_board.PlayerBoard, own_turn: bool) -> float: 
        if own_turn: 
            valid_moves = [move for move in board.get_possible_directions() if board.is_valid_move(move)] 
            if len(valid_moves) == 0: 
                return float('-inf') #us having no moves left on our own turn is very bad
        
        else: #enemy turn
            valid_moves = [move for move in board.get_possible_directions(enemy = True) if board.is_valid_move(move, enemy=True)] 
            if len(valid_moves) == 0: 
                return float('inf') #enemy having no moves left on their turn is very good
            
        head = tuple(board.get_head_location())
        enemy_head = tuple(board.get_head_location(True))

        current_apples = board.get_current_apples()
        current_turn = board.get_turn_count()

        score = 0.0
        apple_proximity = 0.0
        proximity_penalty = 0.0

        #Assign a score of 10000 for eating an apple
        if(own_turn): 
            score = 10000 * board.get_apples_eaten()

            closest_apple = self.find_closest_apple(head, current_apples)

            if closest_apple is not None:
                ax, ay = closest_apple
                dist = self.distance_map[head][ay][ax]
                if dist is not None:
                    apple_proximity += 10000.0 / (dist + 1) - dist
            elif self.future_apples_deque:  # No current apples, consider future
                min_future_dist = float('inf')
                for spawn_turn, ax, ay in self.future_apples_deque:
                    dist = self.distance_map[head][ay][ax]
                    if dist is not None and spawn_turn + 2 <= current_turn + dist:
                        if dist < min_future_dist:
                            min_future_dist = dist
                            apple_proximity += 10000.0 / (dist + 1) - dist * 0.5
                    if dist is not None and spawn_turn > current_turn + dist:
                        break

            
                    # Custom calculation of free spaces in 5x5 area (2 steps including diagonals)
            free_spaces = 0
            head_x, head_y = head
            my_body = [tuple(loc) for loc in board.get_all_locations(enemy=False)]
            enemy_body = [tuple(loc) for loc in board.get_all_locations(enemy=True)]
            for x in range(max(0, head_x - 2), min(self.board_width, head_x + 3)):
                for y in range(max(0, head_y - 2), min(self.board_height, head_y + 3)):
                    if (x, y) != head:  # Exclude the head itself
                        # Check if cell is occupied (wall, my body, or enemy body)
                        if (self.wall_mask[y, x] == 0 and  # Not a wall
                            (x, y) not in my_body and  # Not my body
                            (x, y) not in enemy_body and  # Not enemy body
                            (self.portal_mask[y, x][0] == -1 or self.portal_mask[y, x][1] == -1)):  # Not a portal
                            free_spaces += 1

            max_possible_spaces = 24  # 5x5 area around head (25 cells total, minus head)
            # Polynomial scaling: penalty = max_penalty * (1 - free_spaces/max_spaces)^2
            proximity_ratio = free_spaces / max_possible_spaces
            max_penalty = 10000.0
            proximity_penalty = max_penalty * (1 - proximity_ratio) ** 2
        
        else: 
            ate_apple = np.any(np.all(current_apples == enemy_head, axis=1)) if current_apples.size > 0 else False
            if ate_apple:
                return -10000.0 * board.get_apples_eaten(False)

            closest_apple = self.find_closest_apple(enemy_head, current_apples)

            if closest_apple is not None:
                ax, ay = closest_apple
                dist = self.distance_map[enemy_head][ay][ax]
                if dist is not None:
                    apple_proximity += -10000.0 / (dist + 1) - dist
            elif self.future_apples_deque:  # No current apples, consider future
                min_future_dist = float('inf')
                for spawn_turn, ax, ay in self.future_apples_deque:
                    dist = self.distance_map[enemy_head][ay][ax]
                    if dist is not None and spawn_turn + 2 <= current_turn + dist:
                        if dist < min_future_dist:
                            min_future_dist = dist
                            apple_proximity += -10000.0 / (dist + 1) - dist * 0.5
                    if dist is not None and spawn_turn > current_turn + dist:
                        break
            
            free_spaces = 0
            head_x, head_y = enemy_head
            my_body = [tuple(loc) for loc in board.get_all_locations(enemy=True)]
            enemy_body = [tuple(loc) for loc in board.get_all_locations(enemy=False)]
            for x in range(max(0, head_x - 2), min(self.board_width, head_x + 3)):
                for y in range(max(0, head_y - 2), min(self.board_height, head_y + 3)):
                    if (x, y) != head:  # Exclude the head itself
                        # Check if cell is occupied (wall, my body, or enemy body)
                        if (self.wall_mask[y, x] == 0 and  # Not a wall
                            (x, y) not in enemy_body and  # Not my body
                            (x, y) not in my_body and  # Not enemy body
                            (self.portal_mask[y, x][0] == -1 or self.portal_mask[y, x][1] == -1)):  # Not a portal
                            free_spaces += 1

            max_possible_spaces = 24  # 5x5 area around head (25 cells total, minus head)
            # Polynomial scaling: penalty = max_penalty * (1 - free_spaces/max_spaces)^2
            proximity_ratio = free_spaces / max_possible_spaces
            max_penalty = 10000.0
            proximity_penalty = max_penalty * (1 - proximity_ratio) ** 2


        # # Custom calculation of free spaces in 5x5 area (2 steps including diagonals)
        # free_spaces = 0
        # head_x, head_y = head
        # my_body = [tuple(loc) for loc in board.get_all_locations(enemy=False)]
        # enemy_body = [tuple(loc) for loc in board.get_all_locations(enemy=True)]
        # for x in range(max(0, head_x - 2), min(self.board_width, head_x + 3)):
        #     for y in range(max(0, head_y - 2), min(self.board_height, head_y + 3)):
        #         if (x, y) != head:  # Exclude the head itself
        #             # Check if cell is occupied (wall, my body, or enemy body)
        #             if (self.wall_mask[y, x] == 0 and  # Not a wall
        #                 (x, y) not in my_body and  # Not my body
        #                 (x, y) not in enemy_body and  # Not enemy body
        #                 (self.portal_mask[y, x][0] == -1 or self.portal_mask[y, x][1] == -1)):  # Not a portal
        #                 free_spaces += 1

        # max_possible_spaces = 24  # 5x5 area around head (25 cells total, minus head)
        # # Polynomial scaling: penalty = max_penalty * (1 - free_spaces/max_spaces)^2
        # proximity_ratio = free_spaces / max_possible_spaces
        # max_penalty = 10000.0
        # proximity_penalty = max_penalty * (1 - proximity_ratio) ** 2

        score -= proximity_penalty


        # Free space calculation
        #GROK IDEA AND CODE
        # head_x, head_y = head if own_turn else enemy_head
        # my_body = [tuple(loc) for loc in board.get_all_locations(enemy=False)]
        # enemy_body = [tuple(loc) for loc in board.get_all_locations(enemy=True)]
        # free_spaces = 0
        # for x in range(max(0, head_x - 2), min(self.board_width, head_x + 3)):
        #     for y in range(max(0, head_y - 2), min(self.board_height, head_y + 3)):
        #         if (x, y) != (head_x, head_y):
        #             if (self.wall_mask[y, x] == 0 and
        #                 (x, y) not in my_body and
        #                 (x, y) not in enemy_body and
        #                 (self.portal_mask[y, x][0] == -1 or self.portal_mask[y, x][1] == -1)):
        #                 free_spaces += 1
        # proximity_ratio = free_spaces / 24
        # max_penalty = 20000.0
        # proximity_penalty = max_penalty * (1 - proximity_ratio) ** 2
        # score += proximity_penalty if own_turn else -proximity_penalty

        return score
    

    def minimax_alphabeta(self, board: player_board.PlayerBoard, depth: int, maximizing: bool, alpha: float = float('-inf'), beta: float = float('inf')) -> tuple[float, Action | None]:
        if depth == 0:
            return self.evaluate(board, maximizing), None
        
        valid_moves = None
        if(maximizing): #it's our turn
            valid_moves = [move for move in board.get_possible_directions() if board.is_valid_move(move)] 
        else: 
            valid_moves = [move for move in board.get_possible_directions(enemy = True) if board.is_valid_move(move, enemy=True)] 
        
        if not valid_moves:
            return float('-inf') if maximizing else float('inf'), None
        best_move = None 

        if maximizing: 
            max_eval = float('-inf')
            for move in valid_moves: 
                next_board, success = board.forecast_turn([move])
                if not success: 
                    print("ERORR: Tried applying invalid move")
                eval_score, _ = self.minimax_alphabeta(next_board, depth - 1, False, alpha, beta)
                if eval_score > max_eval: 
                    max_eval = eval_score 
                    best_move = move 
                alpha = max(alpha, eval_score)
                if beta <= alpha: 
                    break
            return max_eval, best_move

        else: #minimizing
            min_eval = float('inf')
            for move in valid_moves: 
                next_board, success = board.forecast_turn([move])
                if not success: 
                    print("ERORR: Tried applying invalid move")
                eval_score, _ = self.minimax_alphabeta(next_board, depth - 1, True, alpha, beta)
                if eval_score < min_eval:
                    min_eval = eval_score
                    best_move = move
                beta = min(beta, eval_score)
                if beta <= alpha:
                    break
            return min_eval, best_move


    def precompute_pairwise_distances(self, board: player_board.PlayerBoard): 
        self.distance_map = {}
        directions = {(-1, 0), (1, 0), (0, 1), (0, -1), (-1, -1), (1, 1), (-1, 1), (1, -1)}
        for x in range(self.board_width): 
            for y in range(self.board_height):
                start = (x, y)
                if self.wall_mask[y, x]:
                    self.distance_map[start] = [[None for _ in range(self.board_width)] 
                                              for _ in range(self.board_height)]
                    continue
                distances = [[None for _ in range(self.board_width)] for _ in range(self.board_height)]
                queue = deque([(x, y, 0)])
                while queue: 
                    fx, fy, dist = queue.popleft()
                    if distances[fy][fx] is not None:
                        continue
                    distances[fy][fx] = dist
                    for dx, dy in directions: 
                        nx, ny = fx + dx, fy + dy
                        if (nx < 0 or nx >= self.board_width or ny < 0 or ny >= self.board_height):
                            continue
                        if self.wall_mask[ny, nx]:
                            distances[ny][nx] = -1
                            continue
                        if distances[ny][nx] is not None:
                            continue
                        px, py = self.portal_mask[ny, nx]
                        if px != -1 and py != -1:
                            queue.append((px, py, dist))
                        else:
                            queue.append((nx, ny, dist + 1))
                self.distance_map[start] = distances
        self.pairwise_distances_precomputed = True