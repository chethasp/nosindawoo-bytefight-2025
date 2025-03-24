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
        self.terminator_mode = False
        self.double_move_allowed = False
        self.triple_move_allowed = False
        self.enemy_double_move_allowed = False
        self.enemy_triple_move_allowed = False

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

        actions = []
        our_length = board.get_length(False)
        enemy_length = board.get_length(True)
        
        head = tuple(board.get_head_location())
        enemy_head = tuple(board.get_head_location(True))
        dist_apart = self.distance_map[head][enemy_head[1]][enemy_head[0]] if self.distance_map[head][enemy_head[1]][enemy_head[0]] is not None else float('inf')
        
        # Set terminator mode and move allowances
        self.terminator_mode = False
        # self.terminator_mode = abs(our_length - enemy_length) >= 8
        self.double_move_allowed = dist_apart <= 4 and our_length >= 4
        self.triple_move_allowed = dist_apart <= 2 and our_length >= 8

        # self.enemy_double_move_allowed = dist_apart <= 5 and enemy_length >= 4
        # self.enemy_triple_move_allowed = dist_apart <= 4 and enemy_length >= 8
        self.enemy_double_move_allowed = False
        self.enemy_triple_move_allowed = False

        self.max_depth = 4

        if not best_move:
            valid_moves = [move for move in board.get_possible_directions() if board.is_valid_move(move)]
            fallback_move = valid_moves[0] if valid_moves else Action.FF
            actions.append(fallback_move)
        else:
            # Add trap action if our length > 3 * enemy length
            if our_length > 2.5 * enemy_length and our_length > 10:
                cost = 0
                if len(best_move) == 1: 
                    cost = 2
                elif len(best_move) == 2: 
                    cost = 6
                cost += 2
                if our_length - cost - 2 >= 2:
                    actions.append(Action.TRAP)  # add trap action to front
            actions.extend(best_move)
        
        return actions

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
                return float('-inf')  # No moves left, we lose
        
        else:  # Enemy turn
            valid_moves = [move for move in board.get_possible_directions(enemy=True) if board.is_valid_move(move, enemy=True)] 
            if len(valid_moves) == 0: 
                return float('inf')  # Enemy has no moves, we win

        head = tuple(board.get_head_location())
        enemy_head = tuple(board.get_head_location(True))

        current_apples = board.get_current_apples()
        current_turn = board.get_turn_count()

        score = 0.0
        proximity_penalty = 0.0

        if own_turn:
            score += 2500 * board.get_length()

            if self.terminator_mode:
                enemy_x, enemy_y = enemy_head
                target_cells = [(enemy_x - 1, enemy_y), (enemy_x + 1, enemy_y)]
                min_dist_to_target = float('inf')
                for tx, ty in target_cells:
                    if (0 <= tx < self.board_width and 0 <= ty < self.board_height and
                        not self.wall_mask[ty, tx]):
                        dist = self.distance_map[head][ty][tx]
                        if dist is not None and dist < min_dist_to_target:
                            min_dist_to_target = dist
                if min_dist_to_target != float('inf'):
                    score += 20000.0 / (min_dist_to_target + 1)
            else:
                score += 10000 * board.get_apples_eaten()
                closest_apple = self.find_closest_apple(head, current_apples)
                if closest_apple is not None:
                    ax, ay = closest_apple
                    dist = self.distance_map[head][ay][ax]
                    if dist is not None:
                        score += 10000.0 / (dist + 1) - dist
                elif self.future_apples_deque:
                    min_future_dist = float('inf')
                    for spawn_turn, ax, ay in self.future_apples_deque:
                        dist = self.distance_map[head][ay][ax]
                        if dist is not None and spawn_turn + 2 <= current_turn + dist:
                            if dist < min_future_dist:
                                min_future_dist = dist
                                score += 10000.0 / (dist + 1) - dist * 0.5
                        if dist is not None and spawn_turn > current_turn + dist:
                            break

            free_spaces = 0
            head_x, head_y = head
            my_body = [tuple(loc) for loc in board.get_all_locations(enemy=False)]
            enemy_body = [tuple(loc) for loc in board.get_all_locations(enemy=True)]
            for x in range(max(0, head_x - 2), min(self.board_width, head_x + 3)):
                for y in range(max(0, head_y - 2), min(self.board_height, head_y + 3)):
                    if (x, y) != head:
                        if (self.wall_mask[y, x] == 0 and
                            (x, y) not in my_body and
                            (x, y) not in enemy_body and
                            (self.portal_mask[y, x][0] == -1 or self.portal_mask[y, x][1] == -1)):
                            free_spaces += 1
            max_possible_spaces = 24
            proximity_ratio = free_spaces / max_possible_spaces
            max_penalty = 10000.0
            proximity_penalty = max_penalty * (1 - proximity_ratio) ** 2
        
        else:  # Enemy turn
            score = -10000 * board.get_apples_eaten(enemy=True)
            score -= 2500 * board.get_length(enemy=True)

            closest_apple = self.find_closest_apple(enemy_head, current_apples)
            if closest_apple is not None:
                ax, ay = closest_apple
                dist = self.distance_map[enemy_head][ay][ax]
                if dist is not None:
                    score += -10000.0 / (dist + 1) - dist
            elif self.future_apples_deque:
                min_future_dist = float('inf')
                for spawn_turn, ax, ay in self.future_apples_deque:
                    dist = self.distance_map[enemy_head][ay][ax]
                    if dist is not None and spawn_turn + 2 <= current_turn + dist:
                        if dist < min_future_dist:
                            min_future_dist = dist
                            score += -10000.0 / (dist + 1) - dist * 0.5
                    if dist is not None and spawn_turn > current_turn + dist:
                        break
            
            free_spaces = 0
            head_x, head_y = enemy_head
            my_body = [tuple(loc) for loc in board.get_all_locations(enemy=True)]
            enemy_body = [tuple(loc) for loc in board.get_all_locations(enemy=False)]
            for x in range(max(0, head_x - 2), min(self.board_width, head_x + 3)):
                for y in range(max(0, head_y - 2), min(self.board_height, head_y + 3)):
                    if (x, y) != head:
                        if (self.wall_mask[y, x] == 0 and
                            (x, y) not in enemy_body and
                            (x, y) not in my_body and
                            (self.portal_mask[y, x][0] == -1 or self.portal_mask[y, x][1] == -1)):
                            free_spaces += 1
            max_possible_spaces = 24
            proximity_ratio = free_spaces / max_possible_spaces
            max_penalty = 10000.0
            proximity_penalty = max_penalty * (1 - proximity_ratio) ** 2

        score -= proximity_penalty
        return score

    def minimax_alphabeta(self, board: player_board.PlayerBoard, depth: int, maximizing: bool, alpha: float = float('-inf'), beta: float = float('inf')) -> tuple[float, list[Action]]:
        if depth == 0:
            return self.evaluate(board, maximizing), []  # Return empty list instead of None
        
        # Generate move sequences based on distance
        if maximizing:  # Our turn
            single_moves = [move for move in board.get_possible_directions() if board.is_valid_move(move)]
            if not single_moves:
                return float('-inf'), []  # No valid moves, we lose
            
            double_moves = []
            triple_moves = []
            if self.double_move_allowed:  # dist_apart <= 6
                for first_move in single_moves:
                    next_board, success = board.forecast_turn([first_move])
                    if not success:
                        continue
                    valid_second_moves = [move for move in next_board.get_possible_directions() if next_board.is_valid_move(move)]
                    for second_move in valid_second_moves:
                        double_moves.append([first_move, second_move])
                        if self.triple_move_allowed:  # dist_apart <= 3
                            next_board2, success = next_board.forecast_turn([second_move])
                            if not success:
                                continue
                            valid_third_moves = [move for move in next_board2.get_possible_directions() if next_board2.is_valid_move(move)]
                            for third_move in valid_third_moves:
                                triple_moves.append([first_move, second_move, third_move])
            
            all_moves = [[move] for move in single_moves] + double_moves + triple_moves
        
        else:  # Enemy turn
            single_moves = [move for move in board.get_possible_directions(enemy=True) if board.is_valid_move(move, enemy=True)]
            if not single_moves:
                return float('inf'), []  # No valid moves, enemy loses
            
            double_moves = []
            triple_moves = []
            if self.enemy_double_move_allowed:  
                for first_move in single_moves:
                    next_board, success = board.forecast_turn([first_move])
                    if not success:
                        continue
                    valid_second_moves = [move for move in next_board.get_possible_directions(enemy=True) if next_board.is_valid_move(move, enemy=True)]
                    for second_move in valid_second_moves:
                        double_moves.append([first_move, second_move])
                        if self.enemy_triple_move_allowed:  
                            next_board2, success = next_board.forecast_turn([second_move])
                            if not success:
                                continue
                            valid_third_moves = [move for move in next_board2.get_possible_directions(enemy=True) if next_board2.is_valid_move(move, enemy=True)]
                            for third_move in valid_third_moves:
                                triple_moves.append([first_move, second_move, third_move])
            
            all_moves = [[move] for move in single_moves] + double_moves + triple_moves
        
        best_move = []

        if maximizing:
            max_eval = float('-inf')
            for move_sequence in all_moves:
                next_board, success = board.forecast_turn(move_sequence)
                if not success:
                    print(f"Debug: Invalid move sequence: {move_sequence}")
                    continue
                eval_score, _ = self.minimax_alphabeta(next_board, depth - 1, False, alpha, beta)
                if eval_score > max_eval:
                    max_eval = eval_score
                    best_move = move_sequence
                alpha = max(alpha, eval_score)
                if beta <= alpha:
                    break
            return max_eval, best_move

        else:  # Minimizing
            min_eval = float('inf')
            for move_sequence in all_moves:
                next_board, success = board.forecast_turn(move_sequence)
                if not success:
                    print(f"Debug: Invalid move sequence: {move_sequence}")
                    continue
                eval_score, _ = self.minimax_alphabeta(next_board, depth - 1, True, alpha, beta)
                if eval_score < min_eval:
                    min_eval = eval_score
                    best_move = move_sequence
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