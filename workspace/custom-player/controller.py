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

        remaining_time = self.time_left()
        if remaining_time < 0.1:
            return [Action.FF]

        current_apples = board.get_current_apples()
        head = tuple(board.get_head_location())
        closest_apple = self.find_closest_apple(head, current_apples)

        print(f"Head: {head}, Closest Apple: {closest_apple}, Apples: {current_apples.tolist()}")
        score, best_move = self.minimax(board, self.max_depth, True, closest_apple=closest_apple)
        print(f"Best Move: {best_move}, Score: {score}")
        if best_move is None:
            valid_moves = self.get_valid_moves(board)
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

    def evaluate(self, board: player_board.PlayerBoard, closest_apple: tuple) -> float:
        """Score: High for eating, strong proximity reward."""
        valid_moves = self.get_valid_moves(board)
        if not valid_moves:
            return float('-inf')

        head = tuple(board.get_head_location())
        apples = board.get_current_apples()

        ate_apple = np.any(np.all(apples == head, axis=1)) if apples.size > 0 else False
        if ate_apple:
            return 10000.0

        if closest_apple is None:
            return 0.0

        ax, ay = closest_apple
        dist = self.distance_map[head][ay][ax]
        if dist is None:
            return 0.0
        # Stronger reward for proximity, penalize distance
        score = 10000.0 / (dist + 1) - dist  # E.g., dist 1 → 4999, dist 10 → 999.9
        print(f"Eval - Head: {head}, Dist to {closest_apple}: {dist}, Score: {score}")
        return score

    def minimax(self, board: player_board.PlayerBoard, depth: int, maximizing: bool, 
                closest_apple: tuple, alpha: float = float('-inf'), beta: float = float('inf')) -> tuple[float, Action | None]:
        if depth == 0:
            return self.evaluate(board, closest_apple), None

        valid_moves = self.get_valid_moves(board)
        if not valid_moves:
            return float('-inf') if maximizing else float('inf'), None

        best_move = None
        if maximizing:
            max_eval = float('-inf')
            for move in valid_moves:
                next_board, success = board.forecast_action(move)
                if not success:
                    continue
                if board.is_my_turn():
                    next_board.end_turn(reverse=True)
                eval_score, _ = self.minimax(next_board, depth - 1, False, closest_apple, alpha, beta)
                if eval_score > max_eval:
                    max_eval = eval_score
                    best_move = move
                alpha = max(alpha, eval_score)
                if beta <= alpha:
                    break
            return max_eval, best_move
        else:
            min_eval = float('inf')
            for move in valid_moves:
                next_board, success = board.forecast_action(move)
                if not success:
                    continue
                if board.is_enemy_turn():
                    next_board.end_turn(reverse=True)
                eval_score, _ = self.minimax(next_board, depth - 1, True, closest_apple, alpha, beta)
                if eval_score < min_eval:
                    min_eval = eval_score
                    best_move = move
                beta = min(beta, eval_score)
                if beta <= alpha:
                    break
            return min_eval, best_move

    def get_valid_moves(self, board: player_board.PlayerBoard) -> list[Action]:
        return [move for move in board.get_possible_directions() if board.is_valid_move(move)]

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