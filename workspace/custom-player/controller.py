from game import player_board
from game.enums import Action
from collections.abc import Callable
import random
class PlayerController:
    # for the controller to read
    def __init__(self, time_left: Callable):
        return

    def bid(self, board:player_board.PlayerBoard, time_left:Callable):
        return 0

    def play(self, board:player_board.PlayerBoard, time_left:Callable):
        possible_moves = board.get_possible_directions()
        final_moves = []
        for move in possible_moves:
            if(board.is_valid_move(move)):
                final_moves.append(move)

        final_move = random.choice(final_moves)

        final_turn = [final_move]

        if(len(final_turn) > 0):
            return final_turn
        return Action.FF