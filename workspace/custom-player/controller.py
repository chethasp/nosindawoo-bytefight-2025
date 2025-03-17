from game import player_board
from game.enums import Action
from collections.abc import Callable
import random

from collections import deque

class PlayerController:
    # for the controller to read
    def __init__(self, time_left: Callable):
        self.time_left = time_left
        self.board_width = None
        self.board_height = None
        self.portal_mask = None
        self.wall_mask = None
        self.future_apples_deque = None
        self.distance_map = None
        self.pairwise_distances_precomputed = False;


    def bid(self, board:player_board.PlayerBoard, time_left:Callable):
        self.board_width = board.get_dim_x() # num of cols 
        self.board_height = board.get_dim_y() # num of rows

        self.portal_mask = board.get_portal_mask(descriptive=True)
        self.wall_mask = board.get_wall_mask()

        future_apples = board.get_future_apples()
        future_apples.sort(key=lambda x: x[0]) #sort future apples according to time
        self.future_apples_deque = deque(future_apples)

        return 0 #return value for bid length


    def play(self, board:player_board.PlayerBoard, time_left:Callable):
        if not self.pairwise_distances_precomputed: 
            self.precompute_pairwise_distances(board)

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
    
    
    def precompute_pairwise_distances(self, board: player_board.PlayerBoard): 
        self.distance_map = {} # (start_x, start_y) -> {(target_x, target_y): distance}
        directions = {(-1, 0), (1, 0), (0, 1), (0, -1), (-1, -1), (1, 1), (-1, 1), (1, -1)}

        for x in range(self.board_width): 
            for y in range(self.board_height): #iterating over all possible start points in the grid
                start = (x, y)
                
                if self.wall_mask[y, x]:
                    self.distance_map[start] = {}  # empty dict for invalid start (start is a wall)
                    continue

                distances = [[None for _ in range(self.board_width)] for _ in range(self.board_height)] #initialize everything with None

                queue = deque([(x, y, 0)])

                while(queue): 
                    fx,fy,dist = queue.popleft()
                    distances[fy][fx] = dist

                    for dx, dy in directions: 
                        nx, ny = fx + dx, fy + dy
                        
                        if(nx < 0 or nx >= self.board_width or ny < 0 or ny >= self.board_height): #out of bounds
                            continue

                        if (distances[ny][nx] is not None): #wall (-1) or alr visited
                            continue

                        #check if wall 
                        if self.wall_mask[ny, nx]: 
                            distances[ny][nx] = -1
                            continue
                    
                        #check if portal, also add the other end of the portal 
                        px, py = self.portal_mask[ny, nx]
                        if(px != -1 and py != -1): #this is a portal 
                            queue.append((px, py, dist))
                        else:  #a regular cell
                            queue.append((nx, ny, dist + 1))

                self.distance_map[start] = distances
