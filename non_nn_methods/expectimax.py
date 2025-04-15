import random
import numpy as np
import copy
from utils import *

class expectimax_agent:
    def __init__(self, depth, player_index):
        self.depth = depth
        self.player_index = player_index
    
    def get_action_value(self, game, move, depth):
        game_clone = copy.deepcopy(game)
        if move == 'pass':
            # NORMAL NODE
            game_clone.action(move)
            return self.expectimax(game_clone, depth - 1)
        if move == 'take':
            # CHANCE NODE
            remain_card = game_clone.remain_card # meaning there's no card left
            if len(remain_card) == game.n_remove_card:
                # if game is over, return the score
                return game_clone.players[self.player_index].calculate_score()
            #loop through all remaining cards
            # calculate and average the value over each child
            for card in remain_card:
                game_clone = copy.deepcopy(game)
                game_clone.action(move, card)

                total_value = 0
                total_value += self.expectimax(game_clone, depth - 1)
            return total_value/len(remain_card)
    
    def expectimax(self, game, depth):
        # base case
        if depth == 0:
            return game.players[self.player_index].calculate_score()
        legal_move = game.get_legal_action()
        # if there's no legal move left, this shouldnt happen though, get action value caught all of it
        if len(legal_move) == 0:
            return game.players[self.player_index].calculate_score()
        if game.turn == self.player_index: 
            # MAX NODE
            max_value = -np.inf
            for move in legal_move:
                value_tmp = self.get_action_value(game, move, depth)
                max_value = max(max_value, value_tmp)
            return max_value
        else:
            # EXPECTATION NODE
            total_value = 0
            for move in legal_move:
                total_value += self.get_action_value(game, move, depth)
            return total_value/len(legal_move)
    
    def get_best_action(self, game):
        legal_move = game.get_legal_action()
        value = -np.inf
        move = None
        for move_tmp in legal_move:
            value_tmp = self.get_action_value(game, move_tmp, self.depth)
            if value_tmp > value:
                value = value_tmp
                move = move_tmp
        return move


if __name__ == '__main__':
    move_encode = {"0": "pass",
                "1": "take"}
    nothanks = game(3)
    player_index = 0
    agent = expectimax_agent(depth = 3, 
                             player_index = player_index)

    while nothanks.is_continue:
        print('------------------------------')
        print(f'''Card: {nothanks.current_card} | Chip in pot: {nothanks.chip_in_pot} | Player: {nothanks.turn} - {nothanks.players[nothanks.turn]}\n'''
    )
        print('------------------------------')
        if nothanks.turn == player_index:
            move = agent.get_best_action(nothanks)
            print(f"Agent action: {move}")
            print('------------------------------')
        else:
            move = move_encode.get(input("""Your turn:
0: pass
1: take
Enter here: """))
        
        nothanks.action(move)