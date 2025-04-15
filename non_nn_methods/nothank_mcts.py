import random
import numpy as np
import heapq
from tqdm import tqdm
import copy
from utils import *

class mcts():
    """This is where the MCTS happens"""
    def __init__(self, 
                 depth = 20):
        """Depth: limit on the number of card flip to look in the future. Not used atm
        """
        self.depth = depth
        self.simulation_iter = 100

    def get_best_uct(self, child_list):
        """Given a list of nodes, get the nodes with the highest uct
        child_list: list of game_state nodes
        """
        index_max = np.argmax([i[0] for i in child_list])
        return child_list[index_max]
    
    def simulation(self, game, iters = 100) -> list[float]:
        """Simulation stage of MCTS. Let the game play by itself iter time and return the accumulated result
        """

        score_list = np.array([0.]*game.n_player)
        for iter_no in range(iters):
            game_tmp = copy.deepcopy(game)            
            score_list_tmp = game_tmp.self_play()
            score_list += score_list_tmp
        return score_list


    def selection(self, game_node, game):
        score_list, total_explored = np.array([0.]*game.n_player), 0
        if len(game_node.child) == 0:
            # EXPANSION
            legal_move = game.get_legal_action()
            for move in legal_move:     
                turn_tmp = game.next_player(move)
                next_node = game_state(parent = game_node,
                                        turn = turn_tmp,
                                       depth = game_node.depth + 1
                                       )
                uct_tmp = next_node.calculate_uct()
                heapq.heappush(game_node.child, [uct_tmp, move, next_node])

            # SIMULATION
            total_explored = 0
            for _, move, child in game_node.child:
                game_clone = copy.deepcopy(game)
                game_clone.action(move)
                score_list_tmp = self.simulation(game_clone, self.simulation_iter)
                child.simulation_score = score_list_tmp
                child.n_explored += self.simulation_iter
                child.win += score_list_tmp[child.parent.turn]

                score_list += score_list_tmp
                total_explored += self.simulation_iter
            
            # Update the current node here too
            game_node.n_explored += total_explored
            if game_node.parent:
                game_node.win += score_list[game_node.parent.turn]
            return score_list, total_explored 

        else:
            #SELECTION
            for child in game_node.child:
                child[0] = child[-1].calculate_uct()
            
            _, move, next_node = self.get_best_uct(game_node.child)
            game_clone = copy.deepcopy(game)
            game_clone.action(move)
            score_list, total_explored = self.selection(next_node, game_clone)
            
            #BACKPROPAGAION
            game_node.n_explored += total_explored
            if game_node.parent:
                game_node.win += score_list[game_node.parent.turn]

            return score_list, total_explored
        
if __name__ == "__main__":
    nothanks = game(3)
    game_node = game_state()
    tree = mcts()

    
    print(nothanks)
    n_selection = 50
    human_player_turn = 0
    move_encode = {"0": "pass",
                "1": "take"}

    while nothanks.is_continue:
        print('------------------------------')
        print(f'''Card: {nothanks.current_card} | Chip in pot: {nothanks.chip_in_pot} | Player: {nothanks.turn} - {nothanks.players[nothanks.turn]}\n'''
)
        print('------------------------------')
        if nothanks.turn == human_player_turn:
            move = move_encode.get(input("""Your turn:
0: pass
1: take
Enter here: """))
        else:
            for _ in tqdm(range(n_selection)):
                tree.selection(game_node= game_node, game= nothanks)
            move = game_node.get_best_move() 
        
        print(f"""Move taken: {move}\n""")
        nothanks.action(move)
        game_node = game_state(turn = nothanks.turn)
    
    for user in nothanks.players:
        print(user, user.calculate_score())