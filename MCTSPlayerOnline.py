import random
# import datetime
import time
from math import log, sqrt
from no_thanks import NoThanksBoard, NoThanksConfig, ACTION_TAKE, ACTION_PASS

import os, psutil

class MCTSPlayerOnline():
    """Monte Carlo Tree Search Player
    Online only (no pre-training)
    """
    def __init__(self, n_players = 3, thinking_time = 1, order = 0):
        assert thinking_time > 0
        self.order = order
        self.n_players = n_players
        self.thinking_time = thinking_time

        self.config = NoThanksConfig(n_players = self.n_players)

        self.max_moves = 200
        self.C = 1.4 # parameter for exploration formula. Higher C means more exploration.

    def make_state_packed(self, coins, cards, card_in_play, coins_in_play, n_cards_in_deck, current_player):
        details = (card_in_play, coins_in_play, n_cards_in_deck, current_player)
        packed_state = tuple(coins), tuple(map(tuple, cards)), details
        return packed_state
        
    def get_action(self, state, legal_actions):
        self.max_depth = 0

        board = NoThanksBoard(self.n_players, self.config)
        
        player = state[2][3]

        if not legal_actions:
            return
        if len(legal_actions) == 1:
            return legal_actions[0]
        
        plays, wins = {}, {}
        games = 0
        calculation_delta = self.thinking_time
        begin = time.perf_counter()
        while time.perf_counter() - begin < calculation_delta:
            board = NoThanksBoard(self.n_players, self.config)
            plays, wins = self.run_simulation(state, board, plays, wins)
            games += 1

        percent_wins, chosen_action = max(
            (100 * wins.get((player, state, action), 0) / 
             plays.get((player, state, action), 1),
             action)
            for action in legal_actions
        )

        return chosen_action

    def run_simulation(self, state, board, plays, wins):
        """Run a single simulation of MCTS from state."""

        visited_actions = set()
        player = board.current_player(state)

        phase = "selection" # "selection", "expansion", "end_expansion", "backpropagation"

        for t in range(1, self.max_moves + 1):
            legal_actions = board.legal_actions(state)

            if all(plays.get((player, state, action)) for action in legal_actions):
                # if we have stats on all of the legal moves, use them
                log_total = log(
                    sum(plays[(player, state, action)] for action in legal_actions))
                value, action = max(
                    ((wins[(player, state, action)] / plays[(player, state, action)]) + self.C *
                        sqrt(log_total / plays[(player, state, action)]), action)
                        for action in legal_actions
                )
            else:
                if phase == "selection":
                    phase = "expansion"
                # otherwise, just pick a random one
                action = random.choice(legal_actions)

            # states_copy.append(state)

            if phase == "expansion" and (player, state, action) not in plays:
                plays[(player, state, action)] = 0
                wins[(player, state, action)] = 0
                if t > self.max_depth:
                    self.max_depth = t
                phase = "end_expansion"

            if phase == "selection" or "phase" == "expansion":
                visited_actions.add((player, state, action))
            elif phase == "end_expansion":
                visited_actions.add((player, state, action))
                phase = "simulation"

            # move to next state
            state = board.next_state(state, action)

            player = board.current_player(state)
            winner = board.winner(state)
            if winner is not None:
                break

        phase = "backpropagation"
        for player, state, action in visited_actions:
            plays[(player, state, action)] += 1
            if player == winner:
                wins[(player, state, action)] += 1

        return plays, wins


class HumanPlayer():
    def __init__(self, n_players = 3, order = 0):
        self.order = order
        self.n_players = n_players
        self.config = NoThanksConfig(n_players = self.n_players)

    def get_action(self, state, legal_actions):
        """Get the action from the human player."""
        board = NoThanksBoard(self.n_players, self.config)
        # board.display_state(state)
        print("Legal actions:", legal_actions)
        userinput = input("Select your action: (0 for take, 1 for pass) ")
        assert userinput in ["0", "1"]
        assert int(userinput) in legal_actions
        
        return int(userinput)
    

if __name__ == "__main__":
    Player_0 = MCTSPlayerOnline(n_players=5, thinking_time=1, order=0)
    Player_1 = MCTSPlayerOnline(n_players=5, thinking_time=1, order=1)
    Player_2 = HumanPlayer(n_players=5, order=2)
    Player_3 = MCTSPlayerOnline(n_players=5, thinking_time=1, order=3)
    Player_4 = MCTSPlayerOnline(n_players=5, thinking_time=1, order=4)
    game = NoThanksBoard(n_players = 5)
    state = game.starting_state(current_player=0)
    state = game.pack_state(state)
    current_player = 0

    for player in [Player_0, Player_1, Player_2, Player_3, Player_4]:
        while current_player == player.order:
            if game.is_ended(state):
                break
                break
            state = game.next_state(state, player.get_action(state, game.legal_actions(state)))
            coins, cards, (card_in_play, coins_in_play, n_cards_in_deck, current_player) = game.unpack_state(state)
            game.display_state(state, human_player=2)
            
    
    game.display_scores(state)
    game.winner(state)
    print("Game ended. Player", game.winner(state), "wins!")