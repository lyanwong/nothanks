import random
import time
from math import log, sqrt
import numpy as np
from no_thanks import NoThanksBoard, NoThanksConfig, ACTION_TAKE, ACTION_PASS
from collections import defaultdict, deque

class Player:
    """The abstract class for a player. A player can be an AI agent (bot) or human."""
    def __init__(self, name, game, turn):
        self.name = name
        self.game = game
        self.turn = turn # starting form 0 as convention in python
        assert self.turn < self.game.n_players, "Player turn out of range."

    def get_action(self, state):
        raise NotImplementedError
    
class Edge:
    """An edge in the MCTS tree. Connects a parent state-action pair to a child state."""
    def __init__(self, n_players, parent, state, action, prior):
        self.n_players = n_players
        self.parent = parent
        self.state = state
        self.action = action
        self.visits = 1 # N(s, a)
        self.total_reward = np.zeros(self.n_players) # W(s, a)
        self.avg_reward = np.zeros(self.n_players) # Q(s, a)
        self.prior = prior # P(s, a)
        
    
    def __str__(self):
        return f"Edge: state={self.state}, action={self.action}, visits={self.visits}, total_reward={self.total_reward}, avg_reward={self.avg_reward}, prior={self.prior}"
    
    def update(self, reward):
        self.visits += 1
        self.total_reward += reward
        self.avg_reward = self.total_reward / self.visits
    
    def update_prior(self, new_prior):
        self.prior = new_prior

    def is_root(self):
        return self.parent == None


class MCTSPlayer(Player):
    def __init__(self, name, game, turn):
        super().__init__(name, game, turn)
        self.lambda_ = 0
        self.reward = self.game.reward_winloss
        self.utility = self.ucb
        self.delay = 1 # self.game.n_players - 1
    
    def get_action(self, state):
        legal_actions = self.game.legal_actions(state)

        assert legal_actions, "No legal actions available."
        if len(legal_actions) == 1:
            return legal_actions[0]

        visited = self.simulate(state, prior=lambda s, a: 0.49 if a == ACTION_TAKE else 0.51)
        random.shuffle(legal_actions)
        action = max(legal_actions, key=lambda a: visited[(state, a)].avg_reward[self.game.current_player(state)] if (state, a) in visited.keys() else 0)
        print(f"Action: {action}, Simulated Frequencies: Takes: {visited[(state, 0)].visits if (state, 0) in visited.keys() else 0}, Passes: {visited[(state, 1)].visits if (state, 1) in visited.keys() else 0}")
        print(f"PUCB scores: Takes: {self.utility(visited[(state, 0)]) if (state, 0) in visited else 0}, Passes: {self.utility(visited[(state, 1)]) if (state, 1) in visited else 0}")
        # print(f"Priors: Takes: {visited[(state, 0)].prior}, Passes: {visited[(state, 1)].prior}")
        print(f"Rewards: Takes: {visited[(state, 0)].avg_reward if (state, 0) in visited else 0}, Passes: {visited[(state, 1)].avg_reward if (state, 1) in visited else 0}")
        return action


    def simulate(self, state, prior, simNum=500, max_moves=200):
        """Simulate a game from the current state.
        Args:
            state: the current state of the game
            prior: the prior probability function of the actions
        """
        visited = {}  # (state, action) -> Edge

        for sim in range(simNum):
            frontier = deque([state])
            current = frontier.pop()  # the current STATE
            prev = [None]  # the trajectory of EDGEs from the current state to the leaf state

            while len(prev) < max_moves and not self.game.is_ended(current):
                legal_actions = self.game.legal_actions(current)
                if not legal_actions:
                    break

                ## SELECT
                # Select the action with the highest utility
                # print("State:", current, "PUCT scores:", [self.utility(visited[(current, a)], sim) for a in legal_actions])
                if not visited: # If the tree is empty, select a random action
                    action = random.choices(legal_actions, weights=[prior(current, a) for a in legal_actions])[0]
                    visited[(current, action)] = Edge(self.game.n_players, prev[-1], current, action, prior(current, action))
                else:
                    action = max(legal_actions, key=lambda a: self.utility(visited[(current, a)], sim)[self.game.current_player(current)] if (current, a) in visited.keys() else 0)
                    if (current, action) not in visited.keys():
                        visited[(current, action)] = Edge(self.game.n_players, prev[-1], current, action, prior(current, action))
                    
                ## EXPAND
                # print("Action:", action)
                next_state = self.game.next_state(current, action)
                frontier.append(next_state)
                prev.append(visited[(current, action)])
                current = next_state

            ## TERMINAL EVALUATE
            terminal_reward = np.array(self.reward(current)) 
            
            ## BACKPROPAGATE
            if len(prev) - self.delay > 0:
                for i in range(len(prev) - self.delay):  
                    edge = prev[i]
                    future = prev[i + self.delay].state  
                    if edge is not None:
                        if (edge.state, edge.action) in visited:
                            temp_reward = self.reward(future)
                            reward = self.lambda_ * temp_reward + (1 - self.lambda_) * terminal_reward
                            visited[(edge.state, edge.action)].update(reward)
            
                for i in range(len(prev) - self.delay, len(prev)):
                    edge = prev[i]
                    if edge is not None:
                        if (edge.state, edge.action) in visited:
                            visited[(edge.state, edge.action)].update(terminal_reward)
            else :
                for i in range(len(prev)):
                    edge = prev[i]
                    if edge is not None:
                        if (edge.state, edge.action) in visited:
                            visited[(edge.state, edge.action)].update(terminal_reward)

        return visited
    
    def pucb(self, edge, sim=1000):
        """Compute the PUCB score of an edge."""
        C = 1.5
        if edge.is_root():  # Handle the root edge
            return edge.avg_reward + C * edge.prior * sqrt(sim) / (1 + edge.visits)

        return edge.avg_reward + C * edge.prior * sqrt(edge.parent.visits) / (1 + edge.visits)

    def ucb(self, edge, sim=1000):
        """Compute the UCB score of an edge."""
        C = 1.4
        if edge.is_root():
            return edge.avg_reward + C * sqrt(log(sim) / edge.visits)
        return edge.avg_reward + C * sqrt(log(edge.parent.visits) / edge.visits)
    

class MCTSPlayerOnline(Player):
    """Monte Carlo Tree Search Player (Online only, no pre-training)"""
    def __init__(self, game, thinking_time=1, turn=0):
        assert thinking_time > 0
        self.turn = turn
        self.game = game
        self.thinking_time = thinking_time
        self.max_moves = 200
        self.C = 1.4  # Exploration parameter for UCB1
        self.max_depth = 0

    def get_action(self, state):
        board = self.game
        player = board.current_player(state)
        legal_actions = board.legal_actions(state)

        if not legal_actions:
            return None
        if len(legal_actions) == 1:
            return legal_actions[0]
        
        # Initialize visit and win counts
        plays, wins = {}, {}
        games = 0
        start_time = time.perf_counter()

        # Run MCTS for the specified thinking time
        while time.perf_counter() - start_time < self.thinking_time:
            self.run_simulation(state, board, plays, wins)
            games += 1

        # Choose the best action based on win rate
        action = max(
            legal_actions,
            key=lambda a: wins.get((player, state, a), 0) / plays.get((player, state, a), 1)
        )
        return action

    def run_simulation(self, state, board, plays, wins):
        """Run a single MCTS simulation."""
        visited = set()
        player = board.current_player(state)

        # === Selection & Expansion ===
        for t in range(1, self.max_moves + 1):
            legal_actions = board.legal_actions(state)

            # Selection using UCB1 if data exists for all actions
            if all(plays.get((player, state, a)) for a in legal_actions):
                log_total = log(sum(plays[(player, state, a)] for a in legal_actions))
                action = max(
                    legal_actions,
                    key=lambda a: (
                        wins[(player, state, a)] / plays[(player, state, a)] +
                        self.C * sqrt(log_total / plays[(player, state, a)])
                    )
                )
            else:
                # Expansion – If any action is unexplored, take a random one
                action = random.choice(legal_actions)
                if (player, state, action) not in plays:
                    plays[(player, state, action)] = 0
                    wins[(player, state, action)] = 0
                    if t > self.max_depth:
                        self.max_depth = t

            visited.add((player, state, action))
            state = board.next_state(state, action)
            player = board.current_player(state)

            # Check for game-ending state
            winner = board.winner(state)
            if winner is not None:
                break

        # === Backpropagation ===
        for player, state, action in visited:
            plays[(player, state, action)] += 1
            if player == winner:
                wins[(player, state, action)] += 1




class HumanPlayer(Player):
    def __init__(self, name, game, turn):
        super().__init__(name, game, turn)

    def change_name(self):
        name = input("Enter your name: ")
        self.name = name
        
    def get_action(self, state):
        """Get the action from the human player."""
        
        legal_actions = self.game.legal_actions(state)
        if len(legal_actions) == 1:
            return legal_actions[0]
        
        print("Legal actions:", legal_actions)
        userinput = input("Select your action: (0 for take, 1 for pass) ")
        while userinput not in ["0", "1"]:
            print("Invalid input. Please try again.")
            userinput = input("Select your action: (0 for take, 1 for pass) ")
        
        return int(userinput)
    



if __name__ == "__main__":
    game = NoThanksBoard(n_players = 3)

    # Player_0 = MCTSPlayerOnline(game=game, thinking_time=1, turn=0)
    # Player_1 = MCTSPlayerOnline(game=game, thinking_time=1, turn=1)
    # Player_2 = HumanPlayer("Human", game, turn=2)
    # Player_3 = MCTSPlayerOnline(game=game, thinking_time=1, turn=3)
    # Player_4 = MCTSPlayerOnline(game=game, thinking_time=1, turn=4)
    # players = [Player_0, Player_1, Player_2, Player_3, Player_4]

    Player_0 = MCTSPlayer(name=None, game=game, turn=0)
    Player_1 = MCTSPlayer(name=None, game=game, turn=1)
    Player_2 = MCTSPlayerOnline(game=game, thinking_time=1, turn=2)
    # Player_2 = HumanPlayer("Human", game, turn=2)
    players = [Player_0, Player_1, Player_2]

    state = game.starting_state(current_player=0)
    state = game.pack_state(state)
    current_player = 0

    while not game.is_ended(state):
        player = players[current_player]
        action = player.get_action(state)
        state = game.next_state(state, action)
        coins, cards, (card_in_play, coins_in_play, n_cards_in_deck, current_player) = game.unpack_state(state)
        game.display_state(state, human_player=2)
        
        # print(game.standard_state(state))
        
    game.display_scores(state)
    winner = game.winner(state)
    print("Game ended. Player", winner, "wins!")
