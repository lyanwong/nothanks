import random
import time
from math import log, sqrt
import numpy as np
from game import NoThanksBoard, NoThanksConfig, ACTION_TAKE, ACTION_PASS
from collections import deque
from network import PolicyValueNet, train_nn
import torch
import json
from tqdm import tqdm

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
        self.reward = self.game.reward_rank
        self.utility = self.pucb
        self.delay = 1 # self.game.n_players - 1
        self.prior = lambda s, a: 0.5
    
    def get_action(self, state):
        legal_actions = self.game.legal_actions(state)

        assert legal_actions, "No legal actions available."
        if len(legal_actions) == 1:
            return legal_actions[0]

        visited = self.simulate(state, prior=self.prior)
        random.shuffle(legal_actions)
        action = max(legal_actions, key=lambda a: visited[(state, a)].avg_reward[self.game.current_player(state)] if (state, a) in visited.keys() else 0)
        print(f"Action: {action}, Simulated Frequencies: Takes: {visited[(state, 0)].visits if (state, 0) in visited.keys() else 0}, Passes: {visited[(state, 1)].visits if (state, 1) in visited.keys() else 0}")
        print(f"PUCB scores: Takes: {self.utility(visited[(state, 0)]) if (state, 0) in visited else 0}, Passes: {self.utility(visited[(state, 1)]) if (state, 1) in visited else 0}")
        print(f"Rewards: Takes: {visited[(state, 0)].avg_reward if (state, 0) in visited else 0}, Passes: {visited[(state, 1)].avg_reward if (state, 1) in visited else 0}")
        return action

    def self_play(self, prior, to_save=None):
        state = self.game.starting_state(current_player=0)
        state = self.game.pack_state(state)
        visited = self.simulate(state, prior=self.prior, simNum=1000, max_moves=float("inf"))

        # Initialize data structures
        data = {"state": [], "action": [], "visits": [], "value": []}
        state_stats = {}

        # Collect data from visited edges
        for (state, action), edge in visited.items():
            data["state"].append(state)
            data["action"].append(action)
            data["visits"].append(edge.visits)
            data["value"].append(edge.avg_reward)

            # Aggregate statistics for each state
            if state not in state_stats:
                state_stats[state] = {
                    "total_visits": 0,
                    "take_visits": 0,
                    "take_value": 0,
                    "pass_value": 0,
                }
            state_stats[state]["total_visits"] += edge.visits
            if action == ACTION_TAKE:
                state_stats[state]["take_visits"] += edge.visits
                state_stats[state]["take_value"] += edge.avg_reward
            elif action == ACTION_PASS:
                state_stats[state]["pass_value"] += edge.avg_reward

        # Compute policy and value for each unique state
        data_clean = {"state": [], "policy": [], "value": []}
        for state, stats in state_stats.items():
            total_visits = stats["total_visits"]
            frequency = stats["take_visits"] / total_visits if total_visits > 0 else 0
            value_take = stats["take_value"] / total_visits if total_visits > 0 else 0
            value_pass = stats["pass_value"] / total_visits if total_visits > 0 else 0
            value = value_take * frequency + value_pass * (1 - frequency)

            data_clean["state"].append(self.game.standard_state(state))
            data_clean["policy"].append(frequency)
            data_clean["value"].append(value)

        # Save data if required
        if to_save is not None:
            with open(to_save, "w") as f:
                json.dump(data_clean, f)

        return data_clean
    
    def update_prior(self, new_prior):
        self.prior = new_prior

    def simulate(self, state, prior, simNum=3000, max_moves=100):
        """Simulate a game from the current state.
        Args:
            state: the current state of the game
            prior: the prior probability function of the actions
        """
        visited = {}  # (state, action) -> Edge

        for sim in tqdm(range(simNum)):
            frontier = deque([state])
            current = frontier.pop()  # the current STATE
            prev = [None]  # the trajectory of EDGEs from the current state to the leaf state
            # print("Simulating game", sim, "visited", len(visited))
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
                    random.shuffle(legal_actions)
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
    
    def rl_train(self, times=1):
        
        batch_size = 256
        model = PolicyValueNet(self.game.n_players, hidden_dim=128)

        for i in range(times):
            print(f"Training model {i + 1}/{times}")

            data = self.self_play(prior=self.prior, to_save=None)
            states = data["state"]
            target_policy = np.array(data["policy"])
            target_value = np.array(data["value"])

            num_samples = len(states)
            num_batches = num_samples // batch_size
            print(f"Total samples: {num_samples}, Batches: {num_batches}")
            
            for batch_idx in range(num_batches):
                # Extract batch data
                batch_start = batch_idx * batch_size
                batch_end = batch_start + batch_size
            
                batch_states = states[batch_start:batch_end]
                batch_policies = target_policy[batch_start:batch_end]
                batch_values = target_value[batch_start:batch_end]
            
                # Reshape states into (M, b) format
                M = np.array([s[0] for s in batch_states])
                b = np.array([s[1] for s in batch_states])

                # print(f"Input M shape: {M.shape}, Input b shape: {b.shape}")
                # print(f"Predicted value shape: {batch_policies.shape}, Target value shape: {batch_values.shape}")
                # Train the model on the batch
                model.train()
                print(f"Training batch {batch_idx + 1}/{num_batches}")
                model = train_nn(model, batch_size, self.game.n_players, 128, (M, b), batch_policies, batch_values)
            
            # Update prior
            model.eval()
            new_prior = lambda s, a: model(
                torch.tensor(self.game.standard_state(s)[0], dtype=torch.float32).unsqueeze(0),  # Add batch dimension
                torch.tensor(self.game.standard_state(s)[1], dtype=torch.float32).unsqueeze(0)   # Add batch dimension
            )[0].item() if a == ACTION_TAKE else 1 - model(
                torch.tensor(self.game.standard_state(s)[0], dtype=torch.float32).unsqueeze(0),  # Add batch dimension
                torch.tensor(self.game.standard_state(s)[1], dtype=torch.float32).unsqueeze(0)   # Add batch dimension
            )[0].item()
            self.update_prior(new_prior)

        torch.save(model.state_dict(), 'policy_value_net.pth')
        return model

class RLTrainedPlayer(Player):
    def __init__(self, name, game, turn):
        super().__init__(name, game, turn)
        self.model = PolicyValueNet(game.n_players, 128)
        self.model.load_state_dict(torch.load('policy_value_net.pth'))
        self.model.eval()
    
    def get_action(self, state):
        M, b = self.game.standard_state(state)
        M = torch.tensor(M, dtype=torch.float32)
        b = torch.tensor(b, dtype=torch.float32)
        policy, _ = self.model(M, b)
        action = ACTION_TAKE if policy > 0.5 else ACTION_PASS

        return action

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

        print("Max depth searched:", self.max_depth, "Games played:", games)
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

def play():
    game = NoThanksBoard(n_players = 3)
    Player_0 = MCTSPlayerOnline(game=game, thinking_time=1, turn=0)
    Player_1 = MCTSPlayerOnline(game=game, thinking_time=1, turn=1)
    Player_2 = HumanPlayer("Human", game, turn=2)
    players = [Player_0, Player_1, Player_2]

    state = game.starting_state(current_player=0)
    state = game.pack_state(state)
    current_player = 0

    while not game.is_ended(state):
        player = players[current_player]
        action = player.get_action(state)
        state = game.next_state(state, action)
        coins, cards, (card_in_play, coins_in_play, n_cards_in_deck, current_player) = game.unpack_state(state)
        game.display_state(state)
        
        # print(game.standard_state(state))
        
    game.display_scores(state)
    winner = game.winner(state)
    print("Game ended. Player", winner, "wins!")

if __name__ == "__main__":
    game = NoThanksBoard(n_players = 3)

    Player = MCTSPlayer("MCTS", game, turn=0)
    Player.rl_train(times=10)