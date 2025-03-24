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
from multiprocess import Pool

class Player:
    """The abstract class for a player. A player can be an AI agent (bot) or human."""
    def __init__(self, name, game, turn):
        self.name = name
        self.game = game
        self.turn = turn # starting form 0 as convention in python
        assert self.turn < self.game.n_players, "Player turn out of range."

    def get_action(self, state):
        raise NotImplementedError
    
class UCTPlayer(Player):
    """Monte Carlo Tree Search Player (UCT, no prior)"""
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
            return None, None
        if len(legal_actions) == 1:
            return legal_actions[0], 0
        
        # Initialize visit and win counts
        plays, wins = {}, {}
        games = 0
        start_time = time.perf_counter()

        # Run MCTS for the specified thinking time
        while time.perf_counter() - start_time < self.thinking_time:
            self.run_simulation(state, board, plays, wins)
            games += 1

        # Choose the best action based on win rate
        random.shuffle(legal_actions)
        action = max(
            legal_actions,
            key=lambda a: plays.get((player, state, a), 1)
        )
        # print("UCT:", 0, wins.get((player, state, 0), 0) / plays.get((player, state, 0), 1))
        # print("UCT:", 1, wins.get((player, state, 1), 0) / plays.get((player, state, 1), 1))
        # print("Max depth searched:", self.max_depth, "Games played:", games)
        return action, wins.get((player, state, action), 0) / plays.get((player, state, action), 1)

    def run_simulation(self, state, board, plays, wins):
        """Run a single MCTS simulation."""
        tree = set()
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

            tree.add((player, state, action))
            state = board.next_state(state, action)
            player = board.current_player(state)

            # Check for game-ending state
            winner = board.winner(state)
            if winner is not None:
                break

        # === Backpropagation ===
        for player, state, action in tree:
            plays[(player, state, action)] += 1
            if player == winner:
                wins[(player, state, action)] += 1
    
class PUCTPlayer(Player):
    """Monte Carlo Tree Search Player (Prior given by NN)"""
    def __init__(self, game, thinking_time=1, turn=0):
        assert thinking_time > 0
        self.turn = turn
        self.game = game
        self.thinking_time = thinking_time
        self.max_moves = 200
        self.C = 4  # Exploration parameter for PUCT
        self.max_depth = 0
        self.prior = lambda state, action: 1 / len(self.game.legal_actions(state))  # Default prior
        self.value = None

    def get_action(self, state):
        board = self.game
        player = board.current_player(state)
        legal_actions = board.legal_actions(state)
        
        if not legal_actions:
            return None, None
        if len(legal_actions) == 1:
            return legal_actions[0], 0
        
        # Initialize visit and win counts
        plays, wins = {}, {}
        games = 0
        start_time = time.perf_counter()

        # Run MCTS for the specified thinking time
        while time.perf_counter() - start_time < self.thinking_time:
            self.run_simulation(state, board, plays, wins)
            games += 1

        # Choose the best action based on win rate
        random.shuffle(legal_actions)
        action = max(
            legal_actions,
            key=lambda a: ( # choose the action with highest visits
                plays.get((player, state, a), 1)
            )
        )
        return action, wins.get((player, state, action), 0) / plays.get((player, state, action), 1)

    def run_simulation(self, state, board, plays, wins):
        """Run a single MCTS simulation."""
        tree = set()
        player = board.current_player(state)

        # === Selection & Expansion ===
        for t in range(1, self.max_moves + 1):
            legal_actions = board.legal_actions(state)

            # Selection using UCB1 if data exists for all actions
            if all(plays.get((player, state, a)) for a in legal_actions):
                total = sum(plays[(player, state, a)] for a in legal_actions)
                # for a in legal_actions:
                #     print("Check Total:", a, wins[(player, state, a)] / plays[(player, state, a)])
                action = max(
                    legal_actions,
                    key=lambda a: (
                        wins[(player, state, a)] / plays[(player, state, a)] +
                        self.C * self.prior(state, a) * sqrt(total / plays[(player, state, a)])
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

            tree.add((player, state, action))
            state = board.next_state(state, action)
            player = board.current_player(state)

            # Check for game-ending state
            winner = board.winner(state)
            if winner is not None:
                break

        # === Backpropagation ===
        for player, state, action in tree:
            plays[(player, state, action)] += 1
            if player == winner:
                wins[(player, state, action)] += 1

class RLTrainedPlayer(Player):
    def __init__(self, game, turn):
        super().__init__(name="RL", game=game, turn=turn)
        self.model = PolicyValueNet(game.n_players, 128)
        self.model.load_state_dict(torch.load('policy_value_net.pth'))
        self.model.eval()
    
    def get_action(self, state):
        legal_actions = self.game.legal_actions(state)
        
        if not legal_actions:
            return None, None
        if len(legal_actions) == 1:
            return legal_actions[0], 0
        
        M, b = self.game.standard_state(state)
        M = torch.tensor(M, dtype=torch.float32).unsqueeze(0)
        b = torch.tensor(b, dtype=torch.float32).unsqueeze(0)
        policy, value = self.model(M, b)
        action = ACTION_TAKE if policy < 0.3 else ACTION_PASS

        return action, value


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
            return legal_actions[0], None
        
        print("Legal actions:", legal_actions)
        userinput = input("Select your action: (0 for take, 1 for pass) ")
        while userinput not in ["0", "1"]:
            print("Invalid input. Please try again.")
            userinput = input("Select your action: (0 for take, 1 for pass) ")
        
        return int(userinput), None


def self_play(game, players, times=1, to_file=None):
    data = {"state": [], "policy": [], "value": []}
    for _ in tqdm(range(times)):
        state = game.starting_state(current_player=0)
        state = game.pack_state(state)
        current_player = 0

        while not game.is_ended(state):
            player = players[current_player]
            action, score = player.get_action(state)
            state = game.next_state(state, action)
            coins, cards, (card_in_play, coins_in_play, n_cards_in_deck, current_player) = game.unpack_state(state)
            if card_in_play is not None:
                data["state"].append(game.standard_state(state))  # Append (M, b) as NumPy arrays
                data["policy"].append(action)
                data["value"].append(score)

    # Convert NumPy arrays in "state" to lists for JSON serialization
    if to_file is not None:
        serializable_data = {
            "state": [(M.tolist(), b.tolist()) for M, b in data["state"]],
            "policy": data["policy"],
            "value": data["value"]
        }
        with open(to_file, "w") as f:
            json.dump(serializable_data, f)

    return data

def parallel_self_play(args):
    game, players, times, file_prefix, process_id = args
    print(f"Process {process_id} started.")
    return self_play(game, players, times, to_file=f"{file_prefix}_process{process_id}.json")

def parallel_self_play_nosave(args):
    game, players, times, process_id = args
    print(f"Process {process_id} started.")
    return self_play(game, players, times)

def rl_train(rounds=10, from_file=None, num_processes=4):
    game = NoThanksBoard(n_players=3)
    Player_0 = PUCTPlayer(game=game, turn=0)
    Player_1 = PUCTPlayer(game=game, turn=1)
    Player_2 = PUCTPlayer(game=game, turn=2)
    players = [Player_0, Player_1, Player_2]

    batch_size = 32
    n_players = 3
    model = PolicyValueNet(n_players, hidden_dim=128)

    for i in range(rounds):
        print(f"Round {i}: The bots are playing...")

        # Parallelize self_play
        num_games = 4
        games_per_process = num_games // num_processes
        args = [(game, players, games_per_process, f"data_round{i}", process_id) for process_id in range(num_processes)]

        # Use tqdm to track progress
        with Pool(num_processes) as pool:
            with tqdm(total=num_processes) as pbar:
                for _ in pool.imap_unordered(parallel_self_play, args):
                    pbar.update(1)

        # Combine results from all processes
        data = {"state": [], "policy": [], "value": []}
        for process_id in range(num_processes):
            file_name = f"data_round{i}_process{process_id}.json"
            with open(file_name, "r") as f:
                process_data = json.load(f)
                data["state"].extend([(np.array(M), np.array(b)) for M, b in process_data["state"]])
                data["policy"].extend(process_data["policy"])
                data["value"].extend(process_data["value"])

        # Combine and shuffle the data
        combined_data = list(zip(data["state"], data["policy"], data["value"]))
        random.shuffle(combined_data)
        data["state"], data["policy"], data["value"] = zip(*combined_data)

        states = data["state"]
        target_policy = np.array(data["policy"])
        target_value = np.array(data["value"])
        print("Training Data prepared.")
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
            b = np.array([np.array(s[1], dtype=np.float32) for s in batch_states], dtype=np.float32)

            # Train the model on the batch
            model.train()
            print(f"Training batch {batch_idx + 1}/{num_batches}")
            model = train_nn(model, batch_size, n_players, 128, (M, b), batch_policies, batch_values)
            
            # Update prior
            model.eval()
            for player in players:
                player.prior = lambda state, action: (
                    model(
                        torch.tensor(game.standard_state(state)[0], dtype=torch.float32).unsqueeze(0),
                        torch.tensor(game.standard_state(state)[1], dtype=torch.float32).unsqueeze(0)
                    )[0].item() if action == ACTION_PASS else 1 - model(
                        torch.tensor(game.standard_state(state)[0], dtype=torch.float32).unsqueeze(0),
                        torch.tensor(game.standard_state(state)[1], dtype=torch.float32).unsqueeze(0)
                    )[0].item()
                )

    torch.save(model.state_dict(), 'policy_value_net.pth')
    return model

def rl_train_nosave(rounds=10, from_file=None, num_processes=4):
    game = NoThanksBoard(n_players=3)
    Player_0 = PUCTPlayer(game=game, turn=0)
    Player_1 = PUCTPlayer(game=game, turn=1)
    Player_2 = PUCTPlayer(game=game, turn=2)
    players = [Player_0, Player_1, Player_2]

    batch_size = 32
    n_players = 3
    model = PolicyValueNet(n_players, hidden_dim=128)

    for i in range(rounds):
        print(f"Round {i}: The bots are playing...")

        # Parallelize self_play
        num_games = 4
        games_per_process = num_games // num_processes
        args = [(game, players, games_per_process, process_id) for process_id in range(num_processes)]

        # Use Pool to collect data from all processes
        with Pool(num_processes) as pool:
            results = pool.map(parallel_self_play_nosave, args)

        # Combine results from all processes
        data = {"state": [], "policy": [], "value": []}
        for result in results:
            data["state"].extend(result["state"])
            data["policy"].extend(result["policy"])
            data["value"].extend(result["value"])

        # Combine and shuffle the data
        combined_data = list(zip(data["state"], data["policy"], data["value"]))
        random.shuffle(combined_data)
        data["state"], data["policy"], data["value"] = zip(*combined_data)

        states = data["state"]
        target_policy = np.array(data["policy"])
        target_value = np.array(data["value"])
        print("Training Data prepared.")
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
            b = np.array([np.array(s[1], dtype=np.float32) for s in batch_states], dtype=np.float32)

            # Train the model on the batch
            model.train()
            print(f"Training batch {batch_idx + 1}/{num_batches}")
            model = train_nn(model, batch_size, n_players, 128, (M, b), batch_policies, batch_values)
            
            # Update prior
            model.eval()
            for player in players:
                player.prior = lambda state, action: (
                    model(
                        torch.tensor(game.standard_state(state)[0], dtype=torch.float32).unsqueeze(0),
                        torch.tensor(game.standard_state(state)[1], dtype=torch.float32).unsqueeze(0)
                    )[0].item() if action == ACTION_PASS else 1 - model(
                        torch.tensor(game.standard_state(state)[0], dtype=torch.float32).unsqueeze(0),
                        torch.tensor(game.standard_state(state)[1], dtype=torch.float32).unsqueeze(0)
                    )[0].item()
                )

    torch.save(model.state_dict(), 'policy_value_net.pth')
    return model

def play():
    game = NoThanksBoard(n_players = 3)
    Player_0 = PUCTPlayer(game=game, turn=0)
    Player_1 = PUCTPlayer(game=game, turn=1)

    model = PolicyValueNet(game.n_players, 128)
    model.load_state_dict(torch.load('policy_value_net.pth'))
    model.eval()

    new_prior = lambda state, action: (
                    model(
                        torch.tensor(game.standard_state(state)[0], dtype=torch.float32).unsqueeze(0),
                        torch.tensor(game.standard_state(state)[1], dtype=torch.float32).unsqueeze(0)
                    )[0].item() if action == ACTION_PASS else 1 - model(
                        torch.tensor(game.standard_state(state)[0], dtype=torch.float32).unsqueeze(0),
                        torch.tensor(game.standard_state(state)[1], dtype=torch.float32).unsqueeze(0)
                    )[0].item()
                )

    # Player_0.prior = new_prior
    Player_1.prior = new_prior

    # Player_0 = RLTrainedPlayer(game=game, turn=0)
    # Player_1 = RLTrainedPlayer(game=game, turn=1)

    Player_2 = UCTPlayer(game, turn=2)
    players = [Player_0, Player_1, Player_2]

    state = game.starting_state(current_player=0)
    state = game.pack_state(state)
    current_player = 0

    while not game.is_ended(state):
        player = players[current_player]
        action, score = player.get_action(state)
        state = game.next_state(state, action)
        coins, cards, (card_in_play, coins_in_play, n_cards_in_deck, current_player) = game.unpack_state(state)
        game.display_state(state)
        
        # print(game.standard_state(state))
        
    game.display_scores(state)
    winner = game.winner(state)
    print("Game ended. Player", winner, "wins!")

if __name__ == "__main__":
    
    # rl_train()

    # rl_train_nosave()
    
    play()