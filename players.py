import time
import random
import numpy as np
from math import sqrt, log
from collections import defaultdict
from abc import ABC, abstractmethod
from itertools import chain
import copy
from game import NoThanksBoard, NoThanksConfig, ACTION_TAKE, ACTION_PASS
import torch
from network import PolicyValueNet, train_nn, ValueNet, train_expvalue
# from policy_net import PolicyOnlyNet
import json
# from log_progress import log_progress
from tqdm import tqdm
from non_nn_methods.utils import *
from non_nn_methods.ppo_model import *


# from matplotlib import pyplot as plt

bar = tqdm

class Player(ABC):
    """The abstract class for a player. A player can be an AI agent (bot) or human."""
    def __init__(self, game, turn):
        self.name = "Player " + str(turn)
        self.game = game
        self.turn = turn  # starting form 0 as convention in python
        assert self.turn < self.game.n_players, "Player turn out of range."

    @abstractmethod
    def get_action(self, state):
        pass

class RandomPlayer(Player):
    def __init__(self, game, turn):
        super().__init__(game, turn)

    def get_action(self, state):
        """Get a random action from the legal actions."""
        legal_actions = self.game.legal_actions(state)
        if len(legal_actions) == 1:
            return legal_actions[0], None
        return random.choice(legal_actions), None
    
class RuleBasedPlayer(Player):
    def __init__(self, game, turn):
        super().__init__(game, turn)
        self.prior = smart_prior_fn(game, p=0.9)

    def get_action(self, state):
        """Get the action from the rule-based player."""
        legal_actions = self.game.legal_actions(state)
        if len(legal_actions) == 1:
            return legal_actions[0], None
        prob = []
        for action in legal_actions:
            prob.append(self.prior(state, action))
        action = np.random.choice(legal_actions, p=prob)
        return action, None

class HumanPlayer(Player):
    def __init__(self, game, turn):
        super().__init__(game, turn)

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

class ExpectimaxPlayer(Player):
    def __init__(self, game, turn, depth=3, use_cache=True):
        super().__init__(game, turn)
        self.depth = depth
        self.cache = {} # store visited nodes and their values
        self.cache_hit = 0
        self.use_cache = use_cache
        self.trained = False

    def get_action(self, state):
        # Choose the best action using expectimax
        best_value = float('-inf')
        best_action = None
        for action in self.game.legal_actions(state):
            next_state = self.game.all_possible_next(state, action)
            value = self.expectimax(next_state, self.depth, perspective=self.turn)
            if value > best_value:
                best_value = value
                best_action = action
        # print(f"Best action: {best_action}, Value: {best_value}")
        return best_action, best_value

    def expectimax(self, state_list, depth, perspective):
        """Recursive Expectimax function.

        Args:
            state_list: list of (probability, state) pairs.
            depth: remaining search depth.
            perspective: the player index to evaluate from.

        Returns:
            Expected value (float) from the perspective player's point of view.
        """
        # Base case: depth limit — evaluate all outcomes and return expected value
        if depth == 0 and len(state_list) >= 1:
            expected_value = 0.0
            for prob, state in state_list:
                packed = self.game.pack_state(state)
                key = (packed, perspective)
                if self.use_cache and key in self.cache:
                    value = self.cache[key]
                    self.cache_hit += 1
                else:
                    value = self.evaluate(state, perspective)
                    self.cache[key] = value
                expected_value += prob * value
            # print(f"State: {state}, Depth {depth}, Expected value: {expected_value} for Player {perspective}")
            return expected_value

        # Stochastic node: multiple possible outcomes
        if len(state_list) > 1:
            expected_value = 0.0
            for prob, state in state_list:
                expected_value += prob * self.expectimax([(1.0, state)], depth - 1, perspective)
            # print(f"State: {state_list}, Depth {depth}, Expected value: {expected_value} for Player {perspective}")
            return expected_value

        # Deterministic node
        # print(f"State: {state_list}, Depth {depth}, Expected value: {0} for Player {perspective}")
        prob, state = state_list[0]
        packed = self.game.pack_state(state)
        key = (packed, perspective)

        # Use cache if available
        if self.use_cache and key in self.cache:
            self.cache_hit += 1
            return self.cache[key]

        # Terminal game state
        if self.game.is_ended(state):
            value = self.evaluate_untrained(state, perspective)
            self.cache[key] = value
            return value

        current_player = self.game.current_player(state)
        legal_actions = self.game.legal_actions(state)
        
        # Maximize if it's our turn
        if current_player == self.turn:
            value = max(
                self.expectimax(self.game.all_possible_next(state, action), depth - 1, self.turn)
                for action in legal_actions
            )
        # The opponents maximize their value
        else:
            opponent_action = max(
                legal_actions, 
                key=lambda a: self.expectimax(self.game.all_possible_next(state, a), depth - 1, current_player)
                )
            value = self.expectimax(self.game.all_possible_next(state, opponent_action), depth - 1, self.turn)

        self.cache[key] = value
        # print(f"State: {state}, Depth {depth}, Expected value: {value} for Player {perspective}")
        return value

    def evaluate(self, state, perspective):
        if self.trained:
            return self.evaluate_trained(state, perspective)
        else:
            return self.evaluate_untrained(state, perspective)

    def evaluate_untrained(self, state, perspective):
        if state is None:
            return 0.0
        score = self.game.compute_scores(state)
        return -score[perspective]
    
    def evaluate_trained(self, state, perspective, load_model='value_net_expectimax.pth'):
        if state == None:
            return 0.0
        coins, cards, (card_in_play, coins_in_play, n_cards_in_deck, current_player) = self.game.unpack_state(state)
        if card_in_play is None:
            return self.evaluate_untrained(state, perspective)
        model = ValueNet(self.game.n_players, hidden_dim=128)
        model.eval()
        model.load_state_dict(torch.load(load_model))
        with torch.no_grad():
            M, b = self.game.standard_state(state)
            M_tensor = torch.tensor(M, dtype=torch.float32).unsqueeze(0)
            b_tensor = torch.tensor(b, dtype=torch.float32).unsqueeze(0)
            value = model(M_tensor, b_tensor).item()
        return value


class BaseMCTSPlayer(Player, ABC):
    def __init__(self, game, turn, thinking_time=1, simNum=0, max_moves=200):
        super().__init__(game, turn)
        self.thinking_time = thinking_time
        self.simNum = simNum
        self.max_moves = max_moves
        self.max_depth = 0

    @abstractmethod
    def get_action(self, state):
        pass

    def score(self, state, player, legal_actions, plays, wins):
        total_ply = sum(plays[("decision", player, state, a)] for a in legal_actions)
        if total_ply == 0:
            return 0
        score = 0
        for action in legal_actions:
            key = ("decision", player, state, action)
            if plays[key]:
                score += (wins[key] / plays[key]) * (plays[key] / total_ply)
        return score


class UCTPlayer(BaseMCTSPlayer):
    def __init__(self, game, turn=0, thinking_time=1, simNum=0):
        super().__init__(game, turn, thinking_time, simNum)
        self.C = 1.4  # Exploration parameter

    def get_action(self, state):
        board = self.game
        player = board.current_player(state)
        legal_actions = board.legal_actions(state)

        if not legal_actions:
            return None, None
        if len(legal_actions) == 1:
            return legal_actions[0], 0

        plays = defaultdict(int)
        wins = defaultdict(int)
        games = 0

        if self.thinking_time > 0 and self.simNum == 0:
            start_time = time.perf_counter()
            while time.perf_counter() - start_time < self.thinking_time:
                self.run_simulation(state, board, plays, wins)
                games += 1
        else:
            for _ in range(self.simNum):
                self.run_simulation(state, board, plays, wins)
                games += 1

        random.shuffle(legal_actions)
        action = max(
            legal_actions,
            key=lambda a: plays.get((player, state, a), 0)
        )

        return action, self.score(state, player, legal_actions, plays, wins)

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


class PUCTPlayer(BaseMCTSPlayer):
    def __init__(self, game, turn=0, thinking_time=1, simNum=0):
        super().__init__(game, turn, thinking_time, simNum)
        self.C = 1.5  # Exploration parameter
        self.prior = lambda state, action: 1 / len(self.game.legal_actions(state))
        self.value = None

    def get_action(self, state):
        board = self.game
        player = board.current_player(state)
        legal_actions = board.legal_actions(state)

        if not legal_actions:
            return None, None
        if len(legal_actions) == 1:
            return legal_actions[0], 0

        plays = defaultdict(int)
        wins = defaultdict(int)
        games = 0

        if self.thinking_time > 0 and self.simNum == 0:
            start_time = time.perf_counter()
            while time.perf_counter() - start_time < self.thinking_time:
                self.run_simulation(state, board, plays, wins)
                games += 1
        else:
            for _ in range(self.simNum):
                self.run_simulation(state, board, plays, wins)
                games += 1

        random.shuffle(legal_actions)
        action = max(
            legal_actions,
            key=lambda a: plays.get((player, state, a), 0)
        )

        return action, self.score(state, player, legal_actions, plays, wins)

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

            tree.add((player, state, action)) # trajectory
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

def nn_prior_fn(model, game):
    def nn_prior(state, action):
        with torch.no_grad():
            M, b = game.standard_state(state)
            M_tensor = torch.tensor(M, dtype=torch.float32).unsqueeze(0)
            b_tensor = torch.tensor(b, dtype=torch.float32).unsqueeze(0)
            policy = model(M_tensor, b_tensor)[0]
            prob = policy.item()
            return (prob if action == ACTION_PASS else 1 - prob)
    return nn_prior

def smart_prior_fn(game, p=0.9):
    def smart_prior(state, action):
        state = game.unpack_state(state)
        coins, cards, (card_in_play, coins_in_play, n_cards_in_deck, current_player) = state 
        other_cards = [i for i in list(chain.from_iterable(cards)) if i not in cards[current_player]]
        good_for_me = any(abs(card_in_play - card) < 2 for card in cards[current_player])
        good_for_them = any(abs(card_in_play - card) < 2 for card in other_cards)
        least_chip = min(coins)
        legal_actions = game.legal_actions(state)

        if good_for_me:
            if good_for_them:
                good_action = ACTION_TAKE
            else:
                good_action = ACTION_TAKE
                if least_chip > 2:
                    good_action = ACTION_PASS
        else:
            good_action = ACTION_PASS
            if coins[current_player] < 2 or abs(coins_in_play - card_in_play) < min(3, card_in_play//2):
                good_action = ACTION_TAKE

        if action not in legal_actions:
            return 0  # Invalid action for the state
        return p if action == good_action else (1 - p)
    
    return smart_prior

def self_play(game, players, times=1, to_file=None, smart=False):
    if smart:
        for player in players:
            player.prior = smart_prior_fn(game)

    data = {"state": [], "policy": [], "value": []}
    for _ in bar(range(times)):
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

def rl_train(rounds=10, num_games=4, simNum=1000, prior=False, ctd_from=0):
    game = NoThanksBoard(n_players=3)
    Player_0 = PUCTPlayer(game=game, turn=0, simNum=simNum)
    Player_1 = PUCTPlayer(game=game, turn=1, simNum=simNum)
    Player_2 = PUCTPlayer(game=game, turn=2, simNum=simNum)
    players = [Player_0, Player_1, Player_2]

    tester = PUCTPlayer(game=game, turn=0, simNum=simNum)

    batch_size = 32
    n_players = 3
    model = PolicyValueNet(n_players, hidden_dim=32)

    if prior == True:
        model.load_state_dict(torch.load(f'policy_value_net_rd{ctd_from}.pth', weights_only=True))
        model.eval()
        for player in players:
            player.prior = nn_prior_fn(model, game)
        print(f"Model {ctd_from} loaded...Priors are updated for players.")

    for i in range(rounds):
        print(f"Round {i}: The bots are playing...")

        # smart = True if i == 0 else False
        smart = False
        data = self_play(game, players, times=num_games, smart=smart)
        
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

        backup_model = copy.deepcopy(model.state_dict())
            
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
            train_nn(model, batch_size, n_players, 32, (M, b), batch_policies, batch_values)

        # Save the model
        torch.save(model.state_dict(), f'policy_value_net_rd{i+ctd_from}.pth')

        print(f"Round {i} completed. Evaluating performance...")
        # Update prior for the tester
        model.eval()
        tester.prior = nn_prior_fn(model, game)

        # Evaluate the performance of the trained model
        winrate = eval_performance(game, tester, players[1:], num_games=30, verbose=False)
        if winrate > 0.4:
            print(f"Winrate of the Target Player: {winrate:.2%}; Model is accepted.")
            for player in players:
                player.prior = nn_prior_fn(model, game)
        else:
            print(f"Winrate of the Target Player: {winrate:.2%}; Model is rejected.")
            # Rollback to the previous model
            model.load_state_dict(backup_model)

def combine_caches(*caches):
    combined = {}
    key_counts = {}

    # Iterate through all caches
    for cache in caches:
        for key, value in cache.items():
            if key in combined:
                combined[key] += value
                key_counts[key] += 1
            else:
                combined[key] = value
                key_counts[key] = 1

    # Compute the average for each key
    for key in combined:
        combined[key] /= key_counts[key]

    return combined

## To-do: Implement Expectimax training with NN
def expectimax_train(batch_size=128, depth=2):
    n_players = 3 
    hidden_dim = 128
    game = NoThanksBoard(n_players = 3)
    Player_0 = ExpectimaxPlayer(game=game, turn=0, depth=depth, use_cache=False)
    Player_1 = ExpectimaxPlayer(game=game, turn=1, depth=depth, use_cache=False)
    Player_2 = ExpectimaxPlayer(game=game, turn=2, depth=depth, use_cache=False)

    players = [Player_0, Player_1, Player_2]
    play(game, players, display=True)
    combined = combine_caches(Player_0.cache, Player_1.cache, Player_2.cache)
    # data = {"state": [], "value": []}
    # for key, value in combined.items():
    #     data["state"].append(game.standard_state(key[0]))
    #     data["value"].append(value)
    states = []
    target_value = []
    for key, value in combined.items():
        state = key[0]
        coins, cards, (card_in_play, coins_in_play, n_cards_in_deck, current_player) = game.unpack_state(state)
        if card_in_play is not None:
            states.append(game.standard_state(state))
            target_value.append(value)

    # Shuffle the data
    data = list(zip(states, target_value))  # Combine states and target_value
    random.shuffle(data)  # Shuffle the combined data
    states, target_value = zip(*data)

    num_samples = len(states)
    num_batches = num_samples // batch_size
    print(f"Total samples: {num_samples}, Batches: {num_batches}")

    model = ValueNet(n_players, hidden_dim)
    losses = []
    for batch_idx in bar(range(num_batches)):
        # Extract batch data
        batch_start = batch_idx * batch_size
        batch_end = batch_start + batch_size
        
        batch_states = states[batch_start:batch_end]
        batch_values = target_value[batch_start:batch_end]
        
        # Reshape states into (M, b) format
        M = np.array([s[0] for s in batch_states])
        b = np.array([np.array(s[1], dtype=np.float32) for s in batch_states], dtype=np.float32)

        # Train the model on the batch
        model.train()
        # print(f"Training batch {batch_idx + 1}/{num_batches}")
        loss = train_expvalue(model, batch_size, n_players, hidden_dim, (M, b), batch_values)
        losses.append(loss)
    
    # Save the model
    torch.save(model.state_dict(), f'value_net_expectimax.pth')
    return losses

def play(game, players, display=True): 
    # print('Before: ', [x.turn for x in players])   
    random.seed(time.time())
    players.sort(key=lambda x: x.turn)
    current_player = players[0].turn
    
    state = game.starting_state(current_player=current_player)
    state = game.pack_state(state)
    # i = 0
    # print('Middle 1: ', [x.turn for x in players])
    while not game.is_ended(state):
        # i += 1
        if display:
            game.display_state(state, players)
        player = players[current_player]
        action, score = player.get_action(state)
        state = game.next_state(state, action)
        coins, cards, (card_in_play, coins_in_play, n_cards_in_deck, current_player) = game.unpack_state(state)
        # print(players[2].cache)
        # if i > 3:
        #     break
        # print(game.standard_state(state))  
    # print('Middle 2: ', [x.turn for x in players])
    winner = game.winner(state)
    rank = game.rank(state)
    if display:
        game.display_state(state, players)
        game.display_scores(state, players)
        print("Game ended. Player", winner, "wins!")
    # print('Middle 3: ', [x.turn for x in players])
    return winner, rank

def play_hybrid(game_lya, game_kien, players, kien_player_turn, model, display=True):    
    # print('Before hybrid: ', [x.turn for x in players])
    random.seed(time.time())
    action_kien_to_lya = {
        0: 1,
        1: 0
    }
    num2act = {0: 'take',
                1: 'pass'
                }

    # action_kien_to_game_kien = {
    #     0: 'pass',
    #     1: 'take'
    # }
    players.sort(key=lambda x: x.turn) # sort by turn
    current_player = players[0].turn # get current player
    
    state = game_lya.starting_state(current_player=current_player)
    state = game_lya.pack_state(state)
    # i = 0
    while not game_lya.is_ended(state):
        # print('ye')
        print('------------------------------')
        print(f'''Card: {game_kien.current_card} | Chip in pot: {game_kien.chip_in_pot} | Player: {game_kien.turn} - {game_kien.players[game_kien.turn]} | {kien_player_turn}\n''')
        print('------------------------------')
        if current_player == kien_player_turn:
            with torch.no_grad():
                if model.gen == 2:
                    current_state = torch.tensor(game_kien.encode_state_gen_2())
                elif model.gen == 3:
                    current_state = torch.tensor(game_kien.encode_state_gen_3(game_kien.get_state))
                elif model.gen == 3.5:
                    current_state = torch.tensor(game_kien.encode_state_gen_3(game_kien.get_state_gen_3_5))
                elif model.gen == 4:
                    current_state = torch.tensor(game_kien.encode_state_gen_3(game_kien.get_state_gen_4))
                elif model.gen == 5:
                    x_card, x_state = game_kien.encode_state_gen_5(game_kien.get_state)
                    x_card = torch.tensor(x_card).float().unsqueeze(1)
                    x_state = torch.tensor(x_state)
                elif model.gen == 5.5:
                    x_card, x_state = game_kien.encode_state_gen_5(game_kien.get_state_gen_3_5)
                    x_card = torch.tensor(x_card).float().unsqueeze(1)
                    x_state = torch.tensor(x_state)
            legal_move = game_kien.get_legal_action() # a list 
            legal_move_mask = torch.tensor([False if move in legal_move else True for move in game_kien.move_encode.values()])
            # print(game_kien, legal_move_mask)
            if model.gen in [5, 5.5]:
                move_raw, log_prob, entropy, value = model.forward(x_card, x_state, legal_move_mask)
            else:
                move_raw, log_prob, entropy, value = model.forward(current_state, legal_move_mask)
            
            move_for_kien = game_kien.move_encode.get(move_raw.item())
            action = action_kien_to_lya.get(move_raw)
        else:
        # i += 1
            if display:
                game_lya.display_state(state, players)
            
            player = players[current_player]
            action, score = player.get_action(state)
            move_for_kien = game_kien.move_encode.get(action_kien_to_lya.get(action))

        print(f"""Move taken: {move_for_kien}\n""")
        # LYAN GAME UPDATE STATE
        state = game_lya.next_state(state, action)
        coins, cards, (card_in_play, coins_in_play, n_cards_in_deck, current_player) = game_lya.unpack_state(state)

            # KIEN GAME FOLLOWS
        print(move_for_kien, action, num2act.get(action), card_in_play)
        game_kien.action(num2act.get(action), card_in_play)
        # print(players[2].cache)
        # if i > 3:
        #     break
        # print(game.standard_state(state))  
    print(game_kien.is_continue, game_kien.calculate_ranking())
    winner = game_lya.winner(state)
    rank = game_lya.rank(state)
    if display:
        game_lya.display_state(state, players)
        game_lya.display_scores(state, players)
        print("Game ended. Player", winner, "wins!")
    # print('After hybrid: ', [x.turn for x in players])
    return winner, rank

def eval_performance(game,
                     target_player, 
                     opponents, 
                     num_games=300, 
                     verbose=False):
    # random.seed(time.time())
    target_player.name = "Target"
    players = [target_player] + opponents
    win = defaultdict(int)
    ranks = defaultdict(int)
    for i in bar(range(num_games)):
        # print('Begin eval: ', [x.turn for x in players])
        # target_player.turn = i % len(players)
        # print(f'Target {i}: {target_player.turn}')
        for j, player in enumerate(players):
            # if player != target_player:
            player.turn = (i + j) % len(players)
        #     print(f'Other player {j}: {player.turn}')
        # print('After assigning turn: ', [x.turn for x in players])
            # print(f"Game {i+1}: Player {player.name} turn: {player.turn}")
        winner, rank = play(game, players, display=False)
        win[players[winner].name] += 1
        for player in players:
            ranks[player.name] += rank[player.turn]
        if verbose and i % 30 == 0:
            print(f"Number of wins for each player: {win}")
            print(f"Rank for each player: {ranks}")
    # print(f"Number of wins for each player: {win}")
    winrate = win[target_player.name] / num_games
    avg_rank = {player.name: ranks[player.name] / num_games for player in players}
    if verbose:
        print(f"Average rank for each player: {avg_rank}")
        print(f"Winrate of the Target Player: {winrate:.2%}")

    return winrate, avg_rank["Target"]

def eval_performance_hybrid(game_lya,
                     target_player, 
                     opponents, 
                     model,
                     num_games=300, 
                     verbose=False):
    # random.seed(time.time())
    target_player.name = "Target"
    players = [target_player] + opponents
    win = defaultdict(int)
    ranks = defaultdict(int)
    win_ppo = 0
    rank_ppo = []
    for i in bar(range(num_games)):
        # target_player.turn = i % len(players)
        for j, player in enumerate(players):
            # if player != target_player:
            player.turn = (i + j) % len(players)
        # turn_order = [pla.turn for pla in players]
        kien_player_turn = 0
        while kien_player_turn == target_player.turn:
            kien_player_turn = random.choice([0,1,2])

            # print(f"Game {i+1}: Player {player.name} turn: {player.turn}")
        # winner, rank = play(game, players, display=False)
        nothanks = game()
        winner, rank = play_hybrid(
            game_lya = game_lya, 
            game_kien = nothanks, 
            players = players, 
            kien_player_turn = kien_player_turn, 
            model = model, 
            display=False)
        
        win[players[winner].name] += 1
        
        # print('check: ', players[winner].turn == winner)

        rank_ppo.append(rank[kien_player_turn])
        if winner == kien_player_turn:
            win_ppo += 1
        # print(f'\n kien_player_index: {kien_player_turn} \nwinner: {winner} \nrank: {rank[kien_player_turn]}')
        
        # print(win_ppo, rank_ppo)
        
        for player in players:
            ranks[player.name] += rank[player.turn]
        if verbose and i % 30 == 0:
            print(f"Number of wins for each player: {win}")
            print(f"Rank for each player: {ranks}")
    # print(f"Number of wins for each player: {win}")
    winrate = win[target_player.name] / num_games
    avg_rank = {player.name: ranks[player.name] / num_games for player in players}
    winrate_ppo = win_ppo/num_games
    avg_rank_ppo = np.mean(rank_ppo)
    if verbose:
        print(f"Average rank for each player: {avg_rank}")
        print(f"Winrate of the Target Player: {winrate:.2%}")
        print(f"Winrate of the PPO Player: {winrate_ppo:.2%}")
        print(f"Average Rank of the PPO Player: {avg_rank_ppo:.2%}")

    return winrate, avg_rank["Target"]
        

if __name__ == "__main__":

    # rl_train(rounds=2, num_games=2, simNum=1000, prior=False, ctd_from=0)

    # losses = expectimax_train(batch_size=64, depth=6)
    # plt.plot(range(len(losses)), losses, color='r')
    # plt.xticks(range(0, len(losses), len(losses)//50), range(1, len(losses) + 1, len(losses)//50)))
    # plt.xlabel('Batch Number')
    # plt.ylabel('Loss')
    # plt.title('Loss vs. Batch Number')
    # plt.grid(True)
    # plt.show()

    game_lya = NoThanksBoard(n_players = 3)
    Player_0 = PUCTPlayer(game=game_lya, turn=1, simNum=500)
    # Player_0 = HumanPlayer(game=game, turn=0)
    Player_1 = UCTPlayer(game=game_lya, turn=1, simNum=500)
    # Player_2 = PUCTPlayer(game, turn=1, simNum=500)
    Player_2 = ExpectimaxPlayer(game=game_lya, turn=1, depth=2, use_cache=False)
    Player_2.trained = False

    model = PolicyValueNet(game_lya.n_players, 32)
    model.load_state_dict(torch.load('policy_value_net_rd4.pth'))
    model.eval()

    N_PLAYER = 3
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    path = f'./non_nn_methods/ppo_weight/model_gen_3_default_rwd_50_iter.pth'
    if 'gen_2' in path:
        model = ppo_gen_2(N_PLAYER).to(device)
    elif 'gen_3' in path:
        model = ppo_gen_3(N_PLAYER).to(device)
        if 'gen_3_5' in path:
            model.gen = 3.5
    elif 'gen_4' in path:
        model = ppo_gen_4(N_PLAYER).to(device)
    elif 'gen_5' in path:
        model = ppo_gen_5(N_PLAYER).to(device)
        if 'gen_5_5' in path:
            model.gen = 5.5   

    model.load_state_dict(torch.load(path,  map_location=torch.device('cpu')))
    

    # # model = PolicyOnlyNet(game.n_players, 128)
    # # model.load_state_dict(torch.load('policy_only_net.pth'))
    # # model.eval()

    # # new_prior = smart_prior_fn(game)
    # new_prior = nn_prior_fn(model, game)
    # # Player_0.prior = smart_prior_fn(game)
    # # Player_1.prior = new_prior
    # # Player_2.prior = new_prior

    # players = [Player_0, Player_1, Player_2]
    # play(game, players, display=True)
    # # print(f"{Player_2.cache_hit} cache hits.")
    # # print(Player_2.cache)

    randPlayer1 = RandomPlayer(game_lya, turn=0)
    randPlayer2 = RandomPlayer(game_lya, turn=2)
    randPlayers = [randPlayer1, randPlayer2]

    rulePlayer1 = RuleBasedPlayer(game_lya, turn=0)
    rulePlayer2 = RuleBasedPlayer(game_lya, turn=2)
    rulePlayers = [rulePlayer1, rulePlayer2]

    player_fill_uct_1 = UCTPlayer(game=game_lya, turn=0, simNum=500)
    player_fill_uct_2 = UCTPlayer(game=game_lya, turn=2, simNum=500)

    player_fill_exp_1 = ExpectimaxPlayer(game=game_lya, turn=0, depth=2, use_cache=False)
    player_fill_exp_1.trained = False

    player_fill_exp_2 = ExpectimaxPlayer(game=game_lya, turn=2, depth=2, use_cache=False)
    player_fill_exp_2.trained = False

    mixPlayers = [player_fill_uct_1, player_fill_uct_2]
    # winrate, avg_rank = eval_performance(game_lya, Player_1, rulePlayers, num_games=300, verbose=True)
    winrate, avg_rank = eval_performance_hybrid(
        game_lya = game_lya, 
        target_player = Player_2, 
        opponents = mixPlayers,
        model = model,
        num_games= 300, 
        verbose=True)

    # winrate, avg_rank = eval_performance(game, Player_2, rulePlayers, num_games=300, verbose=True)

    # # print(f"Winrate of the Target Player: {winrate:.2%}")
