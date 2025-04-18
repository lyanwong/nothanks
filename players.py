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
from network import PolicyValueNet, train_nn
from policy_net import PolicyOnlyNet
import json
from log_progress import log_progress
from tqdm import tqdm
# from multiprocessing import Pool
# from functools import partial

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

        if not state_list:
            return 0.0

        # Base case: depth limit — evaluate all outcomes and return expected value
        if depth == 0:
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
            return expected_value

        # Stochastic node: multiple possible outcomes
        if len(state_list) > 1:
            expected_value = 0.0
            for prob, state in state_list:
                expected_value += prob * self.expectimax([(1.0, state)], depth - 1, perspective)
            # print(f"Depth {depth}, Expected value: {expected_value} for Player {perspective}, State: {state_list}")
            return expected_value

        # Deterministic node
        prob, state = state_list[0]
        packed = self.game.pack_state(state)
        key = (packed, perspective)

        # Use cache if available
        if self.use_cache and key in self.cache:
            self.cache_hit += 1
            return self.cache[key]

        # Terminal game state
        if self.game.is_ended(state):
            value = self.evaluate(state, perspective)
            self.cache[key] = value
            return value

        current_player = self.game.current_player(state)
        legal_actions = self.game.legal_actions(state)

        if not legal_actions:
            value = self.evaluate(state, perspective)
            self.cache[key] = value
            return value

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
        return value

    def evaluate(self, state, perspective):
        if self.trained:
            return self.evaluate_trained(state, perspective)
        else:
            return self.evaluate_untrained(state, perspective)

    def evaluate_untrained(self, state, perspective):
        if isinstance(state, list):
            expected_value = 0.0
            for prob, next_state in state:
                score = self.game.compute_scores(next_state)
                value = -score[perspective]
                expected_value += prob * value
            # print(f"Expected value: {expected_value} for Player {perspective}")
            return expected_value
        # If the state is not a list
        score = self.game.compute_scores(state)
        return -score[perspective]
    
    def evaluate_trained(self, state, perspective, model):
        pass


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

## To-do: Implement Expectimax training with NN
def expectimax_train():
    pass

def play(game, players, display=True):    
    random.seed(time.time())
    players.sort(key=lambda x: x.turn)
    current_player = players[0].turn
    
    state = game.starting_state(current_player=current_player)
    state = game.pack_state(state)
    # i = 0
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
    winner = game.winner(state)
    if display:
        game.display_state(state, players)
        game.display_scores(state, players)
        print("Game ended. Player", winner, "wins!")

    return winner

def eval_performance(game, target_player, opponents, num_games=300, verbose=False):
    # random.seed(time.time())
    target_player.name = "Target"
    players = [target_player] + opponents
    win = defaultdict(int)
    for i in bar(range(num_games)):
        target_player.turn = i % len(players)
        for j, player in enumerate(players):
            if player != target_player:
                player.turn = (i + j) % len(players)
            # print(f"Game {i+1}: Player {player.name} turn: {player.turn}")
        winner = play(game, players, display=False)
        win[players[winner].name] += 1
        if verbose and i in [60, 90, 120, 150, 180, 210, 240, 270]:
            print(f"Number of wins for each player: {win}")
    # print(f"Number of wins for each player: {win}")
    winrate = win[target_player.name] / num_games

    return winrate
        

if __name__ == "__main__":

    # rl_train(rounds=2, num_games=2, simNum=1000, prior=False, ctd_from=0)
    

    game = NoThanksBoard(n_players = 3)
    Player_0 = UCTPlayer(game=game, turn=0, simNum=500)
    # Player_0 = HumanPlayer(game=game, turn=0)
    Player_1 = UCTPlayer(game=game, turn=1, simNum=500)
    # Player_2 = PUCTPlayer(game, turn=2, simNum=500)
    Player_2 = ExpectimaxPlayer(game=game, turn=2, depth=2, use_cache=False)

    # model = PolicyValueNet(game.n_players, 32)
    # model.load_state_dict(torch.load('policy_value_net_rd0.pth'))
    # model.eval()

    # model = PolicyOnlyNet(game.n_players, 128)
    # model.load_state_dict(torch.load('policy_only_net.pth'))
    # model.eval()

    # new_prior = smart_prior_fn(game)
    # new_prior = nn_prior_fn(model, game)
    # Player_0.prior = smart_prior_fn(game)
    # Player_1.prior = new_prior
    # Player_2.prior = new_prior

    players = [Player_0, Player_1, Player_2]
    play(game, players, display=True)
    # print(f"{Player_2.cache_hit} cache hits.")
    # print(Player_2.cache)

    # winrate = eval_performance(game, Player_2, [Player_1, Player_0], num_games=300, verbose=True)

    # print(f"Winrate of the Target Player: {winrate:.2%}")
