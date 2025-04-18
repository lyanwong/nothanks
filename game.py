import random
from dataclasses import dataclass, field
import numpy as np

ACTION_TAKE = 0
ACTION_PASS = 1

def diff(first, second):
    second = set(second)
    return [item for item in first if item not in second]

@dataclass
class NoThanksConfig:
    min_card: int = 3
    max_card: int = 14
    n_omit_cards: int = 3
    n_players: int = 3
    start_coins: int = 4
    # start_coins: int = field(init=False)

    # def __post_init__(self):
    #     self.start_coins = self.calculate_start_coins()

    # def calculate_start_coins(self):
    #     if 3 <= self.n_players <= 5:
    #         return 11
    #     elif self.n_players == 6:
    #         return 9
    #     elif self.n_players == 7:
    #         return 7
    #     else:
    #         raise ValueError("Number of players must be between 3 and 7")

class NoThanksBoard():
    def __init__(self, n_players = 3, config = NoThanksConfig):
        self.n_players = n_players
        self.min_card = config.min_card
        self.max_card = config.max_card
        self.full_deck = list(range(self.min_card, self.max_card+1))
        self.n_omit_cards = config.n_omit_cards
        self.n_cards = self.max_card - self.min_card + 1
        self.start_coins = config.start_coins
        random.seed(999)
    
    # def reward_dict(self):
    #     if self.n_players == 3:
    #         return {1: 1, 2: 0, 3: -1}
    #     elif self.n_players == 4:
    #         return {1: 1, 2: 0.5, 3: -0.5, 4: -1}
    #     elif self.n_players == 5:
    #         return {1: 1, 2: 0.5, 3: 0, 4: -0.5, 5: -1}
    #     elif self.n_players == 6:
    #         return {1: 1, 2: 0.75, 3: 0.5, 4: -0.5, 5: -0.75, 6: -1}
    #     elif self.n_players == 7:
    #         return {1: 1, 2: 0.75, 3: 0.5, 4: 0, 5: -0.5, 6: -0.75, 7: -1}

            
    # state: ((player coins),(player cards),(card in play, coins in play, n_cards_remaining, current player))
    def starting_state(self, current_player = 0):
        coins = [self.start_coins for i in range(self.n_players)]
        cards = [[] for i in range(self.n_players)]

        card_in_play = random.choice(self.full_deck)
        
        coins_in_play = 0
        n_cards_in_deck = self.n_cards - 1 - self.n_omit_cards

        return coins, cards, (card_in_play, coins_in_play, n_cards_in_deck, current_player)

    def next_state(self, state, action):
        if state is None:
            return None
        state = self.unpack_state(state)
        coins, cards, (card_in_play, coins_in_play, n_cards_in_deck, current_player) = state

        if action == ACTION_TAKE:
            cards[current_player].append(card_in_play)
            coins[current_player] += coins_in_play

            all_player_cards = [card for player_cards in cards for card in player_cards]
            cards_in_deck = diff(self.full_deck, all_player_cards)
            current_player = current_player
            
            if cards_in_deck and n_cards_in_deck > 0:   
                # random.shuffle(list(cards_in_deck))
                card_in_play = random.choice(cards_in_deck)
                n_cards_in_deck -= 1
            else:
                card_in_play = None
            coins_in_play = 0

        else:
            coins[current_player] -= 1
            coins_in_play += 1
            current_player += 1
        
        if current_player == self.n_players:
            current_player = 0

        next_state = coins, cards, (card_in_play, coins_in_play, n_cards_in_deck, current_player)
        return self.pack_state(next_state)
    
    def all_possible_next(self, state, action):
        if action == ACTION_PASS:
            return [(1.0, self.next_state(state, action))]
        elif action == ACTION_TAKE:
            next_states = []
            state = self.unpack_state(state)
            coins, cards, (card_in_play, coins_in_play, n_cards_in_deck, current_player) = state
            cards[current_player].append(card_in_play)
            coins[current_player] += coins_in_play
            coins_in_play = 0

            all_player_cards = [card for player_cards in cards for card in player_cards]
            cards_in_deck = diff(self.full_deck, all_player_cards)
            prob = 1.0 / len(cards_in_deck) if cards_in_deck else 1.0
            
            if not cards_in_deck:
                return []
            else:
                n_cards_in_deck -= 1
                for card in cards_in_deck: 
                    card_in_play = card
                    current_player += 1
                    if current_player == self.n_players:
                        current_player = 0
                    next_state = coins, cards, (card_in_play, coins_in_play, n_cards_in_deck, current_player)
                    next_state = self.pack_state(next_state)
                    next_states.append((prob, next_state))
            
            return next_states # (prob, next_state) pairs

    def is_legal(self, state, action):
        coins, cards, (card_in_play, coins_in_play, n_cards_in_deck, current_player) = state

        if card_in_play is None:
            return False
        if coins[current_player] <= 0 and action == ACTION_PASS:
            return False
        else:
            return True

    def legal_actions(self, state):
        if state is None:
            return []
        actions = []
        
        if self.is_legal(state, ACTION_TAKE):
            actions.append(ACTION_TAKE)

        if self.is_legal(state, ACTION_PASS):
            actions.append(ACTION_PASS)

        return actions

    def pack_state(self, state):
        coins, cards, details = state
        packed_state = tuple(coins), tuple(map(tuple, cards)), details
        return packed_state

    def unpack_state(self, packed_state):
        coins, cards, details = packed_state
        coins = list(coins)
        cards = list(map(list, cards))
        return coins, cards, details
    

    def standard_state(self, state):
        """
        Input state (packed or unpacked): ([coins], [[cards]], (card_in_play, coins_in_play, n_cards_in_deck, current_player))
        Transform state into the required format:
        1. Extract the state into M and b where M is the card/coin matrix and b is the vector (card_in_play, coins_in_play, n_cards_in_deck)
        2. Rotate M such that the first row corresponds to the current player
        3. Transform M into array of shape (3, n_players, 33)
        """
        coins, cards, (card_in_play, coins_in_play, n_cards_in_deck, current_player) = state
        
        # Step 1: Build the M matrix (n_players x 34)
        M = []
        for k in range(self.n_players):
            # Initialize the card representation for player k
            card_rep = [0] * self.n_cards  # 33 cards

            # Set 1 for each card player k has
            for card in cards[k]:
                card_rep[card - self.min_card] = 1  # Cards are indexed from 3 to 35

            # Add the number of coins for player k
            M.append(card_rep + [coins[k]])  # 33 card columns + 1 coin column
        
        # Step 2: Build the b vector (card_in_play, coins_in_play, n_cards_in_deck)
        b = np.array([card_in_play, coins_in_play, n_cards_in_deck])

        # Step 3: Rotate M such that the first row is the current player
        M_rotated = M[current_player:] + M[:current_player]  # Rotate the matrix
        
        # Step 4: Transform M into array of shape (3, n_players, 33)
        # where M[0] is the card matrix, M[1] is the coin matrix, M[2] is the card in play
        M_transformed = np.zeros((3, self.n_players, self.n_cards))
        M_rotated = np.array(M_rotated)
        M_transformed[0] = M_rotated[:, :-1]  # Card matrix
        M_transformed[1] = np.repeat(M_rotated[:, -1][:, np.newaxis], self.n_cards, axis=1)  # Coin matrix
        M_transformed[1] = M_transformed[1] / (self.start_coins * self.n_players)  # Normalize coins to be between 0 and 1
        
        card_in_play_onehot = np.zeros(self.n_cards)
        if self.min_card <= card_in_play <= self.max_card:
            card_in_play_onehot[card_in_play - self.min_card] = 1
        M_transformed[2] = np.tile(card_in_play_onehot, (self.n_players, 1))

        return M_transformed, b


    def is_ended(self, state):
        if isinstance(state, list):
            return False
        elif state is None:
            return True
        
        coins, cards, (card_in_play, coins_in_play, n_cards_in_deck, current_player) = state

        if n_cards_in_deck == 0 and card_in_play == None:
            return True
        else:
            return False

    def compute_scores(self, state):
        state = self.unpack_state(state)
        coins, cards, (card_in_play, coins_in_play, n_cards_in_deck, current_player) = state

        scores = []

        for p_idx in range(self.n_players):
            cards[p_idx].sort()

            score = 0
            if cards[p_idx]:
                score += cards[p_idx][0]
                last_card = cards[p_idx][0]

                for card_idx in range(1, len(cards[p_idx])):
                    new_card = cards[p_idx][card_idx]

                    if not new_card == last_card + 1:
                        score += new_card
                    last_card = new_card

            score -= coins[p_idx]

            scores.append(score)

        return scores

    def winner(self, state):
        """Temporary winner: player with the lowest score even if the game is not ended."""
        state = self.unpack_state(state)
        coins, cards, (card_in_play, coins_in_play, n_cards_in_deck, current_player) = state

        if not self.is_ended(state):
            return None
        
        scores = self.compute_scores(state)
        min_score = 1000
        lowest_scorers = []
        # get lowest scorers (could be more than one)
        for i, score in enumerate(scores):
            if score < min_score:
                lowest_scorers = [i]
                min_score = score
            if score <= min_score:
                lowest_scorers.append(i)
        
        # if players are tied on lowest score, get the one with the fewest cards
        if len(lowest_scorers) > 1:
            min_n_cards = 1000
            for i in lowest_scorers:
                n_cards = len(cards[i])
                if n_cards < min_n_cards:
                    lowest_card_players = [i]
                    min_n_cards = n_cards
                elif n_cards <= min_n_cards:
                    lowest_card_players.append(i)

            if len(lowest_card_players) > 1:
                winner = lowest_card_players[0]
            else: # if still tied, pick a random winner (not the official rules)
                winner = random.choice(lowest_card_players) 
        else:
            winner = lowest_scorers[0]

        return winner
    
    # def reward_rank(self, state):
    #     state = self.unpack_state(state)
    #     scores = self.compute_scores(state)
    #     rank = sorted(range(len(scores)), key=lambda k: scores[k])
    #     value = [self.reward_dict()[rank.index(player) + 1] for player in range(self.n_players)]
    #     return np.array(value)
    
    # def reward_winloss(self, state):
    #     state = self.unpack_state(state)
    #     value = [1 if self.winner(state) == player else -1 for player in range(self.n_players)]
    #     return np.array(value)

    # def reward_score(self, state):
    #     state = self.unpack_state(state)
    #     scores = self.compute_scores(state)
    #     rewards = [-score for score in scores]

    #     return np.array(rewards)

    
    def basic_display_state(self, state):
        coins, cards, (card_in_play, coins_in_play, n_cards_in_deck, current_player) = state

        print("Coins:           {0}".format(coins))
        print("Cards:           {0}".format(cards))
        print("Card in play:    {0}".format(card_in_play))
        print("Coins:           {0}".format(coins_in_play))
        print("Player:          {0}".format(current_player))

    def display_scores(self, state, players):
        scores = self.compute_scores(state)
        
        print("")
        print("--- Scores ---")
        for player in players:
            print("{:<10} {:<10}".format(
                player.name, scores[player.turn])
            )
        print("")

    def display_state(self, state, players):
        state = self.unpack_state(state)
        coins, cards, (card_in_play, coins_in_play, n_cards_in_deck, current_player) = state

        scores = self.compute_scores(state)

        def format_cards(card_list):
            return ", ".join(map(str, sorted(card_list)))

        # Reorder players based on player.turn (in-place)
        player_labels = [players[i].name for i in range(self.n_players)]
        card_strings = [format_cards(cards[i]) for i in range(self.n_players)]
        coin_strings = [str(coins[i]) for i in range(self.n_players)]
        score_strings = [str(scores[i]) for i in range(self.n_players)]

        max_card_len = max(20, max(len(card_str) for card_str in card_strings))

        print("")
        print("-" * (15 + max_card_len + 10 + 10 + 10))
        print("")
        print("{:<15} {:<{}} {:<10} {:<10}".format("Player", "Cards", max_card_len, "Coins", "Score"))
        print("-" * (15 + max_card_len + 10 + 10 + 10))

        for i in range(self.n_players):
            print("{:<15} {:<{}} {:<10} {:<10}".format(
                player_labels[i],
                card_strings[i],
                max_card_len,
                coin_strings[i],
                score_strings[i]
            ))

        print("-" * (15 + max_card_len + 10 + 10 + 10))
        print("\t\t In play: [{0}]".format(card_in_play))
        print("\t\t Cards remaining: {0}".format(n_cards_in_deck))
        print("\t\t   Coins: {0}".format(coins_in_play))
        print("")
            
    def pack_action(self, notation):
        if notation == "y" or notation == "Y":
            return ACTION_TAKE
        else:
            return ACTION_PASS

    def current_player(self, state):
        return state[2][3]
    
    def remaining_cards(self, state):
        s = self.unpack(state)

        # All cards in original deck
        full_deck = set(self.initial_deck)  # This should be fixed when game starts

        # Cards already taken by players
        taken_cards = set()
        for player_cards in s["player_cards"]:
            taken_cards.update(player_cards)

        # Card currently in play
        if s["card"] is not None:
            taken_cards.add(s["card"])

        # Cards already revealed (optional: e.g., discard pile if any)
        # taken_cards.update(s.get("discarded_cards", []))

        # Remaining = full_deck - taken
        remaining = full_deck - taken_cards
        return list(remaining)
    

if __name__ == "__main__":
    game = NoThanksBoard()
    start = game.starting_state()
    print(game.all_possible_next(start, ACTION_TAKE))