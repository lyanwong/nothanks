import random
import numpy as np
import copy

class player():
    """Representation of a player in no thanks"""
    def __init__(self, chip:int = 11 ):
        """
        chip: number of chip the player has at the moment
        card: list of cards that the player owns
        """
        self.chip = chip
        self.card = []
    
    def __repr__(self):
        return f'Chip: {self.chip} | Card owned: {self.card}'
        
    def calculate_score(self):
        """Calculate the score of the player based on his card and chip
        Idea: sum the value of all the cards. For a series of incremental values, only take the lowest"""
        sorted_card = sorted(self.card)
        total_score = 0
        if len(sorted_card) == 0:
            return self.chip
        total_score -= sorted_card[0]
        for score, prev_score in zip(sorted_card[1:], sorted_card[:-1]):
            if score - prev_score > 1:
                total_score -= score
        total_score += self.chip
        return total_score
    
class game:
    """Representation of the state of the game"""
    full_deck = [i for i in range(3,15)]
    
    def __init__(self,
                 card: int = None,
                 n_player: int = 3,
                 n_chip: int = 4,
                 n_remove_card: int = 3
                 ):
        """
        card: the card to start the game with
        n_player: number of players
        n_chip: number of chips for each player
        n_remove_card: number of card to burn before playing
        """
        self.n_remove_card = n_remove_card
        self.remain_card = game.full_deck
        self.played_card = []
        self.current_card = self.flip_card(card) if card else self.flip_card()
        
        self.chip_in_pot = 0
        self.turn = 0 # need this to know which player to take
        self.n_player = n_player
        self.n_chip = n_chip

        self.max_score = 1
        self.min_score = 0
        self.score_range = self.max_score - self.min_score
        self.rollout_rule = {'pass': 0.9,
                             'take': 0.1}
        self.is_continue = True
        self.init_player(n_player)
    
    def __len__(self):
        return len(self.remain_card)
    
    def __str__(self):
        return f'Current card: {self.current_card} | Played cards: {self.played_card}'
    def __repr__(self):
        return f'Current card: {self.current_card} | Played cards: {self.played_card}'

    def init_player(self, n_player):
        self.players = [player(chip =  self.n_chip) for _ in range(n_player)]
        
    def set_remain_card(self):
        """Update list of remaining cards"""
        self.remain_card = [i for i in game.full_deck 
                            if i not in self.played_card]

    def set_removed_card(self):
        "Remove 9 cards"
        self.removed_card = self.get_cards(9)
        self.set_remain_card()

    def get_cards(self, n):
        'Randomize a card, if number of card remained == 9'
        if len(self.remain_card) == self.n_remove_card:
            return False
        card = random.sample(self.remain_card, n)
        return card[0] if n == 1 else card
    
    
    def flip_card(self, 
                card: int = None):
        '''Progress the game by flipping a new card'''
        if card == None:
            card = self.get_cards(1)
            if not card:
                return card
        self.played_card.append(card)
        self.set_remain_card()
        self.current_card = card
        return card
    
    def calculate_ranking(self) -> list:
        """Get score for each player based on their ranking"""
        score_list = np.array([player.calculate_score() for player in self.players])
        rank_tmp = score_list.argsort()
        ranking = rank_tmp.argsort()
        step = self.score_range/(self.n_player - 1)
        final_score = [self.min_score + step*rank for rank in ranking]
        return final_score
    
    def get_legal_action(self) -> list[str]:
        """Get the possible action a player can carry out"""
        if not self.is_continue:
            print('Game over, no legal action')
            return []
        current_player = self.players[self.turn]
        if current_player.chip > 0:
            return ['pass', 'take']
        else:
            return ['take']

    def next_player(self, move) -> int:
        """Get index of next player. if take then turn does not change"""
        if move == 'take':
            return self.turn
        elif move == 'pass':
            return self.turn + 1 if self.turn + 1 < self.n_player else 0
            
    def action(self, move: str = 'pass', card: int = None) -> bool:
        """Progress the game
        move: the action of the current player: pass or take
        """
        if not self.is_continue:
            print('Game over, cannot act')
            return False
        current_player = self.players[self.turn]
        if move == 'take':
            current_player.card.append(self.current_card)
            current_player.chip += self.chip_in_pot
            self.chip_in_pot = 0
            is_continue = self.flip_card(card)
            self.is_continue = is_continue
            return self.is_continue

        elif move == 'pass':
            current_player.chip -= 1
            self.chip_in_pot += 1
            self.turn = self.next_player(move)
            return True
        
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
    agent = expectimax_agent(depth = 4, 
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