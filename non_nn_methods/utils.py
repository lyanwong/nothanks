import random
import numpy as np


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
    full_deck = [i for i in range(3,36)]
    
    def __init__(self,
                 card: int = None,
                 n_player: int = 3,
                 n_chip: int = 11,
                 n_remove_card: int = 9
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
        self.min_score = -0.1
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
            
    def action(self, move: str = 0, card: int = None) -> bool:
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
    
    def rollout_policy_rule(self):
        """Rule based policy no.1 of rollout stage 
        Idea: 90% pass, 10% take"""
        legal_action = self.get_legal_action()
        weight = [self.rollout_rule.get(i) for i in legal_action]
        move = random.choices(legal_action, weight)[0]
        return move
    
    def rollout_policy_1(self, verbose = False) ->str:
        """Rule based policy no.2 of rollout stage
        Idea: probability of a "pass" increases linearly as number of chips in pot increase. Reaches 100% if number of chips == 1/2 value of flipped card
        """
        # current_player = self.players[self.turn]
        legal_action = self.get_legal_action()
        const = 0.5
        prob = (self.chip_in_pot/self.current_card)/const
        weight = [prob if i == 'take' else 1 - prob for i in legal_action]
        move = random.choices(legal_action, weight)[0]
        if verbose:
            print(legal_action, weight)
        return move
    
    def rollout_policy_2(self, verbose = False) -> str:
        """if number of chip in pot >= card value -1 -> 90% pass else 1% pass"""
        # current_player = self.players[self.turn]
        legal_action = self.get_legal_action()
        remain = self.current_card//2 - self.chip_in_pot
        if remain <= 1:
            prob = 0.9
        else:
            prob = 0.01

        weight = [prob if i == 'take' else 1 - prob for i in legal_action]
        move = random.choices(legal_action, weight)[0]
        if verbose:
            print(legal_action, weight)
        return move
    
    def rollout_policy_3(self, p = 0.98, verbose = False) -> str:
        """A bit more complicated, explained in comment
        
        p: probability of the selected action
        """
        current_player = self.players[self.turn]
        other_card = [i for i in self.played_card if i not in current_player.card]
        good_for_me = any([abs(self.current_card - card) <= 2 for card in current_player.card])
        good_for_them = any([abs(self.current_card - card) <= 2 for card in other_card])
        least_chip = min([self.players[turn].chip for turn in range(self.n_player) if turn != self.turn])
        legal_action = self.get_legal_action()

        if good_for_me:
            if good_for_them:
                #then you must take or the other guy will take it
                good_action = 'take'
            else:
                #how much can i farm
                # look at the guy with the least chip
                good_action = 'take'
                if least_chip >= 3:
                    # I will farm until the guy with the least chip has fewer than 3 chips
                    good_action = 'pass'
        else:
            # can i afford to pass it till taken?
            good_action = 'pass'
            if current_player.chip <= 2 or self.chip_in_pot >= self.current_card//2:
                # 
                good_action = 'take'

        weight = [p if act == good_action else 1 - p for act in legal_action]
        move = random.choices(legal_action, weight)[0]
        if verbose:
            print(legal_action, weight)
        return move


    def self_play(self, verbose = False):
        """Keeps playing till the game end
        This is where you deploy your rollout policy
        """
        while self.is_continue:
            move = self.rollout_policy_3(verbose = verbose)
            if verbose:
                print(f'''Card: {self.current_card} | Chip in pot: {self.chip_in_pot} | Player: {self.turn} - {self.players[self.turn]}
move: {move}'''
    )
            self.is_continue = self.action(move)        
        score_list = self.calculate_ranking()
        return score_list
    

class game_state:
    """This is node of the tree for later search"""
    def __init__(self, parent = None, 
                turn: int = 0,
                 depth = 0):
        """
        turn: whose turn is it at this node
        depth: number of turns passed. Could be use to limit the depth of the tree
        """

        self.depth = depth
        self.win = 0
        self.n_explored = 0
        self.parent = parent
        self.child = [] # (uct, move, next node)
        self.simulation_score = [0,0,0]
        self.turn = turn # need this as reference when doing backpropagation

    def calculate_uct(self):
        """Calculate its own UCT: Upper Confidence Bound
        Can set your uct policy here"""
        if self.n_explored == 0:
            return np.inf
        else:
            exploitation = self.win/self.n_explored
            exploration = np.sqrt(4*np.log(self.parent.n_explored)/self.n_explored)
            return exploitation + exploration
        
    def get_best_move(self):
        """Get move with the highest win percentage. Used after simulation is finished"""
        win_list = [child_node[-1].win/child_node[-1].n_explored for child_node in self.child]
        return self.child[np.argmax(win_list)][1]
    
