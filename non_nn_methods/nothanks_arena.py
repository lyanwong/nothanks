from players import * 
from non_nn_methods.nothank_mcts import *

# Combine logic - pseudo code

# KIEN_PLAYER = 2
# if current_player == KIEN_PLAYER:
#     move = kiens_game.make_decision()
# else:
#     move = lyans_game.make_decision()

# next_card = lyans_game.proceed(move)
# kiens_game.update(move, next_card)

if __name__ == '__main__':

    # INITIALIZE LYAN

    lyan_game = NoThanksBoard(n_players = 3)
    Player_0 = MCTSPlayer(name=None, game=lyan_game, turn=0)
    Player_1 = MCTSPlayer(name=None, game=lyan_game, turn=1)
    Player_2 = MCTSPlayerOnline(game=lyan_game, thinking_time=1, turn=2)

    players = [Player_0, Player_1, Player_2]

    state = lyan_game.starting_state(current_player=0)
    state = lyan_game.pack_state(state)
    coins, cards, (card_in_play, coins_in_play, n_cards_in_deck, current_player) = lyan_game.unpack_state(state)
    current_player = 0

    # INITIALIZE KIEN

    kien_game = game(card_in_play)
    game_node = game_state()
    tree = mcts()

    # PLAY

    KIEN_PLAYER = 2
    n_selection = 50

    act2num = {'take': 0,
                'pass': 1
                }

    num2act = {0: 'take',
                1: 'pass'
                }

    while not lyan_game.is_ended(state):
        if current_player == KIEN_PLAYER:
            # KIEN GAME MAKE DECISION
            for _ in tqdm(range(n_selection)):
                tree.selection(game_node= game_node, game= kien_game)
            kien_action = game_node.get_best_move()
            action = act2num.get(kien_action)
        else:
            player = players[current_player]
            action = player.get_action(state)
        # LYAN GAME UPDATE STATE
        state = lyan_game.next_state(state, action)
        coins, cards, (card_in_play, coins_in_play, n_cards_in_deck, current_player) = lyan_game.unpack_state(state)
        lyan_game.display_state(state, human_player=2)

        # KIEN GAME FOLLOWS
        kien_game.action(num2act.get(action), card_in_play)
        print(kien_game)
        game_node = game_state(turn = current_player)

    lyan_game.display_scores(state)
    winner = lyan_game.winner(state)
    print("Game ended. Player", winner, "wins!")
