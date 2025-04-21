"""Pitting a PPO model against 2 MCTS players"""

from ppo_model import *
from utils import *
from nothank_mcts import *
from test_code.expectimax import *


N_PLAYER = 3
device = 'cuda' if torch.cuda.is_available() else 'cpu'
ppo_turn = [1]
mcts_turn = [0,2]
expectimax_turn = 2
n_selection = 50


path = f'./ppo_weight/trained_model/model_gen_3_default_rwd_60_iter.pth'
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

game_node = game_state()
tree = mcts()
nothanks = game()

while nothanks.is_continue:
    print('------------------------------')
    print(f'''Card: {nothanks.current_card} | Chip in pot: {nothanks.chip_in_pot} | Player: {nothanks.turn} - {nothanks.players[nothanks.turn]}\n'''
)
    print('------------------------------')

    if nothanks.turn in ppo_turn:
        with torch.no_grad():
            if model.gen == 2:
                current_state = torch.tensor(nothanks.encode_state_gen_2()).to(device)
            elif model.gen == 3:
                current_state = torch.tensor(nothanks.encode_state_gen_3(nothanks.get_state)).to(device)
            elif model.gen == 3.5:
                current_state = torch.tensor(nothanks.encode_state_gen_3(nothanks.get_state_gen_3_5)).to(device)
            elif model.gen == 4:
                current_state = torch.tensor(nothanks.encode_state_gen_3(nothanks.get_state_gen_4)).to(device)
            elif model.gen == 5:
                x_card, x_state = nothanks.encode_state_gen_5(nothanks.get_state)
                x_card = torch.tensor(x_card).float().unsqueeze(1)
                x_state = torch.tensor(x_state)
            elif model.gen == 5.5:
                x_card, x_state = nothanks.encode_state_gen_5(nothanks.get_state_gen_3_5)
                x_card = torch.tensor(x_card).float().unsqueeze(1)
                x_state = torch.tensor(x_state)

            legal_move = nothanks.get_legal_action() # a list 
            legal_move_mask = torch.tensor([False if move in legal_move else True for move in nothanks.move_encode.values()]).to(device)
            if model.gen in [5, 5.5]:
                move_raw, log_prob, entropy, value = model.forward(x_card, x_state, legal_move_mask)
            else:
                move_raw, log_prob, entropy, value = model.forward(current_state, legal_move_mask)
            # move_raw, log_prob, entropy, value = model.forward(current_state, legal_move_mask)
            move = nothanks.move_encode.get(move_raw.item())
    elif nothanks.turn in mcts_turn:
        for _ in tqdm(range(n_selection)):
            tree.selection(game_node= game_node, game= nothanks)
        move = game_node.get_best_move()

    # elif nothanks.turn == expectimax_turn:
    #     move = agent.get_best_action(nothanks)
    
    print(f"""Move taken: {move}\n""")
    nothanks.action(move)    
    game_node = game_state(turn = nothanks.turn)

for player_tmp in nothanks.players:
    print(player_tmp.calculate_score())

print(nothanks.calculate_ranking())
