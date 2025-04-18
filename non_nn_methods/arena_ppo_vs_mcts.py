from ppo_model import *
from utils import *
from nothank_mcts import *
from expectimax import *


N_PLAYER = 3
device = 'cuda' if torch.cuda.is_available() else 'cpu'
ppo_turn = [1]
mcts_turn = [0,2]
expectimax_turn = 2
n_selection = 50


model = ppo_gen_3(N_PLAYER).to(device)
path = f'./ppo_weight/model_gen_3_default_rwd_60_iter.pth'
model.load_state_dict(torch.load(path,  map_location=torch.device('cpu')))

game_node = game_state()
tree = mcts()
nothanks = game()

# agent = expectimax_agent(depth = 3, 
#         player_index = expectimax_turn)



while nothanks.is_continue:
    print('------------------------------')
    print(f'''Card: {nothanks.current_card} | Chip in pot: {nothanks.chip_in_pot} | Player: {nothanks.turn} - {nothanks.players[nothanks.turn]}\n'''
)
    print('------------------------------')

    if nothanks.turn in ppo_turn:
        with torch.no_grad():
            current_state = torch.tensor(nothanks.encode_state_gen_3()).to(device)
            legal_move = nothanks.get_legal_action() # a list 
            legal_move_mask = torch.tensor([False if move in legal_move else True for move in nothanks.move_encode.values()]).to(device)
            move_raw, log_prob, entropy, value = model.forward(current_state, legal_move_mask)
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
