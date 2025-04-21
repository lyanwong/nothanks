from ppo_model import *
import os
from utils import *
import pandas as pd

# model_list = [i for i in os.listdir('/kaggle/working') if i.startswith(model_prefix)]
model_list = [
            'model_gen_3_default_rwd_60_iter.pth',
            'model_gen_3_default_rwd_60_iter.pth',
            'model_gen_3_default_rwd_60_iter.pth',
            #   'model_gen_3_default_rwd_80_iter.pth',
            #   'model_gen_3_default_rwd_50_iter.pth',
            #   'model_gen_5_default_rwd_57_iter.pth',
            #   'model_gen_3_5_default_rwd_62_iter.pth',
            #   'model_gen_3_5_default_rwd_52_iter.pth',
            #   'model_gen_3_5_default_rwd_42_iter.pth'
             ]
model_name_dict = {a:b for a, b in enumerate(model_list)}
n_model = len(model_name_dict)

select_record = {i:0 for i in range(n_model)}
win_record = {i:0 for i in range(n_model)}
move_encode = {"0": "pass",
                "1": "take"}

n_match = 1000
N_PLAYER = 3
device = 'cuda' if torch.cuda.is_available() else 'cpu'

game_length_list = []
max_score_list = []

for _ in tqdm(range(n_match)):
    
    model_index = random.sample(range(n_model), k = 3)        
    for index in model_index:
        select_record[index] += 1
    model_list = []
    for index in model_index:
        model_name = model_name_dict.get(index)
        if 'gen_2' in model_name:
            model = ppo_gen_2(N_PLAYER).to(device)
        elif 'gen_3' in model_name:
            model = ppo_gen_3(N_PLAYER).to(device)
            if 'gen_3_5' in model_name:
                model.gen = 3.5    
        elif 'gen_4' in model_name:
            model = ppo_gen_4(N_PLAYER).to(device)
        elif 'gen_5' in model_name:
            model = ppo_gen_5(N_PLAYER).to(device)
            if 'gen_5_5' in model_name:
                model.gen = 5.5    

        path = f'./ppo_weight/{model_name_dict.get(index)}'
        model.load_state_dict(torch.load(path, map_location=torch.device('cpu')))
        model_list.append(model)
        
    nothanks = game()
    game_length = 0
    while nothanks.is_continue:
        game_length += 1
        print('------------------------------')
        print(f'''Card: {nothanks.current_card} | Chip in pot: {nothanks.chip_in_pot} | Player: {nothanks.turn} - {nothanks.players[nothanks.turn]}\n''')
        print('------------------------------')
        with torch.no_grad():
            model_tmp = model_list[nothanks.turn]
            if model_tmp.gen == 2:
                current_state = torch.tensor(nothanks.encode_state_gen_2()).to(device)
            elif model_tmp.gen == 3:
                current_state = torch.tensor(nothanks.encode_state_gen_3(nothanks.get_state)).to(device)
            elif model_tmp.gen == 3.5:
                current_state = torch.tensor(nothanks.encode_state_gen_3(nothanks.get_state_gen_3_5)).to(device)
            elif model_tmp.gen == 4:
                current_state = torch.tensor(nothanks.encode_state_gen_3(nothanks.get_state_gen_4)).to(device)
            elif model_tmp.gen == 5:
                x_card, x_state = nothanks.encode_state_gen_5(nothanks.get_state)
                x_card = torch.tensor(x_card).float().unsqueeze(1)
                x_state = torch.tensor(x_state)
            elif model_tmp.gen == 5.5:
                x_card, x_state = nothanks.encode_state_gen_5(nothanks.get_state_gen_3_5)
                x_card = torch.tensor(x_card).float().unsqueeze(1)
                x_state = torch.tensor(x_state)
                
            legal_move = nothanks.get_legal_action() # a list 
            legal_move_mask = torch.tensor([False if move in legal_move else True for move in nothanks.move_encode.values()]).to(device)
            if model_tmp.gen in [5, 5.5]:
                move_raw, log_prob, entropy, value = model_tmp.forward(x_card, x_state, legal_move_mask)
            else:
                move_raw, log_prob, entropy, value = model_tmp.forward(current_state, legal_move_mask)
            
            move = nothanks.move_encode.get(move_raw.item())
        print(f"""Move taken: {move}\n""")
        nothanks.action(move)
    for player_tmp in nothanks.players:
        print(player_tmp.calculate_score())
    print(nothanks.calculate_ranking())
    score_list_tmp = [player_tmp.calculate_score() for player_tmp in nothanks.players]
    game_length_list.append(game_length)
    # print(game_length_list)
    max_score_list.append(max(score_list_tmp))
    # print(min_score_list)

    winner_index = np.argmax(nothanks.calculate_ranking())
    win_record[model_index[winner_index]] += 1

pd_result = pd.DataFrame([select_record, win_record]).T\
.rename(columns = {0: 'total',
         1: 'win'
        })\
.assign(win_pct = lambda df: df['win']/df['total'])\
.sort_values('win_pct', ascending = False)\
.reset_index()\
.assign(model_name = lambda df: df['index'].apply(lambda x: model_name_dict.get(x)))

print(pd_result.to_string())

print('Game length: \n', pd.Series(game_length_list).describe())

print('Max score: \n', pd.Series(max_score_list).describe())
