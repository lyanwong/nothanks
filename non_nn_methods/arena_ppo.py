from ppo_model import *
import os
from utils import *
import pandas as pd

# model_list = [i for i in os.listdir('/kaggle/working') if i.startswith(model_prefix)]
model_list = ['model_gen_3_default_rwd_60_iter.pth',
              'model_gen_3_default_rwd_50_iter.pth',
              'model_gen_3_complex_rwd_80_iter.pth',
              'model_gen_2_complx_rwd_40_iter.pth',
              'model_gen_2_complx_rwd_50_iter.pth',
              'model_gen_2_default_rwd_30_iter_.pth',
              'model_gen_2_default_rwd_113_iter.pth'
             ]
model_name_dict = {a:b for a, b in enumerate(model_list)}
n_model = len(model_name_dict)

select_record = {i:0 for i in range(n_model)}
win_record = {i:0 for i in range(n_model)}
move_encode = {"0": "pass",
                "1": "take"}

n_match = 500
N_PLAYER = 3
device = 'cuda' if torch.cuda.is_available() else 'cpu'

for _ in tqdm(range(n_match)):
    
    model_index = random.sample(range(n_model), k = 3)        
    for index in model_index:
        select_record[index] += 1
    model_list = []
    for index in model_index:
        model_name = model_name_dict.get(index)
        if 'gen_2' in model_name:
            model = ppo_gen_2(N_PLAYER).to(device)
        else:
            model = ppo_gen_3(N_PLAYER).to(device)

        path = f'./ppo_weight/{model_name_dict.get(index)}'
        model.load_state_dict(torch.load(path, map_location=torch.device('cpu')))
        model_list.append(model)
        
    nothanks = game()
    while nothanks.is_continue:
        with torch.no_grad():
        
            model_tmp = model_list[nothanks.turn]
            if model_tmp.gen == 2:
                current_state = torch.tensor(nothanks.encode_state_gen_2()).to(device)
            else:
                current_state = torch.tensor(nothanks.encode_state_gen_3()).to(device)
            legal_move = nothanks.get_legal_action() # a list 
            legal_move_mask = torch.tensor([False if move in legal_move else True for move in nothanks.move_encode.values()]).to(device)
            move_raw, log_prob, entropy, value = model_tmp.forward(current_state, legal_move_mask)
            move = nothanks.move_encode.get(move_raw.item())
        nothanks.action(move)    
    
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