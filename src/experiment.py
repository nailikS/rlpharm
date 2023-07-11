import model_test as mt
import pandas as pd
from datetime import datetime

run_folder = r"C:\Users\kilia\MASTER\rlpharm\data\experiments"
erg = mt.run_experiment(max_timesteps=100, 
                        model_folder_path=r"C:\Users\kilia\MASTER\rlpharm\data\model-eval", 
                        mode="best",
                        approx=False, 
                        hybrid=True)

#loop over dict keys
dfs = []
exp_name = f'\\exp_{datetime.now().strftime("%Y_%m_%d-%I_%M")}.csv'
for key in erg.keys():
    d = {}
    d['model'] = key
    d["type"] = key[0:3]
    d['reward'] = erg[key]['reward']
    values = []
    for k in erg[key]['phar'].keys():
        values.extend(erg[key]['phar'][k])
    d['best_pharmacophore'] = str(values)
    #d['best_pharmacophore'] = str(erg[key]['phar'])
    dfs.append(pd.DataFrame(d, index=[0]))
df = pd.concat(dfs)
df.to_csv(run_folder + exp_name, index=False)