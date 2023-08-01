import model_test as mt
import pandas as pd
from datetime import datetime

def experiment():
    run_folder = r"C:\Users\kilia\MASTER\rlpharm\data\experiments"
    erg = mt.run_experiment(max_timesteps=1, 
                            model_folder_path=r"C:\Users\kilia\MASTER\rlpharm\data\model-eval\single", 
                            mode="best",
                            approx=True, 
                            hybrid=False)

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

def experiment_statistics():
    run_folder = r"C:\Users\kilia\MASTER\rlpharm\data\experiments"
    erg = mt.run_experiment(max_timesteps=100, 
                            model_folder_path=r"C:\Users\kilia\MASTER\rlpharm\data\model-eval\1ZD5", 
                            mode="stat",
                            approx=False, 
                            hybrid=False,
                            ran_folder="ran_1ZD5")

    #loop over dict keys
    dfs = []
    exp_name = f'\\exp_{datetime.now().strftime("%Y_%m_%d-%I_%M")}.csv'
    for key in erg.keys():
        d = {} 
        d['model'] = key
        d["type"] = key[0:3]
        d['reward'] = erg[key]['reward']
        d['stats_test'] = erg[key]['info']
        dfs.append(pd.DataFrame(d, index=[0]))
    df = pd.concat(dfs)
    df.to_csv(run_folder + exp_name, index=False)

def generate_samples():
    mt.random_sample_creation(7,1,5,10)


#generate_samples()
experiment_statistics()