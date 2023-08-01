import gymnasium as gym
from gymnasium.envs.registration import register
from stable_baselines3 import A2C, PPO, DQN
import utils
import os
import random
import pandas as pd
import numpy as np
from scipy.stats import ttest_ind
from scipy.stats import levene
from statsmodels.stats.power import TTestIndPower

def register_env(approx=False, hybrid=False, blueprint=None, action_space_type="discrete", evalFlag=""):
    if approx == True: hybrid = False
    if blueprint is None: raise ValueError("No blueprint specified in model-name")
    register(
    # unique identifier for the env `name-version`
    id=f"PharmacophoreEnv-v0-Eval-{action_space_type}{evalFlag}",
    # path to the class for creating the env
    entry_point="customenv:PharmacophoreEnv",
    max_episode_steps=200,
    # Max number of steps per episode, using a `TimeLimitWrapper`
    kwargs={"output": r'C:\\Users\\kilia\\MASTER\\rlpharm\\data\\hitlists\\hitlist', 
            "querys": r'C:\\Users\\kilia\\MASTER\\rlpharm\\data\\querys' + blueprint, 
            "actives_db": r'C:\\Users\\kilia\\MASTER\\rlpharm\\data\\ldb2s\\actives.ldb2',
            "inactives_db": r"C:\\Users\\kilia\\MASTER\\rlpharm\\data\\ldb2s\\inactives.ldb2",
            "data_dir": "C:\\Users\\kilia\\MASTER\\rlpharm\\data\\",
            "ldba": 58, # 58|36
            "ldbi": 177, # 177|112
            "features": "H,HBA,HBD",
            "enable_approximator": approx,
            "hybrid_reward": hybrid,
            "inf_mode": False,
            "threshold": 0.77,
            "render_mode": "console",
            "verbose": 3,
            "ep_length": 200,
            "delta": 0.2,
            "action_space_type": action_space_type
            },
    )

def run_experiment(max_timesteps=None, model_folder_path=None, mode="best", starting_point=None, approx=False, hybrid=False, ran_folder=""):
    """
    Run experiment for all models in a specified folder.
    :param max_timesteps: maximum number of timesteps to run inference for each model
    :param model_folder_path: path to folder containing models
    :param mode: "best" or "mean_best" (best pharmacophore or mean of rewards over all timesteps) 
    :param starting_point: starting point or initial pharmacophore as list of feature tolerances int the same order they are in the pml file
    :param approx: True: enables approximator model in inference steps, default: False
    :param hybrid: True: saves all steps, states and rewards in a csv file, default: False
    """
    env=None
    if max_timesteps is None:
        max_timesteps = 100
    if model_folder_path is None:
        raise ValueError("model_folder must be specified as valid path")
    erg = {}
    model = None
    filenames = os.listdir(model_folder_path)
    for filename in filenames:
        
        blueprint = '\\\\' + filename.split('_')[2] + '.pml'
        aType = None
        match filename.split("_")[1]:
            case "BOX":
                aType = "box"
            case "DISCRETE":
                aType = "discrete"
            case _:
                raise ValueError("model-action-space-type must be specified in the filename as BOX or DISCRETE")
            
        match(filename[0:3]):
            case "DQN":
                model = DQN.load(path=model_folder_path + "\\" + filename, env=env)
            case "PPO":
                model = PPO.load(model_folder_path + "\\" + filename)
            case "A2C":
                model = A2C.load(model_folder_path + "\\" + filename)
            case _:
                raise ValueError("model must be of type DQN, PPO or A2C")
        
        register_env(approx=approx, hybrid=hybrid, blueprint=blueprint, action_space_type=aType)
        register_env(approx=False, hybrid=False, blueprint=blueprint, action_space_type=aType, evalFlag="-test")
        env = gym.make(f"PharmacophoreEnv-v0-Eval-{aType}")    
        test_env = gym.make(f"PharmacophoreEnv-v0-Eval-{aType}-test")  
        obs, _ = env.reset()

        if str(model.action_space) != str(env.action_space):
            raise ValueError("model-action-space-type must match environment-action-space-type")
        
        if starting_point != None:
            env.write_values_to_tree(starting_point, runtime=True)
        
        auc, ef, _, _ = env.screening()
        buffer = {"reward":(auc+auc+ef)/3, "phar":obs}
        print(buffer["reward"])
        accumulated_reward = 0
        n = 0
        if mode == "stat":
            replay_buffer = []
        END = False
        while END == False:
            n += 1
            action, _ = model.predict(obs)
            obs, reward, terminated, truncated, _ = env.step(action)
            END = terminated or n > max_timesteps
            if mode == "stat":
                replay_buffer.append(reward)
            if mode == "mean_best" or mode == "stat":
                accumulated_reward += reward
            if terminated:
                print('terminated')
                print(n)
            if reward > buffer["reward"]:
                buffer["reward"] = reward
                buffer["phar"] = obs
            if truncated:
                print('truncated')
                obs, _ = env.reset()
        if mode == "best":
            if approx:
                buffer["reward"] = str(buffer["reward"]) + "\tactual reward: " + str(check_approximation(buffer["phar"], test_env, blueprint))
            erg[filename] = buffer
        if mode == "stat":
            folder_path = r"C:\Users\kilia\MASTER\rlpharm\data"
            files = os.listdir(folder_path + "\\" + ran_folder)
            ran_file = random.choice(files)
            df = pd.read_csv(os.path.join(folder_path + "\\" + ran_folder, ran_file))
            list2 = df.iloc[:, -1].tolist()
            info = stat_tests(list1=replay_buffer ,list2=list2)
            erg[filename] = {"reward": accumulated_reward/n, "info": info}
        if mode == "mean_best":
            erg[filename] = {"reward": accumulated_reward/n, "phar": buffer["phar"]}
    return erg

def check_approximation(phar, env, blueprint):
    """
    Tests a reward predicted by an AI model on a real Virtual Screening benchmark
    """
    _, _ = env.reset()

    output_file = r'C:\\Users\\kilia\\MASTER\\rlpharm\\data\\hitlists\\hitlist'
    query = r'C:\\Users\\kilia\\MASTER\\rlpharm\\data\\querys' + blueprint[:-4] + '-TMP.pml'
    actives_db = r'C:\\Users\\kilia\\MASTER\\rlpharm\\data\\ldb2s\\actives.ldb2:active'
    inactives_db = r"C:\\Users\\kilia\\MASTER\\rlpharm\\data\\ldb2s\\inactives.ldb2:inactive"

    if isinstance(phar, dict):
        values = []
        for key in phar.keys():
            values.extend(phar[key])
    else:
        values = phar
    print(query)
    env.obs_to_pml(values, query)
    hits, scores, pos, neg = utils.exec_vhts(output_file, query, actives_db, inactives_db)
    print(pos)
    print(neg)
    auc, ef = env.scoring(hits, scores, pos, neg)
    print(phar)
    print(auc)

    print(f"Reward: {(auc+auc+auc+ef)/4}")

    return (auc+auc+auc+ef)/4

def random_sample_creation(n_feat, a, b, n_lists):
    register_env(approx=False, hybrid=False, blueprint="\\\\" + "sEH-1ZD5-mod5-LS-3.02.pml", action_space_type="discrete")
    env = gym.make(f"PharmacophoreEnv-v0-Eval-discrete")
    _, _ = env.reset()
    output_file = r'C:\\Users\\kilia\\MASTER\\rlpharm\\data\\hitlists\\hitlist'
    query = r'C:\\Users\\kilia\\MASTER\\rlpharm\\data\\querys' + "\\\\" + "sEH-1ZD5-mod5-LS-3.02-TMP.pml"
    actives_db = r'C:\\Users\\kilia\\MASTER\\rlpharm\\data\\ldb2s\\actives.ldb2:active'
    inactives_db = r"C:\\Users\\kilia\\MASTER\\rlpharm\\data\\ldb2s\\inactives.ldb2:inactive"
    
    for idx in range(n_lists):
        lists = []        
        for _ in range(100):
            # Generate a list of n random floats rounded to 2 decimal places
            random_floats = [round(random.uniform(a, b), 1) for _ in range(n_feat)]
            env.obs_to_pml(random_floats, query)
            hits, scores, pos, neg = utils.exec_vhts(output_file, query, actives_db, inactives_db)
            auc, ef = env.scoring(hits, scores, pos, neg)
            reward = (auc+auc+auc+ef)/4
            random_floats.append(reward)
            lists.append(random_floats)
        
        df = pd.DataFrame(lists)
        df.to_csv(r'C:\Users\kilia\MASTER\rlpharm\data\ran_1ZD5\random_'+ str(idx) + '.csv', index=False)

def stat_tests(list1, list2):
    info = []

    _, p = levene(list1, list2)
    # If p is less than 0.05, the variances are not equal
    if p < 0.05:
        print("The variances are not equal -> perform welch's ttest")
        ind_t_test = ttest_ind(list1, list2, alternative="greater", equal_var=False)
    else:
        print("The variances are equal.")
        ind_t_test = ttest_ind(list1, list2, alternative="greater")

    # Perform independent t-test, statistic is positive when sample mean of agent is greater than random
    info.append("t statistic: " + str(ind_t_test[0]))
    info.append("p value: " + str(ind_t_test[1]))
    print("t statistic: ", ind_t_test[0])
    print("p value: ", ind_t_test[1])

    # Calculate the means and standard deviations
    mean1, mean2 = np.mean(list1), np.mean(list2)
    std1, std2 = np.std(list1, ddof=1), np.std(list2, ddof=1)

    # Calculate the effect size
    effect_size = (mean1 - mean2) / np.sqrt((std1**2 + std2**2) / 2)
    info.append("eff_size: " + str(effect_size))
    print("eff_size: " + str(effect_size))
    
    # Parameters for power analysis
    alpha = 0.05  # significance level
    power = 0.8  # power level

    # Perform power analysis
    analysis = TTestIndPower()
    sample_size = analysis.solve_power(effect_size, power=power, nobs1=None, ratio=1.0, alpha=alpha)
    print("Sample size: ", sample_size)
    
    return '\t'.join(info)
