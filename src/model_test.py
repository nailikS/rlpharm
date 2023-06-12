import gymnasium as gym
from gymnasium.envs.registration import register
from stable_baselines3 import A2C, PPO, DQN
import utils
import os

def register_env(approx=False, hybrid=True):
    if approx == True: hybrid = False
    register(
    # unique identifier for the env `name-version`
    id="PharmacophoreEnv-v0",
    # path to the class for creating the env
    entry_point="customenv:PharmacophoreEnv",
    max_episode_steps=200,
    # Max number of steps per episode, using a `TimeLimitWrapper`
    kwargs={"output": r'C:\\Users\\kilia\\MASTER\\rlpharm\\data\\hitlists\\hitlist', 
            "querys": r'C:\\Users\\kilia\\MASTER\\rlpharm\\data\\querys\\sEH-1ZD5_mod5_LS_3.02.pml', 
            "actives_db": r'C:\\Users\\kilia\\MASTER\\rlpharm\\data\\ldb2s\\actives.ldb2',
            "inactives_db": r"C:\\Users\\kilia\\MASTER\\rlpharm\\data\\ldb2s\\inactives.ldb2",
            "approximator": r"C:\Users\kilia\MASTER\rlpharm\data\models\approximator\best.pt",
            "ldba": 58, # 58|36
            "ldbi": 177, # 177|112
            "features": "H,HBA,HBD",
            "enable_approximator": approx,
            "hybrid_reward": False,
            "buffer_path": r"C:\Users\kilia\MASTER\rlpharm\data\inference.csv",
            "inf_mode": False,
            "threshold": 1.5,
            "render_mode": "human",
            },
    )

def run_experiment(max_timesteps=None, model_folder_path=None):
    if max_timesteps is None:
        max_timesteps = 100
    if model_folder_path is None:
        raise ValueError("model_folder must be specified as valid path")
    
    # Create environment
    env = gym.make("PharmacophoreEnv-v0")
    filenames = os.listdir(model_folder_path)

    for filename in filenames:
        model = DQN.load(model_folder_path + filename)


        END = False
        obs, _ = env.reset()
        #env.write_values_to_tree([2,2,2,2,2,2,2], runtime=True)
        replay_buffer = {"reward":0, "phar":{}}
        n=0
        while END == False:
            n+=1
            action, _states = model.predict(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            END = terminated or n > 200
            if terminated:
                print('terminated')
                print(n)
            if reward > replay_buffer["reward"]:
                replay_buffer["reward"] = reward
                replay_buffer["phar"] = obs
            if truncated:
                print('truncated')
                obs, _ = env.reset()
        #env.write_values_to_tree([2,2,2,2,2,2,2], runtime=True)

def check_approximation(phar, env, temp_path):

    output_file = r'C:\\Users\\kilia\\MASTER\\rlpharm\\data\\hitlists\\hitlist'
    querys = r'C:\\Users\\kilia\\MASTER\\rlpharm\\data\\querys\\sEH-1ZD5_mod5_LS_3.02_infer_best.pml'
    actives_db = r'C:\\Users\\kilia\\MASTER\\rlpharm\\data\\ldb2s\\actives.ldb2:active'
    inactives_db = r"C:\\Users\\kilia\\MASTER\\rlpharm\\data\\ldb2s\\inactives.ldb2:inactive"

    if isinstance(phar, dict):
        values = []
        for key in phar.keys():
            values.extend(phar[key])
    else:
        values = phar
    
    env.obs_to_pml(values, temp_path, runtime=True)
    auc, ef = env.scoring(*utils.exec_vhts(output_file, querys, actives_db, inactives_db))
    print(phar)
    print(f"Reward: {auc+ef}")
    
    return auc, ef
