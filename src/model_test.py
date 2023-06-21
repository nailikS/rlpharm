import gymnasium as gym
from gymnasium.envs.registration import register
from stable_baselines3 import A2C, PPO, DQN
import utils
import os
import customenv as ce

def register_env(approx=False, hybrid=True, blueprint=None, action_space_type="discrete"):
    if approx == True: hybrid = False
    if blueprint is None: raise ValueError("No blueprint specified in model-name")
    register(
    # unique identifier for the env `name-version`
    id=f"PharmacophoreEnv-v0-Eval-{action_space_type}",
    # path to the class for creating the env
    entry_point="customenv:PharmacophoreEnv",
    max_episode_steps=200,
    # Max number of steps per episode, using a `TimeLimitWrapper`
    kwargs={"output": r'C:\\Users\\kilia\\MASTER\\rlpharm\\data\\hitlists\\hitlist', 
            "querys": r'C:\\Users\\kilia\\MASTER\\rlpharm\\data\\querys' + blueprint, 
            "actives_db": r'C:\\Users\\kilia\\MASTER\\rlpharm\\data\\ldb2s\\actives.ldb2',
            "inactives_db": r"C:\\Users\\kilia\\MASTER\\rlpharm\\data\\ldb2s\\inactives.ldb2",
            "approximator": r"C:\Users\kilia\MASTER\rlpharm\data\models\approximator\best.pt",
            "data_dir": "C:\\Users\\kilia\\MASTER\\rlpharm\\data\\",
            "ldba": 58, # 58|36
            "ldbi": 177, # 177|112
            "features": "H,HBA,HBD",
            "enable_approximator": approx,
            "hybrid_reward": hybrid,
            "inf_mode": False,
            "threshold": 1.55,
            "render_mode": "console",
            "verbose": 3,
            "ep_length": 200,
            "delta": 0.2,
            "action_space_type": action_space_type
            },
    )

def run_experiment(max_timesteps=None, model_folder_path=None, mode="best", starting_point=None, approx=False, hybrid=True):
    """
    Run experiment for all models in a specified folder.
    :param max_timesteps: maximum number of timesteps to run inference for each model
    :param model_folder_path: path to folder containing models
    :param mode: "best" or "mean_best" (best pharmacophore or mean of rewards over all timesteps) 
    :param starting_point: starting point or initial pharmacophore as list of feature tolerances int the same order they are in the pml file
    :param approx: True: enables approximator model in inference steps, default: False
    :param hybrid: True: saves all steps, states and rewards in a csv file, default: True
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
        match(filename[0:3]):
            case "DQN":
                model = DQN.load(model_folder_path + "\\" + filename)
            case "PPO":
                model = PPO.load(model_folder_path + "\\" + filename)
            case "A2C":
                model = A2C.load(model_folder_path + "\\" + filename)
            case _:
                raise ValueError("model must be of type DQN, PPO or A2C")


        blueprint = '\\\\' + filename.split('_')[2] + '.pml'
        if filename.split("_")[1] == "BOX":
            register_env(approx=approx, hybrid=hybrid, blueprint=blueprint, action_space_type="box")
            env = gym.make(f"PharmacophoreEnv-v0-Eval-box")
            # env = ce.PharmacophoreEnv(
            #     output= r'C:\\Users\\kilia\\MASTER\\rlpharm\\data\\hitlists\\hitlist', 
            #     querys= r'C:\\Users\\kilia\\MASTER\\rlpharm\\data\\querys' + blueprint, 
            #     actives_db= r'C:\\Users\\kilia\\MASTER\\rlpharm\\data\\ldb2s\\actives.ldb2',
            #     inactives_db= r"C:\\Users\\kilia\\MASTER\\rlpharm\\data\\ldb2s\\inactives.ldb2",
            #     approximator= r"C:\Users\kilia\MASTER\rlpharm\data\models\approximator\best.pt",
            #     data_dir= "C:\\Users\\kilia\\MASTER\\rlpharm\\data\\",
            #     ldba= 58, # 58|36
            #     ldbi= 177, # 177|112
            #     features= "H,HBA,HBD",
            #     enable_approximator= approx,
            #     hybrid_reward= hybrid,
            #     inf_mode= False,
            #     threshold= 1.55,
            #     render_mode= "console",
            #     verbose= 3,
            #     ep_length= 200,
            #     delta= 0.2,
            #     action_space_type= "box"
            # )

        elif filename.split("_")[1] == "DISCRETE":
            register_env(approx=approx, hybrid=hybrid, blueprint=blueprint, action_space_type="discrete")
            env = gym.make("PharmacophoreEnv-v0-Eval-discrete")
            # env = ce.PharmacophoreEnv(
            #     output= r'C:\\Users\\kilia\\MASTER\\rlpharm\\data\\hitlists\\hitlist', 
            #     querys= r'C:\\Users\\kilia\\MASTER\\rlpharm\\data\\querys' + blueprint, 
            #     actives_db= r'C:\\Users\\kilia\\MASTER\\rlpharm\\data\\ldb2s\\actives.ldb2',
            #     inactives_db= r"C:\\Users\\kilia\\MASTER\\rlpharm\\data\\ldb2s\\inactives.ldb2",
            #     approximator= r"C:\Users\kilia\MASTER\rlpharm\data\models\approximator\best.pt",
            #     data_dir= "C:\\Users\\kilia\\MASTER\\rlpharm\\data\\",
            #     ldba= 58, # 58|36
            #     ldbi= 177, # 177|112
            #     features= "H,HBA,HBD",
            #     enable_approximator= approx,
            #     hybrid_reward= hybrid,
            #     inf_mode= False,
            #     threshold= 1.55,
            #     render_mode= "console",
            #     verbose= 3,
            #     ep_length= 200,
            #     delta= 0.2,
            #     action_space_type= "discrete"
            # )
        else:
            raise ValueError("model-action-space-type must be specified in the filename as BOX or DISCRETE")
        
        obs, _ = env.reset()

        if str(model.action_space) != str(env.action_space):
            raise ValueError("model-action-space-type must match environment-action-space-type")
        
        if starting_point != None:
            env.write_values_to_tree(starting_point, runtime=True)
        
        auc, ef, _, _ = env.screening()
        replay_buffer = {"reward":auc+ef, "phar":obs}
        accumulated_reward = 0
        n = 0
        END = False
        while END == False:
            n += 1
            action, _ = model.predict(obs)
            obs, reward, terminated, truncated, _ = env.step(action)
            END = terminated or n > max_timesteps
            if mode == "mean_best":
                accumulated_reward += reward
            if terminated:
                print('terminated')
                print(n)
            if reward > replay_buffer["reward"]:
                replay_buffer["reward"] = reward
                replay_buffer["phar"] = obs
            if truncated:
                print('truncated')
                obs, _ = env.reset()
        if mode == "best":
            if approx:
                replay_buffer["reward"] = str(replay_buffer["reward"]) + check_approximation(replay_buffer["phar"], env, temp_path)
            erg[filename] = replay_buffer

        if mode == "mean_best":
            erg[filename] = accumulated_reward/n
    return erg

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

    return auc+ef
