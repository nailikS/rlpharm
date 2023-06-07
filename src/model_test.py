import gymnasium as gym
from gymnasium.envs.registration import register
from stable_baselines3 import A2C, PPO, DQN
import utils

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
            "enable_approximator": True,
            "hybrid_reward": False,
            "buffer_path": r"C:\Users\kilia\MASTER\rlpharm\data\inference.csv",
            "inf_mode": False,
            "threshold": 1.6,
            "render_mode": "human",
            },
)
env = gym.make("PharmacophoreEnv-v0")

model = DQN.load(r"C:\Users\kilia\MASTER\rlpharm\src\DQN1685636898.zip")
filename = r'C:\\Users\\kilia\\MASTER\\rlpharm\\src\\temp.pml'

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

output_file = r'C:\\Users\\kilia\\MASTER\\rlpharm\\data\\hitlists\\hitlist'
querys = r'C:\\Users\\kilia\\MASTER\\rlpharm\\data\\querys\\sEH-1ZD5_mod5_LS_3.02_infer_best.pml'
actives_db = r'C:\\Users\\kilia\\MASTER\\rlpharm\\data\\ldb2s\\actives.ldb2:active'
inactives_db = r"C:\\Users\\kilia\\MASTER\\rlpharm\\data\\ldb2s\\inactives.ldb2:inactive"

# register(
#     # unique identifier for the env `name-version`
#     id="PharmacophoreEnv-vTEST",
#     # path to the class for creating the env
#     entry_point="customenv:PharmacophoreEnv",
#     max_episode_steps=200,
#     # Max number of steps per episode, using a `TimeLimitWrapper`
#     kwargs={"output": output_file, 
#             "querys": querys, 
#             "actives_db": actives_db,
#             "inactives_db": inactives_db,
#             "approximator": r"C:\Users\kilia\MASTER\rlpharm\data\models\approximator\best.pt",
#             "ldba": 36, # 58|36
#             "ldbi": 36, # 177|112
#             "features": "H,HBA,HBD",
#             "enable_approximator": False,
#             "hybrid_reward": False,
#             "buffer_path": r"C:\Users\kilia\MASTER\rlpharm\data\inference.csv",
#             "inf_mode": False,
#             "threshold": 1.6,
#             "render_mode": "human",
#             },
# )

values = []
d = dict(replay_buffer["phar"])
for key in d.keys():
    values.extend(d[key])

#env.obs_to_pml(values, filename, runtime=True)
auc, ef = env.scoring(*utils.exec_vhts(output_file, querys, actives_db, inactives_db))


print(replay_buffer)
print(f"Reward = {auc+ef}")
