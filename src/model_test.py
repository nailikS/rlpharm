import gymnasium as gym
from gymnasium.envs.registration import register
from stable_baselines3 import A2C

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
            "enable_approximator": False,
            "hybrid_reward": True,
            "inf_mode": False,
            },
)
env = gym.make("PharmacophoreEnv-v0")

model = A2C.load(r"C:\Users\kilia\MASTER\rlpharm\src\A2C1684660535.zip")

obs, _ = env.reset()
replay_buffer = {"reward":0, "phar":{}}
END = False
while not END:
    action, _states = model.predict(obs)
    obs, reward, terminated, truncated, info = env.step(action)
    END = terminated
    print(obs)
    print(reward)
    if reward > replay_buffer["reward"]:
        replay_buffer["reward"] = reward
        replay_buffer["phar"] = obs
    if truncated:
      print('truncated')
      obs, _ = env.reset()

