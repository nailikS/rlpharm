from stable_baselines3 import PPO
from wandb.integration.sb3 import WandbCallback
import wandb
import time
from stable_baselines3.common.monitor import Monitor
import gymnasium as gym
from gymnasium.envs.registration import register
from stable_baselines3 import A2C
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnNoModelImprovement
#import dataGeneratorCallback as dgc
from stable_baselines3.common.env_util import make_vec_env

register(
    # unique identifier for the env `name-version`
    id="PharmacophoreEnv-v0",
    # path to the class for creating the env
    entry_point="customenv:PharmacophoreEnv",
    max_episode_steps=200,
    # Max number of steps per episode, using a `TimeLimitWrapper`
    kwargs={
        "output": r'C:\\Users\\kilia\\MASTER\\rlpharm\\data\\hitlists\\hitlist', 
        "querys": r'C:\\Users\\kilia\\MASTER\\rlpharm\\data\\querys\\sEH-1ZD5_mod5_LS_3.02.pml', 
        "actives_db": r'C:\\Users\\kilia\\MASTER\\rlpharm\\data\\ldb2s\\actives_mini.ldb2',
        "inactives_db": r"C:\\Users\\kilia\\MASTER\\rlpharm\\data\\ldb2s\\inactives_mini.ldb2",
        "approximator": r"C:\Users\kilia\MASTER\rlpharm\data\models\approximator\best.pt",
        "ldba": 36,
        "ldbi": 112,
        "features": "H,HBA,HBD",
        "enable_approximator": False,
        "hybrid_reward": True,
        "inf_mode": False
        },
)
env = gym.make("PharmacophoreEnv-v0")
obs, _ = env.reset()
env = Monitor(env)
#vec_env = make_vec_env("PharmacophoreEnv-v0", n_envs=6)
# Separate evaluation env
#eval_env = gym.make("PharmacophoreEnv-v0")
# Stop training if there is no improvement after more than 3 evaluations
#stop_train_callback = StopTrainingOnNoModelImprovement(max_no_improvement_evals=3, min_evals=5, verbose=1)
#eval_callback = EvalCallback(eval_env, eval_freq=1000, callback_after_eval=stop_train_callback, verbose=1)

config = {"policy_type": "MultiInputPolicy", "total_timesteps": 10000}
experiment_name = f"{int(time.time())}"

wandb.tensorboard.patch(root_logdir=f"runs/{experiment_name}")
wandb.init(project="repharm", config=config, name=experiment_name, sync_tensorboard=True)

# Define and Train the agent
model = PPO(config["policy_type"], env, verbose=2, tensorboard_log=f"runs/{experiment_name}")

model.learn(config["total_timesteps"], log_interval=100, 
            callback=[WandbCallback(gradient_save_freq=100,
                                   model_save_freq=500, 
                                   model_save_path=f"models/{experiment_name}",
                                   verbose=2
        ),
        #dgc.CustomCallback(r"C:\Users\kilia\MASTER\rlpharm\data\approx.csv")
        ]
)
model.save(f"PPO{experiment_name}")
wandb.finish()