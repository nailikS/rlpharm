from datetime import datetime
from wandb.integration.sb3 import WandbCallback
import wandb
import time
from stable_baselines3.common.monitor import Monitor
import gymnasium as gym
from gymnasium.envs.registration import register
from stable_baselines3 import A2C, PPO, DQN, SAC
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnNoModelImprovement
#import dataGeneratorCallback as dgc
from stable_baselines3.common.env_util import make_vec_env

query = r'C:\\Users\\kilia\\MASTER\\rlpharm\\data\\querys\\sEH-1VJ5mod+3ANTmod_merged(Ref.3ANT)_LS_3.02.pml'
register(
    # unique identifier for the env `name-version`
    id="PharmacophoreEnv-v0",
    # path to the class for creating the env
    entry_point="customenv:PharmacophoreEnv",
    max_episode_steps=200,
    # Max number of steps per episode, using a `TimeLimitWrapper`
    kwargs={
        "output": r'C:\\Users\\kilia\\MASTER\\rlpharm\\data\\hitlists\\hitlist', 
        "querys": query, 
        "actives_db": r'C:\\Users\\kilia\\MASTER\\rlpharm\\data\\ldb2s\\actives_mini.ldb2',
        "inactives_db": r"C:\\Users\\kilia\\MASTER\\rlpharm\\data\\ldb2s\\inactives_mini.ldb2",
        "approximator": r"C:\Users\kilia\MASTER\rlpharm\data\models\approximator\best.pt",
        "data_dir": "C:\\Users\\kilia\\MASTER\\rlpharm\\data\\",
        "ldba": 36,
        "ldbi": 112,
        "features": "H,HBA,HBD",
        "enable_approximator": False,
        "hybrid_reward": True,
        "buffer_path": r"C:\Users\kilia\MASTER\rlpharm\data\3KOOCollection.csv",
        "inf_mode": False,
        "threshold": 1.6,
        "render_mode": "console",
        "verbose": 3,
        "ep_length": 100,
        "delta": 0.15,
        "action_space_type": "discrete",
        },
)
env = gym.make("PharmacophoreEnv-v0")
obs, _ = env.reset()
env = Monitor(env)

# Separate evaluation env
eval_env = gym.make("PharmacophoreEnv-v0")
# Stop training if there is no improvement after more than 3 evaluations
stop_train_callback = StopTrainingOnNoModelImprovement(max_no_improvement_evals=5, min_evals=5, verbose=1)
eval_callback = EvalCallback(eval_env, eval_freq=1000, callback_after_eval=stop_train_callback, verbose=1)

config = {"policy_type": "MultiInputPolicy", "total_timesteps": 10000}
experiment_name = datetime.now().strftime("%Y_%m_%d-%I_%M_%S_%f")
type = "DQN"
bp = query.split('\\')[-1][:-4]
action_space_type = "BOX"
wandb.tensorboard.patch(root_logdir=f"runs/{experiment_name}")
wandb.init(project="repharm", config=config, name=experiment_name, sync_tensorboard=True)

# Define and Train the agent
model = DQN(config["policy_type"], env, verbose=2, tensorboard_log=f"runs/{experiment_name}")

model.learn(config["total_timesteps"], log_interval=10, 
            callback=[WandbCallback(gradient_save_freq=10,
                                   model_save_freq=100, 
                                   model_save_path=f"models/{type}_{experiment_name}",
                                   verbose=2
        ),
        #eval_callback
        #dgc.CustomCallback(r"C:\Users\kilia\MASTER\rlpharm\data\approx.csv")
        ]
)

model.save(f"{type}_{action_space_type}_{bp}_{experiment_name}")
wandb.finish()