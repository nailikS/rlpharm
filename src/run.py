from datetime import datetime
from multiprocessing import Process
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

def train1():
    query = r'C:\\Users\\kilia\\MASTER\\rlpharm\\data\\querys\\sEH-1ZD5-mod5-LS-3.02.pml'
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
            "actives_db": r'C:\\Users\\kilia\\MASTER\\rlpharm\\data\\ldb2s\\actives.ldb2',
            "inactives_db": r"C:\\Users\\kilia\\MASTER\\rlpharm\\data\\ldb2s\\inactives.ldb2",
            "data_dir": "C:\\Users\\kilia\\MASTER\\rlpharm\\data\\",
            "ldba": 58,
            "ldbi": 177,
            "features": "H,HBA,HBD",
            "enable_approximator": False,
            "hybrid_reward": True,
            "buffer_path": r"C:\Users\kilia\MASTER\rlpharm\data\sEH-1ZD5-mod5-LS-3.02.csv",
            "inf_mode": False,
            "threshold": 0.79,
            "render_mode": "console",
            "verbose": 3,
            "ep_length": 100,
            "delta": 0.25,
            "action_space_type": "discrete",
            },
    ) 
    env = gym.make("PharmacophoreEnv-v0")
    obs, _ = env.reset()
    env = Monitor(env)

    config = {"policy_type": "MultiInputPolicy", "total_timesteps": 30000}
    experiment_name = datetime.now().strftime("%Y_%m_%d-%I_%M_%S_%f")
    type = "PPO"
    bp = query.split('\\')[-1][:-4]
    action_space_type = "DISCRETE"
    wandb.tensorboard.patch(root_logdir=f"runs/{experiment_name}")
    wandb.init(project="repharm", config=config, name=experiment_name, sync_tensorboard=True)

    # Define and Train the agent
    model = DQN(config["policy_type"], env, verbose=2, tensorboard_log=f"runs/{experiment_name}")

    model.learn(config["total_timesteps"], log_interval=10, 
                callback=[WandbCallback(gradient_save_freq=100,
                                       model_save_freq=1000, 
                                       model_save_path=f"models/{type}_{experiment_name}",
                                       verbose=2
            ),
            #eval_callback
            #dgc.CustomCallback(r"C:\Users\kilia\MASTER\rlpharm\data\approx.csv")
            ]
    )

    model.save(f"{type}_{action_space_type}_{bp}_{experiment_name}")
    env.env.close()
    wandb.finish()

def train2():
    query = r'C:\Users\kilia\MASTER\rlpharm\data\querys\sEH-1ZD5-mod5-LS-3.02.pml'
    register(
        # unique identifier for the env `name-version`
        id="PharmacophoreEnv-v1",
        # path to the class for creating the env
        entry_point="customenv:PharmacophoreEnv",
        max_episode_steps=200,
        # Max number of steps per episode, using a `TimeLimitWrapper`
        kwargs={
            "output": r'C:\\Users\\kilia\\MASTER\\rlpharm\\data\\hitlists\\hitlist', 
            "querys": query, 
            "actives_db": r'C:\\Users\\kilia\\MASTER\\rlpharm\\data\\ldb2s\\actives.ldb2',
            "inactives_db": r"C:\\Users\\kilia\\MASTER\\rlpharm\\data\\ldb2s\\inactives.ldb2",
            "data_dir": "C:\\Users\\kilia\\MASTER\\rlpharm\\data\\",
            "ldba": 58,
            "ldbi": 177,
            "features": "H,HBA,HBD",
            "enable_approximator": False,
            "hybrid_reward": True,
            "buffer_path": r"C:\Users\kilia\MASTER\rlpharm\data\sEH-1ZD5-mod5-LS-3.02_p.csv",
            "inf_mode": False,
            "threshold": 0.79,
            "render_mode": "console",
            "verbose": 3,
            "ep_length": 100,
            "delta": 0.3,
            "action_space_type": "discrete",
            },
    )
    env = gym.make("PharmacophoreEnv-v1")
    obs, _ = env.reset()
    env = Monitor(env)

    # Separate evaluation env
    #eval_env = gym.make("PharmacophoreEnv-v0")
    # Stop training if there is no improvement after more than 3 evaluations
    #stop_train_callback = StopTrainingOnNoModelImprovement(max_no_improvement_evals=5, min_evals=5, verbose=1)
    #eval_callback = EvalCallback(eval_env, eval_freq=1000, callback_after_eval=stop_train_callback, verbose=1)

    config = {"policy_type": "MultiInputPolicy", "total_timesteps": 30000}
    experiment_name = datetime.now().strftime("%Y_%m_%d-%I_%M_%S_%f")
    type = "PPO"
    bp = query.split('\\')[-1][:-4]
    action_space_type = "DISCRETE"
    wandb.init(project="repharm", config=config, name=experiment_name)

    # Define and Train the agent
    model = DQN(config["policy_type"], env, verbose=2, tensorboard_log=f"runs/{experiment_name}")

    model.learn(config["total_timesteps"], log_interval=100, 
                callback=[WandbCallback(gradient_save_freq=100,
                                    model_save_freq=1000, 
                                    model_save_path=f"models/{type}_{experiment_name}",
                                    verbose=2
            ),
            #eval_callback
            #dgc.CustomCallback(r"C:\Users\kilia\MASTER\rlpharm\data\approx.csv")
            ]
    )

    model.save(f"{type}_{action_space_type}_{bp}_{experiment_name}")
    env.close()
    wandb.finish()


def runInParallel(*fns):
  proc = []
  for fn in fns:
    p = Process(target=fn)
    p.start()
    proc.append(p)
  for p in proc:
    p.join()

if __name__ == '__main__':
    runInParallel(train1, train2)

# train1()
# train2()    