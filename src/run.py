from stable_baselines3 import PPO
from wandb.integration.sb3 import WandbCallback
import wandb
import time
from stable_baselines3.common.monitor import Monitor
import gymnasium as gym
from gymnasium.envs.registration import register


# Example for the CartPole environment
register(
    # unique identifier for the env `name-version`
    id="PharmacophoreEnv-v0",
    # path to the class for creating the env
    entry_point="customenv:PharmacophoreEnv",
    # Max number of steps per episode, using a `TimeLimitWrapper`
    kwargs={"output": r'C:\Users\kilia\MASTER\rlpharm\data\hitlists\hitlist', 
            "querys": r'C:\Users\kilia\MASTER\V2Z1551.pml', 
            "actives_db": r'C:\Users\kilia\MASTER\rlpharm\data\seh_actives_mini.ldb',
            "inactives_db": r"C:\Users\kilia\MASTER\rlpharm\data\seh_inactives_mini.ldb",
            "ldba": 36,
            "ldbi": 112,
            "features": "H,HBA,HBD"},
)

env = gym.make("PharmacophoreEnv-v0")
env = Monitor(env)

config = {"policy_type": "MultiInputPolicy", "total_timesteps": 500000}
experiment_name = f"PPO_{int(time.time())}"

wandb.tensorboard.patch(root_logdir=f"runs/{experiment_name}")
wandb.init(project="repharm", config=config, name=experiment_name, sync_tensorboard=True)
# Define and Train the agent
model = PPO(config["policy_type"], env, verbose=2, tensorboard_log=f"runs/{experiment_name}")

model.learn(config["total_timesteps"], log_interval=1, 
            callback=WandbCallback(gradient_save_freq=100,
                                   model_save_freq=100, 
                                   model_save_path=f"models/{experiment_name}",
                                   verbose=2
        )
)
wandb.finish()