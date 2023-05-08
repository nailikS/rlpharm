import csv
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import VecEnv

class CustomCallback(BaseCallback):
    def __init__(self, csv_file):
        super(CustomCallback, self).__init__()
        self.csv_file = csv_file
        self.writer = None
        
    def _on_training_start(self) -> None:
        
        header = ['Reward']
        for i in range(len(self.getValues())):
            header.append(f"Feature{i}")
        with open(self.csv_file, 'w') as f:
            writer = csv.writer(f)
            writer.writerow(header)
        
    def _on_step(self) -> bool:
        reward = self.locals['rewards'][0]
        flat_obs = self.getValues()
        with open(self.csv_file, 'a') as f:
            writer = csv.writer(f)
            new = [reward]
            new.extend(map(str, flat_obs))
            writer.writerow(new)
        return True

    def getValues(self):
        values = []
        if isinstance(self.training_env, VecEnv):
            obs = np.array(self.training_env.get_attr('last_observation'))
        else:
            obs = np.array(self.training_env.envs[0].last_observation)
        for key in obs[0].keys():
            values.extend(obs[0][key])
        
        return values