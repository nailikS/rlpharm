import gymnasium as gym
import numpy as np
from gymnasium import spaces
import utils
import csv
import torch
import torch.nn as nn

class PharmacophoreEnv(gym.Env):
    """Custom Environment that follows gym interface."""

    #metadata = {"render.modes": ["console"], "features":{0:"H", 1:"HBA", 2:"HBD", 3:"exclusion"}}

    def __init__(self, output, querys, actives_db, inactives_db, approximator, ldba, ldbi, features, enable_approximator=False):
        super().__init__()
        # Define action and observation space
        # They must be gym.spaces objects
        # Example when using discrete actions:
        self.enable_approximator = enable_approximator
        self.approximator = nn.Sequential(
                                nn.Linear(7, 96),
                                nn.ReLU(),
                                nn.Linear(96, 192),
                                nn.ReLU(),
                                nn.Linear(192, 96),
                                nn.ReLU(),
                                nn.Linear(96, 48),
                                nn.ReLU(),
                                nn.Linear(48, 24),
                                nn.ReLU(),
                                nn.Linear(24, 1),
                            )
        self.approximator.load_state_dict(torch.load(approximator))
        self.features = features.split(",")
        self.bounds = {"H":np.array([1, 5], dtype=np.float32), 
                       "HBA":np.array([1, 5], dtype=np.float32), 
                       "HBD":np.array([1, 5], dtype=np.float32),
                       }
        self.codec = {0:"H", 
                      1:"HBA", 
                      2:"HBD"}
        self.threshold = 20 # threshold for reward, TODO: make this a parameter
        self.out_file = output
        self.querys = querys
        self.actives_db = actives_db+":active"
        self.inactives_db = inactives_db+":inactive"
        self.ldba_size = ldba
        self.ldbi_size = ldbi
        self.phar = utils.read_pharmacophores(querys)
        self.phar_modified = utils.read_pharmacophores(querys)
        self.read_featureIds()
        self.initial_os = self.get_observation(initial=True)
        self.counter = 0
        # Calculation of action space size
        anvec = 0
        for i in range(len(self.featureIds)): 
            if i==0 or i==3:
                anvec += len(self.featureIds[i]*2)
            if i==1 or i==2:
                anvec += len(self.featureIds[i]*4)
        
        # Initialization of Spaces
        self.action_space = spaces.Discrete(anvec)
        self.observation_space = spaces.Dict(self.get_observation_space())

    def screen(self):
        """
        Execute VHTS and calculate score
        :return: score query pharmacophores against actives and inactives database        
        """
        if self.enable_approximator:
            obs = self.initial_os if self.counter == 0 else self.last_observation
            values =[]
            for key in obs.keys():
                values.extend(obs[key])
            with torch.no_grad():
                return self.approximator(torch.tensor(values, dtype=torch.float32))
        # currently for one pharmacophore at a time
        self.temp_querys = self.querys[:-4]+"_temp"+self.querys[-4:]
        self.phar_modified.write(self.temp_querys, encoding="utf-8", xml_declaration=True)
        actives, inactives = utils.exec_vhts(output_file=self.out_file, 
                                             querys=self.temp_querys, 
                                             actives_db=self.actives_db, 
                                             inactives_db=self.inactives_db)
        return self.scoring(actives, inactives)
    
    def scoring(self, actives, inactives):
        """
        Calculate score
        :return: score query pharmacophores against actives and inactives database        
        """       
        if actives == 0:
            return 0
        EF = (actives/(actives+inactives))/(self.ldba_size/(self.ldba_size+self.ldbi_size))
        if inactives == 0:
            inactives = 1
        score = (EF + actives) / (inactives)
        return score
    
    def step(self, action):
        # Execute one time step within the environment
        self.phar_modified = utils.action_execution(action, self.featureIds, self.phar_modified)
        
        # new observation (state)
        self.last_observation = self.get_observation(initial=False)

        # Evaluate and calculate reward
        self.reward = self.screen()
        
        # Episode termination conditions
        terminated = self.reward > self.threshold
        
        # Truncated if episode exceeds timestep limit
        truncated = self.counter > 200
        
        # changes made to the pharmacophore in total, returned in info
        diff = {}
        if self.counter % 10 == 0:
            for key in self.last_observation.keys():
                diff[key] = np.subtract(self.last_observation[key], self.initial_os[key])

        # check boundaries
        terminated = []
        for key in self.last_observation.keys():
            terminated.append(not np.logical_and(np.all(self.last_observation[key] >= 1), 
                                                 np.all(self.last_observation[key] <= 5)))
        if np.any(terminated):
            self.reward = 0
        
        self.counter += 1
        
        return self.last_observation, self.reward, np.any(terminated), truncated, {"performance": self.reward, "diff": diff}
    
    def reset(self, seed=None, options=None):
        # Reset the state of the environment to an initial state
        self.counter = 0
        self.reward = self.screen()
        self.last_observation = self.get_observation(initial=True)
        return self.last_observation, {}
    
    def get_observation(self, initial=False):
        os_space = dict()
        for i, f in zip(range(len(self.featureIds)),self.features):
            x = []
            for id in self.featureIds[i]:
                if initial:
                    x.extend([utils.get_tol(self.phar, id)])
                else:
                    x.extend([utils.get_tol(self.phar_modified, id)])
            x = np.array(x, dtype=np.float32)
            os_space[f] = x.flatten()
        return os_space

    def get_observation_space(self):
        d = self.bounds.copy()
        for i in range(len(self.featureIds)):
            feature = self.features[i]
            lower = self.bounds[feature][0]
            upper = self.bounds[feature][1]
            up = []
            down = []
            if feature == "H": # or feature == "exclusion" 
                for _ in self.featureIds[i]:
                    up.extend([upper])
                    down.extend([lower])
                d[feature] = spaces.Box(low=np.array(down), high=np.array(up), shape=(len(self.featureIds[i]),), dtype=np.float32)
            if feature == "HBA" or feature == "HBD":
                for _ in self.featureIds[i]:
                    up.extend([upper, upper])
                    down.extend([lower, lower])
                d[feature] = spaces.Box(low=np.array(down), high=np.array(up), shape=(len(self.featureIds[i])*2,), dtype=np.float32)
        return d

    def read_featureIds(self):
        featureIds = []
        for i in range(len(self.features)):
            featureIds.append([])
            if self.features[i] != "exclusion":
                for elm in self.phar.findall(".//*[@name='"+self.features[i]+"']"):
                    featureIds[i].append(elm.get("featureId"))
            else:
                for elm in self.phar.findall(".//*[@type='"+self.features[i]+"']"):
                    featureIds[i].append(elm.get("featureId"))
        self.featureIds = featureIds

    def generate_examples(self, n=None, csv_file="examples.csv"):
        if n == None:
            n = 1000
        temp_querys = self.querys[:-4]+"_temp"+self.querys[-4:]
        initial_observation, info = self.reset()
        observation = initial_observation.copy()
        modified_phar = utils.read_pharmacophores(self.querys)
        for i in range(n):
            if i == 0:
                header = ['Reward']
                values = []
                for key in initial_observation.keys():
                    values.extend(initial_observation[key])
                for i in range(len(values)):
                    header.append(f"Feature{i}")
                with open(csv_file, 'w') as f:
                    writer = csv.writer(f)
                    writer.writerow(header)

            action = self.action_space.sample()
             # Execute one time step within the environment
            modified_phar = utils.action_execution(action, self.featureIds, modified_phar)
            modified_phar.write(temp_querys, encoding="utf-8", xml_declaration=True)
            # Evaluate and calculate reward
            reward = self.scoring(*utils.exec_vhts(output_file=self.out_file, 
                                                   querys=temp_querys, 
                                                   actives_db=self.actives_db, 
                                                   inactives_db=self.inactives_db))
            
            for i, f in zip(range(len(self.featureIds)),self.features):
                x = []
                for id in self.featureIds[i]:
                    x.extend([utils.get_tol(modified_phar, id)])
                x = np.array(x, dtype=np.float32)
                observation[f] = x.flatten()
            
            terminated = []
            for key in observation.keys():
                terminated.append(not np.logical_and(np.all(observation[key] >= 1), 
                                                     np.all(observation[key] <= 5)))
            if any(terminated):
                observation = initial_observation.copy()
                modified_phar = utils.read_pharmacophores(self.querys)
            else:
                values =[]
                for key in observation.keys():
                        values.extend(observation[key])
                
                with open(csv_file, 'a') as f:
                    writer = csv.writer(f)
                    new = [reward]
                    new.extend(map(str, values))
                    writer.writerow(new)
            
            



    def render(self, mode="console"):
        ...
    def close(self):
        ...

    
    


