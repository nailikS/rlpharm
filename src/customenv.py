import gymnasium as gym
import numpy as np
from gymnasium import spaces
import utils
import csv
import torch
import torch.nn as nn
import time
import pandas as pd
from sklearn.metrics import roc_auc_score

class PharmacophoreEnv(gym.Env):
    """Custom Environment that follows gym interface."""

    # TODO: scoring function should utilize rocAUC
    # TODO: Pharmacophore io functions need to be provided for direct access

    def __init__(self, 
                 output, 
                 querys, 
                 actives_db, 
                 inactives_db, 
                 approximator, 
                 ldba, 
                 ldbi, 
                 features, 
                 enable_approximator=False,
                 hybrid_reward=False,
                 buffer_path="../data/approxCollection.csv",
                 inf_mode=False,
                 ):
        super().__init__()
        # Define action and observation space
        # They must be gym.spaces objects
        # Example when using discrete actions:
        self.inference_mode = inf_mode
        self.enable_approximator = enable_approximator
        # TODO: transfer model specification to config file
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
        self.buffer_path = buffer_path
        self.threshold = 0.7 # threshold for reward, TODO: make this a parameter
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
        if hybrid_reward:
            self.hybrid_reward = True
            self.replay_buffer = pd.read_csv(buffer_path)
        self.timings = []
        anvec = 0
        for i in range(len(self.featureIds)): 
            if i==0 or i==3:
                anvec += len(self.featureIds[i]*2)
            if i==1 or i==2:
                anvec += len(self.featureIds[i]*4)
        
        # Initialization of Spaces
        self.action_space = spaces.Discrete(anvec)
        self.observation_space = spaces.Dict(self.get_observation_space())

    def get_reward(self):
        """
        Execute VHTS and calculate score
        :return: score query pharmacophores against actives and inactives database        
        """
        if self.enable_approximator:
            obs = self.last_observation
            values = []
            for key in obs.keys():
                values.extend(obs[key])
            with torch.no_grad():
                return self.approximator(torch.tensor(values, dtype=torch.float32)).item()
        if self.hybrid_reward:
            obs = self.last_observation
            values = []
            for key in obs.keys():
                values.extend(obs[key])
            matching_rows = self.replay_buffer.loc[(self.replay_buffer[self.replay_buffer.columns[1:-2]] == values).all(axis=1)]
            if not matching_rows.empty:
                return matching_rows.iloc[0, 0], matching_rows.iloc[0, -2], matching_rows.iloc[0, -1]
            else:
                reward, p, n = self.screening()
                new_row = [reward] + values + [p, n] 
                self.replay_buffer.loc[len(self.replay_buffer)] = new_row
                return reward, p, n

    def refresh_buffer(self):
        self.replay_buffer.to_csv(self.buffer_path, index=False)

    def screening(self):
        """
        Handles pharmacophore IO and execution of virtual screening function in utils.py
        :return: rocAUC score for the provided pharmacophore
        """   
        self.temp_querys = self.querys[:-4]+"_temp"+self.querys[-4:]
        self.phar_modified.write(self.temp_querys, encoding="utf-8", xml_declaration=True)
        hits, scores, pos, neg = utils.exec_vhts(output_file=self.out_file, 
                                                querys=self.temp_querys, 
                                                actives_db=self.actives_db, 
                                                inactives_db=self.inactives_db)
        return self.scoring(hits, scores), pos, neg

    def scoring(self, hits, scores):
        """
        Calculate score
        :param hits: list of hit labels (0=FP or 1:TP)
        :param scores: list of pharmacophore fit scores
        :return: rocAUC of the hitlist
        """     
        # Calculate ROC AUC with weighted cost for false positives
        # Set the weight for false positives
        # weight = 10
        # iterate over hits and scores, every time hits is 0, add 9 zeros on that index and multiply the score in the same position by 10
        idx = np.where(np.array(hits) == 0)[0]
        j=0
        for i in idx:
            hits = np.insert(hits, i+j, np.zeros(9))
            scores = np.insert(scores, i+j, np.full(9,scores[i+j])) 
            j += 9
        return roc_auc_score(hits, scores)
    
    def step(self, action):
        truncated = []
        # Execute one time step within the environment
        self.phar_modified = utils.action_execution(action, self.featureIds, self.phar_modified, self.phar)

        # new observation (state)
        self.last_observation = self.get_observation(initial=False)        
        
        # Truncated if episode exceeds timestep limit
        truncated.append(self.counter > 200)
        
        # changes made to the pharmacophore in total, returned in info
        diff = {}
        if self.inference_mode:
            self.refresh_buffer()

        if self.counter % 100 == 0:
            # writes updated replay buffer to filesystem
            print(np.mean(self.timings[-100:]))
            for key in self.last_observation.keys():
                diff[key] = np.subtract(self.last_observation[key], self.initial_os[key])            
        if self.counter % 10 == 0:
            self.refresh_buffer()   
        
        # check boundaries
        for key in self.last_observation.keys():
            truncated.append(not np.logical_and(np.all(self.last_observation[key] >= 1), 
                                                np.all(self.last_observation[key] <= 5)))
        if np.any(truncated):
            self.reward = 0
        
        # Evaluate and calculate reward
        start_time = time.time()
        self.reward, pos, neg = self.get_reward()
        timing = time.time() - start_time
        
        # Episode termination conditions

        primary_ = (self.reward > self.threshold)
        secondary_ = (pos > self.ldba_size/10 and neg == 0)
        terminated = (primary_ or secondary_)

        self.timings.append(timing)
        self.counter += 1
        if terminated: print("terminated")
        if primary_: print("threshold of "+str(self.threshold)+" reached")
        if secondary_: print(f"10% ({self.ldba_size/10}) and no neg")
        # if not self.enable_approximator: print('\n'.join(timings))
        # print(str(self.last_observation)+"\t"+str(truncated)+"\t"+str(self.counter))
        return self.last_observation, self.reward, terminated, np.any(truncated), {"performance": self.reward, "diff": diff}
    
    def reset(self, seed=None, options=None):
        super().reset()
        # Reset the state of the environment to an initial state
        self.counter = 0
        self.phar_modified = None
        return self.get_observation(initial=True), {}
    
    def get_observation(self, initial=False):
        os_space = dict()
        for i, f in zip(range(len(self.featureIds)),self.features):
            x = []              
            if initial:
                if self.inference_mode == True:
                    for id in self.featureIds[i]:
                        x.extend([utils.get_tol(self.phar, id)])
                    x = np.array(x, dtype=np.float64)
                    os_space[f] = np.around(x.flatten(), decimals=1)
                else:
                    for id in self.featureIds[i]:
                        x.extend([utils.get_tol(self.phar, id)])
                    x = np.array(x, dtype=np.float64)
                    rans = np.around(np.random.uniform(low=2, high=4, size=(len(x.flatten()),)), decimals=1)
                    os_space[f] = rans
                    # write all rans to tree
                    if i==0 or i==3:
                        for i, id in enumerate(self.featureIds[i]):
                            self.phar = utils.set_tol(self.phar, id, rans[i])
                    if i==1 or i==2:
                        for id in self.featureIds[i]:    
                            for j in range(0,len(self.featureIds[i])*2,2):
                                self.phar = utils.set_tol(self.phar, id, rans[j], target="origin")
                                self.phar = utils.set_tol(self.phar, id, rans[j+1], target="target")
            else:
                for id in self.featureIds[i]:
                    x.extend([utils.get_tol(self.phar_modified, id)])
                x = np.array(x, dtype=np.float64)
                os_space[f] = np.around(x.flatten(), decimals=1)
        return os_space

    def write_values_to_tree(self, values):
        writes = 0
        for i in range(len(self.featureIds)):
            if i==0 or i==3:
                for i, id in enumerate(self.featureIds[i]):
                    self.phar = utils.set_tol(self.phar, id, values[i])
                    writes += 1
            if i==1 or i==2:
                for id in self.featureIds[i]:    
                    self.phar = utils.set_tol(self.phar, id, values[writes], target="origin")
                    self.phar = utils.set_tol(self.phar, id, values[writes+1], target="target")
                    writes += 2
    
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
        """
        Purely for generating screening observations for training of approximation models
        :param n: number of observations to generate
        :param csv_file: file to write observation spaces + rewards to
        """
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
            
    def obs_to_pml(self, observation, filename=None):
        """
        Writes observation to pml file
        """
        if filename == None:
            filename = self.querys[:-4]+"_temp"+self.querys[-4:]
        
        self.write_values_to_tree(observation)

        self.phar.write(filename, encoding="utf-8", xml_declaration=True)
        



    def render(self, mode="console"):
        ...
    def close(self):
        ...

    
    


