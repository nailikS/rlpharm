from datetime import datetime
import gymnasium as gym
import numpy as np
from gymnasium import spaces
import utils
import csv
import torch
import torch.nn as nn
import time
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os.path
#from itertools import chain
import xml.etree.ElementTree as ET
import xgboost as xgb

class PharmacophoreEnv(gym.Env):
    """Custom Environment that follows gym interface."""
    def __init__(self, 
                 output, 
                 querys, 
                 actives_db, 
                 inactives_db,
                 data_dir, 
                 ldba, 
                 ldbi, 
                 features, 
                 enable_approximator=False,
                 hybrid_reward=None,
                 buffer_path=None,
                 inf_mode=False,
                 threshold=None,
                 render_mode="console",
                 verbose=0,
                 ep_length=100,
                 delta=None,
                 action_space_type=None,
                 model_path=r'C:\Users\kilia\MASTER\rlpharm\notebooks\best_XGB.txt' 
                 ):
        super().__init__()
        if action_space_type is None:
            raise ValueError("action_space_type must be provided")
        else: self.action_space_type = action_space_type        
        self.threshold = 0.76 if threshold == None else threshold
        self.delta = 0.1 if delta == None else delta
        self.inference_mode = inf_mode
        self.enable_approximator = enable_approximator
        self.features = features.split(",")
        self.bounds = [np.array([0, 7], dtype=np.float32), np.array([0, 7], dtype=np.float32), np.array([0, 7], dtype=np.float32)]
        self.codec = {"H":0, "HBA":1, "HBD":2}
        self.buffer_path = buffer_path
        self.threshold = threshold
        # TODO: replace forward slash with double forward slash
        self.out_file = output
        self.querys = querys
        self.actives_db = actives_db+":active"
        self.inactives_db = inactives_db+":inactive"
        self.n_inhibs = ldba
        self.n_decoys = ldbi
        self.max_EF = 1 / (self.n_inhibs / (self.n_inhibs + self.n_decoys))
        self.phar = ET.parse(querys)
        self.phar_modified = ET.parse(querys)
        self.read_featureIds()
        self.counter = 0
        self.render_mode = render_mode
        self.verbose = verbose
        self.episode_length = ep_length
        anvec = 0
        self.timings = []
        self.initial_os = self.get_observation(initial=True)
        self.last_observation = self.initial_os
        log_features = []
        for i in range(len(self.featureIds)): 
            if i==0 or i==3:
                anvec += len(self.featureIds[i]*2)
                log_features.extend(self.featureIds[i])
            if i==1 or i==2:
                anvec += len(self.featureIds[i]*4)
                doubledIDs = []
                for id in self.featureIds[i]:
                    doubledIDs.extend([id+"_orgin", id+"_target"])
                log_features.extend(doubledIDs)
        if hybrid_reward == None:
            self.hybrid_reward = False
        if hybrid_reward == True:
            self.hybrid_reward = True
            if buffer_path == None:
                self.buffer_path = data_dir + "buffer_data\\" + querys.split("\\\\")[-1][:-4] + '_' + datetime.now().strftime("%Y_%m_%d-%I_%M") + ".csv"
                self.replay_buffer = pd.DataFrame(columns=["score", "auc", "ef", "pos", "neg"]+log_features)
                self.replay_buffer.to_csv(self.buffer_path, index=False)
            if not os.path.exists(buffer_path):
                raise ValueError("Buffer creation failed")
            self.replay_buffer = pd.read_csv(buffer_path)
            if len(self.replay_buffer.columns)-5 != len(log_features):
                raise ValueError("Buffer columns do not match the pharmacophore provided")
            else:
                self.replay_buffer = pd.DataFrame(columns=["score", "auc", "ef", "pos", "neg"]+log_features)
                self.buffer_path = data_dir + "buffer_data\\" + buffer_path.split("\\")[-1][:-4] + '_' + datetime.now().strftime("%Y_%m_%d-%I_%M") + ".csv"
                self.replay_buffer.to_csv(self.buffer_path, index=False)

        # Define action and observation space
        if self.action_space_type == "discrete":
            self.action_space = spaces.Discrete(anvec)

        if self.action_space_type == "box":
            self.action_space = self.get_box_action_space()

        self.observation_space = spaces.Dict(self.get_observation_space())
        
        # TODO: transfer model specification to config file
        # approximator setup
        if self.enable_approximator == "xgb":
            self.approximator = xgb.XGBRegressor()
            self.approximator.load_model(model_path)

        if self.enable_approximator == "linear":
            self.approximator = nn.Sequential(
                nn.Linear(len(log_features), 256),
                nn.ReLU(),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Linear(32, 1),
            )
            self.approximator.load_state_dict(torch.load(model_path))
    
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
            # print('shape is:' + str(np.array(values).reshape(1, -1).shape))
            return self.approximator.predict(np.array(values).reshape(1, -1)), 1, 1
        if self.hybrid_reward:
            obs = self.last_observation
            values = []
            for key in obs.keys():
                values.extend(np.round(obs[key], decimals=2))
            matching_rows = self.replay_buffer.loc[(self.replay_buffer[self.replay_buffer.columns[5:]] == values).all(axis=1)]
            if not matching_rows.empty:
                return matching_rows.iloc[0, 0], matching_rows.iloc[0, 3], matching_rows.iloc[0, 4]
            else:
                auc, ef, p, n = self.screening()
                if auc == 0 and ef == 0:
                    return 0, 0, 0
                x = (auc*3 + ef)/4
                new_row = [x, auc, ef, p, n] + values
                self.replay_buffer.loc[len(self.replay_buffer)] = new_row
                return x, p, n
        else:
                auc, ef, p, n = self.screening()
                if auc == 0 and ef == 0:
                    return 0, 0, 0
                x = (auc*3 + ef)/4
                return x, p, n
                     
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
        if hits == [0] and scores == [0] and pos == 0 and neg == 0:
            return 0, 0, 0, 0
        return *self.scoring(hits, scores, pos, neg), pos, neg

    def scoring(self, hits, scores, pos, neg):
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
        a_hits, a_scores = utils.insert_elements(self.n_inhibs-sum(hits), self.n_decoys-(len(hits)-sum(hits)))
        hits.extend(a_hits)
        scores.extend(a_scores)
        idx = np.where(np.array(hits) == 0)[0]
        j=0
        for i in idx:
            hits = np.insert(hits, i+j, np.zeros(9))
            scores = np.insert(scores, i+j, np.full(9,scores[i+j])) 
            j += 9
        
        auc = roc_auc_score(hits, scores)
        if pos == 0 and neg == 0: ef = 0 
        else: ef = ((pos / (pos + neg)) / (self.n_inhibs / (self.n_inhibs + self.n_decoys))) / self.max_EF # normalized by maximum EF
        
        if self.render_mode == "human":
            fpr, tpr, _ = roc_curve(hits, scores)
            self.render(fpr, tpr, auc, ef) 
        
        return auc, ef
    
    def step(self, action):
        # Execute one time step within the environment
        self.action_execution(action)

        # new observation (state)
        self.last_observation = self.get_observation()        
        
        truncated = []
        # Truncated if episode exceeds timestep limit
        truncated.append(self.counter > self.episode_length)
        
        # in inference mode the csv is updated with every step so no observations are lost
        if self.inference_mode:
            self.refresh_buffer()

        # check boundaries
        for key in self.last_observation.keys():
            truncated.append(not np.logical_and(np.all(self.last_observation[key] >= 0), 
                                                np.all(self.last_observation[key] <= 7)))

        if np.any(truncated):
            self.reward = 0
        
        # Evaluate and calculate reward
        start_time = time.time()
        self.reward, pos, neg = self.get_reward()
        self.timings.append(time.time() - start_time)
        secondary_ = False
        
        # Episode termination conditions
        # if NN Model is used for approximation only a reward is returned 
        if not self.enable_approximator:
            secondary_ = (pos > self.n_inhibs//10 and neg == 0)
            # writes updated replay buffer to filesystem
            if self.counter % 10 == 0 and self.hybrid_reward == True:
                self.refresh_buffer()
        
        primary_ = (self.reward > self.threshold) 
        terminated = (primary_ or secondary_)
        self.counter += 1
        
        if self.verbose > 0: # verbosuty level 1
            if self.counter % 100 == 0:
                # Debug message: mean of last 100 reward evals
                print(np.mean(self.timings[-100:]))

        if self.verbose > 1: # verbosity level 2
            if terminated: print("terminated")
            if np.any(truncated): print(truncated)

        if self.verbose > 2: # verbosity level 3
            if primary_: print("threshold of "+str(self.threshold)+" reached")
            if secondary_: print(f"10% ({self.n_inhibs/10}) and no neg")
        
        return self.last_observation, self.reward, terminated, np.any(truncated), {}
    
    def reset(self, seed=None, options=None):
        super().reset()
        # Reset the state of the environment to an initial state
        self.counter = 0
        self.phar_modified = self.phar
        return self.get_observation(initial=True), {}
    
    def get_observation(self, initial=False):
        obs = dict()
        for i, f in zip(range(len(self.featureIds)),self.features):
            x = []              
            if initial:
                if self.inference_mode == True:
                    for id in self.featureIds[i]:
                        x.extend([*self.get_tol(id, initial=True)])
                    x = np.array(x, dtype=np.float32)
                    x[:] = np.round(x.flatten(), decimals=1)
                    obs[f] = x
                else:
                    for id in self.featureIds[i]:
                        x.extend([*self.get_tol(id, initial=True)])
                    x = np.array(x, dtype=np.float32)
                    x[:] = np.round(np.random.uniform(low=self.bounds[i][0], high=self.bounds[i][1], size=(len(x.flatten()),)), 2)
                    obs[f] = x
                    # write all rans to tree
                    if i==0 or i==3:
                        for k, id in enumerate(self.featureIds[i]):

                            self.set_tol(id=id, newval=np.round(x[k], 2), initial=True)
                    if i==1 or i==2:
                        for id in self.featureIds[i]:    
                            for j in range(0,int(len(self.featureIds[i]))*2,2):
                                self.set_tol(id, np.round(x[j], 2), target="origin", initial=True)
                                self.set_tol(id, np.round(x[j+1], 2), target="target", initial=True)
            else:
                for id in self.featureIds[i]:
                    x.extend([*self.get_tol(id, initial=False)])
                x = np.array(x, dtype=np.float32)
                x[:] = np.round(x.flatten(), decimals=1)
                obs[f] = x
        return obs

    def write_values_to_tree(self, values, initial=False):
        writes = 0
        for i in range(len(self.featureIds)):
            if i==0 or i==3:
                for id in self.featureIds[i]:
                    self.set_tol(id, values[writes], initial=initial)
                    writes += 1
            if i==1 or i==2:
                for id in self.featureIds[i]:    
                    self.set_tol(id, values[writes], target="origin", initial=initial)
                    self.set_tol(id, values[writes+1], target="target", initial=initial)
                    writes += 2

    def get_observation_space(self):
        d = {}
        for i in range(len(self.featureIds)):
            feature = self.features[i]
            lower = self.bounds[i][0]
            upper = self.bounds[i][1]
            if feature == "H": # or feature == "exclusion" 
                up = [upper for _ in self.featureIds[i]]
                down = [lower for _ in self.featureIds[i]]
                d[feature] = spaces.Box(low=np.array(down), high=np.array(up), dtype=np.float32)
            
            if feature == "HBA" or feature == "HBD":
                up = [upper for _ in range(len(self.featureIds[i])*2)]
                down = [lower for _ in range(len(self.featureIds[i])*2)]
                d[feature] = spaces.Box(low=np.array(down), high=np.array(up), dtype=np.float32)

        return d
    
    def get_box_action_space(self):
        low = []
        high = []
        for i in range(len(self.featureIds)):
            lower = self.bounds[i][0]
            upper = self.bounds[i][1]
            if self.features[i] == "H":
                low.extend([lower for _ in self.featureIds[i]])
                high.extend([upper for _ in self.featureIds[i]])
            if self.features[i] == "HBA" or self.features[i] == "HBD":
                low.extend([lower for _ in range(len(self.featureIds[i])*2)])
                high.extend([upper for _ in range(len(self.featureIds[i])*2)])
        return spaces.Box(low=np.array(low), high=np.array(high), dtype=np.float32)

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
                    x.extend([self.get_tol(id, initial=True)])
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
        
        self.write_values_to_tree(observation, initial=False)
        
        self.phar_modified.write(filename, encoding="utf-8", xml_declaration=True)
        
    def render(self, fpr, tpr, auc, ef, mode="console"):
        fig = plt.figure(figsize=(10, 5))
        gs = gridspec.GridSpec(1, 2, width_ratios=[5, 1])

        # Plot ROC curve on the first subplot
        ax_roc = plt.subplot(gs[0])
        ax_roc.plot(fpr, tpr, label='ROC curve (AUC = {:.2f})'.format(auc))
        ax_roc.plot([0, 1], [0, 1], linestyle='--', color='r', label='Random')
        ax_roc.set_xlabel('False Positive Rate')
        ax_roc.set_ylabel('True Positive Rate')
        ax_roc.set_title(f'ROC Curve for {self.last_observation}')
        ax_roc.legend()

        # Bar plot on the second subplot
        ax_bar = plt.subplot(gs[1])
        bar_label = 'EF Score'
        bar_height = ef
        bar_color = 'g'

        bar_position = [1]  # Adjust the position as desired

        ax_bar.bar(bar_position, bar_height, color=bar_color, label=bar_label)
        ax_bar.set_title('EF Score')
        ax_bar.set_ylabel('EF Score')
        ax_bar.set_xticks([])  # Remove x-axis ticks for better visualization
        ax_bar.set_ylim(0,4)
        ax_bar.legend()

        # Display the plot
        plt.tight_layout()
        plt.savefig(f"../data/images/{int(time.time())}.png")
        plt.close()
    
    def action_execution(self, action):
        """
        Execute an action 
        either:
        - add or subtract 0.1 to the tolerance of a feature
        or:
        - add or subtract 0.1 to the weight of a feature
        :param action: action to execute
        :return: Path to modified Phar file
        """
        Hm = 2
        HBAm = 4
        HBDm = 4
        H = len(self.featureIds[0])*Hm
        HBA = len(self.featureIds[1])*HBAm
        HBD = len(self.featureIds[2])*HBDm
        if self.action_space_type == "discrete":
            if action < H:
                d = action // Hm #feature number
                r = action % Hm #0: tol+, 1: tol-
                id = self.featureIds[0][d] #feature id
                self.executor(r, id, False, self.delta)
            if action >= H and action < H + HBA:
                d = (action-H) // HBAm #feature number
                r = (action-H) % HBAm #0: tol+, 1: tol-, 2: tol+, 3: tol-
                id = self.featureIds[1][d] #feature id
                self.executor(r, id, True, self.delta)
            if action >= H + HBA:
                d = (action-H-HBA) // HBDm #feature number
                r = (action-H-HBA) % HBDm #0: tol+, 1: tol-, 2: tol+, 3: tol-
                id = self.featureIds[2][d] #feature id
                self.executor(r, id, True, self.delta)
        if self.action_space_type == "box":
            self.write_values_to_tree(action)

    def executor(self, r, id, f=False, delta=0.1):
        """
        Outsourcing of execution code
        :param r: action encoding
        :param feature: feature id
        :return: modified tree
        """
        if f:
            tol = self.get_tol(id, initial=False)
            match r:
                case 0:
                    self.set_tol(id, (float(tol[1]) + delta), target="origin", initial=False)
                case 1:
                    self.set_tol(id, (float(tol[1]) - delta), target="origin", initial=False)
                case 2:
                    self.set_tol(id, (float(tol[0]) + delta), target="target", initial=False)
                case 3:
                    self.set_tol(id, (float(tol[0]) - delta), target="target", initial=False)
                case _:
                    raise ValueError("No valid action specified")

        else:
            tol = self.get_tol(id, initial=False)
            match r:
                case 0:
                    self.set_tol(id=id, newval=(float(tol[0]) + delta), initial=False)
                case 1:
                    self.set_tol(id=id, newval=(float(tol[0]) - delta), initial=False)
                case _: 
                    raise ValueError("No valid action specified")
        
    def set_tol(self, id, newval, target=None, initial=False):
        """
        Set tolerance of a feature
        :param tree: tree of Phar file
        :param id: featureId
        :param newval: new tolerance value
        :param target: target or origin, when dealing with HBA and HBD
        :return: updated tree
        """
        newval = str(round(newval, 2))
        if initial:
            elm = self.phar.find(".//*[@featureId='"+id+"']")
        else:    
            elm = self.phar_modified.find(".//*[@featureId='"+id+"']")

        flag = False
        if (elm.get("name") == "H") or (elm.get("type") == "exclusion"):
            elm.find("./position").set('tolerance', newval)
            flag = True

        if target == "target":
            child = elm.find("./target")
            child.set('tolerance', newval)
            flag = True

        if target == "origin":
            child = elm.find("./origin")
            child.set('tolerance', newval)
            flag = True

        if not flag:
            raise ValueError("No valid target specified")

    def set_weight(self, id, newval, initial=False):
        """
        Set weight of a feature
        :param tree: tree of Phar file
        :param id: featureId
        :param newval: new weight value
        :return: updated tree    
        """
        newval = str(round(newval, 2))
        if initial:
            elm = self.phar.find(".//*[@featureId='"+id+"']")
        else:
            elm = self.phar_modified.find(".//*[@featureId='"+id+"']")
        elm.set("weight", newval)

    def get_tol(self, id:str, initial=False):
        """
        Get tolerance and weight of a feature
        :param tree: tree of Phar file
        :param id: featureId
        :return: tolerance and weight as separate values, when dealing with HBA and HBD, tolerance of target and origin as well as weight are returned
        """
        if initial:
            elm = self.phar.find(".//*[@featureId='"+str(id)+"']")
        else:
            elm = self.phar_modified.find(".//*[@featureId='"+str(id)+"']")
        
        if (elm.get("name") == "H") or (elm.get("type") == "exclusion"):
            child = elm.find("./position")
            return [float(child.get("tolerance"))]
        else:
            child_target = elm.find("./target")
            child_origin = elm.find("./origin")
            return [float(child_target.get('tolerance')), float(child_origin.get('tolerance'))]

    def close(self):
        dirname = os.path.dirname(self.out_file)
        for f in os.listdir(dirname):
            if not f.endswith(".sdf"):
                continue
            os.remove(os.path.join(dirname, f))

