import gymnasium as gym
import numpy as np
from gymnasium import spaces
import utils


class PharmacophoreEnv(gym.Env):
    """Custom Environment that follows gym interface."""

    #metadata = {"render.modes": ["console"], "features":{0:"H", 1:"HBA", 2:"HBD", 3:"exclusion"}}

    def __init__(self, output, querys, actives_db, inactives_db, ldba, ldbi, features):
        super().__init__()
        # Define action and observation space
        # They must be gym.spaces objects
        # Example when using discrete actions:
        self.features = features.split(",")       
        self.bounds = {"H": [1, 4], "HBA": [1, 5], "HBD": [1, 5], "exclusion": [0, 10], "WGHT": [0.1, 3]}
        self.codec = {0:"H", 1:"HBA", 2:"HBD", 3:"exclusion"}
        self.threshold = 10 # threshold for reward, TODO: make this a parameter
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
                anvec += len(self.featureIds[i]*4)
            if i==1 or i==2:
                anvec += len(self.featureIds[i]*6)
        
        # Initialization of Spaces
        self.action_space = spaces.Discrete(anvec)
        self.observation_space = spaces.Dict(self.get_observation_space())

    def screen(self):
        """
        Execute VHTS and calculate score
        :return: score query pharmacophores against actives and inactives database        
        """
        score = 0
        # currently for one pharmacophore at a time
        self.temp_querys = self.querys[:-4]+"_temp"+self.querys[-4:]
        self.phar_modified.write(self.temp_querys, encoding="utf-8", xml_declaration=True)
        actives, inactives = utils.exec_vhts(output_file=self.out_file, 
                                             querys=self.temp_querys, 
                                             actives_db=self.actives_db, 
                                             inactives_db=self.inactives_db)
        # TODO: calculate score
        if actives == 0:
            return 0
        EF = (actives/(actives+inactives))/(self.ldba_size/(self.ldba_size+self.ldbi_size))
        if inactives == 0:
            inactives = 1
        score = (EF + actives) / (inactives)
        return score

    def step(self, action):
        # Execute one time step within the environment
        self.phar_modified = utils.action_execution(action, self.featureIds, self.phar)
        
        # Evaluate and calculate reward
        self.reward = self.screen()
        
        # Always False, not needed for now
        truncated = False
        
        # Episode termination conditions
        if self.reward > self.threshold or self.counter == 200:
            terminated = True
        else: 
            terminated = False
        
        # new observation (state)
        obs = self.get_observation(initial=False)
        
        # changes made to the pharmacophore in total, returned in info
        diff = {}
        if self.counter % 10 == 0:
            for key in obs.keys():
                diff[key] = np.subtract(obs[key], self.initial_os[key])

        self.counter += 1
        return obs, self.reward, terminated, truncated, {"performance": self.reward, "diff": diff}
    
    def reset(self, seed=None, options=None):
        # Reset the state of the environment to an initial state
        self.reward = self.screen()
        self.counter = 0
        return self.get_observation(initial=True), {}
    
    def get_observation(self, initial=False):
        os_space = dict()
        for i, f in zip(range(len(self.featureIds)),self.features):
            x = []
            for id in self.featureIds[i]:
                if initial:
                    x.extend(utils.get_tol_and_weight(self.phar, id))
                else:
                    x.extend(utils.get_tol_and_weight(self.phar_modified, id))
            os_space[f] = np.array(x, dtype=np.float32)
        return os_space

    def get_observation_space(self):
        wght_low = self.bounds["WGHT"][0]
        wght_up = self.bounds["WGHT"][1]
        d = self.bounds.copy()
        d.popitem()
        d.popitem() # removing exclusion and weight from dict
        for i in range(len(self.featureIds)):
            feature = self.features[i]
            lower = self.bounds[feature][0]
            upper = self.bounds[feature][1]
            up = []
            down = []
            if feature == "H": # or feature == "exclusion" 
                for _ in self.featureIds[i]:
                    up.extend([upper, wght_up])
                    down.extend([lower, wght_low])
                d[feature] = spaces.Box(low=np.array(down), high=np.array(up), shape=(len(self.featureIds[i])*2,), dtype=np.float32)
            if feature == "HBA" or feature == "HBD":
                for _ in self.featureIds[i]:
                    up.extend([upper, upper, wght_up])
                    down.extend([lower, lower, wght_low])
                d[feature] = spaces.Box(low=np.array(down), high=np.array(up), shape=(len(self.featureIds[i])*3,), dtype=np.float32)
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

    def render(self, mode="console"):
        ...
    def close(self):
        ...

    
    


