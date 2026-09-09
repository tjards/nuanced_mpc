import numpy as np
import json

class Plant:

    def __init__(self, configs_base):

        # load parameters from config
        with open(f'{configs_base}/config_plant.json') as f:
            cfg = json.load(f)

        A   = np.array(cfg['A'])
        B   = np.array(cfg['B'])
        constraints = cfg['constraints']
        #d   = np.array(cfg['d'])
        x0  = np.array(cfg['x0'], dtype=float)

        self.A = A
        self.B = B
        self.constraints = constraints
        self.d = None
        self.x0 = x0

    def evolve(self, x, u, d, disturb = True):
        
        self.d = d

        if not disturb:
            x_next = self.A @ x.flatten() + self.B @ (u.flatten())
        else:
            x_next = self.A @ x.flatten() + self.B @ (u.flatten() + self.d.flatten())
            
        # soften the constraint
        softener = 1
        x_next[2] = np.clip(x_next[2], softener*self.constraints['x_min'][2], softener*self.constraints['x_max'][2])
        x_next[3] = np.clip(x_next[3], softener*self.constraints['x_min'][3], softener*self.constraints['x_max'][3])

        return x_next