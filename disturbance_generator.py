import json
import numpy as np

class Disturbance:

    def __init__(self, configs_base, field = None, x = None, t = None):

        # load parameters from config
        with open(f'{configs_base}/config_disturbance.json') as f:
            cfg = json.load(f)

        # get disturbance configs 
        self.dist_type = cfg["type"]

        if self.dist_type == 'linear':
            self.d   = np.array(cfg['d_lin'])
        elif self.dist_type == 'field' and field is not None:
            self.d = field.compute_disturbance(x, t)
        else:
            self.d = 0*np.array(cfg['d_lin'])

    def evolve(self, field = None, x = None, t = None):

        if self.dist_type == 'linear':
            pass
        elif self.dist_type == 'field' and field is not None:
            self.d = field.compute_disturbance(x, t)
        else:
            pass

        return self.d