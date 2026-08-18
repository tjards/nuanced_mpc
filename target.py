
import json
import numpy as np

class Target:

    def __init__(self):

        # load target configuration
        with open("configs/config_target.json") as f:
            cfg = json.load(f)

        self.xr             = np.array(cfg["initial_state"], dtype = float)
        self.dynamics       = cfg["dynamics"]
        self.period         = cfg["period"]
        self.random_seed    = cfg["random_seed"]
        self.center         = self.xr[:2].copy() 

        # wander configuration
        if self.dynamics == 'wander':
            
            # all wander configs
            wander_cfg = cfg["wander_params"]

            # pull out temp ranges
            _x_scale    = wander_cfg["x-scale"]
            _y_scale    = wander_cfg["y-scale"]
            _skew       = wander_cfg["skew"]
            _rotation   = wander_cfg["rotation"]
            _x_wobble = wander_cfg["x-wobble"]
            _y_wobble = wander_cfg["y-wobble"]
            _x_wobble_freq = wander_cfg["x-wobble_freq"]
            _y_wobble_freq = wander_cfg["y-wobble_freq"]
            _x_phase = wander_cfg["x-phase"]
            _y_phase = wander_cfg["y-phase"]

            # random-number generator
            self.rng = np.random.default_rng(self.random_seed)

            # define params (* unpacks the range [min, max])
            self.x_scale        = self.rng.uniform(*_x_scale)
            self.y_scale        = self.rng.uniform(*_y_scale)
            self.skew           = self.rng.uniform(*_skew)
            self.rotation       = self.rng.uniform(*_rotation)
            self.x_wobble       = self.rng.uniform(*_x_wobble)
            self.y_wobble       = self.rng.uniform(*_y_wobble)
            self.x_wobble_freq  = self.rng.uniform(*_x_wobble_freq)
            self.y_wobble_freq  = self.rng.uniform(*_y_wobble_freq)
            self.x_phase        = self.rng.uniform(*_x_phase)
            self.y_phase        = self.rng.uniform(*_y_phase)

            # store wander param ranges for per-period re-randomization
            self._wander_ranges = {
                'x_scale': _x_scale, 'y_scale': _y_scale,
                'skew': _skew, 'rotation': _rotation,
                'x_wobble': _x_wobble, 'y_wobble': _y_wobble,
                'x_wobble_freq': _x_wobble_freq, 'y_wobble_freq': _y_wobble_freq,
                'x_phase': _x_phase, 'y_phase': _y_phase,
            }
            self._current_period = -1

    def evolve(self, t):

        if self.dynamics == "wander":

            # complete one figure eight every period seconds
            progress    = (t % self.period) / self.period
            theta       = 2.0 * np.pi * progress

            # re-randomize at the start of each new period
            period_index = int(t // self.period)
            if period_index != self._current_period:
                self._current_period = period_index
                r = self._wander_ranges
                self.x_scale       = self.rng.uniform(*r['x_scale'])
                self.y_scale       = self.rng.uniform(*r['y_scale'])
                self.skew          = self.rng.uniform(*r['skew'])
                self.rotation      = self.rng.uniform(*r['rotation'])
                self.x_wobble      = self.rng.uniform(*r['x_wobble'])
                self.y_wobble      = self.rng.uniform(*r['y_wobble'])
                self.x_wobble_freq = self.rng.uniform(*r['x_wobble_freq'])
                self.y_wobble_freq = self.rng.uniform(*r['y_wobble_freq'])
                self.x_phase       = self.rng.uniform(*r['x_phase'])
                self.y_phase       = self.rng.uniform(*r['y_phase'])
                # drift center to a new location within the space
                #self.center[0] = self.rng.uniform(-2.0, 2.0)
                #self.center[1] = self.rng.uniform(-2.0, 2.0)

            # base figure-eight shape
            x           = self.x_scale * np.sin(theta)
            y           = self.y_scale * np.sin(2.0 * theta)

            # wobble fades to zero at the beginning and end of period
            envelope = np.sin(np.pi * progress) ** 2
            x += (envelope * self.x_wobble * np.sin(self.x_wobble_freq * theta + self.x_phase))
            y += (envelope * self.y_wobble * np.sin(self.y_wobble_freq * theta + self.y_phase))

            # apply skew
            x_skewed = x + self.skew * y
            y_skewed = y

            # apply rotation
            cos_rotation = np.cos(self.rotation)
            sin_rotation = np.sin(self.rotation)
            x_rotated = (cos_rotation * x_skewed - sin_rotation * y_skewed)
            y_rotated = (sin_rotation * x_skewed + cos_rotation * y_skewed)

            # update x and y 
            self.xr[0] = self.center[0] + x_rotated
            self.xr[1] = self.center[1] + y_rotated

        return self.xr.copy()




