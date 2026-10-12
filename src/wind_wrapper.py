"""Wind wrapper for LunarLander-v3.

Gymnasium only lets you set wind_power and turbulence_power when the env is
built. This wrapper draws fresh values on every reset so that each episode
has its own wind. The wind values are NOT added to the observation, so the
agent has to infer the push from how the lander moves.
"""
import gymnasium as gym
import numpy as np


class RandomWindWrapper(gym.Wrapper):
    def __init__(self, env, wind_range= (0.0, 15.0), turb_range=(0.0, 1.5)):
        super().__init__(env)
        assert env.unwrapped.enable_wind, "build the env with enable_wind=True"
        self.wind_range = wind_range
        self.turb_range = turb_range
        self._rng = np.random.default_rng()

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self._rng = np.random.default_rng(seed + 12345)
        base = self.env.unwrapped
        base.wind_power = float(self._rng.uniform(*self.wind_range))
        base.turbulence_power = float(self._rng.uniform(*self.turb_range))
        obs, info =self.env.reset(seed=seed, options=options)
        info["wind_power"] = base.wind_power
        info["turbulence_power"] = base.turbulence_power
        return obs, info


def make_env(condition="calm", wind=None, render_mode=None):
    """condition: 'calm' (no wind), 'rand' (randomised every episode),
    or 'fixed' (a single wind level, turbulence = wind / 10)."""
    import warnings
    warnings.filterwarnings("ignore", message=".*recommended to be between.*")
    if condition =="calm":
        return gym.make("LunarLander-v3", render_mode=render_mode)
    if condition == "rand":
        env = gym.make("LunarLander-v3", enable_wind=True, render_mode=render_mode)
        return RandomWindWrapper(env)
    if condition == "fixed":
        if wind == 0:
            return gym.make("LunarLander-v3", render_mode=render_mode)
        return gym.make("LunarLander-v3", enable_wind=True, wind_power=float(wind),
                        turbulence_power=float(wind) / 10.0, render_mode=render_mode)
    raise ValueError(condition)
