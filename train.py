"""Train one agent.  Example :
    python train.py --algo ppo --condition rand --seed 0 --steps 1000000
"""
import argparse, json, os, time
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback#import libraries from stable_baselines3
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor

from src.wind_wrapper import make_env
from src.double_dqn import DoubleDQN
from src.config import PPO_KWARGS, PPO_N_ENVS, DQN_KWARGS


class CheckpointEval(BaseCallback):
    """Every `every` env steps, run n greedy episodes in the training condition."""
    def __init__(self, condition, seed, every=10_000, n_eps=10, best_path=None):
        super().__init__()
        self.best_path, self.best, self.best_step =best_path, -np.inf, 0
        self.condition, self.every, self.n_eps = condition, every, n_eps
        self.env = make_env(condition)
        self.next_eval = every
        self.eval_seed = 10_000_000 + seed * 100_000
        self.rows = []

    def _run_eval(self):
        rets = []
        for _ in range(self.n_eps):
            obs,_ =self.env.reset(seed=self.eval_seed); self.eval_seed += 1
            done, total = False, 0.0
            while not done:
                a, _ =self.model.predict(obs, deterministic=True)
                obs, r, term, trunc, _ = self.env.step(a)
                total += r; done = term or trunc
            rets.append(total)
        m = float(np.mean(rets))
        self.rows.append(dict(step=int(self.num_timesteps), mean=m, std=float(np.std(rets))))
        # Keep the best checkpoint, judged ONLY in the agent's own training condition.
        # Added after the first DQN runs collapsed late in training (see README).
        if self.best_path and m > self.best:
            self.best = m
            self.best_step = int(self.num_timesteps)
            self.model.save(self.best_path)

    def _on_step(self):
        if self.num_timesteps >= self.next_eval:
            self._run_eval()
            self.next_eval += self.every
        return True


def main():
    p =argparse.ArgumentParser()
    p.add_argument("--algo", choices=["ppo", "dqn"], required=True)
    p.add_argument("--condition", choices=["calm", "rand"], required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--steps", type=int, default=1_000_000)
    p.add_argument("--eval-every", type=int, default=10_000)
    p.add_argument("--out", default=".")
    a = p.parse_args()
    torch.set_num_threads(1)
    name =f"{a.algo}_{a.condition}_s{a.seed}"
    os.makedirs(f"{a.out}/models", exist_ok=True); os.makedirs(f"{a.out}/results/curves", exist_ok=True)

    if a.algo == "ppo":
        venv = make_vec_env(lambda: make_env(a.condition), n_envs=PPO_N_ENVS, seed=a.seed)
        model =PPO("MlpPolicy", venv, seed=a.seed, verbose=0,
                    tensorboard_log=f"{a.out}/results/tb", **PPO_KWARGS)
    else:
        env = Monitor(make_env(a.condition)); env.reset(seed=a.seed)
        model = DoubleDQN("MlpPolicy", env, seed=a.seed, verbose=0,
                          tensorboard_log=f"{a.out}/results/tb", **DQN_KWARGS)

    cb =CheckpointEval(a.condition, a.seed, every=a.eval_every,
                        best_path=f"{a.out}/models/{name}_best")
    t0 = time.time()
    model.learn(total_timesteps=a.steps, callback=cb, tb_log_name=name)
    wall =time.time() - t0
    model.save(f"{a.out}/models/{name}")
    with open(f"{a.out}/results/curves/{name}.json", "w") as f:
        json.dump(dict(name=name, algo=a.algo, condition=a.condition, seed=a.seed,
                       steps =a.steps, wall_seconds=wall, best_score=cb.best,
                       best_step=cb.best_step, curve=cb.rows), f)
    print(f"{name} done in {wall/60:.1f} min, last checkpoint {cb.rows[-1]['mean']:.1f}")


if __name__ == "__main__":
    main()
