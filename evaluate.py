"""Freeze each trained policy and test it at every wind level.
100 greedy episodes per (agent, wind level). Evaluation seeds start at 20,000,000
so they never overlap with training or checkpoint seeds. Every agent sees the
same seeds at a given wind level, so comparisons are paired.

    python evaluate.py --outdir results/eval models/*_best.zip
    python evaluate.py models/ppo_calm_s0.zip --outdir results/eval_final --winds 0 15
"""
import argparse
import os,sys, json
import numpy as np
import torch
from stable_baselines3 import PPO
from src.double_dqn import DoubleDQN
from src.wind_wrapper import make_env
from src.config import EVAL_WINDS

N_EPS =100


def run_episode(model, env, seed):
    obs, _ = env.reset(seed=seed)
    total, main_fires, steps, done = 0.0, 0, 0, False
    while not done:
        a, _ = model.predict(obs, deterministic=True)
        a = int(a)
        main_fires += (a == 2)
        obs, r, term, trunc, _ = env.step(a) #set the steps
        total += r; steps += 1; done = term or trunc
    outcome = "timeout" if trunc and not term else ("crash" if r == -100 else "landed")
    return dict(ret= total, outcome=outcome, main_fires=main_fires, steps=steps)


def main():
    p =argparse.ArgumentParser()
    p.add_argument("models", nargs="+")
    p.add_argument("--outdir", default="results/eval")
    p.add_argument("--winds", type=float, nargs="+", default=EVAL_WINDS)
    a = p.parse_args()
    torch.set_num_threads(1)
    os.makedirs(a.outdir, exist_ok=True)
    for path in a.models:
        name =os.path.basename(path).replace(".zip", "")
        cls = PPO if name.startswith("ppo") else DoubleDQN
        model = cls.load(path, device="cpu")
        rows =[]
        for w in a.winds:
            wi = EVAL_WINDS.index(w) if w in EVAL_WINDS else int(w)
            env = make_env("fixed", wind=w)
            for ep in range(N_EPS):
                res = run_episode(model, env, seed=20_000_000 + wi * 1000 + ep)
                res.update(wind=w, episode=ep)
                rows.append(res)
            env.close()
            sub =[r["ret"] for r in rows if r["wind"]== w]
            print(f"{name} wind={w}: mean {np.mean(sub):.1f}", flush=True)
        with open(f"{a.outdir}/{name.replace('_best', '')}.json", "w") as f:
            json.dump(rows, f)


if __name__ == "__main__":
    main()
