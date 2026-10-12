# Does Training in Wind Make a Lunar Lander Sturdier?

Part 2 code for my reinforcement learning project (PhDAI-830-A01, Fall 2026).
Asma Batool Lalamiah.

The question: if an agent learns to land in calm conditions, does it still land
when the wind picks up, and does training with random wind from the start help?
I compare PPO and Double DQN, each trained calm and with randomized wind, on
Gymnasium's `LunarLander-v3` (discrete actions).

## What is in here

| File | What it does |
|---|---|
| `src/wind_wrapper.py` | Gymnasium wrapper that draws a new wind power (0–15) and turbulence (0–1.5) at every reset. Wind is **not** part of the observation. |
| `src/double_dqn.py` | Subclass of SB3's `DQN` that swaps in the Double DQN target (online net picks the action, target net scores it). |
| `src/config.py` | Hyperparameters from RL Baselines3 Zoo, identical for calm and randomized runs. Test wind levels. |
| `train.py` | Trains one agent. Every 10,000 steps runs 10 greedy episodes in the agent's own training condition, logs the score, and saves the best checkpoint. |
| `run_all.sh` | Runs all 20 jobs (2 algorithms × 2 conditions × 5 seeds), two at a time. |
| `evaluate.py` | Freezes each best checkpoint and runs 100 greedy episodes at each of 7 wind levels (0, 5, 10, 15, 17.5, 20, 22.5; turbulence = wind/10). Evaluation seeds start at 20,000,000 and never overlap with training. |
| `analyze.py` | Interquartile mean (IQM) with 95% bootstrap CIs, success / crash / timeout rates, fuel use, sample efficiency, all figures. |
| `make_demo.py` | Side-by-side demo videos of the calm-trained and wind-trained agent flying the exact same episode. |
| `results/` | Learning curves (`curves/`), raw evaluation episodes (`eval/`), summary CSVs, TensorBoard logs. |
| `figures/` | Plots used in the slides. |
| `demos/` | Demo videos. |
| `models/` | Best checkpoints (`*_best.zip`) and final policies. |

## How to run it

Tested on Python 3.13, CPU only.

```bash
pip install -r requirements.txt        # needs swig for Box2D: pip install swig first
./run_all.sh                           # ~8 h on 2 CPU cores (DQN is the slow part)
python evaluate.py --outdir results/eval models/*_best.zip    #14,000 test episodes
python analyze.py                      # tables in results/, plots in figures/
python make_demo.py --algo ppo --seed 1 --winds 0 15 20 --episodes 1
```

One run on its own:

```bash
python train.py --algo ppo --condition rand --seed 0 --steps 1000000
```

## Changes from the Part 1 plan

1. **Evaluate the best checkpoint, not the final policy.** In the first DQN runs the
   agent reached ~200–285 and then collapsed to around −500 to −1,000 late in
   training, with the TD loss blowing up. Testing the final weights would have
   measured the collapse rather than what the agent had learned. Every run (PPO
   and DQN alike) now saves the checkpoint with the best score **in its own
   training condition**. No wind-sweep data is used for this choice. The final
   weights are kept in `models/` as well.
2. **rliable is replaced by my own IQM and bootstrap code.** rliable would not
   build on Python 3.13. `analyze.py` reimplements what I needed: per-seed
   means, IQM (25% trimmed mean) across seeds, and percentile bootstrap CIs
   (10,000 resamples).
3. **Learning-curve plots show the median with an interquartile band** instead of
   mean ± SD, because the DQN collapses made the standard deviation meaningless.

Everything else follows the Part 1 design: same hyperparameters across
conditions, 1M environment steps per run, 5 seeds, 7 wind levels, 100 test
episodes per level.

## Outcome definitions

* **Success**: episode return ≥ 200.
* **Crash**: the episode ended with the −100 penalty (hit the ground hard or left the screen).
* **Timeout**: the episode hit 1,000 steps without landing or crashing (hovering).
* **Fuel**: main-engine firings per episode.
* **Steps to 200** : first checkpoint where the rolling mean of the last 5 checkpoint scores reaches 200.

## Results in brief

IQM return across 5 seeds (95% bootstrap CI), 100 test episodes per seed per wind level.
Full tables: `results/summary.csv`, `results/robustness_drop.csv`, `results/efficiency.csv`.

| Agent | Wind 0 | Wind 15 | Drop 0→15 | Success @0 | Success @15 | Landed safely @15 | Crashed @15 |
|---|---|---|---|---|---|---|---|
| PPO-Calm | 258 [253, 263] | 110 [86, 140] | 148 | 98.6% | 46.8% | 50.6% | 47.4% |
| PPO-Rand | 204 [175, 232] | 142 [129, 148] | 62 | 71.6% | 47.4% | 63.8% | 29.4% |
| DQN-Calm | 241 [227, 263] | 118 [63, 146] | 124 | 90.4% | 48.2% | 51.0% | 46.2% |
| DQN-Rand | 233 [142, 271] | 100 [−81, 181] | 132 | 77.8% | 43.4% | 47.0% | 45.2% |

* Calm-trained agents lose about half their success rate by wind 15.
* For PPO, random-wind training cut the score drop by more than half and crashes by more than a third,
  but cost ~21% of calm-air score, ~36% more main-engine use, and some hovering in calm air.
* Success rate (≥ 200) at wind 15 is the same for all agents: wind-trained PPO lands more often,
  but many of its landings use enough fuel to score just under 200.
* DQN reached 200 sooner in calm air (338k vs 504k steps) but was unstable: 7 of 10 DQN runs
  ended far below their best, and DQN gained nothing from random wind.

## AI use

I used Claude (Anthropic) to help write and debug the code, run the experiments, and draft the
slides and narration. I reviewed the code, checked the numbers against the raw result files,
and made the final decisions on the design and on what to report.
