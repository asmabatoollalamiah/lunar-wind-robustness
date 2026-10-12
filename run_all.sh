#!/usr/bin/env bash
# Runs all 20 training jobs (2 algos x 2 conditions x 5 seeds),2 at a time.
mkdir -p results/logs
for s in 0 1 2 3 4; do for c in calm rand; do for a in dqn ppo; do
  echo "$a $c $s"; done; done; done |
xargs -P 2 -L 1 bash -c 'python3 train.py --algo $0 --condition $1 --seed $2 --steps 1000000 > results/logs/$0_$1_s$2.log 2>&1'
