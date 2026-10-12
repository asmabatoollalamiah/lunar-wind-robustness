"""Turn raw curves + evaluation episodes into tables and figures.

rliable would not build on Python 3.13, so the interquartile mean (IQM) and the
stratified bootstrap CI from Agarwal et al. (2021) are re-implemented here:
  * each run (seed) is summarised by its mean over 100 test episodes,
  * IQM = mean of the middle 50% of run scores (scipy trim_mean, 0.25),
  * 95% CI = percentile bootstrap over runs (10,000 resamples).
"""
import glob, json, os #imported glob, json, os
import numpy as np
import pandas as pd
from scipy.stats import trim_mean
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.config import EVAL_WINDS

AGENTS =["ppo_calm", "ppo_rand", "dqn_calm", "dqn_rand"]
LABEL = {"ppo_calm": "PPO-Calm", "ppo_rand": "PPO-Rand",
         "dqn_calm": "DQN-Calm", "dqn_rand": "DQN-Rand"}
COLOR = {"ppo_calm": "#7FA7D9", "ppo_rand": "#1F4E99",
         "dqn_calm": "#F2A98A", "dqn_rand": "#C2410C"}
STYLE = {"ppo_calm": "--", "ppo_rand": "-", "dqn_calm": "--", "dqn_rand": "-"}
RNG = np.random.default_rng(0)


def iqm(x):
    return float(trim_mean(np.asarray(x), 0.25))


def boot_ci(x, stat, n=10_000):
    x =np.asarray(x)
    idx = RNG.integers(0, len(x), size=(n, len(x)))
    vals = np.array([stat(x[i]) for i in idx])
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def load_eval():
    frames =[]
    for f in glob.glob("results/eval/*.json"):
        name =os.path.basename(f)[:-5]
        algo, cond, s = name.split("_")
        df = pd.DataFrame(json.load(open(f)))
        df["agent"], df["seed"] = f"{algo}_{cond}", int(s[1:])
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def load_curves():
    out ={}
    for f in glob.glob("results/curves/*.json"):
        d = json.load(open(f))
        out[d["name"]] = d
    return out


def summary_table(ev):
    ev =ev.copy()
    ev["success"] = (ev["ret"] >= 200).astype(float)
    ev["crash"] = (ev["outcome"] == "crash").astype(float)
    ev["timeout"] = (ev["outcome"] == "timeout").astype(float)
    per_run = ev.groupby(["agent", "seed", "wind"]).agg(
        ret=("ret", "mean"), success=("success", "mean"), crash=("crash", "mean"),
        timeout=("timeout", "mean"), main_fires=("main_fires", "mean")).reset_index()
    rows =[]
    for (agent, wind), g in per_run.groupby(["agent", "wind"]):
        lo, hi = boot_ci(g["ret"], iqm)
        slo, shi = boot_ci(g["success"], np.mean)
        rows.append(dict(agent=agent, wind=wind, iqm_return=iqm(g["ret"]), ret_lo=lo, ret_hi=hi,
                         success=g["success"].mean(), succ_lo=slo, succ_hi=shi,
                         crash=g["crash"].mean(), timeout=g["timeout"].mean(),
                         main_fires=g["main_fires"].mean(),
                         seed_sd=g["ret"].std(ddof=1)))
    return per_run, pd.DataFrame(rows)


def steps_to_200(curve, window=5):
    m = np.array([c["mean"] for c in curve]); steps = [c["step"] for c in curve]
    roll =pd.Series(m).rolling(window).mean().values
    hit =np.where(roll >= 200)[0]
    return steps[hit[0]] if len(hit) else np.nan


def main():
    os.makedirs("figures", exist_ok=True)
    plt.rcParams.update({"font.size": 12, "axes.spines.top": False, "axes.spines.right": False,
                         "font.family": "DejaVu Sans"})
    curves =load_curves()

    #--- learning curves ---
    fig, axes =plt.subplots(1, 2, figsize=(12, 4.4), sharey=True)
    eff_rows = []
    for ax, algo in zip(axes, ["ppo", "dqn"]):
        for cond in ["calm", "rand"]:
            agent =f"{algo}_{cond}"
            runs = [curves[k] for k in sorted(curves) if k.startswith(agent + "_")]
            if not runs:
                continue
            L = min(len(r["curve"]) for r in runs)
            steps = np.array([c["step"] for c in runs[0]["curve"][:L]])
            M =np.array([[c["mean"] for c in r["curve"][:L]] for r in runs])
            Ms = pd.DataFrame(M.T).rolling(5, min_periods=1).mean().values.T
            mu = np.median(Ms, 0)
            q1, q3 = np.percentile(Ms, 25, 0), np.percentile(Ms, 75, 0)
            ax.plot(steps / 1e6, mu, STYLE[agent],color =COLOR[agent], lw=2, label=LABEL[agent])
            ax.fill_between(steps / 1e6, q1, q3, color=COLOR[agent], alpha=0.2)
            for r in runs:
                eff_rows.append(dict(agent=agent, seed=r["seed"],
                                     steps_to_200=steps_to_200(r["curve"]),
                                     final_ckpt=np.mean([c["mean"] for c in r["curve"][-10:]]),
                                     wall_min=r["wall_seconds"] / 60))
        ax.axhline(200, color="#555", lw=1, ls=":")
        ax.set_title(algo.upper() + (" (Double DQN)" if algo == "dqn" else ""))
        ax.set_xlabel("Environment steps (millions)")
        ax.set_ylim(-700, 320); ax.legend(frameon=False,loc="lower left", ncol=2)
    axes[0].set_ylabel("Checkpoint return (10 eps)")
    fig.suptitle("Learning curves: median and interquartile band over 5 seeds (smoothed over 5 checkpoints), in each agent's own training condition",
                 fontsize=12)
    fig.tight_layout(); fig.savefig("figures/learning_curves.png",dpi=180); plt.close(fig)
    eff = pd.DataFrame(eff_rows); eff.to_csv("results/efficiency.csv", index=False)

    if not glob.glob("results/eval/*.json"):
        return
    ev =load_eval()
    per_run, tab =summary_table(ev)
    per_run.to_csv("results/per_run.csv",index=False); tab.to_csv("results/summary.csv",index=False)#convert to cvs

    #--- robustness: return vs wind ---
    fig, axes =plt.subplots(1, 2, figsize=(12, 4.4))
    for agent in AGENTS:
        t =tab[tab.agent == agent].sort_values("wind")
        if t.empty: continue
        axes[0].plot(t.wind, t.iqm_return, STYLE[agent], marker="o", color=COLOR[agent], lw=2, label=LABEL[agent])
        axes[0].fill_between(t.wind, t.ret_lo, t.ret_hi, color=COLOR[agent], alpha=0.15)
        axes[1].plot(t.wind, 100 * t.success, STYLE[agent], marker="o", color=COLOR[agent], lw=2, label=LABEL[agent])
        axes[1].fill_between(t.wind, 100 * t.succ_lo, 100 * t.succ_hi, color=COLOR[agent], alpha=0.15)
    for ax in axes:
        ax.axvspan(15, 23.5, color="#999",alpha=0.10)
        ax.text(19.2, ax.get_ylim()[1] if False else 0, "", ha="center")
        ax.set_xlabel("Test wind power (turbulence = wind / 10)"); ax.set_xlim(-0.5, 23.5)
    axes[0].axhline(200, color ="#555",lw=1, ls=":")
    axes[0].set_ylabel("IQM return (95% CI)"); axes[0].set_title("Return across wind levels")
    axes[1].set_ylabel("Success rate, % (score ≥ 200)"); axes[1].set_title("Success rate across wind levels")
    axes[1].set_ylim(0, 100); axes[1].legend(frameon=False,loc="lower left")
    for ax in axes:
        y0, y1 =ax.get_ylim(); ax.text(19.25, y1 - 0.05 * (y1 - y0), "beyond training range",
                                         ha="center", va="top", fontsize=10, color="#555")
    fig.tight_layout(); fig.savefig("figures/robustness.png", dpi=180); plt.close(fig)

    #--- outcome breakdown at wind 15 and 22.5 ---
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2), sharey=True)
    for ax, w in zip(axes, [0, 15, 20]):
        t = tab[tab.wind == w].set_index("agent").reindex(AGENTS)
        landed = 1 - t.crash - t.timeout
        x = np.arange(len(AGENTS))
        ax.bar(x, 100 * landed, color="#2E7D32", label="Landed / came to rest")
        ax.bar(x, 100 * t.crash, bottom=100 * landed, color="#C62828", label="Crashed / flew off")
        ax.bar(x, 100 * t.timeout, bottom=100 * (landed + t.crash), color="#9E9E9E", label="Hovered to time limit")
        #set xticks, title and ylabel
        ax.set_xticks(x, [LABEL[a].replace("-", "\n") for a in AGENTS]); ax.set_title(f"Wind {w:g}" + (" (beyond training range)" if w > 15 else ""))
    axes[0].set_ylabel("% of test episodes"); fig.legend(*axes[0].get_legend_handles_labels(), frameon=False, loc="lower center", ncol=3, fontsize=11)
    fig.tight_layout(rect=(0, 0.08, 1, 1)); fig.savefig("figures/outcomes.png", dpi=180); plt.close(fig)

    #--- seed stability ---
    fig, ax =plt.subplots(figsize=(7, 4.2))
    for i, agent in enumerate(AGENTS):
        g = per_run[(per_run.agent == agent) & (per_run.wind == 0)]
        ax.scatter(np.full(len(g), i) + RNG.uniform(-0.08, 0.08, len(g)), g.ret, s=60,color=COLOR[agent], edgecolor="k", zorder=3)
    #set axline, xticks,titile
    ax.axhline(200, color ="#555", lw=1, ls=":")
    ax.set_xticks(range(4), [LABEL[a] for a in AGENTS]); ax.set_ylabel("Mean return at wind 0 (per seed)")
    ax.set_title("Seed-to-seed spread at wind 0 (each dot = one seed)")
    fig.tight_layout(); fig.savefig("figures/seeds.png", dpi=180); plt.close(fig)

    #-- print key numbers ---
    pd.set_option("display.width", 200)
    print(tab.round(3).to_string())
    print(eff.groupby("agent").agg(["mean", "std", "min", "max"]).round(1).to_string())
    drop = []
    for agent in AGENTS:
        t = tab[tab.agent == agent].set_index("wind")
        drop.append(dict(agent=agent, ret_w0=t.loc[0, "iqm_return"], ret_w15=t.loc[15, "iqm_return"],
                         drop=t.loc[0, "iqm_return"] - t.loc[15, "iqm_return"],
                         succ_w0=t.loc[0, "success"], succ_w15=t.loc[15, "success"],
                         fuel_w0=t.loc[0, "main_fires"], fuel_w15=t.loc[15, "main_fires"]))
    drop = pd.DataFrame(drop); drop.to_csv("results/robustness_drop.csv", index=False)
    print(drop.round(3).to_string())


if __name__ == "__main__":
    main()
