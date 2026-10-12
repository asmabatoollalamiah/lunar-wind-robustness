"""Record side-by-side demo videos : calm-trained vs wind-trained agent,
same episode seed, same wind. Writes MP4s to demos /.

    python make_demo.py --algo ppo --seed 0 --winds 0 15 20 --episodes 3
"""
import argparse, os
import numpy as np
import imageio.v2 as imageio
from PIL import Image,ImageDraw, ImageFont
from stable_baselines3 import PPO
from src.double_dqn import DoubleDQN
from src.wind_wrapper import make_env

FONT= "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"


def font(sz):
    try: return ImageFont.truetype(FONT, sz)
    except OSError: return ImageFont.load_default()


def rollout_frames(model, wind, seed):
    env =make_env("fixed", wind=wind, render_mode="rgb_array")
    obs, _ = env.reset(seed= seed)
    frames, total, done = [env.render()], 0.0, False
    while not done:
        a, _ = model.predict(obs, deterministic=True)
        obs, r, term, trunc, _ =env.step(int(a)); total += r; done = term or trunc
        frames.append(env.render())
    env.close()
    end = "TIMED OUT" if (trunc and not term) else ("CRASHED" if r == -100 else "LANDED")
    return frames, total, end


def label(frame, title, sub, color):
    im =Image.fromarray(frame); d = ImageDraw.Draw(im)
    d.rectangle([0, 0, im.width, 54], fill=(20, 20, 30))
    d.text((12, 6), title, font=font(20), fill=color)
    d.text((12, 31), sub, font=font(15), fill=(230, 230, 230))
    return np.array(im)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--algo", default="ppo"); p.add_argument("--seed", type=int, default=0)
    p.add_argument("--winds", type=float, nargs="+", default=[0, 15, 20])
    p.add_argument("--episodes", type=int, default=2)
    p.add_argument("--pairs", nargs="*", default=None,
                   help="explicit clips as wind:episode_seed, e.g. 15:30015009")
    p.add_argument("--name",default=None)
    a = p.parse_args()
    cls = PPO if a.algo == "ppo" else DoubleDQN
    calm = cls.load(f"models/{a.algo}_calm_s{a.seed}_best", device="cpu")
    rand = cls.load(f"models/{a.algo}_rand_s{a.seed}_best", device="cpu")
    os.makedirs("demos", exist_ok=True)
    name = a.name or f"{a.algo}_calm_vs_rand_s{a.seed}"
    out = imageio.get_writer(f"demos/{name}.mp4", fps=50, codec="libx264",
                             quality=7, macro_block_size=8)
    A = a.algo.upper()
    if a.pairs:
        clips =[(float(x.split(":")[0]), int(x.split(":")[1])) for x in a.pairs]
    else:
        clips = [(w, 30_000_000 + int(w * 10) * 100 + ep) for w in a.winds for ep in range(a.episodes)]
    for ci, (w, seed) in enumerate(clips):
        if True:
            fc, rc, ec = rollout_frames(calm, w, seed)
            fr, rr, er = rollout_frames(rand, w, seed)
            n =max(len(fc), len(fr)) + 50
            H, W = fc[0].shape[0], fc[0].shape[1] * 2 + 8
            card = Image.new("RGB", (W, H), (20, 20, 30)); d = ImageDraw.Draw(card)
            d.text((W // 2, H // 2 - 22), f"Clip {ci + 1}  ·  wind {w:g}  ·  turbulence {w / 10:g}",
                   font=font(30), fill=(255, 255, 255), anchor="mm")
            d.text((W // 2, H // 2 + 22), "same episode seed for both agents" +
                   ("  ·  beyond training range" if w > 15 else ""),
                   font=font(18), fill=(190, 190, 200), anchor="mm")
            for _ in range(75):
                out.append_data(np.array(card))  # append the data to output
            for i in range(n):
                f1 = fc[min(i, len(fc) - 1)]; f2 = fr[min(i, len(fr) - 1)]
                done1,done2 = i >= len(fc) - 1, i >= len(fr) - 1
                s1 = f"wind {w:g}  |  " + (f"{ec}  score {rc:.0f}" if done1 else "flying...")
                s2 = f"wind {w:g}  |  " + (f"{er}  score {rr:.0f}" if done2 else "flying...")
                L =label(f1, f"{A}-Calm (trained without wind)", s1, (127, 167, 217))
                R = label(f2, f"{A}-Rand (trained with random wind)", s2, (255, 170, 120))
                gap = np.full((L.shape[0], 8, 3), 255, np.uint8)
                frame =np.concatenate([L, gap, R], axis=1)
                out.append_data(frame)
                if ci == 2 and i == min(len(fc), len(fr)) // 2:
                    Image.fromarray(frame).save(f"demos/{name}_still.png")#save the image
            print(f"clip {ci + 1} wind {w} seed {seed}: calm {ec} {rc:.0f} | rand {er} {rr:.0f}", flush=True)
    out.close()


if __name__ == "__main__":
    main()
