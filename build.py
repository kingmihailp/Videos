#!/usr/bin/env python3
"""Flora0world HUB - 3 minute cinematic preview. Usage:
    python3 build.py test     # contact sheet of key frames -> out/contact.png
    python3 build.py audio    # voice-over + music -> out/audio.wav, out/subs.ass
    python3 build.py render   # full video -> out/flora0world_hub_preview.mp4
"""
import math
import os
import subprocess
import sys
from multiprocessing import Pool

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

import scenes as S

W, H, FPS, DUR = S.W, S.H, 24, 180
BAR = 34
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")

# name, start, end, (fade-in secs, colour), (fade-out secs, colour)
BLACK, WHITE = (0, 0, 0), (255, 222, 170)
TIMELINE = [
    ("cabinet", 0, 46, (1.8, BLACK), (1.6, WHITE)),
    ("title", 46, 56, (1.4, WHITE), (0.9, BLACK)),
    ("meadow", 56, 86, (1.0, BLACK), (1.0, BLACK)),
    ("night", 86, 111, (1.0, BLACK), (1.0, BLACK)),
    ("map", 111, 146, (1.0, BLACK), (1.0, BLACK)),
    ("travel", 146, 166, (1.0, BLACK), (1.0, BLACK)),
    ("macro", 166, 175, (1.0, BLACK), (1.2, BLACK)),
    ("title", 175, 180, (1.2, BLACK), (1.5, BLACK)),
]

_yy, _xx = np.mgrid[0:H, 0:W]
VIG = (1 - 0.5 * (((_xx / W * 2 - 1) ** 2) * 0.8 + ((_yy / H * 2 - 1) ** 2) * 1.1)).clip(0.35, 1)[..., None].astype(np.float32)
GRADE = {"cabinet": (1.05, 0.97, 0.9), "meadow": (1.04, 1.0, 0.95), "night": (0.92, 1.0, 1.1),
         "map": (1.0, 1.0, 1.0), "travel": (1.03, 1.0, 1.0), "macro": (1.03, 1.02, 0.95), "title": (0.95, 1.02, 1.05)}


def render_frame(i):
    t = i / FPS
    name, start, end, fin, fout = next(s for s in TIMELINE if s[1] <= t < s[2]) if t < DUR else TIMELINE[-1]
    lt = t - start
    a = S.SCENES[name](lt)
    a = a * np.array(GRADE[name], np.float32)
    bright = np.clip(a - 150, 0, 255)
    bl = np.asarray(S.to_img(bright).filter(ImageFilter.GaussianBlur(3)), np.float32)
    a = a + 0.55 * bl
    a = (a - 128) * 1.06 + 128
    a *= VIG
    a += np.random.default_rng(i).normal(0, 4, (H, W, 1)).astype(np.float32)
    k_in = S.smooth(lt / fin[0])
    k_out = S.smooth((end - t) / fout[0])
    k = min(k_in, k_out)
    if k < 1:
        base = fin[1] if k_in <= k_out else fout[1]
        a = a * k + np.array(base, np.float32) * (1 - k)
    a = np.clip(a, 0, 255).astype(np.uint8)
    a[:BAR] = 0
    a[-BAR:] = 0
    return a.tobytes()


def cmd_test():
    os.makedirs(OUT, exist_ok=True)
    times = [10, 30, 43.5, 50, 62, 70.8, 78, 90, 100, 108, 118, 130, 144, 152, 160, 170, 178]
    sheet = Image.new("RGB", (W * 3, H * ((len(times) + 2) // 3)))
    for n, t in enumerate(times):
        fr = np.frombuffer(render_frame(int(t * FPS)), np.uint8).reshape(H, W, 3)
        sheet.paste(Image.fromarray(fr), ((n % 3) * W, (n // 3) * H))
    sheet.save(os.path.join(OUT, "contact.png"))
    print("saved contact.png", times)


def cmd_render():
    os.makedirs(OUT, exist_ok=True)
    ass = os.path.join(OUT, "subs.ass")
    wav = os.path.join(OUT, "audio.wav")
    assert os.path.exists(ass) and os.path.exists(wav), "run `build.py audio` first"
    dst = os.path.join(OUT, "flora0world_hub_preview.mp4")
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS),
           "-i", "-", "-i", wav,
           "-vf", f"scale=1920:1080:flags=neighbor,subtitles={ass}:fontsdir=/usr/share/fonts/truetype/dejavu,format=yuv420p",
           "-c:v", "libx264", "-crf", "28", "-preset", "slow", "-tune", "grain", "-c:a", "aac", "-b:a", "192k",
           "-t", str(DUR), "-movflags", "+faststart", dst]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    n = DUR * FPS
    with Pool(4) as pool:
        for i, fr in enumerate(pool.imap(render_frame, range(n), chunksize=6)):
            p.stdin.write(fr)
            if i % 240 == 0:
                print(f"frame {i}/{n}", flush=True)
    p.stdin.close()
    p.wait()
    print("done", dst)


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "test"
    if mode == "test":
        cmd_test()
    elif mode == "audio":
        import audio
        audio.main(OUT)
    elif mode == "render":
        cmd_render()
