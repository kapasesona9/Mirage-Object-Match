"""Video quality assessment, decode planning and selective enhancement for Mirage.

Drop next to app.py. Everything here is cheap: ~8 single-frame grabs + one ffprobe packet scan.
cv2 is optional; without it enhancement is skipped (assessment still works, numpy only).
"""
from __future__ import annotations

import os
import statistics
import subprocess

import numpy as np

try:
    import cv2
except Exception:  # pragma: no cover
    cv2 = None


def _env(name, default):
    try:
        return float(os.environ.get(name, default))
    except ValueError:
        return float(default)


def _grab_gray(path, t, src_w, src_h, out_w=640):
    out_h = max(2, int(src_h * out_w / src_w) // 2 * 2)
    r = subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin", "-ss", f"{t:.2f}", "-i", path,
         "-frames:v", "1", "-vf", f"scale={out_w}:{out_h},format=gray", "-f", "rawvideo", "pipe:1"],
        capture_output=True, timeout=30)
    if len(r.stdout) != out_w * out_h:
        return None
    return np.frombuffer(r.stdout, np.uint8).reshape(out_h, out_w).astype(np.float32)


def _frame_metrics(g):
    lap = 4 * g[1:-1, 1:-1] - g[:-2, 1:-1] - g[2:, 1:-1] - g[1:-1, :-2] - g[1:-1, 2:]
    sharp = float(lap.var())
    noise = float(np.median(np.abs(lap)) / 0.6745 / np.sqrt(20))  # Immerkaer-style sigma estimate
    return sharp, float(g.mean()), noise


def _gop_seconds(path, window=120):
    """Median keyframe spacing from packet flags (no decoding)."""
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "packet=pts_time,flags",
             "-read_intervals", f"%+{window}", "-of", "csv=p=0", path],
            capture_output=True, text=True, timeout=60).stdout
    except Exception:
        return None
    keys = []
    for line in out.splitlines():
        parts = line.split(",")
        if len(parts) >= 2 and "K" in parts[1]:
            try:
                keys.append(float(parts[0]))
            except ValueError:
                pass
    keys.sort()
    gaps = [b - a for a, b in zip(keys, keys[1:]) if b > a]
    return statistics.median(gaps) if gaps else None


def assess(path, info, base_rate=3.0):
    """Returns a profile dict: measured quality + chosen decode mode / sample rate / enhancement."""
    dur, w, h = info["duration"], info["width"], info["height"]
    samples = []
    for frac in (0.05, 0.2, 0.35, 0.5, 0.65, 0.8, 0.95):
        g = _grab_gray(path, info["start_time"] + dur * frac, w, h)
        if g is not None:
            samples.append(_frame_metrics(g))
    if samples:
        sharp = statistics.median(s[0] for s in samples)
        bright = statistics.median(s[1] for s in samples)
        noise = statistics.median(s[2] for s in samples)
    else:
        sharp, bright, noise = 100.0, 120.0, 2.0
    gop = _gop_seconds(path)

    blurry = sharp < _env("MIRAGE_BLUR_LAP_VAR", 60)
    dark = bright < _env("MIRAGE_DARK_MEAN", 60)
    noisy = noise > _env("MIRAGE_NOISE_SIGMA", 6)
    low_res = max(w, h) < 640

    # Decode plan: very long files with sparse keyframes are the only case where fps decoding is hopeless.
    mode, rate = "fps", base_rate
    if dur >= _env("MIRAGE_AUTO_KEYFRAME_SEC", 1800):
        if gop and gop <= _env("MIRAGE_AUTO_MAX_GOP", 2.5):
            mode = "keyframes"
        else:
            rate = min(base_rate, _env("MIRAGE_LONG_RATE", 1.0))
    if blurry or low_res:
        rate = max(rate, min(base_rate, 3))  # never undersample blurry footage: motion blur needs more looks

    return {
        "sharpness": round(sharp, 1), "brightness": round(bright, 1), "noise": round(noise, 2),
        "gop_seconds": round(gop, 2) if gop else None,
        "resolution": f"{w}x{h}", "blurry": blurry, "dark": dark, "noisy": noisy, "low_res": low_res,
        "mode": mode, "rate": rate,
        "enhance": bool(cv2 is not None and (dark or noisy)) and _env("MIRAGE_ENHANCE", 1) > 0,
    }


def enhance(frame, profile, bgr):
    """Selective, bounded-cost enhancement. Called only on frames that survive motion gating.
    Contrast (CLAHE on luminance) for dark footage, light bilateral denoise for noisy footage.
    No sharpening: it amplifies compression artefacts and creates false detections."""
    if cv2 is None or not profile.get("enhance"):
        return frame
    if profile.get("noisy"):
        frame = cv2.bilateralFilter(frame, 5, 35, 35)
    if profile.get("dark"):
        to_lab = cv2.COLOR_BGR2LAB if bgr else cv2.COLOR_RGB2LAB
        from_lab = cv2.COLOR_LAB2BGR if bgr else cv2.COLOR_LAB2RGB
        lab = cv2.cvtColor(frame, to_lab)
        lab[:, :, 0] = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(lab[:, :, 0])
        frame = cv2.cvtColor(lab, from_lab)
    return frame
