"""Scan a very long local video without uploading it.

  python run_long.py --reference ref.jpg --video "D:\\cctv\\day1.mp4" --mode person --index 0
"""
import argparse, json, mimetypes, os, time
import requests


def hms(s):
    s = int(s)
    return f"{s // 3600:02d}:{s % 3600 // 60:02d}:{s % 60:02d}"


p = argparse.ArgumentParser()
p.add_argument("--reference", required=True)
p.add_argument("--video", required=True)
p.add_argument("--mode", default="person")
p.add_argument("--match-method", default="body")
p.add_argument("--index", type=int, default=0)
p.add_argument("--server", default="http://127.0.0.1:7860")
a = p.parse_args()

mime = mimetypes.guess_type(a.reference)[0] or "image/jpeg"
with open(a.reference, "rb") as f:
    r = requests.post(f"{a.server}/api/reference", files={"reference": (os.path.basename(a.reference), f, mime)},
                      data={"mode": a.mode})
r.raise_for_status()
selected = r.json()["objects"][a.index]

t = time.perf_counter()
with open(a.reference, "rb") as f:
    r = requests.post(f"{a.server}/api/locate-video",
                      files={"reference": (os.path.basename(a.reference), f, mime)},
                      data={"mode": a.mode, "match_method": a.match_method, "object": json.dumps(selected),
                            "video_path": a.video, "compact": "1"}, timeout=None)
r.raise_for_status()
out = r.json()
print(f"Scanned {hms(out['video']['duration'])} of video in {time.perf_counter() - t:.0f}s")
print(json.dumps(out["performance"], indent=2))
for x in out["appearances"]:
    print(f"#{x['id']}  {hms(x['start'])} - {hms(x['end'])}  best {hms(x['best_timestamp'])}  score {x['best_score']}")
