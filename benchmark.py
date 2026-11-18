"""A/B benchmark: ORIGINAL app (port 7861) vs OPTIMIZED app (port 7860).

  python benchmark.py --reference ref.jpg --video clip1.mp4 clip8min.mp4 --mode person --index 0 \
      --repeat 3 --csv results.csv [--truth truth.json]

truth.json (optional hand labels):  {"clip1.mp4": [[12.0, 20.5], [40.0, 41.2]]}
Run each server alone if you want clean VRAM/CPU numbers. A throwaway warm-up run is done per server.
"""
import argparse, csv, json, mimetypes, os, statistics, time
import requests

try:
    from requests_toolbelt import MultipartEncoder  # streams the upload instead of buffering it
except Exception:
    MultipartEncoder = None


def _mime(p, default):
    return mimetypes.guess_type(p)[0] or default


def reference_objects(base, ref, mode):
    with open(ref, "rb") as f:
        r = requests.post(f"{base}/api/reference", data={"mode": mode},
                          files={"reference": (os.path.basename(ref), f, _mime(ref, "image/jpeg"))})
    r.raise_for_status()
    return r.json()["objects"]


def locate(base, a, video, selected):
    fields = {"mode": a.mode, "match_method": a.match_method, "object": json.dumps(selected), "compact": "1"}
    with open(a.reference, "rb") as rf, open(video, "rb") as vf:
        files = {"reference": (os.path.basename(a.reference), rf, _mime(a.reference, "image/jpeg")),
                 "target": (os.path.basename(video), vf, _mime(video, "video/mp4"))}
        t = time.perf_counter()
        if MultipartEncoder:
            enc = MultipartEncoder(fields={**fields, **files})
            r = requests.post(f"{base}/api/locate-video", data=enc, headers={"Content-Type": enc.content_type}, timeout=None)
        else:
            r = requests.post(f"{base}/api/locate-video", files=files, data=fields, timeout=None)
        wall = time.perf_counter() - t
    r.raise_for_status()
    return r.json(), wall


def merge(intervals, pad=1.0):
    out = []
    for s, e in sorted((s, e) for s, e in intervals):
        if out and s <= out[-1][1] + pad:
            out[-1][1] = max(out[-1][1], e)
        else:
            out.append([s, e])
    return out


def overlap(a, b):
    """Total overlapping seconds between two sorted, merged interval lists."""
    i = j = 0
    total = 0.0
    while i < len(a) and j < len(b):
        total += max(0.0, min(a[i][1], b[j][1]) - max(a[i][0], b[j][0]))
        if a[i][1] < b[j][1]:
            i += 1
        else:
            j += 1
    return total


def length(x):
    return sum(e - s for s, e in x)


def compare(ref_iv, test_iv, min_overlap, short=2.0):
    """ref_iv = baseline or ground truth; test_iv = system under test."""
    inter = overlap(ref_iv, test_iv)
    union = length(ref_iv) + length(test_iv) - inter
    hit = lambda iv: overlap([iv], test_iv) >= min_overlap * max(iv[1] - iv[0], 1e-6)
    shorts = [iv for iv in ref_iv if iv[1] - iv[0] < short]
    return {
        "iou": inter / union if union else 1.0,
        "event_recall": sum(hit(iv) for iv in ref_iv) / len(ref_iv) if ref_iv else 1.0,
        "short_event_recall": sum(hit(iv) for iv in shorts) / len(shorts) if shorts else None,
        "seconds_recall": inter / length(ref_iv) if length(ref_iv) else 1.0,
        "seconds_precision": inter / length(test_iv) if length(test_iv) else 1.0,
        "unmatched_events": sum(1 for iv in test_iv if overlap([iv], ref_iv) == 0),
    }


def intervals(resp):
    return merge([(x["start"], x["end"]) for x in resp["appearances"]])


def timed_runs(base, a, video, selected, repeat):
    locate(base, a, video, selected)  # warm-up, discarded (CUDA init, cudnn autotune, page cache)
    walls, last = [], None
    for _ in range(repeat):
        last, w = locate(base, a, video, selected)
        walls.append(w)
    return last, walls


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--reference", required=True)
    p.add_argument("--video", required=True, nargs="+")
    p.add_argument("--mode", default="person")
    p.add_argument("--match-method", default="body")
    p.add_argument("--index", type=int, default=0)
    p.add_argument("--repeat", type=int, default=3)
    p.add_argument("--min-overlap", type=float, default=0.5)
    p.add_argument("--baseline", default="http://127.0.0.1:7861")
    p.add_argument("--optimized", default="http://127.0.0.1:7860")
    p.add_argument("--truth", help="JSON of hand-labelled intervals per video filename")
    p.add_argument("--csv", default="benchmark_results.csv")
    a = p.parse_args()
    truth = json.load(open(a.truth)) if a.truth else {}
    sel_b = reference_objects(a.baseline, a.reference, a.mode)[a.index]
    sel_o = reference_objects(a.optimized, a.reference, a.mode)[a.index]

    rows = []
    for video in a.video:
        name = os.path.basename(video)
        base, bw = timed_runs(a.baseline, a, video, sel_b, a.repeat)
        opt, ow = timed_runs(a.optimized, a, video, sel_o, a.repeat)
        bm, om = statistics.median(bw), statistics.median(ow)
        bi, oi = intervals(base), intervals(opt)
        perf = opt.get("performance", {})
        row = {
            "video": name, "video_s": round(opt["video"]["duration"], 1),
            "baseline_median_s": round(bm, 2), "optimized_median_s": round(om, 2),
            "optimized_min_s": round(min(ow), 2), "optimized_max_s": round(max(ow), 2),
            "speedup": round(bm / om, 2), "realtime_x": round(opt["video"]["duration"] / om, 1),
            "baseline_events": len(bi), "optimized_events": len(oi),
            "decoder": perf.get("decoder"), "decode_mode": perf.get("decode_mode"),
            "frames_analyzed": perf.get("frames_analyzed"), "frames_skipped": perf.get("frames_skipped"),
            "decode_wait_s": perf.get("decode_wait_seconds"), "inference_s": perf.get("inference_seconds"),
            "server_wall_s": perf.get("wall_seconds"),
        }
        for k, v in compare(bi, oi, a.min_overlap).items():
            row[f"vs_baseline_{k}"] = None if v is None else round(v, 3)
        if name in truth:
            gt = merge(truth[name])
            for label, iv in (("baseline", bi), ("optimized", oi)):
                for k, v in compare(gt, iv, a.min_overlap).items():
                    row[f"{label}_vs_truth_{k}"] = None if v is None else round(v, 3)
        row["resources"] = json.dumps(perf.get("resources"))
        row["quality_profile"] = json.dumps(perf.get("quality_profile"))
        rows.append(row)
        print(json.dumps(row, indent=2))

    keys = sorted({k for r in rows for k in r}, key=lambda k: list(rows[0]).index(k) if k in rows[0] else 999)
    with open(a.csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    print(f"\nWrote {a.csv}. Note: 'vs_baseline' is agreement, not accuracy; only *_vs_truth rows measure accuracy.")


if __name__ == "__main__":
    main()
