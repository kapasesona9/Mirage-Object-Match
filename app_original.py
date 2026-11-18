from __future__ import annotations

import collections
import concurrent.futures
import contextlib
import difflib
import functools
import json
import math
import os
import queue
import re
import shutil
import subprocess
import tempfile
import threading
import time
import uuid
from pathlib import Path

import numpy as np
from flask import Flask, jsonify, request, send_from_directory
from PIL import Image, ImageOps

try:  # optional helper module (quality assessment + selective enhancement)
    import mirage_quality as mq
except ImportError:  # app still works without it
    mq = None

ROOT = Path(__file__).resolve().parent
app = Flask(__name__, static_folder=str(ROOT / "static"), static_url_path="")


# --------------------------------------------------------------------------------------
# Configuration (all tunable via environment variables)
# --------------------------------------------------------------------------------------
def _env_float(name, default):
    try:
        return float(os.environ.get(name, default))
    except ValueError:
        return float(default)


def _env_flag(name, default):
    return os.environ.get(name, "1" if default else "0").strip().lower() not in {"0", "off", "false", "no", ""}


_VIDEO_MAX_BYTES = int(_env_float("MIRAGE_VIDEO_MAX_MB", 51200)) * 1024 * 1024
_VIDEO_SAMPLE_RATE = _env_float("MIRAGE_SAMPLE_RATE", 3)
if float(_VIDEO_SAMPLE_RATE).is_integer():
    _VIDEO_SAMPLE_RATE = int(_VIDEO_SAMPLE_RATE)
app.config["MAX_CONTENT_LENGTH"] = _VIDEO_MAX_BYTES + 22 * 1024 * 1024

# "auto"      : measure the video and choose fps vs keyframes + sample rate (needs mirage_quality.py).
# "fps"       : decode every frame, keep `_VIDEO_SAMPLE_RATE` per second.
# "keyframes" : decode ONLY I-frames (-skip_frame nokey). Much faster on long CCTV files, but
#               sampling density then equals the camera's GOP length.
_DECODE_MODE = os.environ.get("MIRAGE_DECODE_MODE", "auto").strip().lower()
_DEDUP_ENABLED = _env_flag("MIRAGE_DEDUP", True)
_DEDUP_MEAN = _env_float("MIRAGE_DEDUP_MEAN", 1.5)
_DEDUP_BLOCK = _env_float("MIRAGE_DEDUP_BLOCK", 10.0)
_DEDUP_MAX_GAP = _env_float("MIRAGE_DEDUP_MAX_GAP_SEC", 2.0)
_QUEUE_DEPTH = max(2, int(_env_float("MIRAGE_QUEUE_DEPTH", 8)))
_SAVE_WORKERS = max(1, int(_env_float("MIRAGE_SAVE_WORKERS", 3)))
_SCREENSHOT_DIR = ROOT / "screenshots"

_model = None
_model_lock = threading.Lock()
_person_model = None
_person_model_lock = threading.Lock()
_person_inference_lock = threading.Lock()
_object_inference_lock = threading.Lock()
_ALLOWED_MIMES = {"image/jpeg", "image/png", "image/webp", "image/gif", "image/bmp"}
_VIDEO_EXTENSIONS = {".mp4", ".mov", ".webm"}
_STOP_WORDS = {"a", "an", "the", "of", "in", "on", "with", "and", "background"}
_PERSON_MATCH_METHODS = {"body", "face"}
_PERSON_THRESHOLDS = {"body": _env_float("MIRAGE_BODY_THRESHOLD", 0.85), "face": _env_float("MIRAGE_FACE_THRESHOLD", 0.45)}
_PTS_RE = re.compile(r"pts_time:\s*(\S+)")
_MB = 1024 * 1024


# --------------------------------------------------------------------------------------
# Models (loaded once, reused for every request)
# --------------------------------------------------------------------------------------
def get_model():
    global _model
    if _model is None:
        with _model_lock:
            if _model is None:
                os.environ.setdefault("CACTUS_TELEMETRY", "0")
                import cactus

                _model = cactus.Mirage()
    return _model


def _onnx_providers():
    """Prefer the CUDA execution provider when onnxruntime-gpu is installed."""
    override = os.environ.get("MIRAGE_ORT_PROVIDERS")
    if override:
        return [item.strip() for item in override.split(",") if item.strip()]
    try:
        import onnxruntime as ort

        available = set(ort.get_available_providers())
    except Exception:
        return None
    providers = [p for p in ("CUDAExecutionProvider",) if p in available]
    return providers + ["CPUExecutionProvider"] if providers else None


def get_person_model():
    global _person_model
    if _person_model is None:
        with _person_model_lock:
            if _person_model is None:
                os.environ.setdefault("ORT_DISABLE_TELEMETRY", "1")
                from insightface.app import PersonAnalysis

                providers = _onnx_providers()
                model = None
                if providers:
                    try:
                        model = PersonAnalysis(name="cheetah_s", providers=providers)
                    except TypeError:
                        model = None
                if model is None:
                    model = PersonAnalysis(name="cheetah_s")
                model.prepare()
                _person_model = model
    return _person_model


def _warm_models():
    for loader in (get_person_model, get_model):
        try:
            loader()
        except Exception:
            app.logger.exception("Model preload failed for %s (will retry lazily)", loader.__name__)


# --------------------------------------------------------------------------------------
# Person helpers
# --------------------------------------------------------------------------------------
def _person_image(path):
    with Image.open(path) as opened:
        rgb = np.asarray(ImageOps.exif_transpose(opened).convert("RGB"))
    return rgb[:, :, ::-1].copy(), (rgb.shape[1], rgb.shape[0])


def _normal_box(box, width, height):
    if box is None:
        return None
    try:
        left, top, right, bottom = (float(value) for value in box[:4])
    except (TypeError, ValueError, IndexError):
        return None
    if width <= 0 or height <= 0 or not all(math.isfinite(value) for value in (left, top, right, bottom)):
        return None
    normalized = [
        round(max(0, min(1000, left * 1000 / width))),
        round(max(0, min(1000, top * 1000 / height))),
        round(max(0, min(1000, right * 1000 / width))),
        round(max(0, min(1000, bottom * 1000 / height))),
    ]
    return normalized if _valid_bbox({"bbox": normalized}) else None


def _person_feature(person, method):
    if method == "face":
        face = getattr(person, "face", None)
        return getattr(face, "embedding", None) if face is not None else None
    return getattr(person, "reid_feature", None)


def _person_record(person, index, width, height):
    body_box = _normal_box(getattr(person, "body_bbox", None), width, height)
    face = getattr(person, "face", None)
    face_box = _normal_box(getattr(face, "bbox", None), width, height) if face is not None else None
    bbox = body_box or face_box
    if bbox is None:
        return None
    score = getattr(person, "det_score", None)
    if score is None and face is not None:
        score = getattr(face, "det_score", None)
    try:
        score = float(score) if score is not None else None
        if score is not None and not math.isfinite(score):
            score = None
    except (TypeError, ValueError):
        score = None
    return {
        "id": index + 1,
        "name": f"Person {index + 1}",
        "bbox": bbox,
        "face_bbox": face_box,
        "score": round(score, 4) if score is not None else None,
        "body_available": getattr(person, "reid_feature", None) is not None,
        "face_available": getattr(face, "embedding", None) is not None if face is not None else False,
    }


def _person_observations(model, image, size):
    width, height = size
    observations = model.get(image)
    records = []
    by_id = {}
    for index, person in enumerate(observations):
        record = _person_record(person, index, width, height)
        if record is not None:
            records.append(record)
            by_id[str(record["id"])] = person
    return records, by_id


def _cosine_similarity(first, second):
    if first is None or second is None:
        return None
    try:
        a = np.asarray(first, dtype=np.float64).ravel()
        b = np.asarray(second, dtype=np.float64).ravel()
    except (TypeError, ValueError):
        return None
    if a.size == 0 or a.size != b.size:
        return None
    a_norm = float(np.sqrt(a @ a))
    b_norm = float(np.sqrt(b @ b))
    if a_norm == 0 or b_norm == 0 or not math.isfinite(a_norm) or not math.isfinite(b_norm):
        return None
    return max(-1.0, min(1.0, float(a @ b) / (a_norm * b_norm)))


def _bbox_iou(first, second):
    if not _valid_bbox(first) or not _valid_bbox(second):
        return 0.0
    a_left, a_top, a_right, a_bottom = (float(value) for value in first["bbox"][:4])
    b_left, b_top, b_right, b_bottom = (float(value) for value in second["bbox"][:4])
    intersection = max(0.0, min(a_right, b_right) - max(a_left, b_left)) * max(0.0, min(a_bottom, b_bottom) - max(a_top, b_top))
    area_a = (a_right - a_left) * (a_bottom - a_top)
    area_b = (b_right - b_left) * (b_bottom - b_top)
    union = area_a + area_b - intersection
    return intersection / union if union > 0 else 0.0


def _selected_person_feature(reference_records, reference_observations, selected, method):
    nearest = max(reference_records, key=lambda record: _bbox_iou(record, selected), default=None)
    if nearest is None or _bbox_iou(nearest, selected) < 0.5:
        raise ValueError("The selected person was not found again in the reference image. Detect the reference people again.")
    person = reference_observations.get(str(nearest["id"]))
    if person is None:
        raise ValueError("The selected person was not found again in the reference image. Detect the reference people again.")
    feature = _person_feature(person, method)
    if feature is None:
        label = "face" if method == "face" else "full-body"
        raise ValueError(f"The selected person has no usable {label} features for this matching method.")
    return feature


def _rank_people(reference_feature, target_observations, target_records, method, all_scores=None):
    threshold = _PERSON_THRESHOLDS[method]
    candidates = []
    for record in target_records:
        person = target_observations.get(str(record["id"]))
        feature = _person_feature(person, method) if person is not None else None
        score = _cosine_similarity(reference_feature, feature)
        if all_scores is not None and score is not None:
            all_scores.append({"target_id": record["id"], "score": round(score, 4)})
        if score is None or score < threshold:
            continue
        candidate = {
            "target_id": record["id"],
            "score": round(score, 4),
            "name_score": None,
            "appearance_score": None,
            "person_score": round(score, 4),
            "match_method": method,
        }
        candidate[f"{method}_score"] = round(score, 4)
        candidates.append(candidate)
    candidates.sort(key=lambda item: item["score"], reverse=True)
    return candidates


# --------------------------------------------------------------------------------------
# Object helpers
# --------------------------------------------------------------------------------------
def _normalized_objects(result):
    objects = result.get("objects", []) if isinstance(result, dict) else []
    if not isinstance(objects, list):
        return []
    return [
        item
        for item in objects
        if isinstance(item, dict)
        and item.get("name")
        and str(item.get("position", "")).lower() != "background"
        and "background" not in str(item.get("name", "")).lower()
    ]


def _object_text(item):
    values = [item.get("name", "")]
    for key in ("description", "attributes", "properties"):
        value = item.get(key)
        if isinstance(value, str):
            values.append(value)
        elif isinstance(value, dict):
            values.extend(str(part) for part in value.values())
    return " ".join(values)


def _semantic_similarity(first, second):
    normalize = lambda value: re.sub(r"[^a-z0-9 ]+", " ", value.lower()).strip()
    left, right = normalize(_object_text(first)), normalize(_object_text(second))
    left_words = {word for word in left.split() if word not in _STOP_WORDS}
    right_words = {word for word in right.split() if word not in _STOP_WORDS}
    union = left_words | right_words
    overlap = len(left_words & right_words) / len(union) if union else 0.0
    sequence = difflib.SequenceMatcher(None, left, right).ratio()
    return max(overlap, sequence)


def _object_crop(image, item):
    box = item.get("bbox")
    if not isinstance(box, list) or len(box) < 4:
        return None
    try:
        left, top, right, bottom = (float(value) for value in box[:4])
    except (TypeError, ValueError):
        return None
    if not all(0 <= value <= 1000 for value in (left, top, right, bottom)) or right <= left or bottom <= top:
        return None
    width, height = image.size
    x1, y1 = round(left * width / 1000), round(top * height / 1000)
    x2, y2 = round(right * width / 1000), round(bottom * height / 1000)
    pad_x, pad_y = round((x2 - x1) * 0.04), round((y2 - y1) * 0.04)
    crop = image.crop((max(0, x1 - pad_x), max(0, y1 - pad_y), min(width, x2 + pad_x), min(height, y2 + pad_y)))
    return crop if crop.width and crop.height else None


def _valid_bbox(item):
    box = item.get("bbox") if isinstance(item, dict) else None
    if not isinstance(box, list) or len(box) < 4:
        return False
    try:
        left, top, right, bottom = (float(value) for value in box[:4])
    except (TypeError, ValueError):
        return False
    return (
        all(math.isfinite(value) and 0 <= value <= 1000 for value in (left, top, right, bottom))
        and right > left
        and bottom > top
    )


def _appearance_descriptor(crop):
    if crop is None:
        return None
    rgb = crop.convert("RGB")
    area = max(1, rgb.width * rgb.height)
    pixels = np.asarray(rgb)
    hist = np.stack([np.bincount(pixels[..., c].ravel(), minlength=256) for c in range(3)]).astype(np.float64) / area
    thumb = np.asarray(rgb.resize((16, 16), Image.Resampling.BILINEAR), dtype=np.float32)
    return hist, thumb


def _descriptor_similarity(first, second):
    if first is None or second is None:
        return None
    histogram_overlap = float(np.minimum(first[0], second[0]).sum() / 3)
    histogram_overlap = max(0.0, min(1.0, histogram_overlap))
    thumbnail_similarity = max(0.0, 1.0 - float(np.abs(first[1] - second[1]).mean() / 255))
    return 0.55 * histogram_overlap + 0.45 * thumbnail_similarity


def _appearance_similarity(first_crop, second_crop):
    return _descriptor_similarity(_appearance_descriptor(first_crop), _appearance_descriptor(second_crop))


def _make_matches(reference_objects, target_objects, reference_image, target_image, reference_cache=None):
    target_descriptors = [_appearance_descriptor(_object_crop(target_image, item)) for item in target_objects]
    matches = []
    for reference_index, reference in enumerate(reference_objects):
        if reference_cache is not None and reference_index in reference_cache:
            reference_descriptor = reference_cache[reference_index]
        else:
            reference_descriptor = _appearance_descriptor(_object_crop(reference_image, reference))
            if reference_cache is not None:
                reference_cache[reference_index] = reference_descriptor
        candidates = []
        for target_index, target in enumerate(target_objects):
            name_score = _semantic_similarity(reference, target)
            appearance_score = _descriptor_similarity(reference_descriptor, target_descriptors[target_index])
            if appearance_score is None:
                score = name_score
            else:
                score = 0.68 * name_score + 0.32 * appearance_score
            candidates.append({
                "target_id": target.get("id") if target.get("id") is not None else target_index + 1,
                "score": round(score, 4),
                "name_score": round(name_score, 4),
                "appearance_score": round(appearance_score, 4) if appearance_score is not None else None,
            })
        candidates.sort(key=lambda item: item["score"], reverse=True)
        matches.append({
            "reference_id": reference.get("id") if reference.get("id") is not None else reference_index + 1,
            "candidates": candidates,
        })
    return matches


@app.get("/")
def index():
    return send_from_directory(app.static_folder, "index.html")


def _save_upload(upload):
    suffix = Path(upload.filename).suffix.lower()
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as temp:
        upload.save(temp, buffer_size=1024 * 1024)
    return temp.name


def _valid_upload(upload):
    return upload is not None and bool(upload.filename) and upload.mimetype in _ALLOWED_MIMES


def _describe_options():
    return {"sections": ["objects"], "fields": ["bbox", "score", "position"], "max_objects": 64}


def _upload_size(upload):
    upload.stream.seek(0, os.SEEK_END)
    size = upload.stream.tell()
    upload.stream.seek(0)
    return size


def _parse_selected_object(raw):
    try:
        selected = json.loads(raw or "{}")
    except json.JSONDecodeError as exc:
        raise ValueError("Select a detected reference object.") from exc
    name = str(selected.get("name", "")).strip() if isinstance(selected, dict) else ""
    if not name or not _valid_bbox(selected):
        raise ValueError("Select a detected reference object with a valid bounding box.")
    return selected


# --------------------------------------------------------------------------------------
# Video: probing, streaming decode, motion gating
# --------------------------------------------------------------------------------------
class _DecodeError(RuntimeError):
    pass


def _probe_video(path):
    if not shutil.which("ffprobe") or not shutil.which("ffmpeg"):
        raise RuntimeError("Video search needs ffmpeg and ffprobe on PATH. Install FFmpeg, then restart this app.")
    try:
        result = subprocess.run(
            [
                "ffprobe", "-v", "error", "-select_streams", "v:0",
                "-show_entries",
                "format=duration,format_name,start_time:stream=codec_type,width,height:stream_tags=rotate:stream_side_data=rotation",
                "-of", "json", path,
            ],
            capture_output=True,
            text=True,
            check=False,
            timeout=20,
        )
    except subprocess.TimeoutExpired as exc:
        raise ValueError("Could not inspect this video. Try a valid MP4, MOV, or WebM file.") from exc
    if result.returncode:
        raise ValueError("This video is invalid or unsupported. Choose a valid MP4, MOV, or WebM file.")
    try:
        info = json.loads(result.stdout)
        duration = float(info["format"]["duration"])
        format_names = set(str(info["format"].get("format_name", "")).lower().split(","))
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError("Could not read the video metadata. Choose a valid MP4, MOV, or WebM file.") from exc
    if not math.isfinite(duration) or duration <= 0:
        raise ValueError("The video has no readable duration. Choose a valid MP4, MOV, or WebM file.")
    if Path(path).suffix.lower() == ".webm":
        supported_container = bool(format_names & {"matroska", "webm"})
    else:
        supported_container = bool(format_names & {"mov", "mp4", "m4a", "3gp", "3g2", "mj2"})
    if not supported_container:
        raise ValueError("This video container is unsupported. Choose an MP4, MOV, or WebM file.")

    streams = info.get("streams") or []
    try:
        stream = streams[0]
        width, height = int(stream["width"]), int(stream["height"])
    except (IndexError, KeyError, TypeError, ValueError) as exc:
        raise ValueError("The file has no readable video stream. Choose a valid MP4, MOV, or WebM file.") from exc
    rotation = (stream.get("tags") or {}).get("rotate")
    for side in stream.get("side_data_list") or []:
        if rotation is None and isinstance(side, dict) and "rotation" in side:
            rotation = side["rotation"]
    try:
        if rotation is not None and abs(int(float(rotation))) % 180 == 90:
            width, height = height, width
    except (TypeError, ValueError):
        pass
    try:
        start_time = max(0.0, float(info["format"].get("start_time") or 0.0))
    except (TypeError, ValueError):
        start_time = 0.0
    return {"duration": duration, "width": width, "height": height, "start_time": start_time}


def _scaled_size(width, height, limit=1280):
    scale = min(1.0, limit / width, limit / height)
    return max(2, int(width * scale) // 2 * 2), max(2, int(height * scale) // 2 * 2)


@functools.lru_cache(maxsize=1)
def _cuda_decode_available():
    setting = os.environ.get("MIRAGE_HWACCEL", "auto").strip().lower()
    if setting in {"0", "off", "none", "cpu", "false"}:
        return False
    try:
        out = subprocess.run(["ffmpeg", "-hide_banner", "-hwaccels"], capture_output=True, text=True, timeout=10).stdout
    except Exception:
        return False
    if "cuda" not in out:
        return False
    return setting == "cuda" or shutil.which("nvidia-smi") is not None


def _decode_commands(video_path, mode, size, pix_fmt, rate_value):
    width, height = size
    head = ["ffmpeg", "-hide_banner", "-nostdin"]
    tail = ["-an", "-sn", "-dn", "-pix_fmt", pix_fmt, "-fps_mode", "passthrough", "-f", "rawvideo", "pipe:1"]
    if mode == "keyframes":
        return [("cpu-keyframes", head + ["-loglevel", "info", "-skip_frame", "nokey", "-i", video_path,
                                          "-vf", f"showinfo,scale={width}:{height}"] + tail)]
    rate = f"{rate_value:g}"
    attempts = []
    if _cuda_decode_available():
        attempts.append(("cuda", head + ["-loglevel", "error", "-hwaccel", "cuda", "-hwaccel_output_format", "cuda",
                                         "-i", video_path,
                                         "-vf", f"fps={rate},scale_cuda={width}:{height},hwdownload,format=nv12"] + tail))
    attempts.append(("cpu", head + ["-loglevel", "error", "-i", video_path,
                                    "-vf", f"fps={rate},scale={width}:{height}"] + tail))
    return attempts


def _iter_raw_frames(cmd, width, height, keyframes, stop, stats):
    """Stream raw frames from ffmpeg through a pipe. The yielded array is a reused buffer."""
    frame_bytes = width * height * 3
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=frame_bytes * 2)
    pts_queue = queue.Queue()
    tail_lines = collections.deque(maxlen=20)

    def drain_stderr():
        for raw in iter(proc.stderr.readline, b""):
            line = raw.decode("utf-8", "replace")
            match = _PTS_RE.search(line) if keyframes else None
            if match:
                try:
                    pts_queue.put(float(match.group(1)))
                except ValueError:
                    pts_queue.put(None)
            else:
                tail_lines.append(line.strip())

    reader = threading.Thread(target=drain_stderr, daemon=True)
    reader.start()
    buffer = bytearray(frame_bytes)
    view = memoryview(buffer)
    array = np.frombuffer(buffer, dtype=np.uint8).reshape(height, width, 3)
    index = 0
    try:
        while not stop.is_set():
            started = time.perf_counter()
            got = 0
            while got < frame_bytes:
                count = proc.stdout.readinto(view[got:])
                if not count:
                    break
                got += count
            if got < frame_bytes:
                break
            pts = None
            if keyframes:
                try:
                    pts = pts_queue.get(timeout=10)
                except queue.Empty as exc:
                    raise _DecodeError("ffmpeg did not report frame timestamps") from exc
            stats["decode_s"] += time.perf_counter() - started
            yield index, pts, array
            index += 1
        if not stop.is_set():
            proc.wait(timeout=30)
            if proc.returncode and index == 0:
                reader.join(timeout=2)
                raise _DecodeError(" | ".join(tail_lines) or f"ffmpeg exited with {proc.returncode}")
            if proc.returncode:
                app.logger.warning("ffmpeg stopped early (code %s) after %d frames: %s",
                                   proc.returncode, index, " | ".join(tail_lines))
    finally:
        if proc.poll() is None:
            proc.kill()
        proc.wait()
        reader.join(timeout=2)
        for stream in (proc.stdout, proc.stderr):
            with contextlib.suppress(Exception):
                stream.close()


@functools.lru_cache(maxsize=8)
def _signature_grid(height, width):
    rows = np.linspace(0, height, min(18, height), endpoint=False).astype(np.intp)
    cols = np.linspace(0, width, min(32, width), endpoint=False).astype(np.intp)
    counts = (np.diff(np.append(rows, height))[:, None] * np.diff(np.append(cols, width))[None, :]).astype(np.float32)
    return rows, cols, counts


def _frame_signature(array):
    rows, cols, counts = _signature_grid(array.shape[0], array.shape[1])
    summed = np.add.reduceat(array[:, :, 1], rows, axis=0, dtype=np.uint32)
    summed = np.add.reduceat(summed, cols, axis=1)
    return summed / counts


def _put(target, item, stop):
    while not stop.is_set():
        try:
            target.put(item, timeout=0.25)
            return True
        except queue.Full:
            continue
    return False


def _empty(target):
    with contextlib.suppress(queue.Empty):
        while True:
            target.get_nowait()


def _save_frame(frame, path):
    Image.fromarray(frame).save(path, "JPEG", quality=90)
    return path


def _default_profile():
    return {"mode": "keyframes" if _DECODE_MODE == "keyframes" else "fps", "rate": _VIDEO_SAMPLE_RATE, "enhance": False}


def _enhance(frame, profile, bgr):
    if mq is None or not profile.get("enhance"):
        return frame
    try:
        return mq.enhance(frame, profile, bgr)
    except Exception:
        return frame


def _producer(video_path, info, size, mode, pix_fmt, out_queue, stop, stats, saver, save_dir, rate, profile):
    """Decode -> motion-gate -> enhance (only if needed) -> (optionally) write JPEG for the object model."""
    keyframes = mode == "keyframes"
    try:
        attempts = _decode_commands(video_path, mode, size, pix_fmt, rate)
        last_signature, last_ts = None, None
        for attempt_no, (label, cmd) in enumerate(attempts):
            produced = 0
            try:
                with contextlib.closing(_iter_raw_frames(cmd, size[0], size[1], keyframes, stop, stats)) as frames:
                    for index, pts, array in frames:
                        produced += 1
                        ts = max(0.0, pts - info["start_time"]) if keyframes and pts is not None else index / rate
                        process = True
                        if _DEDUP_ENABLED:
                            started = time.perf_counter()
                            signature = _frame_signature(array)
                            if last_signature is not None and (ts - last_ts) < _DEDUP_MAX_GAP:
                                diff = np.abs(signature - last_signature)
                                if float(diff.mean()) < _DEDUP_MEAN and float(diff.max()) < _DEDUP_BLOCK:
                                    process = False
                            stats["dedup_s"] += time.perf_counter() - started
                        if process:
                            if _DEDUP_ENABLED:
                                last_signature, last_ts = signature, ts
                            frame = _enhance(array.copy(), profile, pix_fmt == "bgr24")
                            future = None
                            if save_dir is not None:
                                future = saver.submit(_save_frame, frame, os.path.join(save_dir, f"frame_{index:08d}.jpg"))
                            if not _put(out_queue, ("frame", index, ts, frame, future), stop):
                                return
                        elif not _put(out_queue, ("skip", index, ts), stop):
                            return
                stats["decoder"] = label
                break
            except _DecodeError as exc:
                if produced or attempt_no == len(attempts) - 1:
                    raise
                app.logger.warning("Decoder %s failed (%s); falling back to the next decoder", label, exc)
    except Exception as exc:
        _put(out_queue, ("error", exc), stop)
    finally:
        _put(out_queue, ("done",), stop)


def _drain(source, producer):
    while True:
        try:
            item = source.get(timeout=1)
        except queue.Empty:
            if not producer.is_alive() and source.empty():
                return
            continue
        if item[0] == "done":
            return
        if item[0] == "error":
            raise item[1]
        yield item


def _fast_tmp_dir():
    return "/dev/shm" if os.path.isdir("/dev/shm") and os.access("/dev/shm", os.W_OK) else None


def _video_appearances(frames):
    appearances = []
    current = []
    for frame in frames:
        if frame["candidate"] is None:
            if current:
                appearances.append(current)
                current = []
            continue
        if current and frame["sample_index"] != current[-1]["sample_index"] + 1:
            appearances.append(current)
            current = []
        current.append(frame)
    if current:
        appearances.append(current)

    return [
        {
            "id": index + 1,
            "start": round(samples[0]["timestamp"], 3),
            "end": round(samples[-1]["timestamp"], 3),
            "best_timestamp": round(max(samples, key=lambda item: item["score"])["timestamp"], 3),
            "best_score": max(item["score"] for item in samples),
            "sample_count": len(samples),
            "best_frame": max(samples, key=lambda item: item["score"]),
        }
        for index, samples in enumerate(appearances)
    ]


# --------------------------------------------------------------------------------------
# Event-based, asynchronous screenshots (opt-in: form field screenshots=1 or MIRAGE_SCREENSHOTS=1)
# --------------------------------------------------------------------------------------
def _extract_screenshot(video_path, timestamp, out_path):
    result = subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin", "-ss", f"{timestamp:.3f}", "-i", video_path,
         "-frames:v", "1", "-q:v", "2", "-y", str(out_path)],
        capture_output=True, text=True, timeout=120,
    )
    if result.returncode or not Path(out_path).exists():
        raise RuntimeError(result.stderr.strip() or "ffmpeg screenshot failed")
    return str(out_path)


class _ScreenshotWorker:
    def __init__(self, video_path, workers=2):
        self.video_path = video_path
        self.job_id = uuid.uuid4().hex
        self.dir = _SCREENSHOT_DIR / self.job_id
        self.dir.mkdir(parents=True, exist_ok=True)
        self.pool = concurrent.futures.ThreadPoolExecutor(max_workers=workers, thread_name_prefix="mirage-shot")
        self.futures = {}
        self.count = 0
        self._best = None
        self._last_index = None

    def observe(self, record):
        if record["candidate"] is None:
            self._flush()
            return
        if self._best is not None and record["sample_index"] != self._last_index + 1:
            self._flush()
        if self._best is None or record["score"] > self._best["score"]:
            self._best = record
        self._last_index = record["sample_index"]

    def _flush(self):
        if self._best is None:
            return
        self.count += 1
        timestamp = self._best["timestamp"]
        name = f"appearance_{self.count:04d}_t{timestamp:.3f}.jpg"
        self.futures[self.count] = (name, self.pool.submit(_extract_screenshot, self.video_path, timestamp, self.dir / name))
        self._best = None
        self._last_index = None

    def finish(self):
        self._flush()
        urls = {}
        for appearance_id, (name, future) in self.futures.items():
            try:
                future.result(timeout=180)
                urls[appearance_id] = f"/screenshots/{self.job_id}/{name}"
            except Exception:
                app.logger.exception("Screenshot for appearance %s failed", appearance_id)
        self.pool.shutdown(wait=True)
        return urls

    def abort(self):
        self.pool.shutdown(wait=False, cancel_futures=True)


def _prune_screenshots(max_age_seconds=24 * 3600):
    if not _SCREENSHOT_DIR.exists():
        return
    cutoff = time.time() - max_age_seconds
    for child in _SCREENSHOT_DIR.iterdir():
        with contextlib.suppress(OSError):
            if child.is_dir() and child.stat().st_mtime < cutoff:
                shutil.rmtree(child, ignore_errors=True)


@app.get("/screenshots/<job>/<path:name>")
def serve_screenshot(job, name):
    if not re.fullmatch(r"[0-9a-f]{32}", job):
        return jsonify(error="Not found."), 404
    return send_from_directory(_SCREENSHOT_DIR / job, name)


# --------------------------------------------------------------------------------------
# Resource sampling for the benchmark report
# --------------------------------------------------------------------------------------
class _ResourceSampler:
    def __init__(self, interval=1.0):
        self.interval = interval
        self._stop = threading.Event()
        self._thread = None
        self.samples = collections.defaultdict(list)
        try:
            import psutil

            self._psutil, self._proc = psutil, psutil.Process()
        except Exception:
            self._psutil = self._proc = None
        self._nvml = self._gpu = None
        try:
            import pynvml

            pynvml.nvmlInit()
            self._nvml, self._gpu = pynvml, pynvml.nvmlDeviceGetHandleByIndex(0)
        except Exception:
            pass

    def _sample(self):
        if self._psutil:
            try:
                self.samples["cpu_percent"].append(self._psutil.cpu_percent(None))
                rss = self._proc.memory_info().rss
                for child in self._proc.children(recursive=True):
                    with contextlib.suppress(Exception):
                        rss += child.memory_info().rss
                self.samples["rss_mb"].append(rss / _MB)
            except Exception:
                pass
        if self._nvml:
            try:
                self.samples["gpu_util_percent"].append(self._nvml.nvmlDeviceGetUtilizationRates(self._gpu).gpu)
                self.samples["vram_used_mb"].append(self._nvml.nvmlDeviceGetMemoryInfo(self._gpu).used / _MB)
            except Exception:
                pass

    def _run(self):
        while not self._stop.wait(self.interval):
            self._sample()

    def start(self):
        if self._psutil:
            self._psutil.cpu_percent(None)
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2)
        self._sample()
        return {
            key: {"avg": round(sum(values) / len(values), 1), "peak": round(max(values), 1)}
            for key, values in self.samples.items() if values
        } or None


# --------------------------------------------------------------------------------------
# Routes
# --------------------------------------------------------------------------------------
@app.post("/api/reference")
def detect_reference_objects():
    reference = request.files.get("reference")
    mode = request.form.get("mode", "object")
    if reference is None or not reference.filename:
        return jsonify(error="Choose a reference image."), 400
    if reference.mimetype not in _ALLOWED_MIMES:
        return jsonify(error="Use a JPEG, PNG, WebP, GIF, or BMP image."), 415
    if mode not in {"object", "person"}:
        return jsonify(error="Choose Object or Person search mode."), 400
    path = None
    try:
        path = _save_upload(reference)
        if mode == "person":
            model = get_person_model()
            with _person_inference_lock:
                image, size = _person_image(path)
                people, _observations = _person_observations(model, image, size)
            return jsonify(mode=mode, result={"people": people}, objects=people)
        with _object_inference_lock:
            result = get_model().describe(path, **_describe_options())
        objects = _normalized_objects(result)
        return jsonify(mode=mode, result=result, objects=objects)
    except Exception:
        app.logger.exception("Reference analysis failed")
        return jsonify(error="We couldn’t analyze this reference photo. Check the file and try again."), 500
    finally:
        if path and os.path.exists(path):
            os.unlink(path)


@app.post("/api/locate")
def locate_selected_object():
    reference = request.files.get("reference")
    target = request.files.get("target")
    mode = request.form.get("mode", "object")
    match_method = request.form.get("match_method", "body")
    if not _valid_upload(reference) or not _valid_upload(target):
        return jsonify(error="Choose both the reference image and the target image."), 400
    if mode not in {"object", "person"}:
        return jsonify(error="Choose Object or Person search mode."), 400
    if mode == "person" and match_method not in _PERSON_MATCH_METHODS:
        return jsonify(error="Choose Body appearance or Face identity matching."), 400
    try:
        selected = _parse_selected_object(request.form.get("object", "{}"))
    except (ValueError, TypeError) as exc:
        return jsonify(error=f"Invalid reference object: {exc}"), 400

    paths = []
    try:
        paths.append(_save_upload(reference))
        paths.append(_save_upload(target))

        if mode == "person":
            model = get_person_model()
            with _person_inference_lock:
                reference_bgr, reference_size = _person_image(paths[0])
                target_bgr, target_size = _person_image(paths[1])
                reference_people, reference_observations = _person_observations(model, reference_bgr, reference_size)
                target_people, target_observations = _person_observations(model, target_bgr, target_size)
                reference_feature = _selected_person_feature(reference_people, reference_observations, selected, match_method)
                all_scores = []
                candidates = _rank_people(reference_feature, target_observations, target_people, match_method, all_scores)
            return jsonify({
                "mode": mode,
                "match_method": match_method,
                "match_threshold": _PERSON_THRESHOLDS[match_method],
                "reference": {"objects": [selected]},
                "target": {"result": {"people": target_people}, "objects": target_people},
                "keywords": ["person"],
                "matches": [{"reference_id": selected.get("id", 1), "candidates": candidates}],
                "all_scores": sorted(all_scores, key=lambda item: item["score"], reverse=True),
            })

        with Image.open(paths[0]) as opened:
            reference_image = ImageOps.exif_transpose(opened).convert("RGB")
        with Image.open(paths[1]) as opened:
            target_image = ImageOps.exif_transpose(opened).convert("RGB")
        name = str(selected["name"]).strip()
        with _object_inference_lock:
            target_result = get_model().describe(paths[1], keywords=[name], **_describe_options())
        target_objects = _normalized_objects(target_result)
        matches = _make_matches([selected], target_objects, reference_image, target_image)
        return jsonify({
            "mode": mode,
            "reference": {"objects": [selected]},
            "target": {"result": target_result, "objects": target_objects},
            "keywords": [name],
            "matches": matches,
        })
    except ValueError as exc:
        return jsonify(error=str(exc)), 400
    except Exception:
        app.logger.exception("Target image search failed")
        return jsonify(error="We couldn’t search this image. Check your files and try again."), 500
    finally:
        for path in paths:
            if os.path.exists(path):
                os.unlink(path)


def _search_video(video_path, reference_path, mode, match_method, selected, info, stats, shots):
    """Pipeline: [thread] ffmpeg decode + motion gate + selective enhance (+JPEG writer pool) -> bounded
    queue -> [this thread] model inference. Skipped frames reuse the previous record."""
    size = _scaled_size(info["width"], info["height"])

    # Quality assessment + decode plan. An explicit MIRAGE_DECODE_MODE=fps|keyframes always wins.
    try:
        profile = mq.assess(video_path, info, _VIDEO_SAMPLE_RATE) if mq else _default_profile()
    except Exception:
        app.logger.exception("Quality assessment failed; using default decode settings")
        profile = _default_profile()
    if _DECODE_MODE in {"fps", "keyframes"}:
        profile = dict(profile, mode=_DECODE_MODE, rate=_VIDEO_SAMPLE_RATE)
    keyframes = profile["mode"] == "keyframes"
    rate = profile["rate"]
    stats["mode"], stats["rate"], stats["profile"] = profile["mode"], rate, profile

    pix_fmt = "bgr24" if mode == "person" else "rgb24"
    stop = threading.Event()
    out_queue = queue.Queue(maxsize=_QUEUE_DEPTH)
    frames = []
    state = {"last": None}

    with tempfile.TemporaryDirectory(prefix="mirage-video-frames-", dir=_fast_tmp_dir()) as frame_dir, \
            concurrent.futures.ThreadPoolExecutor(max_workers=_SAVE_WORKERS, thread_name_prefix="mirage-save") as saver:
        producer = threading.Thread(
            target=_producer,
            args=(video_path, info, size, "keyframes" if keyframes else "fps", pix_fmt, out_queue, stop, stats,
                  saver, frame_dir if mode == "object" else None, rate, profile),
            daemon=True,
        )

        if mode == "person":
            model = get_person_model()
            reference_cache = None
            guard = _person_inference_lock
        else:
            with Image.open(reference_path) as opened:
                reference_image = ImageOps.exif_transpose(opened).convert("RGB")
            reference_cache = {}
            keyword = str(selected["name"]).strip()
            guard = contextlib.nullcontext()

        def infer_person(frame, _future):
            people, observations = _person_observations(model, frame, size)
            candidates = _rank_people(reference_feature, observations, people, match_method)
            top = candidates[0] if candidates else None
            target = next((item for item in people if str(item["id"]) == str(top["target_id"])), None) if top else None
            return target, top, True

        def infer_object(frame, future):
            path = future.result()
            try:
                frame_image = Image.fromarray(frame)
                with _object_inference_lock:
                    result = get_model().describe(path, keywords=[keyword], **_describe_options())
            finally:
                with contextlib.suppress(OSError):
                    os.unlink(path)
            objects = [item for item in _normalized_objects(result) if _valid_bbox(item)]
            candidates = _make_matches([selected], objects, reference_image, frame_image, reference_cache)[0]["candidates"]
            top = candidates[0] if candidates else None
            target = None
            if top:
                target = next(
                    (item for i, item in enumerate(objects)
                     if str(item.get("id") if item.get("id") is not None else i + 1) == str(top["target_id"])),
                    None,
                )
            return target, top, False

        infer = infer_person if mode == "person" else infer_object
        try:
            with guard:
                if mode == "person":
                    reference_bgr, reference_size = _person_image(reference_path)
                    reference_records, reference_observations = _person_observations(model, reference_bgr, reference_size)
                    reference_feature = _selected_person_feature(reference_records, reference_observations, selected, match_method)
                producer.start()
                for item in _drain(out_queue, producer):
                    if item[0] == "skip":
                        _, index, ts = item
                        base = state["last"] or {"candidate": None, "bbox": None, "score": None,
                                                 "name_score": None, "appearance_score": None}
                        record = dict(base, sample_index=index, timestamp=round(ts, 3), reused=True)
                        stats["skipped"] += 1
                    else:
                        _, index, ts, frame, future = item
                        started = time.perf_counter()
                        target, top, is_person = infer(frame, future)
                        stats["inference_s"] += time.perf_counter() - started
                        record = {
                            "sample_index": index,
                            "timestamp": round(ts, 3),
                            "candidate": target,
                            "bbox": target.get("bbox") if target else None,
                            "score": top["score"] if top else None,
                            "name_score": None if is_person else (top["name_score"] if top else None),
                            "appearance_score": (top["score"] if is_person else top["appearance_score"]) if top else None,
                        }
                        if is_person:
                            record["match_method"] = match_method
                        state["last"] = record
                        stats["analyzed"] += 1
                    frames.append(record)
                    if shots is not None:
                        shots.observe(record)
        finally:
            stop.set()
            _empty(out_queue)
            if producer.is_alive():
                producer.join(timeout=60)
            saver.shutdown(wait=True, cancel_futures=True)

    if not frames:
        raise ValueError("No video frames could be read. Try a valid MP4, MOV, or WebM file.")
    return frames


@app.post("/api/locate-video")
def locate_object_in_video():
    reference = request.files.get("reference")
    video = request.files.get("target") or request.files.get("video")
    mode = request.form.get("mode", "object")
    match_method = request.form.get("match_method", "body")
    if not _valid_upload(reference):
        return jsonify(error="Choose a valid reference image."), 400
    if mode not in {"object", "person"}:
        return jsonify(error="Choose Object or Person search mode."), 400
    if mode == "person" and match_method not in _PERSON_MATCH_METHODS:
        return jsonify(error="Choose Body appearance or Face identity matching."), 400

    # Very long videos: pass `video_path` (a file on this machine) instead of uploading it.
    local_path = request.form.get("video_path", "").strip().strip('"')
    try:
        if local_path:
            if not os.path.isfile(local_path) or Path(local_path).suffix.lower() not in _VIDEO_EXTENSIONS:
                return jsonify(error="video_path must be an existing MP4, MOV, or WebM file."), 400
            video_name = Path(local_path).name
        else:
            if video is None or not video.filename:
                return jsonify(error="No target video file was received. Re-select an MP4, MOV, or WebM video and try again."), 400
            if Path(video.filename).suffix.lower() not in _VIDEO_EXTENSIONS:
                return jsonify(error="Choose an MP4, MOV, or WebM video."), 415
            if _upload_size(video) > _VIDEO_MAX_BYTES:
                return jsonify(error=f"Videos may be up to {_VIDEO_MAX_BYTES // _MB} MB. Choose a smaller clip."), 413
            video_name = video.filename
        if _upload_size(reference) > 20 * 1024 * 1024:
            return jsonify(error="The reference image may be up to 20 MB."), 413
        selected = _parse_selected_object(request.form.get("object", "{}"))
    except (ValueError, TypeError) as exc:
        return jsonify(error=f"Invalid reference object: {exc}"), 400
    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        return jsonify(error="Video search needs ffmpeg and ffprobe on PATH. Install FFmpeg, then restart this app."), 503

    want_shots = request.form.get("screenshots", "1" if _env_flag("MIRAGE_SCREENSHOTS", False) else "0").lower() in {"1", "true", "yes"}
    compact = request.form.get("compact", "0").lower() in {"1", "true", "yes"}
    video_path = None
    reference_path = None
    shots = None
    sampler = _ResourceSampler()
    try:
        reference_path = _save_upload(reference)
        video_path = local_path or _save_upload(video)
        wall_start = time.perf_counter()
        info = _probe_video(video_path)
        duration = info["duration"]
        stats = {"decode_s": 0.0, "dedup_s": 0.0, "inference_s": 0.0, "analyzed": 0, "skipped": 0, "decoder": None,
                 "mode": None, "rate": None, "profile": None}
        if want_shots:
            _prune_screenshots()
            shots = _ScreenshotWorker(video_path)
        sampler.start()
        frames = _search_video(video_path, reference_path, mode, match_method, selected, info, stats, shots)
        appearances = _video_appearances(frames)
        shot_started = time.perf_counter()
        if shots is not None:
            urls = shots.finish()
            for appearance in appearances:
                appearance["screenshot"] = urls.get(appearance["id"])
        shot_wait = time.perf_counter() - shot_started
        wall = time.perf_counter() - wall_start
        resources = sampler.stop()

        sampled = stats["analyzed"] + stats["skipped"]
        performance = {
            "wall_seconds": round(wall, 3),
            "video_seconds": round(duration, 3),
            "realtime_factor": round(duration / wall, 1) if wall > 0 else None,
            "decoder": stats["decoder"],
            "decode_mode": stats["mode"],
            "quality_profile": stats["profile"],
            "decode_wait_seconds": round(stats["decode_s"], 3),
            "motion_gate_seconds": round(stats["dedup_s"], 3),
            "inference_seconds": round(stats["inference_s"], 3),
            "screenshot_wait_seconds": round(shot_wait, 3),
            "frames_sampled": sampled,
            "frames_analyzed": stats["analyzed"],
            "frames_skipped": stats["skipped"],
            "inference_fps": round(stats["analyzed"] / stats["inference_s"], 2) if stats["inference_s"] > 0 else None,
            "resources": resources,
        }
        app.logger.info("Video search performance: %s", json.dumps(performance))
        effective_rate = stats["rate"] if stats["mode"] != "keyframes" else round(sampled / duration, 3)
        payload = {
            "mode": mode,
            "match_method": match_method if mode == "person" else None,
            "match_threshold": _PERSON_THRESHOLDS[match_method] if mode == "person" else None,
            "reference": {"objects": [selected]},
            "video": {"filename": Path(video_name).name, "duration": duration},
            "keywords": ["person"] if mode == "person" else [str(selected["name"]).strip()],
            "sample_rate": effective_rate,
            "appearances": appearances,
            "performance": performance,
        }
        if not compact:
            payload["frames"] = frames
        return jsonify(payload)
    except ValueError as exc:
        app.logger.info("Mirage video search rejected input: %s", exc)
        if shots is not None:
            shots.abort()
        return jsonify(error=str(exc)), 400
    except Exception:
        app.logger.exception("Video search failed")
        if shots is not None:
            shots.abort()
        return jsonify(error="We couldn’t search this video. Check your files and try again."), 500
    finally:
        sampler.stop()
        for path in (reference_path, None if local_path else video_path):
            if path and os.path.exists(path):
                os.unlink(path)


@app.errorhandler(413)
def images_too_large(_error):
    return jsonify(error=f"The upload is too large. Images may be up to 20 MB and videos up to {_VIDEO_MAX_BYTES // _MB} MB."), 413


if __name__ == "__main__":
    port = int(_env_float("MIRAGE_PORT", 7860))
    print(f"Mirage Object Match: http://127.0.0.1:{port}")
    if _env_flag("MIRAGE_PRELOAD", True):
        threading.Thread(target=_warm_models, name="mirage-preload", daemon=True).start()
    app.run(host="127.0.0.1", port=port, debug=False, threaded=True)
