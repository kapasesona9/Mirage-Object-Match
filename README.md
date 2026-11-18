# Mirage Object Match

A local staged matching page for objects and people. Upload a reference image, choose Object or Person, select the detected object or person, then search a target image or video. Person mode offers full-body appearance matching and face identity matching. Image candidates are ranked and shown beside the reference box. Video search samples frames, groups consecutive matches into appearances, and lets you jump to the best-scoring frame in each range.

## Start

```sh
python3 -m venv .venv
./.venv/bin/pip install -r requirements.txt
./.venv/bin/python app.py
```

Open <http://127.0.0.1:7860>. Select or drop a reference image (JPEG, PNG, WebP, GIF, or BMP, up to 20 MB), choose **Object** or **Person**, detect the reference regions, and select one. Person mode lets you search using either body appearance or face identity. Choose **Image** to search a still image, or **Video** to search an MP4, MOV, or WebM clip up to 200 MB. Video analysis uses local `ffprobe` and `ffmpeg` from `PATH`, samples 3 frames per second, and resizes frames to at most 1280 pixels on the longest side.

Object analysis uses Mirage. Person analysis uses InsightFace's Cheetah S model for person detection, body ReID, and face recognition. Python dependencies install with `requirements.txt`; the Cheetah S model weights download on first person analysis and are reused locally. The pretrained Cheetah model weights are licensed for non-commercial research use; this demo is intended for personal, non-commercial use unless you supply appropriately licensed weights. Person features are held in memory for the active search and are not saved as profiles. Cactus and ONNX Runtime telemetry are disabled by default.

Object scores combine Mirage's object-name similarity with a lightweight crop appearance comparison. Person scores use either face-feature similarity or body ReID similarity and must meet the selected model threshold. All displayed scores are estimates, not calibrated probabilities or guaranteed identity matches; inspect the highlighted boxes. Inference and comparison run on the local machine. Keep `app.py` bound to localhost unless you deliberately want to expose the service to other devices.
