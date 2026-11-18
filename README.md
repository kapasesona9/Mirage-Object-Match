Mirage Object Match

Find a specific person or object in images, folders, and long videos using a single reference photo.

Pick someone in a reference image, choose a matching method, and Mirage reports every moment they appear in the target video, with timestamps and match scores. It is built to handle everything from short clips to day-long CCTV recordings.

Features
Person search with two matching methods:
Face match using face embeddings (best when a face is clearly visible)
Body appearance using full-body re-identification (better for blurry or distant footage)
Object search driven by the reference object's description and appearance
Image, video, and folder search from a simple web interface
Fast video pipeline
Streaming FFmpeg decode through a pipe (no per-frame files)
Optional CUDA decode and GPU inference when available
Motion gating: near-identical frames reuse the previous result
Keyframe-only decoding for very long recordings
Producer/consumer threading so decode and inference overlap
Automatic quality assessment (mirage_quality.py): measures sharpness, brightness, noise, resolution, and keyframe spacing, then picks the decode mode and sample rate. Contrast and denoise are applied only to dark or noisy footage
Large local files: scan a video straight from disk by path, with no upload
Event screenshots (optional): one best-frame screenshot per appearance, saved asynchronously
A/B benchmark (benchmark.py): compares two builds with warm-up, repeated runs, event recall, short-event recall, and optional ground-truth scoring
Requirements
Python 3.10+
FFmpeg and ffprobe on your PATH
Optional: an NVIDIA GPU with onnxruntime-gpu for much faster person matching
Optional: opencv-python-headless (selective enhancement), psutil and pynvml (resource stats), requests-toolbelt (benchmark uploads)
Installation
bash
git clone <your-repo-url>
cd <your-repo>
python -m venv venv
# Windows: venv\Scripts\activate    Linux/Mac: source venv/bin/activate
pip install -r requirements.txt
Run
bash
python app.py

Open http://127.0.0.1:7860, add a reference photo, click Show people, select a person, then choose Image, Video, or Folder and search.

The server listens on 127.0.0.1 only.

Scanning very long videos

Browser uploads are impractical for multi-gigabyte files. Run the scan from disk instead:

bash
python run_long.py --reference ref.jpg --video "D:\cctv\day1.mp4" --mode person --match-method face --index 0

Recommended server settings for 24-hour footage:

bash
MIRAGE_DECODE_MODE=keyframes
MIRAGE_DEDUP_MEAN=2.5
MIRAGE_DEDUP_BLOCK=12
MIRAGE_DEDUP_MAX_GAP_SEC=8

Keyframe mode samples once per camera keyframe interval, so events shorter than that interval can be missed. Validate on a short clip of your own footage first.

Configuration

All settings are environment variables.

Variable	Default	Purpose
MIRAGE_PORT	7860	Server port
MIRAGE_DECODE_MODE	auto	auto, fps, or keyframes
MIRAGE_SAMPLE_RATE	3	Frames per second analyzed in fps mode
MIRAGE_VIDEO_MAX_MB	200	Maximum upload size
MIRAGE_BODY_THRESHOLD	0.85	Body match cosine threshold
MIRAGE_FACE_THRESHOLD	0.45	Face match cosine threshold
MIRAGE_DEDUP	1	Enable motion gating
MIRAGE_DEDUP_MEAN / MIRAGE_DEDUP_BLOCK	1.5 / 10	Motion gating sensitivity
MIRAGE_DEDUP_MAX_GAP_SEC	2	Longest gap that may be skipped
MIRAGE_HWACCEL	auto	auto, cuda, or off
MIRAGE_ORT_PROVIDERS	auto	Override ONNX Runtime providers
MIRAGE_SCREENSHOTS	0	Save a screenshot per appearance
MIRAGE_PRELOAD	1	Load models at startup
API
Endpoint	Purpose
POST /api/reference	Detect people or objects in a reference image
POST /api/locate	Match the selected person or object in a target image
POST /api/locate-video	Search a video (upload, or pass video_path for a local file)

/api/locate-video returns the appearance intervals plus a performance block with decode, inference, and resource timings and the chosen quality profile.

Benchmarking

Run your original build on port 7861 and the optimized build on port 7860:

bash
python benchmark.py --reference ref.jpg --video clip1.mp4 clip2.mp4 --mode person --index 0 --repeat 3 --csv results.csv

Add --truth truth.json with hand-labelled intervals to measure accuracy. Without it the report shows agreement with the baseline, which is not the same as accuracy.

Project structure
app.py               Flask app, models, video pipeline, API
mirage_quality.py    Video quality assessment and selective enhancement
benchmark.py         A/B benchmark with CSV output
run_long.py          Scan a large local video by path
static/              Web interface
Responsible use

This tool matches people across images and video. Use it only with the consent of the people involved or where you have a lawful basis, and follow the privacy, surveillance, and biometric laws that apply to you. Matching is statistical, so treat results as leads to verify, not as proof of identity.

License

Add your license here.
