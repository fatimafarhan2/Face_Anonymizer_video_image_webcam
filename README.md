# Face Anonymizer

A Python-based tool to automatically detect, track, and blur faces in images, videos, and webcam streams using MediaPipe, OpenCV, and DeepSORT.

## Features

- **Multiple Input Modes**: Process images, videos, or real-time webcam streams
- **Fast Face Detection**: Uses MediaPipe's face detection model for efficient and accurate face recognition
- **Multi-Object Tracking**: Maintains consistent identity across video frames using DeepSORT, so the same face keeps the same ID as it moves
- **Flexible Anonymization**: Applies blur filter to detected/tracked faces
- **Experiment Tracking**: Logs detection and tracking metrics (confidence threshold, model variant, detection counts, track counts) across runs using MLflow, for comparing configurations
- **Easy to Use**: Simple command-line interface with sensible defaults

## Requirements

- Python 3.7+
- OpenCV (`cv2`)
- MediaPipe (`mediapipe`)
- deep-sort-realtime
- mlflow

Note: mediapipe requires `protobuf<4.0`, while some mlflow dependencies (databricks-sdk, opentelemetry-proto) prefer newer protobuf versions. If you plan to log results to MLflow, run detection/tracking (`face_anonymizer_test.py`) and MLflow logging (`log_to_mlflow.py`) as separate steps rather than in the same script, so both libraries never need to share a process. See "Experiment Tracking" below.

## Installation

1. Clone or download this repository:
```bash
git clone https://github.com/fatimafarhan2/Face_Anonymizer_video_image_webcam.git
cd face_anonymizer
```

2. Install required dependencies:
```bash
pip install opencv-python mediapipe deep-sort-realtime mlflow
pip install "protobuf==3.20.3"
```

3. Create output directory (optional - will be created automatically):
```bash
mkdir output
```

## Usage

### Image Processing

Process a single image file:
```bash
python face_anonymizer_test.py --mode image --filepath "./data/testImg.png"
```

### Video Processing

Process a video file (includes face tracking):
```bash
python face_anonymizer_test.py --mode video --filepath "./data/testvideo.mp4"
```

Output will be saved as `output.mp4`, and a JSON file with run metrics (`run_results_<video>_model<N>_conf<X>.json`) will be saved in the project root.

### Webcam Stream

Process real-time webcam feed (includes face tracking):
```bash
python face_anonymizer_test.py --mode webcam
```

Press `q` to quit the webcam stream.

### Experiment Tracking

After running one or more video-mode sessions with different `MODEL_SELECTION`/`MIN_DETECTION_CONFIDENCE` values (edit these constants near the top of `face_anonymizer_test.py`), log all results to MLflow:
```bash
python log_to_mlflow.py
```

View logged runs via a Python query (this avoids a protobuf conflict between mediapipe and mlflow's UI dependencies on some setups):
```bash
python -c "import mlflow; print(mlflow.search_runs(experiment_names=['face_anonymizer_tracking']))"
```

## Project Structure

face_anonymizer/
├── face_anonymizer.py # Simple single-image processing script
├── face_anonymizer_test.py # Main multi-mode script (image/video/webcam, tracking, JSON export)
├── log_to_mlflow.py # Logs saved JSON run results into MLflow
├── data/ # Input images/videos directory
├── jsonfiles/ # Per-run metric exports (detections, tracking counts, config)
├── output/ # Output directory for processed files
├── mlflow.db # Local MLflow tracking database
└── README.md # This file



## How It Works

1. **Face Detection**: Uses MediaPipe's `FaceDetection` solution with configurable:
   - Model selection (0 = short-range, best under 2m; 1 = full-range, best up to 10m)
   - Minimum detection confidence (default 0.7)

2. **Face Tracking** (video/webcam modes): Detected faces are passed to a DeepSORT tracker, which assigns and maintains a consistent ID per face across frames. Only tracks with a fresh detection in the current frame are drawn, to avoid blurring stale/ghost predictions.

3. **Face Anonymization**: Applies a 30x30 blur kernel to detected/tracked face regions

4. **Output**:
   - Images saved as `output.png`
   - Videos saved as `output.mp4`
   - Run metrics saved as JSON, for later MLflow logging

## Configuration

You can modify the following near the top of `face_anonymizer_test.py`:

- `MODEL_SELECTION`: 0 (short-range) or 1 (full-range, up to 10m)
- `MIN_DETECTION_CONFIDENCE`: Detection confidence threshold (0.0-1.0)
- `MAX_AGE`: How many frames a track survives without a fresh detection before being dropped
- Blur kernel size: currently `(30, 30)` in the blur calls, adjust for stronger/weaker blur

## Known Limitations

- Tracking uses motion/IOU-based matching (no appearance embedder), since enabling the embedder introduced a torch/torchvision version conflict in this environment. As a result, fast-moving faces or long occlusions can occasionally cause the same person to be reassigned a new ID.
- MediaPipe occasionally produces brief false-positive detections on non-face objects (e.g. a single frame), which can momentarily create a spurious track.
- MLflow's web UI has a known static-asset serving issue on some Windows setups; results are verified via `mlflow.search_runs()` instead.

## Example

```bash
# Process an image
python face_anonymizer_test.py --mode image --filepath "./data/photo.jpg"

# Process a video (with tracking)
python face_anonymizer_test.py --mode video --filepath "./data/video.mp4"

# Use webcam (default, with tracking)
python face_anonymizer_test.py

# Log all saved run results to MLflow
python log_to_mlflow.py
```

## Notes

- Ensure input files are in a readable format (PNG, JPG for images; MP4, AVI for videos)
- Output and jsonfiles directories are created automatically if they don't exist
- Webcam mode requires a connected camera device
- Processing speed depends on image resolution and number of detected faces

## Contributing

Contributions are welcome! Feel free to submit pull requests or report issues.