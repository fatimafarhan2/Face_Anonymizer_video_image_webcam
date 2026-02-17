# Face Anonymizer

A Python-based tool to automatically detect and blur faces in images, videos, and webcam streams using MediaPipe and OpenCV.

## Features

- **Multiple Input Modes**: Process images, videos, or real-time webcam streams
- **Fast Face Detection**: Uses MediaPipe's face detection model for efficient and accurate face recognition
- **Flexible Anonymization**: Applies blur filter to detected faces
- **Easy to Use**: Simple command-line interface with sensible defaults

## Requirements

- Python 3.7+
- OpenCV (`cv2`)
- MediaPipe (`mediapipe`)

## Installation

1. Clone or download this repository:
```bash
git clone <repository-url>
cd face_anonymizer
```

2. Install required dependencies:
```bash
pip install opencv-python mediapipe
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

Process a video file:
```bash
python face_anonymizer_test.py --mode video --filepath "./data/testvideo.mp4"
```

Output will be saved as `output.mp4`

### Webcam Stream

Process real-time webcam feed:
```bash
python face_anonymizer_test.py --mode webcam
```

Press `q` to quit the webcam stream.

## Project Structure

```
face_anonymizer/
├── face_anonymizer.py          # Simple image processing script
├── face_anonymizer_test.py     # Advanced multi-mode script (recommended)
├── face_anonymizer_test.py     # Test/reference script
├── data/                       # Input images/videos directory
├── output/                     # Output directory for processed files
└── README.md                   # This file
```

## How It Works

1. **Face Detection**: Uses MediaPipe's `FaceDetection` solution with:
   - Short-range model (0-2 meters optimal range)
   - Minimum detection confidence: 50%

2. **Face Anonymization**: Applies a 30x30 blur kernel to detected face regions

3. **Output**: 
   - Images saved as `output.png`
   - Videos saved as `output.mp4`

## Configuration

You can modify the following parameters in the code:

- `model_selection`: 0 (short-range, default) or 1 (full-range up to 10m)
- `min_detection_confidence`: Detection confidence threshold (0.0-1.0)
- Blur kernel size: Currently set to `(30, 30)` - adjust for stronger/weaker blur effect

## Example

```bash
# Process an image
python face_anonymizer_test.py --mode image --filepath "./data/photo.jpg"

# Process a video
python face_anonymizer_test.py --mode video --filepath "./data/video.mp4"

# Use webcam (default)
python face_anonymizer_test.py
```

## Notes

- Ensure input files are in a readable format (PNG, JPG for images; MP4, AVI for videos)
- The output directory is created automatically if it doesn't exist
- Webcam mode requires a connected camera device
- Processing speed depends on image resolution and number of detected faces

## License

[Add your license here]

## Contributing

Contributions are welcome! Feel free to submit pull requests or report issues.
