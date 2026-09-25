import cv2
import mediapipe as mp
import os
import argparse
import json
from deep_sort_realtime.deepsort_tracker import DeepSort

args = argparse.ArgumentParser()

# mode can be "image" or "video" or "webcam"
args.add_argument("--mode", default="webcam")
# FILE PATH for image or video If mode is "image" or "video"
args.add_argument("--filepath", default=None)
args = args.parse_args()


output_dir = "./output"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)


def get_detections(img, face_detection):
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    out = face_detection.process(img_rgb)
    H, W, _ = img.shape
    detections = []

    if out.detections is not None:
        for detection in out.detections:
            bbox = detection.location_data.relative_bounding_box
            x1, y1, w, h = bbox.xmin, bbox.ymin, bbox.width, bbox.height
            x1, y1, w, h = int(x1 * W), int(y1 * H), int(w * W), int(h * H)
            confidence = detection.score[0]
            detections.append(([x1, y1, w, h], confidence, "face"))

    return detections


def blur_detections(img, detections):
    H, W, _ = img.shape
    for (x1, y1, w, h), conf, cls in detections:
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(W, x1 + w), min(H, y1 + h)
        img[y1:y2, x1:x2, :] = cv2.blur(img[y1:y2, x1:x2, :], (30, 30))
    return img


# detect face
mp_face_detection = mp.solutions.face_detection

MODEL_SELECTION = 0
MIN_DETECTION_CONFIDENCE = 0.7
MAX_AGE = 30

# model_selection=0: short-range model, best for faces within 2 meters from the camera
# model_selection=1: full-range model, best for faces within 10 meters from the camera
with mp_face_detection.FaceDetection(model_selection=MODEL_SELECTION,
                                      min_detection_confidence=MIN_DETECTION_CONFIDENCE) as face_detection:

    if args.mode in ["image"]:
        img = cv2.imread(args.filepath)

        detections = get_detections(img, face_detection)
        img = blur_detections(img, detections)

        cv2.imwrite(os.path.join(output_dir, 'output.png'), img)

    elif args.mode in ["video"]:
        cap = cv2.VideoCapture(args.filepath)
        ret, frame = cap.read()
        output_video = cv2.VideoWriter(
            os.path.join(output_dir, 'output.mp4'),
            cv2.VideoWriter_fourcc(*'mp4v'), 25,
            (frame.shape[1], int(frame.shape[0]))
        )

        tracker = DeepSort(max_age=MAX_AGE)

        total_frames = 0
        total_detections = 0
        frames_with_zero_detections = 0
        unique_track_ids = set()

        while ret:
            detections = get_detections(frame, face_detection)
            total_frames += 1
            total_detections += len(detections)
            if len(detections) == 0:
                frames_with_zero_detections += 1

            tracks = tracker.update_tracks(detections, frame=frame)

            for track in tracks:
                if not track.is_confirmed() or track.time_since_update > 0:
                    continue
                unique_track_ids.add(track.track_id)

                H, W, _ = frame.shape
                l, t, r, b = map(int, track.to_ltrb())
                l, t = max(0, l), max(0, t)
                r, b = min(W, r), min(H, b)

                if r <= l or b <= t:
                    continue

                frame[t:b, l:r] = cv2.blur(frame[t:b, l:r], (30, 30))
                cv2.putText(frame, f"ID {track.track_id}", (l, max(0, t - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            output_video.write(frame)
            ret, frame = cap.read()

        cap.release()
        output_video.release()

        # save results to JSON 

        # logging happens in a separate script that never imports mediapipe)
        results = {
            "model_selection": MODEL_SELECTION,
            "min_detection_confidence": MIN_DETECTION_CONFIDENCE,
            "max_age": MAX_AGE,
            "source_video": args.filepath,
            "total_frames": total_frames,
            "total_detections": total_detections,
            "avg_detections_per_frame": total_detections / total_frames if total_frames else 0,
            "frames_with_zero_detections": frames_with_zero_detections,
            "unique_track_ids": len(unique_track_ids),
        }

        result_filename = f"run_results_{os.path.basename(args.filepath)}_model{MODEL_SELECTION}_conf{MIN_DETECTION_CONFIDENCE}.json"
        with open(result_filename, "w") as f:
            json.dump(results, f, indent=2)

        print(f"Saved run results to {result_filename}")

    elif args.mode in ["webcam"]:
        cap = cv2.VideoCapture(0)
        ret, frame = cap.read()

        tracker = DeepSort(max_age=MAX_AGE)

        while ret:
            detections = get_detections(frame, face_detection)
            tracks = tracker.update_tracks(detections, frame=frame)

            for track in tracks:
                if not track.is_confirmed() or track.time_since_update > 0:
                    continue
                H, W, _ = frame.shape
                l, t, r, b = map(int, track.to_ltrb())
                l, t = max(0, l), max(0, t)
                r, b = min(W, r), min(H, b)

                if r <= l or b <= t:
                    continue

                frame[t:b, l:r] = cv2.blur(frame[t:b, l:r], (30, 30))
                cv2.putText(frame, f"ID {track.track_id}", (l, max(0, t - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            cv2.imshow("frame", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            ret, frame = cap.read()

        cap.release()
