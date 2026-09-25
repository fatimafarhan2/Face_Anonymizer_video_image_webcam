import json
import glob
import mlflow

mlflow.set_experiment("face_anonymizer_tracking")

for filepath in glob.glob("run_results_*.json"):
    with open(filepath) as f:
        data = json.load(f)

    with mlflow.start_run():
        for key in ["model_selection", "min_detection_confidence", "max_age", "source_video"]:
            mlflow.log_param(key, data[key])
        for key in ["total_frames", "total_detections", "avg_detections_per_frame",
                    "frames_with_zero_detections", "unique_track_ids"]:
            mlflow.log_metric(key, data[key])

print("logged all runs")