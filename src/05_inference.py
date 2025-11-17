from ultralytics import YOLO
import os
import cv2
from tqdm import tqdm
import json

def run_inference():
    # --- Cấu hình ---
    FINAL_MODEL_PATH = '../runs/detect/train_zalo/weights/best.pt'
    TEST_DIR = '../data/zalo_dataset/samples/' # Sửa lại đường dẫn này nếu tập test nằm ở nơi khác
    OUTPUT_JSON_PATH = '../submission.json'
    CONF_THRESHOLD = 0.25

    # --- Logic ---
    model = YOLO(FINAL_MODEL_PATH)
    all_results = []
    video_ids = [name for name in os.listdir(TEST_DIR) if os.path.isdir(os.path.join(TEST_DIR, name))]

    for video_id in tqdm(video_ids, desc="Processing test videos"):
        video_path = os.path.join(TEST_DIR, video_id, 'drone_video.mp4')
        if not os.path.exists(video_path): continue

        preds = model.predict(source=video_path, stream=True, conf=CONF_THRESHOLD, imgsz=640, device=0)
        
        video_annotations = {"bboxes": []}
        frame_idx = 0
        for result in preds:
            boxes = result.boxes.cpu().numpy()
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                video_annotations["bboxes"].append({
                    "frame": frame_idx, "x1": x1, "y1": y1, "x2": x2, "y2": y2
                })
            frame_idx += 1
        
        all_results.append({
            "video_id": video_id, "annotations": [video_annotations]
        })

    with open(OUTPUT_JSON_PATH, 'w') as f:
        json.dump(all_results, f, indent=4)

    print(f"Submission file created at: {OUTPUT_JSON_PATH}")

if __name__ == '__main__':
    run_inference()