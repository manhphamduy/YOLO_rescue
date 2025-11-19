from ultralytics import YOLO
import os
from tqdm import tqdm
import json

def run_inference():
    """
    Chạy model YOLOv8 đã train trên toàn bộ tập public test và tạo file submission.json.
    Với mỗi video, script sẽ tự động xác định class cần tìm và chỉ giữ lại các phát hiện của class đó.
    """
    # --- Cấu hình ---
    # Đường dẫn đến model cuối cùng, mạnh nhất
    FINAL_MODEL_PATH = '../runs/detect/train_zalo/weights/best.pt'
    
    # Đường dẫn đến thư mục 'samples' trong public_test
    TEST_DIR = '../public_test/public_test/samples/' # <-- Đảm bảo đường dẫn này đúng
    
    # Tên file JSON output để nộp bài
    OUTPUT_JSON_PATH = '../submission.json'
    
    # Các siêu tham số cho inference
    CONF_THRESHOLD = 0.25 # Ngưỡng tin cậy. Có thể cần tinh chỉnh (giảm để tìm nhiều hơn, tăng để chắc chắn hơn)
    IMG_SIZE = 640        # Kích thước ảnh đầu vào, nên giống với lúc train

    # --- 1. Tải model và kiểm tra ---
    print("--- Starting Inference Process ---")
    if not os.path.exists(FINAL_MODEL_PATH):
        print(f"FATAL ERROR: Model file not found at '{FINAL_MODEL_PATH}'.")
        print("Please make sure you have successfully run '04_train_zalo.py'.")
        return
        
    model = YOLO(FINAL_MODEL_PATH)
    
    # Lấy map từ tên class sang ID (ví dụ: {'BlackBox': 1, 'LifeJacket': 4})
    class_name_to_id = {v: k for k, v in model.names.items()}
    print(f"Model loaded successfully. Found classes: {model.names}")

    # --- 2. Lấy danh sách video và xử lý ---
    try:
        video_ids = [name for name in os.listdir(TEST_DIR) if os.path.isdir(os.path.join(TEST_DIR, name))]
    except FileNotFoundError:
        print(f"FATAL ERROR: Test directory not found at '{TEST_DIR}'. Please check the path.")
        return

    print(f"Found {len(video_ids)} videos to process in the test set.")
    all_results = []

    for video_id in tqdm(video_ids, desc="Processing test videos"):
        video_path = os.path.join(TEST_DIR, video_id, 'drone_video.mp4')
        if not os.path.exists(video_path):
            print(f"Warning: 'drone_video.mp4' not found for {video_id}. Skipping.")
            continue
        
        # --- 3. LOGIC LỌC CLASS CHO TỪNG VIDEO ---
        # Xác định class mục tiêu từ video_id (ví dụ: 'BlackBox_0' -> 'BlackBox')
        target_class_name = "_".join(video_id.split('_')[:-1])
        target_class_id = class_name_to_id.get(target_class_name)

        if target_class_id is None:
            print(f"Warning: Could not determine target class for video '{video_id}'. Skipping.")
            continue
        
        print(f"\nProcessing '{video_id}', searching for class '{target_class_name}' (ID: {target_class_id})...")

        # Chạy predict trên toàn bộ video
        preds = model.predict(source=video_path, stream=True, conf=CONF_THRESHOLD, imgsz=IMG_SIZE, device=0)
        
        video_annotations = {"bboxes": []}
        frame_idx = 0
        for result in preds:
            # Lọc các box có class ID trùng với target_class_id
            for box in result.boxes:
                pred_class_id = int(box.cls[0])
                if pred_class_id == target_class_id:
                    # Lấy tọa độ và định dạng lại
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    video_annotations["bboxes"].append({
                        "frame": frame_idx, "x1": x1, "y1": y1, "x2": x2, "y2": y2
                    })
            frame_idx += 1
        
        all_results.append({
            "video_id": video_id, "annotations": [video_annotations]
        })

    # --- 4. Lưu file submission ---
    with open(OUTPUT_JSON_PATH, 'w') as f:
        json.dump(all_results, f, indent=4)

    print(f"\n--- Inference complete! ---")
    print(f"Submission file created at: {OUTPUT_JSON_PATH}")

if __name__ == '__main__':
    run_inference()