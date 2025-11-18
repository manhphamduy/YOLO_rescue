from ultralytics import YOLO
import os

def train_on_visdrone():
    """
    Huấn luyện model trên dataset VisDrone.
    - Lần đầu chạy: Bắt đầu fine-tuning từ model yolov8m.pt (đã train trên COCO).
    - Nếu bị gián đoạn và chạy lại: Tự động tiếp tục (resume) từ checkpoint gần nhất.
    """
    
    # 1. Xác định các đường dẫn cần thiết
    visdrone_run_dir = '../runs/detect/train_visdrone'
    checkpoint_path = os.path.join(visdrone_run_dir, 'weights', 'last.pt')
    
    # 2. Logic lựa chọn model để tải
    model_to_load = ''
    if os.path.exists(checkpoint_path):
        # Nếu có checkpoint của lần train VisDrone, ưu tiên resume từ đó.
        print(f"--- Found VisDrone training checkpoint at {checkpoint_path}. Resuming training. ---")
        model_to_load = checkpoint_path
    else:
        # Nếu không, bắt đầu một phiên fine-tuning mới từ model COCO gốc.
        print(f"--- No VisDrone checkpoint found. Starting new training from yolov8m.pt. ---")
        model_to_load = 'yolov8m.pt'

    model = YOLO(model_to_load)

    # 3. Huấn luyện model
    # Khi tải một checkpoint (last.pt), ultralytics sẽ tự động resume.
    # Khi tải một model đã hoàn thành (yolov8m.pt), nó sẽ bắt đầu training mới.
    model.train(
        data='../datasets_yolo/visdrone/visdrone.yaml',
        epochs=50,
        imgsz=640,
        batch=16,
        project='../runs/detect',
        name='train_visdrone',
        exist_ok=True, # Cho phép ghi vào thư mục 'train_visdrone' đã có
        
        # Data Augmentation
        degrees=5.0,
        translate=0.1,
        scale=0.1,
        fliplr=0.5,
        mosaic=1.0,
        
        device=0
    )
    print("--- VisDrone Training Finished ---")

if __name__ == '__main__':
    train_on_visdrone()