from ultralytics import YOLO
import os

def train_on_zalo():
    """
    Huấn luyện model trên dataset Zalo.
    - Lần đầu chạy: Bắt đầu fine-tuning từ model đã train trên VisDrone.
    - Nếu bị gián đoạn và chạy lại: Tự động tiếp tục (resume) từ checkpoint gần nhất.
    """
    
    # 1. Xác định các đường dẫn cần thiết
    visdrone_model_path = '../runs/detect/train_visdrone/weights/best.pt'
    zalo_run_dir = '../runs/detect/train_zalo'
    checkpoint_path = os.path.join(zalo_run_dir, 'weights', 'last.pt')
    
    # 2. Logic lựa chọn model để tải
    # Ultralytics xử lý việc này rất thông minh với tham số `resume`.
    # Tuy nhiên, để code rõ ràng và tránh lỗi, chúng ta sẽ không dùng `resume=True`.
    # Thay vào đó, chúng ta sẽ tự quyết định nên tải model nào.
    
    model_to_load = ''
    if os.path.exists(checkpoint_path):
        # Nếu có checkpoint của lần train Zalo, ưu tiên resume từ đó.
        print(f"--- Found Zalo training checkpoint at {checkpoint_path}. Resuming training. ---")
        model_to_load = checkpoint_path
    elif os.path.exists(visdrone_model_path):
        # Nếu không, bắt đầu một phiên fine-tuning mới từ model VisDrone.
        print(f"--- No Zalo checkpoint found. Starting new fine-tuning from VisDrone model: {visdrone_model_path}. ---")
        model_to_load = visdrone_model_path
    else:
        # Trường hợp xấu nhất: không có model nào, bắt đầu từ model COCO gốc.
        print("--- WARNING: No VisDrone model or Zalo checkpoint found. Starting from scratch with yolov8m.pt. ---")
        model_to_load = 'yolov8m.pt'

    model = YOLO(model_to_load)

    # 3. Huấn luyện model
    # Khi tải một model đã có tiến trình (như last.pt), ultralytics sẽ tự động resume.
    # Khi tải một model đã hoàn thành (như best.pt từ VisDrone), nó sẽ tự động bắt đầu training mới.
    # Vì vậy, chúng ta không cần dùng 'resume=True' nữa.
    model.train(
        # --- Cấu hình cơ bản ---
        data='../datasets_yolo/zalo/zalo.yaml',
        epochs=100,
        imgsz=640,
        batch=8,
        project='../runs/detect',
        name='train_zalo',
        exist_ok=True, # Rất quan trọng, cho phép ghi vào thư mục 'train_zalo' đã có
        
        # --- Data Augmentation ---
        degrees=15, translate=0.1, scale=0.2, shear=5,
        perspective=0.001, flipud=0.5, fliplr=0.5,
        mosaic=1.0, mixup=0.15, copy_paste=0.1,
        
        # --- Các tham số quan trọng khác ---
        patience=30,
        optimizer='AdamW',
        lr0=1e-3,
        lrf=1e-2,
        cache=False,
        device=0
    )
    print("--- Zalo AI Training Finished ---")

if __name__ == '__main__':
    train_on_zalo()