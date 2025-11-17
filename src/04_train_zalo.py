from ultralytics import YOLO

def train_on_zalo():
    # 1. Tải model đã được fine-tune trên VisDrone
    visdrone_model_path = '../runs/detect/train_visdrone/weights/best.pt'
    model = YOLO(visdrone_model_path)

    # 2. Tiếp tục huấn luyện trên dataset Zalo
    print("--- Starting Stage 2 Fine-tuning on Zalo AI Dataset ---")
    model.train(
        # --- Cấu hình cơ bản ---
        data='../datasets_yolo/zalo/zalo.yaml',
        epochs=100, imgsz=640, batch=8,
        project='../runs/detect', name='train_zalo', exist_ok=True,
        
        # --- Checkpointing ---
        resume=True,

        # --- Data Augmentation (mạnh mẽ hơn cho fine-tuning) ---
        degrees=15, translate=0.1, scale=0.2, shear=5,
        perspective=0.001, flipud=0.5, fliplr=0.5,
        mosaic=1.0, mixup=0.15, copy_paste=0.1,
        
        # --- Các tham số quan trọng khác ---
        patience=30, optimizer='AdamW', lr0=1e-3, lrf=1e-2,
        cache=False,
        device=0
    )
    print("--- Zalo AI Training Finished ---")

if __name__ == '__main__':
    train_on_zalo()