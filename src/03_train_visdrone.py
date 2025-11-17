from ultralytics import YOLO

def train_on_visdrone():
    # 1. Tải model YOLOv8m đã pre-train trên COCO
    model = YOLO('yolov8m.pt')

    # 2. Huấn luyện model
    print("--- Starting Stage 1 Training on VisDrone Dataset ---")
    model.train(
        data='../datasets_yolo/visdrone/visdrone.yaml',
        epochs=50,
        imgsz=640,
        batch=16,
        project='../runs/detect',
        name='train_visdrone',
        exist_ok=True,
        # Data Augmentation (YOLOv8 đã bật mặc định với các giá trị tốt)
        # Các tham số này chỉ để minh họa, bạn có thể tinh chỉnh
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