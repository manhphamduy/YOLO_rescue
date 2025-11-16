import os
import json
import cv2
from tqdm import tqdm
import yaml
from collections import defaultdict
import random
import shutil

# --- Cấu hình ---
ZALO_DATA_ROOT = '../data/zalo_dataset' # Thư mục chứa 'annotations' và 'samples'
OUTPUT_DIR = '../datasets_yolo/zalo/'
ANNOTATIONS_FILE = os.path.join(ZALO_DATA_ROOT, 'annotations', 'annotations.json')
SAMPLES_DIR = os.path.join(ZALO_DATA_ROOT, 'samples')
VAL_SPLIT_RATIO = 0.15

def prepare_zalo_dataset():
    print("Preparing Zalo AI dataset...")
    if os.path.exists(OUTPUT_DIR):
        shutil.rmtree(OUTPUT_DIR)

    temp_img_dir = os.path.join(OUTPUT_DIR, 'all_images')
    temp_lbl_dir = os.path.join(OUTPUT_DIR, 'all_labels')
    train_img_dir = os.path.join(OUTPUT_DIR, 'train', 'images')
    train_lbl_dir = os.path.join(OUTPUT_DIR, 'train', 'labels')
    val_img_dir = os.path.join(OUTPUT_DIR, 'val', 'images')
    val_lbl_dir = os.path.join(OUTPUT_DIR, 'val', 'labels')
    for path in [train_img_dir, train_lbl_dir, val_img_dir, val_lbl_dir, temp_img_dir, temp_lbl_dir]:
        os.makedirs(path, exist_ok=True)

    with open(ANNOTATIONS_FILE, 'r') as f:
        all_videos_data = json.load(f)

    all_class_names = sorted(list(set(["_".join(vid['video_id'].split('_')[:-1]) for vid in all_videos_data if "_".join(vid['video_id'].split('_')[:-1])])))
    class_name_to_id = {name: i for i, name in enumerate(all_class_names)}
    print(f"Found classes: {class_name_to_id}")

    for video_data in tqdm(all_videos_data, desc="Extracting frames & labels"):
        video_id = video_data['video_id']
        video_path_in_samples = os.path.join(SAMPLES_DIR, video_id, 'drone_video.mp4')
        if not os.path.exists(video_path_in_samples): continue
        
        frames_data = defaultdict(list)
        class_name = "_".join(video_id.split('_')[:-1])
        class_id = class_name_to_id[class_name]
        for tracked_obj in video_data['annotations']:
            for bbox in tracked_obj['bboxes']:
                frames_data[bbox['frame']].append(bbox)

        cap = cv2.VideoCapture(video_path_in_samples)
        frame_idx = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            if frame_idx in frames_data:
                img_h, img_w, _ = frame.shape
                frame_filename = f"{video_id}_frame_{frame_idx}.jpg"
                label_filename = f"{video_id}_frame_{frame_idx}.txt"
                cv2.imwrite(os.path.join(temp_img_dir, frame_filename), frame)
                yolo_labels = []
                for bbox in frames_data[frame_idx]:
                    x1, y1, x2, y2 = bbox['x1'], bbox['y1'], bbox['x2'], bbox['y2']
                    w, h = x2 - x1, y2 - y1
                    x_center, y_center = (x1 + w / 2) / img_w, (y1 + h / 2) / img_h
                    norm_w, norm_h = w / img_w, h / img_h
                    yolo_labels.append(f"{class_id} {x_center:.6f} {y_center:.6f} {norm_w:.6f} {norm_h:.6f}")
                with open(os.path.join(temp_lbl_dir, label_filename), 'w') as f:
                    f.write('\n'.join(yolo_labels))
            frame_idx += 1
        cap.release()

    print("Splitting data into train and validation sets...")
    all_images = sorted(os.listdir(temp_img_dir))
    random.shuffle(all_images)
    split_idx = int(len(all_images) * (1 - VAL_SPLIT_RATIO))
    train_files, val_files = all_images[:split_idx], all_images[split_idx:]

    for file_list, (img_dest, lbl_dest) in [(train_files, (train_img_dir, train_lbl_dir)), (val_files, (val_img_dir, val_lbl_dir))]:
        for file in tqdm(file_list, desc=f"Moving {len(file_list)} files"):
            shutil.move(os.path.join(temp_img_dir, file), os.path.join(img_dest, file))
            shutil.move(os.path.join(temp_lbl_dir, file.replace('.jpg', '.txt')), os.path.join(lbl_dest, file.replace('.jpg', '.txt')))
    
    shutil.rmtree(temp_img_dir)
    shutil.rmtree(temp_lbl_dir)

    yaml_data = {
        'path': os.path.abspath(OUTPUT_DIR), 'train': 'train/images', 'val': 'val/images',
        'names': {i: name for name, i in class_name_to_id.items()}
    }
    with open(os.path.join(OUTPUT_DIR, 'zalo.yaml'), 'w') as f:
        yaml.dump(yaml_data, f, default_flow_style=False, sort_keys=False)

    print("Zalo data preparation finished!")

if __name__ == '__main__':
    prepare_zalo_dataset()