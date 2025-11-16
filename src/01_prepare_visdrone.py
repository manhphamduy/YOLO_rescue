import os
from PIL import Image
from tqdm import tqdm
import yaml
import shutil

# --- Cấu hình ---
VISDRONE_ROOT = '../data/visdrone_dataset' # Đường dẫn đến thư mục chứa VisDrone gốc
OUTPUT_DIR = '../datasets_yolo/visdrone/'
# Các class được chọn lọc từ VisDrone phù hợp với bối cảnh UAV
CLASSES = ['pedestrian', 'person', 'car', 'van', 'bus', 'truck', 'motor', 'bicycle']
class_map = {str(i+1): i for i in range(len(CLASSES))} 

# --- Logic ---
def convert_visdrone_to_yolo(split):
    print(f"Processing '{split}' split for VisDrone...")
    img_dir = os.path.join(VISDRONE_ROOT, f'VisDrone2019-DET-{split}', 'images')
    ann_dir = os.path.join(VISDRONE_ROOT, f'VisDrone2019-DET-{split}', 'annotations')
    
    out_img_dir = os.path.join(OUTPUT_DIR, split, 'images')
    out_lbl_dir = os.path.join(OUTPUT_DIR, split, 'labels')
    os.makedirs(out_img_dir, exist_ok=True)
    os.makedirs(out_lbl_dir, exist_ok=True)

    for filename in tqdm(os.listdir(ann_dir), desc=f"Converting {split} set"):
        if not filename.endswith('.txt'): continue
        
        img_path = os.path.join(img_dir, filename.replace('.txt', '.jpg'))
        if not os.path.exists(img_path): continue
        
        shutil.copy(img_path, out_img_dir)
        
        img = Image.open(img_path)
        img_w, img_h = img.size
        
        yolo_labels = []
        with open(os.path.join(ann_dir, filename), 'r') as f:
            for line in f.readlines():
                parts = line.strip().split(',')
                if int(parts[4]) == 0 or parts[5] not in class_map: continue
                
                class_id = class_map[parts[5]]
                x1, y1, w, h = map(int, parts[0:4])
                
                x_center = (x1 + w / 2) / img_w
                y_center = (y1 + h / 2) / img_h
                norm_w = w / img_w
                norm_h = h / img_h
                
                yolo_labels.append(f"{class_id} {x_center:.6f} {y_center:.6f} {norm_w:.6f} {norm_h:.6f}")
        
        with open(os.path.join(out_lbl_dir, filename), 'w') as f:
            f.write('\n'.join(yolo_labels))

if __name__ == '__main__':
    if os.path.exists(OUTPUT_DIR):
        shutil.rmtree(OUTPUT_DIR)
    convert_visdrone_to_yolo('train')
    convert_visdrone_to_yolo('val')

    yaml_data = {
        'path': os.path.abspath(OUTPUT_DIR), 'train': 'train/images', 'val': 'val/images',
        'names': {i: name for i, name in enumerate(CLASSES)}
    }
    with open(os.path.join(OUTPUT_DIR, 'visdrone.yaml'), 'w') as f:
        yaml.dump(yaml_data, f, default_flow_style=False, sort_keys=False)

    print("VisDrone data preparation finished!")