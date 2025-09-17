import json
import os
import shutil
import random
from pathlib import Path
from tqdm import tqdm

# Paths
coco_json = "data/taco/annotations.json"
images_root = "data/taco"
output_dir = "data"

# Train/Val split ratio
train_split = 0.8

# Load COCO JSON
with open(coco_json) as f:
    coco = json.load(f)

images = {img["id"]: img for img in coco["images"]}
annotations = coco["annotations"]
categories = {cat["id"]: cat["name"] for cat in coco["categories"]}

# Output folders
for split in ["train", "val"]:
    os.makedirs(f"{output_dir}/{split}/images", exist_ok=True)
    os.makedirs(f"{output_dir}/{split}/labels", exist_ok=True)

# Assign images to train/val
image_ids = list(images.keys())
random.shuffle(image_ids)
split_index = int(len(image_ids) * train_split)
train_ids = set(image_ids[:split_index])
val_ids = set(image_ids[split_index:])

def coco_to_yolo_bbox(bbox, img_w, img_h):
    x, y, w, h = bbox
    x_center = (x + w / 2) / img_w
    y_center = (y + h / 2) / img_h
    w /= img_w
    h /= img_h
    return [x_center, y_center, w, h]

# Group annotations by image
image_to_anns = {}
for ann in annotations:
    image_to_anns.setdefault(ann["image_id"], []).append(ann)

# Convert
for img_id, img_info in tqdm(images.items(), desc="Converting"):
    file_name = img_info["file_name"]
    img_w, img_h = img_info["width"], img_info["height"]
    
    # Choose split
    split = "train" if img_id in train_ids else "val"
    img_src = os.path.join(images_root, file_name)
    img_dst = os.path.join(output_dir, split, "images", os.path.basename(file_name))
    shutil.copy(img_src, img_dst)

    # Label file
    label_path = os.path.join(output_dir, split, "labels", os.path.splitext(os.path.basename(file_name))[0] + ".txt")
    with open(label_path, "w") as f:
        for ann in image_to_anns.get(img_id, []):
            class_id = ann["category_id"] - 1  # YOLO expects 0-indexed class ids
            bbox_yolo = coco_to_yolo_bbox(ann["bbox"], img_w, img_h)
            f.write(f"{class_id} {' '.join(map(str, bbox_yolo))}\n")
