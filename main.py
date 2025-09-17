import os
import sys

# Add yolov5 repo to path (make sure you cloned YOLOv5 into ML_PROJECT/yolov5)
YOLO_DIR = os.path.join(os.path.dirname(__file__), "yolov5")
if YOLO_DIR not in sys.path:
    sys.path.append(YOLO_DIR)

from yolov5 import train, detect


def train_yolov5():
    """
    Train YOLOv5 on the TACO dataset
    """
    train.run(
        data="data.yaml",        # dataset config file
        imgsz=640,               # image size
        batch=16,                # adjust based on your GPU memory
        epochs=50,               # number of training epochs
        weights="yolov5s.pt",    # use pretrained yolov5 small weights
        project="runs/train",    # output directory
        name="taco_yolo_baseline", # experiment name
    )


def detect_yolov5(weights="runs/train/taco_yolo_baseline/weights/best.pt", source="data/val/images"):
    """
    Run inference with trained YOLOv5
    """
    detect.run(
        weights=weights,
        source=source,     # run on validation images (or change to 'data/test/images')
        imgsz=640,
        conf_thres=0.25,   # confidence threshold
    )


if __name__ == "__main__":
    # Step 1: Train the model
    train_yolov5()

    # Step 2: Run inference
    detect_yolov5()
