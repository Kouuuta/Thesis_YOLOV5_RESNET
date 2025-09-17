## Setup

1. pip install -r requirements.txt

   pip install git+https://github.com/philferriere/cocoapi.git#subdirectory=PythonAPI

   git clone https://github.com/ultralytics/yolov5 --- KAILANGAN NIYO TO I CLONE OKAY??

   pip install -r yolov5/requirements.txt

2. Pang start training pero adjustable yan

   python train.py --img 640 --batch 16 --epochs 50 --data ../ML_PROJECT/data.yaml --weights yolov5s.pt

3. Pag tapos na yung training, testing ng model to

   python detect.py --weights runs/train/exp/weights/best.pt --source data/val/images

4. Run detection on your validation images and save outputs in

   runs/detect/exp/

## YUNG FOLDER NA "MODELS" EMPTY LANG YAN FOR ANO YAN OWN EXPERIMENTS NA WHICH IS PAPASOK NA UNG RESNET 50
