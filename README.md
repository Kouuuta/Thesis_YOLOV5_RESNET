## Setup

1. https://zenodo.org/records/3587843 download the dataset ZIP FILE, after nyan kunin lang yung folder na "data" then ignore other file or folder
   
   pip install -r requirements.txt

   pip install git+https://github.com/philferriere/cocoapi.git#subdirectory=PythonAPI

   git clone https://github.com/ultralytics/yolov5 --- KAILANGAN NIYO TO I CLONE OKAY??

   pip install -r yolov5/requirements.txt

4. Pang start training pero adjustable yan

   python train.py --img 640 --batch 16 --epochs 50 --data ../ML_PROJECT/data.yaml --weights yolov5s.pt

5. Pag tapos na yung training, testing ng model to

   python detect.py --weights runs/train/exp/weights/best.pt --source data/val/images

6. Run detection on your validation images and save outputs in

   runs/detect/exp/

## YUNG FOLDER NA "MODELS" EMPTY LANG YAN FOR ANO YAN OWN EXPERIMENTS NA WHICH IS PAPASOK NA UNG RESNET 50

## Run this sa terminal ng VSC para macheck if ginagamit ung GPU or hinde 
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'No GPU')"

