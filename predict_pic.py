import warnings

warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('models/test.pt')
    model.predict(source='test_pic',
                  imgsz=640,
                  save=True,
                  conf=0.25
                  )

