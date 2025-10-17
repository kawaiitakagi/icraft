WEIGHTS_PATH = "../weights/yolov8s-cls.pt"
IMAGE_PATH = "../2_compile/qtset/imagenet/ILSVRC2012_val_00000441.JPEG"

import sys
sys.path.append(R"../0_yolov8_cls")
from ultralytics import YOLO
# from PIL import Image

if __name__ == "__main__":
    model = YOLO(WEIGHTS_PATH)
    print("model is loaded!")

    results = model.predict(source = IMAGE_PATH, save = False, save_txt = False)  # predict on an image
