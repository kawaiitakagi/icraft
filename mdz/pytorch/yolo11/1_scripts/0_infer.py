WEIGHTS_PATH = "../weights/yolo11s.pt"
IMAGE_PATH = "../2_compile/qtset/coco/000000000632.jpg"
import sys
sys.path.append(R"../0_yolo11")
import optparse
from ultralytics import YOLO
if __name__ == "__main__":
    parser4pred = optparse.OptionParser()
    parser4pred.add_option('--model', type=str, default=WEIGHTS_PATH, help='path to model file')
    parser4pred.add_option('--save', type=str, default=False, help='save results') # 是否保存结果
    parser4pred.add_option('--show', type=str, default=True, help='show results') # 是否可可视化结果
    parser4pred.add_option('--source', type=str, default=IMAGE_PATH, help='source directory for images or videos')
    import os

    print("Current working directory:", os.getcwd())
    options, args  = parser4pred.parse_args()
    options = eval(str(options))
    model = YOLO(model=options['model'])  
    results = model(**options)  # predict on an image