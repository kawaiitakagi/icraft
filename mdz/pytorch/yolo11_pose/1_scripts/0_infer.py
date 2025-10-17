
import sys
sys.path.append(R"../0_yolov11_pose")
from ultralytics import YOLO
from PIL import Image

# 该脚本用于加载官方权重进行推理并可视化
# Load a model
model = YOLO('../weights/yolo11s-pose.pt')  # load an official model

# Predict with the model
results = model("https://ultralytics.com/images/bus.jpg")  # predict on an image

# Show the results
for r in results:
    im_array = r.plot()  # plot a BGR numpy array of predictions
    im = Image.fromarray(im_array[..., ::-1])  # RGB PIL image
    im.show()  # show image