#通过 opencv 读取了一张图像，并送入模型中推理得到输出 results，
#results 中保存着不同任务的结果，我们这里是检测任务，因此只需要拿到对应的 boxes 即可
import cv2
import torch 
import argparse
import sys 
sys.path.append(R'../0_yolov10')
from ultralytics import YOLOv10

def hsv2bgr(h, s, v):
    h_i = int(h * 6)
    f = h * 6 - h_i
    p = v * (1 - s)
    q = v * (1 - f * s)
    t = v * (1 - (1 - f) * s)
    
    r, g, b = 0, 0, 0

    if h_i == 0:
        r, g, b = v, t, p
    elif h_i == 1:
        r, g, b = q, v, p
    elif h_i == 2:
        r, g, b = p, v, t
    elif h_i == 3:
        r, g, b = p, q, v
    elif h_i == 4:
        r, g, b = t, p, v
    elif h_i == 5:
        r, g, b = v, p, q

    return int(b * 255), int(g * 255), int(r * 255)

def random_color(id):
    h_plane = (((id << 2) ^ 0x937151) % 100) / 100.0
    s_plane = (((id << 3) ^ 0x315793) % 100) / 100.0
    return hsv2bgr(h_plane, s_plane, 1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='YOLOv10 PyTorch Inference.', add_help=True)
    parser.add_argument('--weights', type=str, default='../weights/yolov10n.pt', help='model path(s) for inference.')
    parser.add_argument('--source', type=str, default='../0_yolov10/ultralytics/assets/zidane.jpg', help='the source path, e.g. image-file.')
    parser.add_argument('--res', type=str, default='predict.jpg', help='the res path.')
    args = parser.parse_args()
    weights = args.weights #权重
    img_path = args.source #推理图片
    res_path = args.res #结果存放路径
    # load model 
    model = YOLOv10(weights)
    # read img
    img = cv2.imread(img_path)
    # get pred res
    results = model(img)[0]
    names   = results.names
    boxes   = results.boxes.data.tolist()
    # visualize results
    for obj in boxes:
        left, top, right, bottom = int(obj[0]), int(obj[1]), int(obj[2]), int(obj[3])
        confidence = obj[4]
        label = int(obj[5])
        color = random_color(label)
        cv2.rectangle(img, (left, top), (right, bottom), color=color ,thickness=2, lineType=cv2.LINE_AA)
        caption = f"{names[label]} {confidence:.2f}"
        w, h = cv2.getTextSize(caption, 0, 1, 2)[0]
        cv2.rectangle(img, (left - 3, top - 33), (left + w + 10, top), color, -1)
        cv2.putText(img, caption, (left, top - 5), 0, 1, (0, 0, 0), 2, 16)
    # save res
    cv2.imwrite(res_path, img)
    print("save at ",res_path)    
