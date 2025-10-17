# -*- coding:utf-8 -*-
# 该脚本用来加载权重，并执行模型前向推理，实现花分类
import os
import json

import torch
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
import sys
sys.path.append(R"../0_SeResNet")
from senet.se_resnet import se_resnet34


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    data_transform = transforms.Compose(
        [
         transforms.Resize(256),
         transforms.CenterCrop(224),
         transforms.ToTensor(),
         transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225])])
    # 文件及参数配置
    img_path = R"../2_compile/qtset/flower/1.jpg"
    weights_path = "../weights/SeResNet34.pth"
    class_indict = {
                "0": "daisy",
                "1": "dandelion",
                "2": "roses",
                "3": "sunflowers",
                "4": "tulips"
            }
    # load image
    assert os.path.exists(img_path), "file: '{}' dose not exist.".format(img_path)
    img = Image.open(img_path)
    plt.imshow(img)
    img = data_transform(img)
    # expand batch dimension
    img = torch.unsqueeze(img, dim=0)

    # prepare class_indict
    print('class_indict =',class_indict)
    # create model
    model = se_resnet34(num_classes=5).to(device)

    # load model weights
    assert os.path.exists(weights_path), "file: '{}' dose not exist.".format(weights_path)
    model.load_state_dict(torch.load(weights_path, map_location=device))

    # prediction
    model.eval()

    # predict class
    print('image-size',img.size())
    result = model(img.to(device)).cpu()
    print('result =',result)
    output = torch.squeeze(model(img.to(device))).cpu()
    predict = torch.softmax(output, dim=0)
    predict_cla = torch.argmax(predict).numpy()
    print('*'*10,'predict_class','*'*10)
    print_res = "class: {}   prob: {:.3}".format(class_indict[str(predict_cla)],
                                                 predict[predict_cla].detach().numpy())
    plt.title(print_res)
    for i in range(len(predict)):
        print("class: {:10}   prob: {:.3}".format(class_indict[str(i)],
                                                  predict[i].detach().numpy()))
    plt.show()


if __name__ == '__main__':
    main()
