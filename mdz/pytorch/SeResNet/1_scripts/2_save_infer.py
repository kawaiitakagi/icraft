# -*- coding:utf-8 -*-
# 该脚本用导出的模型进行前向推理，实现花分类
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
    pt_path = '../2_compile/fmodel/SeResNet34_224x224.pt'
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
    # load model
    assert os.path.exists(pt_path), "file: '{}' dose not exist.".format(pt_path)
    model = torch.load(pt_path,map_location=device)
    print('Model Load Done')
    # prediction
    model.eval()

    # predict class
    print('image-size',img.size())
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