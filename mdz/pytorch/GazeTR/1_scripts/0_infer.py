import os, sys
sys.path.append(R"../0_GazeTR")
base_dir = os.getcwd()
sys.path.insert(0, base_dir)
import model
import torch
import cv2
import argparse
from torchvision import transforms


def parser_args():
    parser = argparse.ArgumentParser(description='Pytorch Basic Model Training')
    parser.add_argument('--weight', default="../weights/Iter_80_trans6.pt",type=str)
    parser.add_argument("--img_path", type=str, default="../3_deploy/modelzoo/GazeTR/io/input/1.jpg")
    args = parser.parse_args()

    return args


def main():

    modelpath = args.weight
    logpath = os.path.join("log/infer/")
    if not os.path.exists(logpath):
        os.makedirs(logpath)

    preprocess = transforms.Compose([
    transforms.ToTensor()
    ])

    # ----------------------Load Model------------------------------
    net = model.Model()
    statedict = torch.load(
        modelpath,
        map_location="cpu"
    )
    net; net.load_state_dict(statedict); net.eval()

    # ----------------------Load IMG------------------------------
    img = cv2.imread(args.img_path)
    data = {}
    data["face"] = preprocess(img).unsqueeze(0)

    with torch.no_grad():

        gaze = net(data)
        gaze = gaze[0].cpu().detach().numpy()
        print(f'[face] pitch: {gaze[1]:.2f}, yaw: {gaze[0]:.2f}')


if __name__ == "__main__":
    args = parser_args()
    main()

