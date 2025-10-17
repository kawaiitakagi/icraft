import os, sys
sys.path.append(R"../0_GazeTR")
base_dir = os.getcwd()
sys.path.insert(0, base_dir)
import model
from model import Model,TransformerEncoderLayer
import torch
import cv2
import argparse
import numpy as np
from easydict import EasyDict as edict
from torchvision import transforms

RED = '\033[31m'       # 设置前景色为红色
RESET = '\033[0m'      # 重置所有属性到默认值
ver = torch.__version__
assert ("2.0.1" in ver) or ("1.9" in ver), f"{RED}Unsupported PyTorch version: {ver}{RESET}"


pos_feature = torch.tensor(np.fromfile(f'./pos_feature_50x32.ftmp', dtype=np.float32).reshape(50,1,32))

def parser_args():
    parser = argparse.ArgumentParser(description='Pytorch Basic Model Training')
    parser.add_argument('--weight', default="../weights/GazeTR-H-ETH.pt",type=str)
    # parser.add_argument('--weight', default="../weights/Iter_80_trans6.pt",type=str)
    parser.add_argument("--img_path", type=str, default="../3_deploy/modelzoo/GazeTR/io/input/1.jpg")
    parser.add_argument('--output-path', '-o', type=str, default="../2_compile/fmodel/")
    parser.add_argument('--trace_model', type=str, default="GazeTR_1x3x224x224.pt")
    args = parser.parse_args()
    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)
    return args


def new_forward(self, x):
    # feature = self.base_model(x_in["face"])#修改点1：取消字典形式输入
    feature = self.base_model(x)
    batch_size = feature.size(0)
    feature = feature.flatten(2)
    feature = feature.permute(2, 0, 1)
    
    # cls = self.cls_token.repeat( (1, batch_size, 1))#修改点2：去除repeat算子，当b=1时，无需repeat
    cls = self.cls_token
    feature = torch.cat([cls, feature], 0)
    
    # 修改点3：将以下部分导出为ftmp，当做全局变量给网络
    # position = torch.from_numpy(np.arange(0, 50))
    # pos_feature = self.pos_embedding(position)# 仍然不支持算子：embeding，解决办法：由于pos_feature是固定常量，将其保存为ftmp，shape=[50,32]
    # 导出脚本
    # pos_feature.numpy().astype(np.float32).tofile("./pos_feature_50x32.ftmp")


    # feature is [HW, batch, channel]
    feature = self.encoder(feature, pos_feature)

    feature = feature.permute(1, 2, 0)

    feature = feature[:,:,0]

    gaze = self.feed(feature)
    return gaze
Model.forward = new_forward


#修改点2：去除repeat
def new_pos_embed(self, src, pos):
    batch_pos = pos.unsqueeze(1)#.repeat(1, src.size(1), 1)  #修改点2：去除repeat算子，当b=1时，无需repeat
    return src + batch_pos
TransformerEncoderLayer.pos_embed = new_pos_embed

# 修改点4：去除self.pos_embed内部的repeat和unsqueeze
def new_trans_encoder_forward(self, src, pos):
            # src_mask: Optional[Tensor] = None,
            # src_key_padding_mask: Optional[Tensor] = None):
            # pos: Optional[Tensor] = None):

    # q = k = self.pos_embed(src, pos)#内部存在不支持算子repeat和unsqueeze
    # 当b=1时，等效替换为：
    q = k = src + pos
    src2 = self.self_attn(q, k, value=src)[0]
    src = src + self.dropout1(src2)
    src = self.norm1(src)

    src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
    src = src + self.dropout2(src2)
    src = self.norm2(src)
    return src
TransformerEncoderLayer.forward = new_trans_encoder_forward

def main():

    modelpath = args.weight

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

        # gaze = net(data)
        
        # 导出模型
        tin = torch.randn(1,3,224,224)
        net.eval()
        trace_model = args.output_path + args.trace_model
        tmodel = torch.jit.trace(net,tin)
        tmodel.save(trace_model)
        print("successful export model to ", trace_model)



if __name__ == "__main__":
    args = parser_args()
    main()

