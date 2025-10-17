import sys 
sys.path.append("../0_U2Net_portrait")
import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import glob
import argparse
from data_loader import RescaleT
from data_loader import ToTensorLab
from data_loader import SalObjDataset
from model import U2NET # full size version 173.6 MB
from model import U2NETP # small version u2net 4.7 MB
from model.u2net import U2NET
import torch.nn.functional as F

RED = '\033[31m'       # 设置前景色为红色
RESET = '\033[0m'      # 重置所有属性到默认值
ver = torch.__version__
assert ("1.6" in ver) or ("1.9" in ver), f"{RED}Unsupported PyTorch version: {ver}{RESET}"

def parser():
    parser = argparse.ArgumentParser(description="image and portrait composite")
    parser.add_argument('-s',action='store',default=20,dest='sigma')
    parser.add_argument('-a',action='store',default=0,dest='alpha')
    parser.add_argument('--img_dir',default='../0_U2Net_portrait/test_data/test_portrait_images/your_portrait_im',help="image path")
    parser.add_argument('--res_dir',default='./results',help="results save path")
    parser.add_argument('--model',default='../weights/u2net_portrait.pth',help="model path")
    parser.add_argument('--export_dir', type=str, default=R'../2_compile/fmodel', help='traced model path')
    args = parser.parse_args()
    return args

def _upsample_like(src,tar):
    src = F.upsample(src,size=tar.shape[2:],mode='bilinear')
    return src
def new_forward(self,x):
    hx = x
    #stage 1
    hx1 = self.stage1(hx)
    hx = self.pool12(hx1)
    #stage 2
    hx2 = self.stage2(hx)
    hx = self.pool23(hx2)
    #stage 3
    hx3 = self.stage3(hx)
    hx = self.pool34(hx3)
    #stage 4
    hx4 = self.stage4(hx)
    hx = self.pool45(hx4)
    #stage 5
    hx5 = self.stage5(hx)
    hx = self.pool56(hx5)
    #stage 6
    hx6 = self.stage6(hx)
    hx6up = _upsample_like(hx6,hx5)

    #-------------------- decoder --------------------
    hx5d = self.stage5d(torch.cat((hx6up,hx5),1))
    hx5dup = _upsample_like(hx5d,hx4)
    hx4d = self.stage4d(torch.cat((hx5dup,hx4),1))
    hx4dup = _upsample_like(hx4d,hx3)
    hx3d = self.stage3d(torch.cat((hx4dup,hx3),1))
    hx3dup = _upsample_like(hx3d,hx2)
    hx2d = self.stage2d(torch.cat((hx3dup,hx2),1))
    hx2dup = _upsample_like(hx2d,hx1)
    hx1d = self.stage1d(torch.cat((hx2dup,hx1),1))

    #side output
    d1 = self.side1(hx1d)
    d2 = self.side2(hx2d)
    d2 = _upsample_like(d2,d1)
    d3 = self.side3(hx3d)
    d3 = _upsample_like(d3,d1)
    d4 = self.side4(hx4d)
    d4 = _upsample_like(d4,d1)
    d5 = self.side5(hx5d)
    d5 = _upsample_like(d5,d1)
    d6 = self.side6(hx6)
    d6 = _upsample_like(d6,d1)
    d0 = self.outconv(torch.cat((d1,d2,d3,d4,d5,d6),1))
    # return F.sigmoid(d0), F.sigmoid(d1), F.sigmoid(d2), F.sigmoid(d3), F.sigmoid(d4), F.sigmoid(d5), F.sigmoid(d6)
    # 修改点：将多输出改为单输出
    return  F.sigmoid(d0)

def main():
    args = parser()
    if(not os.path.exists(args.res_dir)):
        os.mkdir(args.res_dir)

    img_name_list = glob.glob(args.img_dir+'/*')
    print("Number of images: ", len(img_name_list))

    # --------- dataloader ---------
    test_salobj_dataset = SalObjDataset(img_name_list = img_name_list,
                                        lbl_name_list = [],
                                        transform=transforms.Compose([RescaleT(512),
                                                                      ToTensorLab(flag=0)])
                                        )
    test_salobj_dataloader = DataLoader(test_salobj_dataset,
                                        batch_size=1,
                                        shuffle=False,
                                        num_workers=1)

    # --------- model define ---------

    print("...load U2NET---173.6 MB")
    net = U2NET(3,1)
    net.load_state_dict(torch.load(args.model, map_location=torch.device('cpu')))
    if torch.cuda.is_available():
        net.cuda()
    net.eval()

    # export model
    U2NET.forward = new_forward
    input = torch.randn(1,3,512,512).cpu()
    model_traced = torch.jit.trace(net, input)
    if not os.path.exists(args.export_dir):
        os.makedirs(args.export_dir)
    model_traced.save(args.export_dir + "/"+ "u2net_portrait_512x512_traced_new.pt")
    print("traced model saved in ", args.export_dir+ "/"+ "u2net_portrait_512x512_traced_new.pt")



if __name__ == "__main__":
    main()
