import sys 
sys.path.append("../0_U2Net_det")
import os
from skimage import io, transform
import torch
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms#, utils
# import torch.optim as optim

import numpy as np
from PIL import Image
import glob

from data_loader import RescaleT
from data_loader import ToTensorLab
from data_loader import SalObjDataset
from model import U2NET # full size version 173.6 MB
from model.u2net import U2NET
import torch.nn.functional as F

RED = '\033[31m'       # 设置前景色为红色
RESET = '\033[0m'      # 重置所有属性到默认值
ver = torch.__version__
assert ("1.6" in ver) or ("1.9" in ver), f"{RED}Unsupported PyTorch version: {ver}{RESET}"

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
    #return F.sigmoid(d0), F.sigmoid(d1), F.sigmoid(d2), F.sigmoid(d3), F.sigmoid(d4), F.sigmoid(d5), F.sigmoid(d6)
    # 修改点：将多输出改为单输出
    return F.sigmoid(d0)
    

# normalize the predicted SOD probability map
def normPRED(d):
    ma = torch.max(d)
    mi = torch.min(d)
    dn = (d-mi)/(ma-mi)
    return dn

def save_output(image_name,pred,d_dir):
    predict = pred
    predict = predict.squeeze()
    predict_np = predict.cpu().data.numpy()
    im = Image.fromarray(predict_np*255).convert('RGB')
    img_name = image_name.split(os.sep)[-1]
    image = io.imread(image_name)
    imo = im.resize((image.shape[1],image.shape[0]),resample=Image.BILINEAR)
    pb_np = np.array(imo)
    aaa = img_name.split(".")
    bbb = aaa[0:-1]
    imidx = bbb[0]
    for i in range(1,len(bbb)):
        imidx = imidx + "." + bbb[i]
    imo.save(d_dir+imidx+'.png')

def main():
    # --------- 1. get image path and name ---------
    model_name='u2net'
    image_dir = '../0_U2Net_det/test_data/test_human_images'
    model_dir = "../weights/u2net.pth"
    export_dir = "../2_compile/fmodel"
    export_name= "u2net_det_320x320_traced_new.pt"
    if(not os.path.exists(export_dir)):
        os.makedirs(export_dir)
    img_name_list = glob.glob(image_dir + os.sep + '*')
    
    # --------- 2. dataloader ---------
    #1. dataloader
    test_salobj_dataset = SalObjDataset(img_name_list = img_name_list,
                                        lbl_name_list = [],
                                        transform=transforms.Compose([RescaleT(320),
                                                                      ToTensorLab(flag=0)])
                                        )
    test_salobj_dataloader = DataLoader(test_salobj_dataset,
                                        batch_size=1,
                                        shuffle=False,
                                        num_workers=1)

    # --------- 3. model define ---------
    if(model_name=='u2net'):
        print("...load U2NET---173.6 MB")
        net = U2NET(3,1)

    if torch.cuda.is_available():
        net.load_state_dict(torch.load(model_dir))
        net.cuda()
    else:
        net.load_state_dict(torch.load(model_dir, map_location='cpu'))
    net.eval()

     # export model
    U2NET.__call__ = new_forward
    input = torch.randn(1,3,320,320).cpu()
    model_traced = torch.jit.trace(net, input)
    model_traced.save(export_dir + "/"+ export_name)
    print("traced model saved in ", export_dir+ "/"+ export_name)

if __name__ == "__main__":
    main()
