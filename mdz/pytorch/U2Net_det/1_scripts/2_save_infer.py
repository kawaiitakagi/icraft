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
import torch.nn.functional as F


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
    # ---------  get image path and name ---------
    model_name='u2net'
    image_dir = '../0_U2Net_det/test_data/test_human_images'
    prediction_dir = './results/save_infer/'
    if(not os.path.exists(prediction_dir)):
        os.makedirs(prediction_dir)
    model_pt = "../2_compile/fmodel/u2net_det_320x320_traced_new.pt"
    img_name_list = glob.glob(image_dir + os.sep + '*')
    
    # ---------  dataloader ---------
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

    # ---------  加载traced模型 ---------
    print("...load U2NET_det---")
    net = torch.jit.load(model_pt)
    if torch.cuda.is_available():
        net.cuda()
    net.eval()

    # --------- inference for each image ---------
    for i_test, data_test in enumerate(test_salobj_dataloader):

        print("inferencing:",img_name_list[i_test].split(os.sep)[-1])

        inputs_test = data_test['image']
        inputs_test = inputs_test.type(torch.FloatTensor)

        if torch.cuda.is_available():
            inputs_test = Variable(inputs_test.cuda())
        else:
            inputs_test = Variable(inputs_test)

        d1= net(inputs_test)
        d1 = F.sigmoid(d1)#后处理完成sigmoid操作

        # normalization
        pred = d1[:,0,:,:]
        pred = normPRED(pred)

        # save results to test_results folder
        save_output(img_name_list[i_test], pred, prediction_dir)

        del d1

if __name__ == "__main__":
    main()
