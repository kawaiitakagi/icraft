import sys 
sys.path.append("../0_U2Net_portrait")
import os
import torch
from skimage import io, transform
from skimage.filters import gaussian
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import argparse
import numpy as np
import glob
from data_loader import RescaleT
from data_loader import ToTensorLab
from data_loader import SalObjDataset



# normalize the predicted SOD probability map
def normPRED(d):
    ma = torch.max(d)
    mi = torch.min(d)
    dn = (d-mi)/(ma-mi)
    return dn

def save_output(image_name,pred,d_dir,sigma=2,alpha=0.5):
    predict = pred
    predict = predict.squeeze()
    predict_np = predict.cpu().data.numpy()

    image = io.imread(image_name)
    pd = transform.resize(predict_np,image.shape[0:2],order=2)
    pd = pd/(np.amax(pd)+1e-8)*255
    pd = pd[:,:,np.newaxis]
    ## fuse the orignal portrait image and the portraits into one composite image
    ## 1. use gaussian filter to blur the orginal image
    sigma=sigma
    image = gaussian(image, sigma=sigma, preserve_range=True)
    ## 2. fuse these orignal image and the portrait with certain weight: alpha
    alpha = alpha
    im_comp = image*alpha+pd*(1-alpha)
    # print(im_comp.shape)
    img_name = image_name.split(os.sep)[-1]
    aaa = img_name.split(".")
    bbb = aaa[0:-1]
    imidx = bbb[0]
    for i in range(1,len(bbb)):
        imidx = imidx + "." + bbb[i]
    io.imsave(d_dir+'/'+imidx+'_sigma_' + str(sigma) + '_alpha_' + str(alpha) + '_composite.png',(im_comp).astype(np.uint8))

def parser():
    parser = argparse.ArgumentParser(description="image and portrait composite")
    parser.add_argument('-s',action='store',default=20,dest='sigma')
    parser.add_argument('-a',action='store',default=0,dest='alpha')
    parser.add_argument('--img_dir',default='../0_U2Net_portrait/test_data/test_portrait_images/your_portrait_im',help="image path")
    parser.add_argument('--res_dir',default='./results/save_infer',help="results save path")
    parser.add_argument('--model_pt', type=str, default=R"../2_compile/fmodel/u2net_portrait_512x512_traced_new.pt", help='torchscript model path')
    args = parser.parse_args()
    return args

def main():
    args = parser()
    if(not os.path.exists(args.res_dir)):
        os.makedirs(args.res_dir)

    img_name_list = glob.glob(args.img_dir+'/*')
    print("Number of images: ", len(img_name_list))

    # ---------  dataloader ---------
    test_salobj_dataset = SalObjDataset(img_name_list = img_name_list,
                                        lbl_name_list = [],
                                        transform=transforms.Compose([RescaleT(512),
                                                                      ToTensorLab(flag=0)])
                                        )
    test_salobj_dataloader = DataLoader(test_salobj_dataset,
                                        batch_size=1,
                                        shuffle=False,
                                        num_workers=1)

    # ---------  加载traced模型 ---------

    print("...load U2NET_potrait---")
    net = torch.jit.load(args.model_pt)
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

        # normalization
        pred = 1.0 - d1[:,0,:,:]
        pred = normPRED(pred)
        # save results to test_results folder
        save_output(img_name_list[i_test],pred,args.res_dir,sigma=float(args.sigma),alpha=float(args.alpha))

        del d1

if __name__ == "__main__":
    main()
