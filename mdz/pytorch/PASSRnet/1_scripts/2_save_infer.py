from models import *
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
from utils import *
import argparse
import os
from torchvision import transforms



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--testset_dir', type=str, default='data')
    parser.add_argument('--dataset', type=str, default='test')
    parser.add_argument('--scale_factor', type=int, default=4)
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--res', type=str, default='results/save_infer/')
    parser.add_argument('--model_path', type=str, default=R"../2_compile/fmodel/PASSRnet_traced_1x3x125x178.pt", help='torchscript model path')
    parser.add_argument('--lr_imgsz', nargs='+', type=int, default=[125,178], help='lr image size([h,w])')#不同输入图片尺寸不同，需要导出不同尺寸的模型

    return parser.parse_args()

def test(test_loader, cfg):
    net = PASSRnet(cfg.scale_factor).to(cfg.device)
    # cudnn.benchmark = True
    # pretrained_dict = torch.load('./log/x' + str(cfg.scale_factor) + '/PASSRnet_x' + str(cfg.scale_factor) + '.pth',map_location=torch.device('cpu'))
    # net.load_state_dict(pretrained_dict)
    # 加载trace model
    net = torch.jit.load(cfg.model_path)

    psnr_list = []

    with torch.no_grad():
        for idx_iter, (HR_left, _, LR_left, LR_right) in enumerate(test_loader):
            HR_left, LR_left, LR_right = Variable(HR_left).to(cfg.device), Variable(LR_left).to(cfg.device), Variable(LR_right).to(cfg.device)
            scene_name = test_loader.dataset.file_list[idx_iter]
            #前向
            # SR_left = net(LR_left, LR_right, is_training=0)
           
            SR_left = net(LR_left, LR_right)#LR_left.shape=[1, 3, 94, 310],LR_right.shape=[1, 3, 94, 310],不同图片size不同
            SR_left = torch.clamp(SR_left, 0, 1)
            psnr_list.append(cal_psnr(HR_left[:,:,:,64:], SR_left[:,:,:,64:]))#计算峰值信噪比

            ## save results
            if not os.path.exists(cfg.res+cfg.dataset):
                os.makedirs(cfg.res+cfg.dataset)
            if not os.path.exists(cfg.res+cfg.dataset+'/'+scene_name):
                os.makedirs(cfg.res+cfg.dataset+'/'+scene_name)
            SR_left_img = transforms.ToPILImage()(torch.squeeze(SR_left.data.cpu(), 0))
            SR_left_img.save(cfg.res+cfg.dataset+'/'+scene_name+'/img_0.png')
            print("results save in ", cfg.res+cfg.dataset)
        ## print results
        print(cfg.dataset + ' mean psnr: ', float(np.array(psnr_list).mean()))

def main(cfg):
    test_set = TestSetLoader(dataset_dir=cfg.testset_dir + '/' + cfg.dataset, scale_factor=cfg.scale_factor)
    test_loader = DataLoader(dataset=test_set, num_workers=1, batch_size=1, shuffle=False)
    result = test(test_loader, cfg)
    return result

if __name__ == '__main__':
    cfg = parse_args()
    main(cfg)
