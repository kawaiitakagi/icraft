import sys
sys.path.insert(0, '../0_A2Net/')
from models.model import BaseNet
import torch
import torch.backends.cudnn as cudnn
import torch.optim.lr_scheduler

import dataset as myDataLoader
import Transforms as myTransforms
import os, time
import numpy as np
from argparse import ArgumentParser

RED = '\033[31m'       # 设置前景色为红色
RESET = '\033[0m'      # 重置所有属性到默认值
ver = torch.__version__
assert ("1.6" in ver) or ("1.9" in ver), f"{RED}Unsupported PyTorch version: {ver}{RESET}"


@torch.no_grad()
def val(args, val_loader, model, epoch):
    model.eval()
    print(len(val_loader))
    for iter, batched_inputs in enumerate(val_loader):

        img, target = batched_inputs
        pre_img = img[:, 0:3]
        post_img = img[:, 3:6]

        if args.onGPU == True:
            pre_img = pre_img.cuda()
            target = target.cuda()
            post_img = post_img.cuda()

        pre_img_var = torch.autograd.Variable(pre_img).float()
        post_img_var = torch.autograd.Variable(post_img).float()

        # export model
        input1 = torch.randn(1,3,256,256)
        input2 = torch.randn(1,3,256,256)
        output, output2, output3, output4 = model(pre_img_var, post_img_var) # run the model
        torch.onnx.export(model,(input1,input2),args.model_onnx,opset_version=11,example_outputs=(output,output2,output3,output4))
        print("successful export model to ",args.model_onnx)
        # pred = torch.where(output > 0.5, torch.ones_like(output), torch.zeros_like(output)).long()
        break #只跑一张图片
    return 0, 0


def ValidateSegmentation(args):
    torch.backends.cudnn.benchmark = True
    SEED = 2333
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)

    model = BaseNet(3, 1)

    args.savedir = args.savedir + '_' + args.file_root + '_iter_' + str(args.max_steps) + '_lr_' + str(args.lr) + '/'
    args.vis_dir = './Predict/' + args.file_root + '/'

    if args.file_root == 'LEVIR':
        args.file_root = '../0_A2Net/samples'
        # args.file_root = '/home/guan/Documents/Datasets/ChangeDetection/LEVIR-CD_256_patches'
    elif args.file_root == 'BCDD':
        args.file_root = '/home/guan/Documents/Datasets/ChangeDetection/BCDD'
    elif args.file_root == 'SYSU':
        args.file_root = '/home/guan/Documents/Datasets/ChangeDetection/SYSU'
    elif args.file_root == 'CDD':
        args.file_root = '/home/guan/Documents/Datasets/ChangeDetection/CDD'
    elif args.file_root == 'testLEVIR':
        args.file_root = '../0_A2Net/samples'
    else:
        raise TypeError('%s has not defined' % args.file_root)

    if not os.path.exists(args.savedir):
        os.makedirs(args.savedir)

    if not os.path.exists(args.vis_dir):
        os.makedirs(args.vis_dir)

    if args.onGPU:
        model = model.cuda()

    total_params = sum([np.prod(p.size()) for p in model.parameters()])
    print('Total network parameters (excluding idr): ' + str(total_params))

    mean = [0.406, 0.456, 0.485, 0.406, 0.456, 0.485]
    std = [0.225, 0.224, 0.229, 0.225, 0.224, 0.229]

    # compose the data with transforms
    valDataset = myTransforms.Compose([
        myTransforms.Normalize(mean=mean, std=std),
        myTransforms.Scale(args.inWidth, args.inHeight),
        myTransforms.ToTensor()
    ])

    test_data = myDataLoader.Dataset("test", file_root=args.file_root, transform=valDataset)
    testLoader = torch.utils.data.DataLoader(
        test_data, shuffle=False,
        batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=False)

    if args.onGPU:
        cudnn.benchmark = True

    logFileLoc = args.savedir + args.logFile
    if os.path.isfile(logFileLoc):
        logger = open(logFileLoc, 'a')
    else:
        logger = open(logFileLoc, 'w')
        logger.write("Parameters: %s" % (str(total_params)))
        logger.write(
            "\n%s\t%s\t%s\t%s\t%s\t%s" % ('Epoch', 'Kappa', 'IoU', 'F1', 'R', 'P'))
    logger.flush()

    # load the model
    model_file_name = args.savedir + 'best_model.pth'
    # state_dict = torch.load(model_file_name)
    state_dict = torch.load(model_file_name, map_location=torch.device('cpu'))
    model.load_state_dict(state_dict)
    loss_test, score_test = val(args, testLoader, model, 0)



if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--file_root', default="LEVIR", help='Data directory | LEVIR | BCDD | SYSU ')
    parser.add_argument('--inWidth', type=int, default=256, help='Width of RGB image')
    parser.add_argument('--inHeight', type=int, default=256, help='Height of RGB image')
    parser.add_argument('--max_steps', type=int, default=40000, help='Max. number of iterations')
    parser.add_argument('--num_workers', type=int, default=3, help='No. of parallel threads')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size')
    parser.add_argument('--step_loss', type=int, default=100, help='Decrease learning rate after how many epochs')
    parser.add_argument('--lr', type=float, default=5e-4, help='Initial learning rate')
    parser.add_argument('--lr_mode', default='poly', help='Learning rate policy, step or poly')
    parser.add_argument('--savedir', default='../0_A2Net/results', help='Directory to save the results')
    parser.add_argument('--resume', default=None, help='Use this checkpoint to continue training | '
                                                       './results_ep100/checkpoint.pth.tar')
    parser.add_argument('--logFile', default='testLog.txt',
                        help='File that stores the training and validation logs')
    parser.add_argument('--onGPU', default=False, type=lambda x: (str(x).lower() == 'true'),
                        help='Run on CPU or GPU. If TRUE, then GPU.')
    parser.add_argument('--weight', default='', type=str, help='pretrained weight, can be a non-strict copy')
    parser.add_argument('--ms', type=int, default=0, help='apply multi-scale training, default False')
    parser.add_argument('--export_dir', default='../2_compile/fmodel', help='Directory to save the model')#export path
    parser.add_argument('--model_onnx', default='../2_compile/fmodel/A2Net_256x256_traced.onnx', help='save the onnx model path')

    args = parser.parse_args()
    if not os.path.exists(args.export_dir):
        os.makedirs(args.export_dir)

    print('Called with args:')
    print(args)

    ValidateSegmentation(args)
