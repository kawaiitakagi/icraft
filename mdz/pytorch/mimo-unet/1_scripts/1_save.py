import sys
sys.path.append(R"../0_mimo-unet")
# import os
import torch
from torchvision.transforms import functional as F
# import numpy as np
from utils import Adder
from data import test_dataloader
# from skimage.metrics import peak_signal_noise_ratio
import time
from models.MIMOUNet import build_net

def _eval(model, model_path):
    state_dict = torch.load(model_path,map_location=torch.device('cpu'))
    model.load_state_dict(state_dict['model'])
    device = torch.device('cpu')
    dataloader = test_dataloader('../3_deploy/modelzoo/mimo-unet/io', batch_size=1, num_workers=0)
    adder = Adder()
    model.eval()
    with torch.no_grad():
        psnr_adder = Adder()

        # Hardware warm-up
        for iter_idx, data in enumerate(dataloader):
            input_img, label_img, _ = data
            input_img = input_img.to(device)
            tm = time.time()
            _ = model(input_img)
            _ = time.time() - tm

            if iter_idx == 20:
                break

        # Main Evaluation
        for iter_idx, data in enumerate(dataloader):
            input_img, label_img, name = data

            input_img = input_img.to(device)

            tm = time.time()
            # print(input_img.shape)
            # im = torch.randn(1, 3, 720, 1280, dtype = torch.float32)
            torch.onnx.export(model, input_img,"../2_compile/fmodel/mimo-unet_720x1280.onnx",opset_version=11)
            torch.jit.save(torch.jit.trace(model, input_img),"../2_compile/fmodel/mimo-unet_720x1280.pt")

            # pred = model(input_img)[2]
            # elapsed = time.time() - tm
            # adder(elapsed)
            # pred_clip = torch.clamp(pred, 0, 1)
            # pred_numpy = pred_clip.squeeze(0).cpu().numpy()
            # label_numpy = label_img.squeeze(0).cpu().numpy()
            # save_path = '../3_deploy/modelzoo/mimo-unet/io/result'
            # if not os.path.exists(save_path):
            #     os.makedirs(save_path)
            # save_name = os.path.join(save_path, name[0])
            # pred_clip += 0.5 / 255
            # pred = F.to_pil_image(pred_clip.squeeze(0).cpu(), 'RGB')
            # pred.save(save_name)
            # psnr = peak_signal_noise_ratio(pred_numpy, label_numpy, data_range=1)
            # psnr_adder(psnr)
        #     print('%d iter PSNR: %.2f time: %f' % (iter_idx + 1, psnr, elapsed))

        # print('==========================================================')
        # print('The average PSNR is %.2f dB' % (psnr_adder.average()))
        # print("Average time: %f" % adder.average())

if __name__ == '__main__':
    model = build_net("MIMO-UNet")
    model_path = '../weights/MIMO-UNet.pkl'
    _eval(model,model_path)
    print('save done!')
