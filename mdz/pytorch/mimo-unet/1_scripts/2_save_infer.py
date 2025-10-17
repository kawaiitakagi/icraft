import sys
sys.path.append(R"../0_mimo-unet")
import torch
import numpy as np
import cv2

def main():

    # 加载经过跟踪的模型
    loaded_model = torch.jit.load("../2_compile/fmodel/mimo-unet_720x1280.pt")
    loaded_model.eval()
    # 加载特定的图像
    image_path = '../3_deploy/modelzoo/mimo-unet/io/input/GOPR0384_11_00-000001.png'
    # 使用OpenCV加载图像
    img = cv2.imread(image_path)
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # 将颜色通道顺序转换为RGB
    input_tensor = torch.tensor(img/255).permute(2, 0, 1).unsqueeze(0).float()


    # 使用模型进行推理
    with torch.no_grad():
        output =loaded_model(input_tensor)[2]

        # pred_clip = torch.clamp(output, 0, 1)
        # pred_numpy = pred_clip.squeeze(0).cpu().numpy()
        # pred_clip += 0.5 / 255

        output_image_cv = (output.squeeze(0).cpu().numpy().transpose(1, 2, 0))*255

        # 保存输出图像
        print(output_image_cv.shape)
        print(output_image_cv[0][0])
        cv2.imwrite("../3_deploy/modelzoo/mimo-unet/io/save_infer.png", output_image_cv)


if __name__ == '__main__':
    main()
    print('done!')
