import sys
sys.path.append(R"../0_rfdn")
import torch
import cv2
import numpy as np

def main():

    # 加载经过跟踪的模型
    loaded_model = torch.jit.load("../2_compile/fmodel/rfdn_160x240.pt")
    loaded_model.eval()
    # 加载特定的图像
    image_path = '../2_compile/qtset/sr/3096x2.png'
    # 使用OpenCV加载图像
    img = cv2.imread(image_path)
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # 将颜色通道顺序转换为RGB
    input_tensor = torch.tensor(img).permute(2, 0, 1).unsqueeze(0).float()
    idx_scale = torch.zeros(1)

    # 使用模型进行推理
    with torch.no_grad():
        output =loaded_model(input_tensor,idx_scale)

    # 将输出张量转换为NumPy数组并转换通道顺序为BGR
    # output_image_cv = np.clip(output.squeeze(0).cpu().numpy().transpose(1, 2, 0) * 255, 0, 255).astype(np.uint8)
    output_image_cv = (output.squeeze(0).cpu().numpy().transpose(1, 2, 0))
    # output_image_cv = cv2.cvtColor(output_image_cv, cv2.COLOR_RGB2BGR)

    # 保存输出图像
    print(output_image_cv.shape)
    print(output_image_cv[0][0])
    cv2.imwrite("../3_deploy/modelzoo/rfdn/io/test/save_infer_3096x2.png", output_image_cv)


if __name__ == '__main__':
    main()
