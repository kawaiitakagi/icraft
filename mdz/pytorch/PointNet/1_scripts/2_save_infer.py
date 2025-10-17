"""
@Description: Main script for testing a single point cloud data by torchscript model
"""
import argparse
from datetime import datetime
import sys 
sys.path.append(R'../0_PointNet')
from logger import logging
from utils.data import get_single_data
import torch 
seed_value = 42
torch.manual_seed(seed_value)
import random 
random.seed = seed_value
CLASS= {'airplane': 0, 'bathtub': 1, 'bed': 2, 'bench': 3, 'bookshelf': 4, 'bottle': 5, 'bowl': 6, 'car': 7, 'chair': 8, 'cone': 9, 'cup': 10, 'curtain': 11, 'desk': 12, 'door': 13, 'dresser': 14, 'flower_pot': 15, 'glass_box': 16, 'guitar': 17, 'keyboard': 18, 'lamp': 19, 'laptop': 20, 'mantel': 21, 'monitor': 22, 'night_stand': 23, 'person': 24, 'piano': 25, 'plant': 26, 'radio': 27, 'range_hood': 28, 'sink': 29, 'sofa': 30, 'stairs': 31, 'stool': 32, 'table': 33, 'tent': 34, 'toilet': 35, 'tv_stand': 36, 'vase': 37, 'wardrobe': 38, 'xbox': 39}
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default= '../2_compile/fmodel/pointnet_traced_3x1024.pt', help='model path')
    parser.add_argument('--source', type=str, default= './Data/ModelNet40/airplane/test/airplane_0627.off', help='test_data_path')
    parser.add_argument('--device', type=str, default= 'cpu', help='computing_device')
    opt = parser.parse_args()

    device = opt.device
    data_path = opt.source
    model_path = opt.weights
    
    logging.info(
        f"Selected Computing Device:\t       {device}\n"
        f"TEST INFO:\n"
        f"\tTest Data Path:                {data_path}\n"
    )


    # let's test the trained network
    logging.info("Testing started...")
    pointnet = torch.jit.load(model_path)
    _data = get_single_data(data_path)
    _input = _data.float().to(device)
    _input = torch.unsqueeze(_input, 0)
    _output, __, __ = pointnet(_input.transpose(1, 2))
    print("_output =",_output.shape)
    _, pred = torch.max(_output.data, 1) # value,index=pred
    pred = pred.cpu().numpy()[0]
    for category in CLASS.keys():
        if CLASS[category] == pred:
            break
    print(f"TEST RESULT:\n\tcategory =",category)

if __name__ == '__main__':
    main()
