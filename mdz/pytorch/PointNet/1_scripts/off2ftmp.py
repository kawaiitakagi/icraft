"""
@Description: Main script for reading off file and converting to ftmp
"""
# %% Import packages
import numpy as np
import sys 
sys.path.append(R'../0_PointNet')
from logger import logging
import os 
from utils.data import get_single_data
def main(file_path) -> None:
    paths = os.walk(file_path)
    for path,dir_lst,file_lst in paths:
        for file_name in file_lst:
            data_path = os.path.join(path,file_name)# off file path 
            
            save_dir = path.replace('test','ftmp')
            save_name = file_name.replace('.off','.ftmp')
            save_path = os.path.join(save_dir,save_name)# ftmp file path 

            if not os.path.exists(save_dir):
                os.makedirs(save_dir)

            raw_data = get_single_data(data_path)# read off data & sample & normalize
            norm_pointcloud = raw_data.numpy().transpose((1,0))
            norm_pointcloud.astype(np.float32).tofile(save_path)# save norm_pointcloud data
            print('save at',save_path)
            
    logging.info("Convert off file to ftmp!")


if __name__ == '__main__':
    # file_path : off文件所在文件夹路径
    file_path = R'./Data/ModelNet40/airplane/test'
    main(file_path)
