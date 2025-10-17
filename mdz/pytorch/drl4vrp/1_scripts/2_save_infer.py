# 该脚本用来加载模型并执行一张图像的前向推理
import sys
sys.path.append(R"../0_drl4vrp")

import os
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
import onnx
import onnxruntime
import time

import env
from model import DRL4TSP
from tasks.tsp import TSPDataset

device = torch.device('cpu')

def net_forward(sess,static, dynamic):
    tour_idx = []
    # --------------load Input--------------
    # 5 input: input1/input2/input3/input4/mask
    steps = 20 #total iteration steps
    total_time = 0
    static = static.numpy()                              # static
    dynamic = dynamic.numpy()                            # dynamic
    decoder_input = np.zeros((1,2,1),dtype=np.float32)   # decoder_input
    last_hh = np.zeros((1,1,128),dtype=np.float32)       # last_hh
    mask = np.ones((1,20),dtype=np.float32)              # mask
    # --------------Session Forward--------------
    for i in range(steps):
        start = time.time()
        # step = 0, use original inputs
        # step = {1-19}, use updated results
        if i == 0:
            ptr,gru_out = sess.run(None,input_feed= {"static":static,"dynamic":dynamic,"decoder_input":decoder_input,"last_hh":last_hh,"mask":mask})
        else:
            ptr,gru_out = sess.run(None,input_feed= {"static":static,"dynamic":dynamic,"decoder_input":decoder_update,"last_hh":last_hh_update,"mask":mask_update})
        
        # update results:decoder_input = decoder_update,gru_out=last_hh_update,mask=mask_update
        mask[0][ptr] = 0
        decoder_update = static[:,:,ptr].reshape(1,2,1)
        last_hh_update = gru_out
        mask_update = mask 
        # collect results
        tour_idx.append(ptr[0])
        # calc time
        end = time.time()
        total_time += end - start
    print('tour_idx =',tour_idx,'\nTotal time = ',total_time,'ms')
    return tour_idx

def validate(data_loader, model_path,reward_fn, render_fn=None, save_dir='.',
             num_plot=5):
    """Used to monitor progress on a validation set & optionally plot solution."""
    rewards = []
    # --------------load Model & Create Session--------------
    onnx_model = onnx.load(model_path)
    onnx.checker.check_model(onnx_model)
    print('Load Done')
    sess = onnxruntime.InferenceSession(model_path, providers=['CPUExecutionProvider'])
    print('Create Session Done')
    # --------------Validate results--------------
    for batch_idx, batch in enumerate(data_loader):
        print('*'*40,batch_idx,'*'*40)
        static, dynamic, x0 = batch

        static = static.to(device)
        dynamic = dynamic.to(device)
        x0 = x0.to(device) if len(x0) > 0 else None
        #----------------------------net forward--------------
        test_tour_indices = net_forward(sess, static, dynamic)
        print('idx =',test_tour_indices)
        tour_indices = torch.tensor(test_tour_indices).unsqueeze(0)
        reward = reward_fn(static, tour_indices).mean().item()
        rewards.append(reward)

        # Uncomment the following code will save the visualization results  
        
        if render_fn is not None and batch_idx < num_plot:
            name_path = 'batch_'+str(batch_idx)+'.gif'
            path = os.path.join(save_dir, name_path)
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            render_fn(static, tour_indices, path)
            
    return np.mean(rewards)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Combinatorial Optimization')
    parser.add_argument('--seed', default=12345, type=int)
    parser.add_argument('--task', default='tsp')
    parser.add_argument('--nodes', dest='num_nodes', default=20, type=int)
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--test-size',default=100, type=int)
    parser.add_argument('--model-path',default="../3_deploy/modelzoo/drl4vrp/imodel/drl_tsp_step1.onnx", type=str)

    args = parser.parse_args()
    print(args)
   
    # Goals from paper:
    # TSP20, 3.97
    # TSP50, 6.08
    # TSP100, 8.44
    
    test_data = TSPDataset(args.num_nodes, args.test_size, args.seed + 2)
    test_dir = 'test_save_infer'
    test_loader = DataLoader(test_data, args.batch_size, False, num_workers=0)
    model_path = args.model_path
    out = validate(test_loader, model_path,env.reward, env.render, test_dir, num_plot=5)
    print('Results save at: ', test_dir)
    print('Average tour length: ', out)
