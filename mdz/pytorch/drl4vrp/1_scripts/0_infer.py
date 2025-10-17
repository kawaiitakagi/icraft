# 该脚本用来加载模型并执行一张图像的前向推理
import sys
sys.path.append(R"../0_drl4vrp")

import os
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader

import env
from model import DRL4TSP
from tasks import tsp
from tasks.tsp import TSPDataset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#device = torch.device('cpu')


def validate(data_loader, actor, reward_fn, render_fn=None, save_dir='.',
             num_plot=5):
    """Used to monitor progress on a validation set & optionally plot solution."""
    actor.eval()
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    rewards = []
    for batch_idx, batch in enumerate(data_loader):
        print('*'*40,batch_idx,'*'*40)
        static, dynamic, x0 = batch

        static = static.to(device)
        dynamic = dynamic.to(device)
        x0 = x0.to(device) if len(x0) > 0 else None

        with torch.no_grad():
            tour_indices, _ = actor.forward(static, dynamic, x0)
            print('idx =',tour_indices)

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
    parser.add_argument('--checkpoint', default="../weights/tsp20")
    parser.add_argument('--task', default='tsp')
    parser.add_argument('--nodes', dest='num_nodes', default=20, type=int)
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--hidden', dest='hidden_size', default=128, type=int)
    parser.add_argument('--dropout', default=0.1, type=float)
    parser.add_argument('--layers', dest='num_layers', default=1, type=int)
    parser.add_argument('--test-size', default=100, type=int)

    args = parser.parse_args()
    print(args)
    # Goals from paper:
    # TSP20, 3.97
    # TSP50, 6.08
    # TSP100, 8.44

    STATIC_SIZE = 2 # (x, y)
    DYNAMIC_SIZE = 1 # dummy for compatibility
    update_fn = None
    actor = DRL4TSP(STATIC_SIZE,
                    DYNAMIC_SIZE,
                    args.hidden_size,
                    update_fn,
                    tsp.update_mask,
                    args.num_layers,
                    args.dropout).to(device)
    if args.checkpoint:
        path = os.path.join(args.checkpoint, 'actor.pt')
        actor.load_state_dict(torch.load(path, device))

    test_data = TSPDataset(args.num_nodes, args.test_size, args.seed + 2)

    test_dir = 'test_infer'
    test_loader = DataLoader(test_data, args.batch_size, False, num_workers=0)
    out = validate(test_loader, actor, env.reward, env.render, test_dir, num_plot=5)
    print('Results save at: ', test_dir)
    print('Average tour length: ', out)
