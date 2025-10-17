# 该脚本用来导出onnx模型
import os
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import sys
sys.path.append(R"../0_drl4vrp")
from model import DRL4TSP
from tasks import tsp
def new_forward(self, static, dynamic, mask, decoder_input=None, last_hh=None):
    
    batch_size, input_size, sequence_size = static.size()

    if decoder_input is None:
        decoder_input = self.x0.expand(batch_size, -1, -1)

    
    max_steps = 1
    
    static_hidden = self.static_encoder(static)
    dynamic_hidden = self.dynamic_encoder(dynamic)

    for _ in range(max_steps):

        if not mask.byte().any():
            break

        # ... but compute a hidden rep for each element added to sequence
        decoder_hidden = self.decoder(decoder_input)

        probs, last_hh = self.pointer(static_hidden,
                                        dynamic_hidden,
                                        decoder_hidden, last_hh)
        probs = F.softmax(probs + mask.log(), dim=1)

        # When training, sample the next step according to its probability.
        # During testing, we can take the greedy approach and choose highest
        if self.training:
            m = torch.distributions.Categorical(probs)

            # Sometimes an issue with Categorical & sampling on GPU; See:
            # https://github.com/pemami4911/neural-combinatorial-rl-pytorch/issues/5
            ptr = m.sample()
            while not torch.gather(mask, 1, ptr.data.unsqueeze(1)).byte().all():
                ptr = m.sample()
            logp = m.log_prob(ptr)
        else:
            prob, ptr = torch.max(probs, 1)  # Greedy
            logp = prob.log()
    return ptr, last_hh
DRL4TSP.forward = new_forward
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Combinatorial Optimization')
    parser.add_argument('--seed', default=12345, type=int)
    parser.add_argument('--checkpoint', default="../weights/tsp20/actor.pt")
    parser.add_argument('--dst', default="../3_deploy/modelzoo/drl4vrp/imodel/drl_tsp_step1.onnx")
    parser.add_argument('--nodes', dest='num_nodes', default=20, type=int)

    args = parser.parse_args()
    print(args)
    device = torch.device('cpu')
    
    # set seed
    seed = np.random.randint(args.seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    
    
    STATIC_SIZE = 2 # (x, y)
    DYNAMIC_SIZE = 1 # dummy for compatibility
    HIDDEN_SIZE = 128
    update_fn = None
    actor = DRL4TSP(STATIC_SIZE,
                    DYNAMIC_SIZE,
                    HIDDEN_SIZE,
                    update_fn,
                    tsp.update_mask,
                    num_layers=1,
                    dropout=0.1).to(device)
    # load weights
    actor.load_state_dict(torch.load(args.checkpoint, device))
    actor.eval()
    # input
    static =torch.rand((1, 2, args.num_nodes))
    dynamic = torch.zeros((1,1,args.num_nodes))
    x0 = torch.zeros((1,2,1))
    last_hh = torch.zeros((1,1,128))
    mask = torch.ones((1,20))
    # net forward
    tour_indices, _ = actor.forward(static, dynamic, mask,x0,last_hh)
    # export onnx model
    torch.onnx.export(actor,
                        (static, dynamic, mask,x0,last_hh),
                        args.dst,
                        export_params = True,
                        opset_version=11,
                        input_names = ['static','dynamic','mask','decoder_input','last_hh'],
                        output_names = ['ptr','gru_out']  
                    )
    print('Traced model save at:',args.dst)
    
    
