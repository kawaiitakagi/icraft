import sys
sys.path.append(R"../0_edsr")
import torch
import utility
import data
import model
import loss
from option import args
from trainer import Trainer
import os

torch.manual_seed(args.seed)
checkpoint = utility.checkpoint(args)

def main():
    global model
    if args.data_test == ['video']:
        from videotester import VideoTester
        model = model.Model(args, checkpoint)
        t = VideoTester(args, model, checkpoint)
        t.test()
    else:
        if checkpoint.ok:
            loader = data.Data(args)
            _model = model.Model(args, checkpoint)
            _loss = loss.Loss(args, checkpoint) if not args.test_only else None
            t = Trainer(args, loader, _model, _loss, checkpoint)
            while not t.terminate():
                t.train()
                t.test()

            checkpoint.done()
if __name__ == '__main__':
    main()

'''
cd 1_scripts
run 
python .\0_infer.py  --patch_size 48 --n_resblocks 10 --n_feats 32 --res_scale 1 --pre_train  ../weights/edsr_gelu.pt --test_only --save_results
python .\0_infer.py  --patch_size 48 --n_resblocks 10 --n_feats 32 --res_scale 1 --pre_train  ../weights/edsr_big_gelu.pt --test_only --save_results
'''