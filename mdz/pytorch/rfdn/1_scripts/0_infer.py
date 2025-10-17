import sys
sys.path.append(R"../0_rfdn")
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
run python 0_infer.py --model RFDN  --scale 2   --pre_train ../weights/model_best.pt --test_only --data_test fly --save_results
'''