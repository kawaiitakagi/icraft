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
            # model.Model.forward = new_forward
            _model = model.Model(args, checkpoint)
            print('-----------------model trace --------------------')
            # x = torch.randn(1, 3, 160, 240)
            _loss = loss.Loss(args, checkpoint) if not args.test_only else None
            t = Trainer(args, loader, _model, _loss, checkpoint)
            _model.eval()
            
            im = torch.randn(1, 3, 160, 240, dtype = torch.float32)
            idx_scale = torch.ones(1)
            torch.onnx.export(_model, (im,idx_scale),"../2_compile/fmodel/rfdn_160x240.onnx",opset_version=11)
            trace_model = torch.jit.trace(_model, (im,idx_scale))
            torch.jit.save(trace_model,"../2_compile/fmodel/rfdn_160x240.pt")

            while not t.terminate():
                t.train()
                t.test()

            checkpoint.done()



if __name__ == '__main__':
    main()

'''
cd 1_scripts
run python 1_save.py --model RFDN  --scale 2   --pre_train   ../weights/model_best.pt --test_only --data_test fly
'''
