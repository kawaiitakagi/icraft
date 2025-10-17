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
            # model.Model.forward = new_forward
            _model = model.Model(args, checkpoint)
            print('-----------------model trace --------------------')
            # x = torch.randn(1, 3, 160, 240)
            _loss = loss.Loss(args, checkpoint) if not args.test_only else None
            t = Trainer(args, loader, _model, _loss, checkpoint)
            _model.eval()
            
            im = torch.randn(1, 3, 160, 240, dtype = torch.float32)
            torch.onnx.export(_model, im,"../2_compile/fmodel/edsr_gelu_160x240.onnx",opset_version=11)
            # torch.onnx.export(_model, im,"../2_compile/fmodel/edsr_big_gelu_160x240.onnx",opset_version=11)
            trace_model = torch.jit.trace(_model, im)
            torch.jit.save(trace_model,"../2_compile/fmodel/edsr_gelu_160x240.pt")
            # torch.jit.save(trace_model,"../2_compile/fmodel/edsr_big_gelu_160x240.pt")

            while not t.terminate():
                t.train()
                t.test()

            checkpoint.done()



if __name__ == '__main__':
    main()

'''
cd 1_scripts
run 
python .\1_save.py --patch_size 48 --n_resblocks 10 --n_feats 32 --res_scale 1 --pre_train  ../weights/edsr_big_gelu.pt --test_only
python .\1_save.py --patch_size 48 --n_resblocks 10 --n_feats 32 --res_scale 1 --pre_train  ../weights/edsr_gelu.pt --test_only

'''
