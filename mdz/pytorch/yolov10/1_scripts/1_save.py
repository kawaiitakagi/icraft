#将模型导出为torch_script
import torch 
import argparse
import json
import sys 
sys.path.append(R'../0_yolov10')
from ultralytics import YOLOv10
from ultralytics.nn.tasks import BaseModel
from ultralytics.nn.modules.head import v10Detect
from ultralytics.engine.exporter import Exporter,try_export
from ultralytics.utils import (LOGGER, __version__, colorstr)
def new_predict_once(self, x, profile=False, visualize=False, embed=None):
    y, dt, embeddings = [], [], []  # outputs
    for m in self.model:
        if m.f != -1:  # if not from previous layer
            x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]  # from earlier layers
        # for export 
        if m.i == 23 :  # 为了在cat前输出
            return m(x)
        else:
            x = m(x)  # run
        y.append(x if m.i in self.save else None)  # save output
    return x
BaseModel._predict_once = new_predict_once
def new_Detect_forward(self,x):
    y = []
    for i in range(self.nl):
        t1 = self.one2one_cv2[i](x[i]) #cv2 - 64
        t2 = self.one2one_cv3[i](x[i]) #cv3 - 80
        y.append(t2)
        y.append(t1)
    return y
v10Detect.forward = new_Detect_forward
@try_export
def new_export_torchscript(self, prefix=colorstr("TorchScript:")):
    """YOLOv10 TorchScript model export."""
    LOGGER.info(f"\n{prefix} starting export with torch {torch.__version__}...")
    f = self.file.with_suffix(".torchscript")
    trace_path = TRACE_PATH # traced path
    im = torch.ones(1, 3, 640, 640, dtype = torch.float32) # dummy input size
    ts = torch.jit.trace(self.model, im, strict=False)
    extra_files = {"config.txt": json.dumps(self.metadata)}  # torch._C.ExtraFilesMap()
    if self.args.optimize:  # https://pytorch.org/tutorials/recipes/mobile_interpreter.html
        LOGGER.info(f"{prefix} optimizing for mobile...")
        from torch.utils.mobile_optimizer import optimize_for_mobile

        optimize_for_mobile(ts)._save_for_lite_interpreter(trace_path, _extra_files=extra_files)
    else:
        ts.save(trace_path, _extra_files=extra_files)
    return trace_path, None
Exporter.export_torchscript = new_export_torchscript

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='YOLOv10 PyTorch Inference.', add_help=True)
    parser.add_argument('--weights', type=str, default='../weights/yolov10n.pt', help='model path(s) for inference.')
    parser.add_argument('--save_path', type=str, default='../2_compile/fmodel/yolov10n_640x640.pt', help='model path(s) for inference.')
    args = parser.parse_args()
    weights = args.weights #权重
    TRACE_PATH = args.save_path# traced path
    # load model 
    model = YOLOv10(weights)
    
    # export traced model
    success = model.export()
    print('Model save at:',success)
      
