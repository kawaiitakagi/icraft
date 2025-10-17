WEIGHTS_PATH = "../weights/yolov8s-cls.pt"
TRACED_MODEL_PATH = "../2_compile/fmodel/yolov8s-cls-224x224.pt"

import sys
sys.path.append(R"../0_yolov8_cls")
import torch
import json
from ultralytics.engine.exporter import Exporter ,try_export
from ultralytics.utils import (LOGGER, __version__, colorstr)

# ## When you don't want softmax op ##
# from ultralytics.nn.modules.head import Classify
# def new_Classify_forward(self, x):
#     """Performs a forward pass without softmax."""
#     if isinstance(x, list):
#         x = torch.cat(x, 1)
#     x = self.linear(self.drop(self.pool(self.conv(x)).flatten(1)))
#     return x 
# Classify.forward = new_Classify_forward

@try_export
def new_export_torchscript(self, prefix=colorstr('TorchScript:')):
    """YOLOv8 TorchScript model export."""
    LOGGER.info(f'\n{prefix} starting export with torch {torch.__version__}...')
    f = self.file.with_suffix('.torchscript')
    im = torch.ones(1, 3, 224, 224, dtype = torch.float32)
    ts = torch.jit.trace(self.model.cpu(), im, strict=False)
    extra_files = {'config.txt': json.dumps(self.metadata)}  # torch._C.ExtraFilesMap()
    if self.args.optimize:  # https://pytorch.org/tutorials/recipes/mobile_interpreter.html
        LOGGER.info(f'{prefix} optimizing for mobile...')
        from torch.utils.mobile_optimizer import optimize_for_mobile
        optimize_for_mobile(ts)._save_for_lite_interpreter(str(f), _extra_files=extra_files)
    else:
        ts.save(TRACED_MODEL_PATH, _extra_files=extra_files)
    return f, None
Exporter.export_torchscript = new_export_torchscript
from ultralytics import YOLO
import optparse
if __name__ == "__main__":
    parser4pred = optparse.OptionParser()
    parser4pred.add_option('--format', type=str, default="TorchScript", help='format to export to')
    parser4pred.add_option('--model', type=str, default=WEIGHTS_PATH, help='path to model file')
    options, args  = parser4pred.parse_args()
    options = eval(str(options))
    model = YOLO(model=options['model']) 
    success = model.export(**options)  # export the model 