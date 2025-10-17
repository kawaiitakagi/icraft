WEIGHTS_PATH = "../weights/yolo11s.pt"
TRACED_MODEL_PATH = "../2_compile/fmodel/yolo11s_640x640.pt"
import sys
sys.path.append(R"../0_yolo11")
from ultralytics.utils import torch_utils
from ultralytics.utils.torch_utils import TORCH_1_9,TORCH_2_0
import torch
RED = '\033[31m'       # 设置前景色为红色
RESET = '\033[0m'      # 重置所有属性到默认值
ver = torch.__version__
assert ("1.6" in ver) or ("1.9" in ver), f"{RED}Unsupported PyTorch version: {ver}{RESET}"
# bug1 :Inplace update to inference tensor outside InferenceMode is not allowed.
def new_smart_inference_mode():
    """Applies torch.inference_mode() decorator if torch>=1.9.0 else torch.no_grad() decorator."""

    def decorate(fn):
        """Applies appropriate torch decorator for inference mode based on torch version."""
        if TORCH_2_0 and torch.is_inference_mode_enabled():
            return fn  # already in inference_mode, act as a pass-through
        else:
            return (torch.inference_mode if TORCH_2_0 else torch.no_grad)()(fn)

    return decorate
torch_utils.smart_inference_mode  = new_smart_inference_mode

from ultralytics.nn.modules.head import Detect
import torch
from ultralytics.utils.tal import dist2bbox, make_anchors
def new_Detect_forward(self, x):
    """Concatenates and returns predicted bounding boxes and class probabilities."""
    cls_dfl_head = []
    for i in range(self.nl):
        # (self.cv3[i](x[i]) 是 类别头输出
        # (self.cv2[i](x[i]) 是 检测头输出
        cls_dfl_head.append(self.cv3[i](x[i]))
        cls_dfl_head.append(self.cv2[i](x[i]))
        # x[i] = torch.cat((self.cv2[i](x[i]), self.cv3[i](x[i])), 1)
    return cls_dfl_head
Detect.forward = new_Detect_forward

from ultralytics.nn.tasks import BaseModel
from ultralytics.utils.plotting import feature_visualization
def new_predict_once(self, x, profile=False, visualize=False, embed=None):
    """
    Perform a forward pass through the network.

    Args:
        x (torch.Tensor): The input tensor to the model.
        profile (bool):  Print the computation time of each layer if True, defaults to False.
        visualize (bool): Save the feature maps of the model if True, defaults to False.

    Returns:
        (torch.Tensor): The last output of the model.
    """
    y, dt = [], []  # outputs
    for m in self.model:
        if m.f != -1:  # if not from previous layer
            x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]  # from earlier layers
        if profile:
            self._profile_one_layer(m, x, dt)
        if m.i == 23 :  # 为了在cat前输出
            return m(x)
        else:
            x = m(x)  # run
        y.append(x if m.i in self.save else None)  # save output
        if visualize:
            feature_visualization(x, m.type, m.i, save_dir=visualize)
    return x
BaseModel._predict_once = new_predict_once
import json
from ultralytics.engine.exporter import Exporter ,try_export
from ultralytics.utils import (LOGGER, __version__, colorstr)
@try_export
def new_export_torchscript(self, prefix=colorstr('TorchScript:')):
    """YOLO11 TorchScript model export."""
    LOGGER.info(f'\n{prefix} starting export with torch {torch.__version__}...')
    f = self.file.with_suffix('.torchscript')
    im = torch.ones(1, 3, 640, 640, dtype = torch.float32)
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
    success = model.export(**options)  # export the model to ONNX format
