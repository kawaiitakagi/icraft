"""
@Description: Main script for exporting torchscript model
"""

# %% Import packages
from datetime import datetime
import torch 
# 需要修改network.py
from network import PointNet
import sys 
sys.path.append(R'../0_PointNet')
from logger import logging
from utils import (
    get_configurations,
    get_device, 
)

def main(config_path,save_path) -> None:
    datetime_now = datetime.now().strftime('%Y%m%d-%H%M%S')
    # %% set configurations
    configs = get_configurations(
        config_yaml_path=config_path,
        datetime_now=datetime_now
    )
    device = get_device(compute_device=configs['computing_device'])

    logging.info(
        f"Selected Computing Device:\t       {configs['computing_device']}\n"
        "\n"

        f"TEST INFO:\n"
        f"\tTrained Model Path:            {configs['single_test']['trained_model_path']}\n"
        f"\tTest Data Path:                {configs['single_test']['test_data_path']}\n"
        "\n"
    )

    # %% what kinds of classes do we have?

    # %% let's test the trained network
    logging.info("Exporting started...")
    num_classes = 40
    # load network
    pointnet = PointNet(num_classes=num_classes, device=device)
    loaded_model = torch.load(configs['single_test']['trained_model_path'], map_location=device)
    if 'epoch' in loaded_model.keys():
            epochs_trained = loaded_model['epoch']
    # network state
    if 'network_state_dict' in loaded_model.keys():
        pointnet.load_state_dict(loaded_model['network_state_dict'])
    else:
        pointnet.load_state_dict(loaded_model)
    pointnet.to(device)
    # prepare dummy input
    dummy_input = torch.randn(1,3,1024,dtype=torch.float32)

    network = pointnet
    network.eval()
    _output, __, __ = network(dummy_input)
    print("output shape=",_output.shape)
    # export traced model
    script_model = torch.jit.trace(network,dummy_input)
    
    torch.jit.save(script_model,save_path)
    print('Traced Model save at:',save_path)


if __name__ == '__main__':
    YAML_CONFIG_PATH = "./configs_1.yaml" # config path 
    save_path = "../2_compile/fmodel/pointnet_traced_3x1024.pt"# traced model save path 
    main(YAML_CONFIG_PATH,save_path)
