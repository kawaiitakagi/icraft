"""
@Description: Main script for testing a single point cloud data
"""

# %% Import packages
from datetime import datetime
import sys 
sys.path.append(R'../0_PointNet')
from logger import logging
from utils import (
    pointnet_model,
    get_configurations,
    get_device,
    single_data_test
)


def main() -> None:
    datetime_now = datetime.now().strftime('%Y%m%d-%H%M%S')

    # %% set configurations
    YAML_CONFIG_PATH = "./configs_1.yaml"
    configs = get_configurations(
        config_yaml_path=YAML_CONFIG_PATH,
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
    classes= {'airplane': 0, 'bathtub': 1, 'bed': 2, 'bench': 3, 'bookshelf': 4, 'bottle': 5, 'bowl': 6, 'car': 7, 'chair': 8, 'cone': 9, 'cup': 10, 'curtain': 11, 'desk': 12, 'door': 13, 'dresser': 14, 'flower_pot': 15, 'glass_box': 16, 'guitar': 17, 'keyboard': 18, 'lamp': 19, 'laptop': 20, 'mantel': 21, 'monitor': 22, 'night_stand': 23, 'person': 24, 'piano': 25, 'plant': 26, 'radio': 27, 'range_hood': 28, 'sink': 29, 'sofa': 30, 'stairs': 31, 'stool': 32, 'table': 33, 'tent': 34, 'toilet': 35, 'tv_stand': 36, 'vase': 37, 'wardrobe': 38, 'xbox': 39}
    
    logging.info(f"Classes: {classes}")

    # %% let's test the trained network
    logging.info("Testing started...")
    num_classes = 40
    pointnet, optimizer, epochs_trained = pointnet_model(
        num_classes=num_classes,
        device=device,
        mode='test',
        model=configs['single_test']['trained_model_path']
    )
    pred = single_data_test(
        network=pointnet,
        classes=classes,
        data_path=configs['single_test']['test_data_path'],
        device=device
    )
    print(f"Prediction: {pred}")
    logging.info("Testing DONE!")


if __name__ == '__main__':
    main()
