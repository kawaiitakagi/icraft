# -*- coding: utf-8 -*-
"""
.. codeauthor:: Mona Koehler <mona.koehler@tu-ilmenau.de>
.. codeauthor:: Daniel Seichter <daniel.seichter@tu-ilmenau.de>
"""
import copy
import os
import pickle
from torch.utils.data import DataLoader

from src import preprocessing
from src.datasets import SUNRGBD


def prepare_data(dataset,ckpt_dir=None, with_input_orig=False, split=None):
    train_preprocessor_kwargs = {}
    if dataset == 'sunrgbd':
        Dataset = SUNRGBD
        dataset_kwargs = {}
        valid_set = 'test'
    else:
        raise ValueError(f"Unknown dataset: `{dataset}`")


    if split in ['valid', 'test']:
        valid_set = split

    depth_mode = 'refined'

    # train data
    train_data = Dataset(
        data_dir=None,
        split='train',
        depth_mode=depth_mode,
        with_input_orig=with_input_orig,
        **dataset_kwargs
    )

    train_preprocessor = preprocessing.get_preprocessor(
        height=480,
        width=640,
        depth_mean=train_data.depth_mean,
        depth_std=train_data.depth_std,
        depth_mode=depth_mode,
        phase='train',
        **train_preprocessor_kwargs
    )

    train_data.preprocessor = train_preprocessor

    if ckpt_dir is not None:
        pickle_file_path = os.path.join(ckpt_dir, 'depth_mean_std.pickle')
        if os.path.exists(pickle_file_path):
            with open(pickle_file_path, 'rb') as f:
                depth_stats = pickle.load(f)
            print(f'Loaded depth mean and std from {pickle_file_path}')
            print(depth_stats)
        else:
            # dump depth stats
            depth_stats = {'mean': train_data.depth_mean,
                           'std': train_data.depth_std}
            with open(pickle_file_path, 'wb') as f:
                pickle.dump(depth_stats, f)
    else:
        depth_stats = {'mean': train_data.depth_mean,
                       'std': train_data.depth_std}

    # valid data
    valid_preprocessor = preprocessing.get_preprocessor(
        height=480,
        width=640,
        depth_mean=depth_stats['mean'],
        depth_std=depth_stats['std'],
        depth_mode=depth_mode,
        phase='test'
    )
    valid_full_res = False
    if valid_full_res:
        valid_preprocessor_full_res = preprocessing.get_preprocessor(
            depth_mean=depth_stats['mean'],
            depth_std=depth_stats['std'],
            depth_mode=depth_mode,
            phase='test'
        )
    valid_data = Dataset(
        data_dir=None,
        split=valid_set,
        depth_mode=depth_mode,
        with_input_orig=with_input_orig,
        **dataset_kwargs
    )

    valid_data.preprocessor = valid_preprocessor

    dataset_dir = None

    if dataset_dir is None:
        # no path to the actual data was passed -> we cannot create dataloader,
        # return the valid dataset and preprocessor object for inference only
        if valid_full_res:
            return valid_data, valid_preprocessor_full_res
        else:
            return valid_data, valid_preprocessor

    # create the data loaders
    train_loader = DataLoader(train_data,
                              batch_size=8,
                              num_workers=8,
                              drop_last=True,
                              shuffle=True)

    # for validation we can use higher batch size as activations do not
    # need to be saved for the backwards pass
    batch_size_valid = None or 8
    print('batch_size_valid',batch_size_valid)
    
    valid_loader = DataLoader(valid_data,
                              batch_size=batch_size_valid,
                              num_workers=8,
                              shuffle=False)
    if valid_full_res:
        valid_loader_full_res = copy.deepcopy(valid_loader)
        valid_loader_full_res.dataset.preprocessor = valid_preprocessor_full_res
        return train_loader, valid_loader, valid_loader_full_res

    return train_loader, valid_loader
