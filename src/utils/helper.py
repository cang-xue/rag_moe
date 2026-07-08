from torch.utils.data import ConcatDataset, DataLoader, TensorDataset
import torch
from torch import Tensor
import logging
import numpy as np
import pandas as pd
import os
import sys
import pickle
import random
import torch_geometric
from src.utils.scaler import StandardScaler


def get_dataloader(datapath, batch_size, input_dim, output_dim, mode='train', include_metadata=False, return_time_meta=False):
    data = {}
    processed = {}
    results = {}
    raw_meta = {}

    for category in ['train', 'val', 'test']:
        cat_data = np.load(os.path.join(datapath, category + '.npz'))
        data['x_' + category] = cat_data['x']
        data['y_' + category] = cat_data['y']
        if include_metadata or return_time_meta:
            raw_meta[category] = {key: cat_data[key] for key in cat_data.files if key not in ('x', 'y')}

    meta_keys = _legacy_meta_keys(raw_meta) if return_time_meta else (_metadata_keys(raw_meta) if include_metadata else [])

    # we use different the scalers for each node
    scalers = []
    for i in range(data['x_train'].shape[2]):
        scalers.append(StandardScaler(mean=data['x_train'][:, :, i, 0].mean(),
                                      std=data['x_train'][:, :, i, 0].std()))

    # Normalize each node
    for category in ['train', 'val', 'test']:
        for i in range(data['x_train'].shape[2]):
            data['x_' + category][:, :, i, :1] = scalers[i].transform(data['x_' + category][:, :, i, :1])
            data['y_' + category][:, :, i, :1] = scalers[i].transform(data['y_' + category][:, :, i, :1])

        new_x = Tensor(data['x_' + category])[..., :input_dim]
        new_y = Tensor(data['y_' + category])[..., :output_dim]

        fields = [new_x, new_y]
        for meta_key in meta_keys:
            fields.append(_metadata_tensor(raw_meta[category], meta_key, len(new_x), new_x.shape[1]))
        processed[category] = TensorDataset(*fields)

    results['train_loader'] = DataLoader(processed['train'], batch_size, shuffle=True)
    results['val_loader'] = DataLoader(processed['val'], batch_size, shuffle=False)
    results['test_loader'] = DataLoader(processed['test'], batch_size, shuffle=False)
    results['batch_meta_keys'] = meta_keys
    for loader in (results['train_loader'], results['val_loader'], results['test_loader']):
        loader.batch_meta_keys = meta_keys

    print('train: {}\t valid: {}\t test:{}'.format(len(results['train_loader'].dataset),
                                                   len(results['val_loader'].dataset),
                                                   len(results['test_loader'].dataset)))
    results['scalers'] = scalers
    return results


def reassign_train_val_to_source_test(data, batch_size):
    updated = dict(data)
    updated['train_loader'] = DataLoader(
        ConcatDataset([data['train_loader'].dataset, data['val_loader'].dataset]),
        batch_size,
        shuffle=True,
    )
    updated['val_loader'] = DataLoader(data['test_loader'].dataset, batch_size, shuffle=False)
    for loader in (updated['train_loader'], updated['val_loader'], updated['test_loader']):
        loader.batch_meta_keys = data.get('batch_meta_keys', [])
    return updated


def split_batch(batch, meta_keys=None):
    X, label = batch[0], batch[1]
    meta_keys = list(meta_keys or _infer_batch_meta_keys(batch))
    batch_meta = {key: value for key, value in zip(meta_keys, batch[2:])}
    return X, label, batch_meta


def move_batch_meta(batch_meta, device):
    moved = {}
    for key, value in (batch_meta or {}).items():
        moved[key] = value.to(device) if hasattr(value, 'to') else value
    return moved


def _metadata_keys(raw_meta):
    if not raw_meta:
        return []
    keys = ['sample_ids']
    if _all_categories_have(raw_meta, ('x_hour',)):
        keys.append('x_hour')
    if _all_categories_have(raw_meta, ('x_minute',)):
        keys.append('x_minute')
    if _all_categories_have(raw_meta, ('x_weekday', 'x_dow', 'x_dayofweek')):
        keys.append('x_weekday')
    return keys


def _legacy_meta_keys(raw_meta):
    if not raw_meta:
        return []
    keys = []
    if _all_categories_have(raw_meta, ('x_hour',)):
        keys.append('x_hour')
    if _all_categories_have(raw_meta, ('x_minute',)):
        keys.append('x_minute')
    if _all_categories_have(raw_meta, ('x_weekday', 'x_dow', 'x_dayofweek')):
        keys.append('x_weekday')
    keys.append('sample_ids')
    return keys


def _infer_batch_meta_keys(batch):
    if len(batch) == 5:
        return ['x_hour', 'x_minute', 'sample_ids']
    if len(batch) == 6:
        return ['x_hour', 'x_minute', 'x_weekday', 'sample_ids']
    return []


def _all_categories_have(raw_meta, aliases):
    return all(any(alias in values for alias in aliases) for values in raw_meta.values())


def _metadata_tensor(category_meta, meta_key, length, seq_len):
    aliases = {
        'sample_ids': ('sample_ids', 'sample_idx', 'rag_index'),
        'x_hour': ('x_hour',),
        'x_minute': ('x_minute',),
        'x_weekday': ('x_weekday', 'x_dow', 'x_dayofweek'),
    }[meta_key]
    for alias in aliases:
        if alias in category_meta:
            return torch.as_tensor(category_meta[alias], dtype=torch.long)
    if meta_key == 'sample_ids':
        return torch.arange(length, dtype=torch.long)
    return torch.full((length, seq_len), -1, dtype=torch.long)


def check_device(device=None):
    if device is None:
        print("`device` is missing, try to train and evaluate the model on default device.")
        if torch.cuda.is_available():
            print("cuda device is available, place the model on the device.")
            return torch.device("cuda")
        else:
            print("cuda device is not available, place the model on cpu.")
            return torch.device("cpu")
    else:
        if isinstance(device, torch.device):
            return device
        else:
            return torch.device(device)


def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)


def get_num_nodes(dataset):
    print(dataset)
    d = {'Delivery_SH': 30,
        'Delivery_HZ': 31,
        'Delivery_CQ': 30,
        'Delivery_YT': 30,
        'Delivery_JL': 14,
        'Delivery_LA': 17,
        'Delivery_NY': 22,
        'Delivery_SF': 19,
        'Delivery_SH_kriging': 30,
        'Delivery_HZ_kriging': 31,
        'Delivery_CQ_kriging': 30,
        'Delivery_SH_long': 30,
        'Delivery_HZ_long': 31,
         }
    assert dataset in d.keys()
    return d[dataset]


def get_null_value(dataset):
    d = {'Delivery': -1.0}
    assert dataset[:8] in d.keys()
    return d[dataset[:8]]
