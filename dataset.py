import os
import random
import glob
import json
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from tqdm import tqdm

class MultiRegionSlidingWindowDataset(Dataset):
    def __init__(self, file_paths, params_filename, =4, time_stride=1, patch_size=64, patch_stride=32):
        self.patch_size = patch_size
        self.seq_len = seq_len
        self.file_paths = file_paths

        self.region_names = [os.path.basename(os.path.dirname(os.path.dirname(p))) for p in file_paths]
        self.region_data_memmaps = [np.load(path, mmap_mode='r') for path in self.file_paths]
        
        self.region_stats = []
        for path in self.file_paths:
            json_path = os.path.join(os.path.dirname(path), params_filename)
            try:
                with open(json_path, 'r') as f:
                    self.region_stats.append(json.load(f))
            except FileNotFoundError:
                print(f"error")
                self.region_stats.append(None)

        self.indices = []
        for region_idx, data in enumerate(tqdm(self.region_data_memmaps, desc="build the index")):
            T, C, H, W = data.shape
            if T <  or H < patch_size or W < patch_size:
                print(f"skip this area")
                continue
            
            time_indices = [t for t in range(0, T -  + 1, time_stride)]
            spatial_indices = [(i, j) for i in range(0, H-patch_size+1, patch_stride) for j in range(0, W-patch_size+1, patch_stride)]
            
            for t in time_indices:
                for (i, j) in spatial_indices:
                    self.indices.append((region_idx, t, i, j))

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        region_idx, t, i, j = self.indices[idx]
        data_array = self.region_data_memmaps[region_idx]
        patch_seq = data_array[t:t+self., :, i:i+self.patch_size, j:j+self.patch_size]
        
        return {
            'data': torch.tensor(np.copy(patch_seq), dtype=torch.float32),
            'region_idx': region_idx,
            'index': idx,
            'row': i,
            'col': j
        }


def denormalize_batch(batch_data, batch_region_indices, all_region_stats, channel_order, channel_to_denorm=-1):
    denorm_data = batch_data.clone()
    for i in range(batch_data.shape[0]):
        sample_data_channel = batch_data[i, channel_to_denorm]
        region_idx = batch_region_indices[i].item()
        stats_for_region = all_region_stats[region_idx]
        target_channel_name = channel_order[channel_to_denorm]
        mean = stats_for_region[target_channel_name]['mean_after_cleaning']
        std = stats_for_region[target_channel_name]['std_after_cleaning']
        denorm_channel = (sample_data_channel * std) + mean
        denorm_data[i, channel_to_denorm] = denorm_channel
    return denorm_data

def create_split_dataloaders(
    root_path, data_subfolder, data_filename, params_filename,
    train_regions, dataset_params, batch_size,seed=42
):
    def get_paths_from_regions(regions_list):
        paths = []
        for region_name in regions_list:
            path = os.path.join(root_path, region_name, data_subfolder, data_filename)
            if os.path.exists(path):
                paths.append(path)
            else:
                print(f"skip this area")
        return paths

    train_val_files = get_paths_from_regions(train_regions)
    
    train_loader, val_loader, train_val_dataset = None, None, None
    if train_val_files:
        train_val_dataset = MultiRegionSlidingWindowDataset(train_val_files, params_filename, **dataset_params)
        dataset_size = len(train_val_dataset)
        indices = list(range(dataset_size))

        train_ratio = 0.7
        val_ratio = 0.2
        test_ratio = 0.1

        random.seed(seed)
        random.shuffle(indices)

        train_end = int(train_ratio * dataset_size)
        val_end = int((train_ratio + val_ratio) * dataset_size)

        train_indices = indices[:train_end]
        val_indices = indices[train_end:val_end]
        test_indices_internal = indices[val_end:]

        if train_ratio > 0:
            train_dataset = Subset(train_val_dataset, train_indices)
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=8)

        if val_ratio > 0:
            val_dataset = Subset(train_val_dataset, val_indices)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=8)

        test_dataset_internal = Subset(train_val_dataset, test_indices_internal)
        test_loader = DataLoader(test_dataset_internal, batch_size=batch_size, shuffle=False, num_workers=8)

    return train_loader, val_loader, test_loader, train_val_dataset
