import os
import json
import numpy as np
import torch
from tqdm import tqdm
from sklearn.metrics import r2_score

from dataset import create_split_dataloaders
from SmaAt_UNet import SmaAt_UNet

def set_seed(seed, deterministic=False):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.benchmark = True

def test_model(model, loader):
    model.eval()
    all_preds, all_trues, all_indices = [], [], []
    with torch.no_grad():
        for batch_dict in tqdm(loader, desc=f"Evaluating on {loader.dataset.dataset.__class__.__name__}"):
            batch = batch_dict['data']
            region_indices = batch_dict['region_idx']
            input_frames = batch[:, 0:-1, :, :, :]
            target_frame = batch[:, -1, :, :, :]
            B, T, C, H, W = input_frames.shape
            input_frames_reshaped = input_frames.view(B, T * C, H, W)
            xb, yb = input_frames_reshaped, target_frame  
            pred = model(xb)   
            all_preds.append(pred.detach().cpu().numpy())
            all_trues.append(yb.detach().cpu().numpy())
            all_indices.append(region_indices.cpu().numpy())
    final_preds = np.concatenate(all_preds, axis=0)
    final_trues = np.concatenate(all_trues, axis=0)
    final_indices = np.concatenate(all_indices, axis=0)
    return final_preds, final_trues, final_indices

def denormalize_numpy_batch(batch_data, batch_region_indices, all_region_stats, channel_order):
    denorm_data = np.copy(batch_data)
    for i in range(batch_data.shape[0]):
        region_idx = batch_region_indices[i]
        stats_for_region = all_region_stats[region_idx]
        for c, channel_name in enumerate(channel_order):
            mean = stats_for_region[channel_name]['mean_after_cleaning']
            std = stats_for_region[channel_name]['std_after_cleaning']
            denorm_data[i, c, :, :] = (batch_data[i, c, :, :] * std) + mean
    return denorm_data

if __name__ == "__main__":
    MODEL_CHECKPOINT_PATH = "best_model.pt"
    SAVE_DIR = "convL_test" 
    os.makedirs(SAVE_DIR, exist_ok=True)
    set_seed(42)

    ROOT_PATH = "NPP"
    DATA_SUBFOLDER = "factor_prj"
    DATA_FILENAME = "combined_standardized_final.npy"
    PARAMS_FILENAME = "channel_statistics_final.json"

    TRAIN_REGIONS = ['01Jiuzhaigou', '02Milin','03Ludian','04NewGuinea', '05Palu', '06Kaikoura','07Chile', '08Iwate','09Kashmir', '10wenchuan']

    CHANNEL_ORDER = ['Elevation', 'Slope', 'ET', 'FPAR', 'LAI', 'PGA', 'Precipitation', 'SoilMoisture', 'Temperature', 'NPP'] 
    DATASET_PARAMS = { "seq_len": 4, "time_stride": 1, "patch_size": 64, "patch_stride": 32 }
    BATCH_SIZE = 8

    _, _, test_loader, train_val_dataset = create_split_dataloaders(
        ROOT_PATH, DATA_SUBFOLDER, DATA_FILENAME, PARAMS_FILENAME,
        TRAIN_REGIONS,
        DATASET_PARAMS, BATCH_SIZE
    )
        
    all_val_region_stats = train_val_dataset.region_stats

    input_timesteps = DATASET_PARAMS["seq_len"] - 1
    original_channels = len(CHANNEL_ORDER)
    in_channels = input_timesteps * original_channels
    out_channels = original_channels
    model = SmaAt_UNet(in_channels, out_channels)
    checkpoint = torch.load(MODEL_CHECKPOINT_PATH)
    model.load_state_dict(checkpoint['model_state_dict'])

    predicted_data_norm, true_data_norm, region_indices = test_model(model, test_loader)
    predicted_data_denorm = denormalize_numpy_batch(predicted_data_norm, region_indices, all_val_region_stats, CHANNEL_ORDER)
    true_data_denorm = denormalize_numpy_batch(true_data_norm, region_indices, all_val_region_stats, CHANNEL_ORDER)

    preds_save_path = os.path.join(SAVE_DIR, 'preds.npy')
    trues_save_path = os.path.join(SAVE_DIR, 'trues.npy')
    
    np.save(preds_save_path, predicted_data_denorm)
    np.save(trues_save_path, true_data_denorm)

    preds_npp_denorm = predicted_data_denorm[:, -1, :, :].flatten()
    trues_npp_denorm = true_data_denorm[:, -1, :, :].flatten()

    r2 = r2_score(trues_npp_denorm, preds_npp_denorm)
