# -*-coding:utf-8-*-
import os
import random
import time
import json
import numpy as np
import torch
from torch import optim, nn
from tqdm import tqdm
from matplotlib import pyplot as plt
from sklearn.metrics import r2_score

from dataset import create_split_dataloaders
from SmaAt_UNet import SmaAt_UNet


def set_seed(seed, deterministic=False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.benchmark = True

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def loss_func(y_pred, y_true):
    return nn.functional.mse_loss(y_pred, y_true)

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

def plot_prediction_vs_truth(pred, truth, save_path=None, show=False, title=None):
    pred = pred.squeeze()
    truth = truth.squeeze()
    if pred.ndim > 2: pred = pred[-1]
    if truth.ndim > 2: truth = truth[-1]
    v_abs_max = np.max([np.abs(truth), np.abs(pred)])
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    im1 = axes[0].imshow(truth, cmap='coolwarm', vmin=-v_abs_max, vmax=v_abs_max)
    axes[0].set_title("Ground Truth (NPP)")
    fig.colorbar(im1, ax=axes[0], orientation='vertical', shrink=0.8)
    im2 = axes[1].imshow(pred, cmap='coolwarm', vmin=-v_abs_max, vmax=v_abs_max)
    axes[1].set_title("Prediction (NPP)")
    fig.colorbar(im2, ax=axes[1], orientation='vertical', shrink=0.8)
    for ax in axes: ax.axis('off')
    if title: plt.suptitle(title)
    if save_path:
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(save_path, dpi=150)
        plt.close()
    elif show:
        plt.show()

def validate(model, val_loader, loss_func):
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_trues = []
    all_indices = []

    with torch.no_grad():
        for batch_dict in val_loader:
            batch = batch_dict['data']
            region_indices = batch_dict['region_idx']
            
            input_frames = batch[:, 0:-1, :, :, :]
            target_frame = batch[:, -1, :, :, :]
            B, T, C, H, W = input_frames.shape
            input_frames_reshaped = input_frames.view(B, T * C, H, W)
            xb, yb = input_frames_reshaped, target_frame
            y_pred = model(xb)
            loss = loss_func(y_pred, yb)
            total_loss += loss.item()

            all_preds.append(y_pred.detach().cpu())
            all_trues.append(yb.detach().cpu())
            all_indices.append(region_indices.cpu())

    all_preds = torch.cat(all_preds, dim=0)
    all_trues = torch.cat(all_trues, dim=0)
    all_indices = torch.cat(all_indices, dim=0)

    return total_loss / len(val_loader), all_preds, all_trues, all_indices


def fit(epochs, model, loss_func, opt, train_dl, valid_dl, all_val_stats, channel_order, 
        save_every: int = None, earlystopping: int = None, lr_scheduler=None):
    
    train_losses = []
    val_losses = []
    start_time = time.time()
    best_val_loss = float('inf')
    earlystopping_counter = 0
    save_path = "convL"
    weights_save_path = os.path.join(save_path, 'weights')
    vis_save_path = os.path.join(save_path, 'vis')
    os.makedirs(weights_save_path, exist_ok=True)
    os.makedirs(vis_save_path, exist_ok=True)

    for epoch in tqdm(range(epochs), desc="Epochs", leave=True):
        model.train()
        train_loss = 0.0
        for batch_dict in train_dl:
            batch = batch_dict['data']
            input_frames = batch[:, 0:-1, :, :, :]
            target_frame = batch[:, -1, :, :, :]
            B, T, C, H, W = input_frames.shape
            input_frames_reshaped = input_frames.view(B, T * C, H, W)
            xb, yb = input_frames_reshaped, target_frame
            pred = model(xb)
            loss = loss_func(pred, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            train_loss += loss.item()
        train_loss /= len(train_dl)
        train_losses.append(train_loss)
        
        val_loss, preds_all_val, trues_all_val, indices_all_val = validate(model, valid_dl, loss_func)
        val_losses.append(val_loss)

        preds_denorm = denormalize_batch(preds_all_val, indices_all_val, all_val_stats, channel_order, -1)
        trues_denorm = denormalize_batch(trues_all_val, indices_all_val, all_val_stats, channel_order, -1)

        preds_npp = preds_denorm[:, -1, :, :].cpu().numpy().flatten()
        trues_npp = trues_denorm[:, -1, :, :].cpu().numpy().flatten()

        val_r2 = r2_score(trues_npp, preds_npp)
        print(f"r2: {val_r2:.4f}")

        num_vis_samples = 5
        vis_preds_norm = preds_all_val[:num_vis_samples]
        vis_trues_norm = trues_all_val[:num_vis_samples]
        vis_indices = indices_all_val[:num_vis_samples]
        
        vis_preds_denorm = denormalize_batch(vis_preds_norm, vis_indices, all_val_stats, channel_order, -1)
        vis_trues_denorm = denormalize_batch(vis_trues_norm, vis_indices, all_val_stats, channel_order, -1)

        for i in range(len(vis_preds_denorm)):
            vis_pred = vis_preds_denorm[i].numpy()
            vis_true = vis_trues_denorm[i].numpy()
            vis_img_path = os.path.join(vis_save_path, f"epoch_{epoch+1}_sample_{i}.png")
            plot_prediction_vs_truth(vis_pred, vis_true, save_path=vis_img_path,
                                     title=f"Epoch {epoch+1} | Sample {i} | Val Loss: {val_loss:.4f} (Denormalized)")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            earlystopping_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': opt.state_dict(),
                'val_loss': val_loss,
            }, os.path.join(weights_save_path, f"best_model.pt"))
            print(f"Epoch {epoch+1}: New best model saved with Val Loss: {val_loss:.6f}")
        else:
            if earlystopping is not None:
                earlystopping_counter += 1
                if earlystopping_counter >= earlystopping:
                    print(f"Early stopping triggered: no improvement for {earlystopping} epochs.")
                    break
        
        if lr_scheduler is not None: lr_scheduler.step(val_loss)
        print(f"Epoch: {epoch+1:3d} | Time: {(time.time() - start_time)/60:.2f} min | "
              f"Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f} | "
              f"LR: {get_lr(opt):.6f} | Patience: {earlystopping_counter}/{earlystopping if earlystopping else '-'}")


if __name__ == "__main__":
    set_seed(42)
    ROOT_PATH = "NPP"
    DATA_SUBFOLDER = "factor_prj"
    DATA_FILENAME = "combined_standardized_final.npy"
    PARAMS_FILENAME = "channel_statistics_final.json"

    TRAIN_REGIONS = ['01Jiuzhaigou', '02Milin','03Ludian','04NewGuinea', '05Palu', '06Kaikoura','07Chile', '08Iwate','09Kashmir', '10wenchuan']
    
    DATASET_PARAMS = {
        "seq_len": 4, "time_stride": 1,
        "patch_size": 64, "patch_stride": 32,
    }
    BATCH_SIZE = 8

    train_loader, val_loader, _, train_val_dataset = create_split_dataloaders(
        ROOT_PATH, DATA_SUBFOLDER, DATA_FILENAME, PARAMS_FILENAME,
        TRAIN_REGIONS,
        DATASET_PARAMS, BATCH_SIZE
    )

    learning_rate = 0.001
    epochs = 200
    earlystopping = 15
    save_every = 10
    
    input_timesteps = DATASET_PARAMS["seq_len"] - 1
    sample_batch = next(iter(train_loader))
    original_channels = sample_batch['data'].shape[2]
    
    in_channels = input_timesteps * original_channels
    out_channels = original_channels
    
    model = SmaAt_UNet(in_channels, out_channels)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    opt = optim.Adam(model.parameters(), lr=learning_rate)
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=0.1, patience=10, verbose=True)
    
    all_validation_stats = train_val_dataset.region_stats
    channel_order = ['Elevation', 'Slope', 'ET', 'FPAR', 'LAI', 'PGA', 'Precipitation', 'SoilMoisture', 'Temperature', 'NPP']

    fit(epochs, model, loss_func, opt, train_loader, val_loader, 
        all_val_stats=all_validation_stats, channel_order=channel_order,
        save_every=None, earlystopping=earlystopping, lr_scheduler=lr_scheduler)
