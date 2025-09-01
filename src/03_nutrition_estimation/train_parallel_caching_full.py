#!/usr/bin/env python
# %%
import argparse
import os
import sys
import pandas as pd
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import pickle
import time
from tqdm import tqdm
from collections import Counter # For counting image types

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from torchvision import models, transforms
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler


os.environ['NCCL_DEBUG'] = 'WARN'
PROCESS_DEVICE = None

def setup_distributed(backend='nccl'):
    """Initializes the distributed environment."""
    global PROCESS_DEVICE
    if "RANK" not in os.environ or "WORLD_SIZE" not in os.environ or "LOCAL_RANK" not in os.environ:
        print("Distributed environment variables (RANK, WORLD_SIZE, LOCAL_RANK) not found. Running in non-distributed mode.")
        PROCESS_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return 0, 1, PROCESS_DEVICE, False

    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])

    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    PROCESS_DEVICE = device

    print(f"Initializing process group: backend={backend}, rank={rank}, world_size={world_size}, local_rank={local_rank}, device={device}")
    dist.init_process_group(backend=backend)
    return rank, world_size, device, True

def cleanup_distributed():
    if dist.is_initialized():
        dist.destroy_process_group()

def reduce_tensor(tensor, world_size):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= world_size
    return rt

def parse_arguments():
    parser = argparse.ArgumentParser(description="Train nutrition estimation models with DDP.")
    parser.add_argument('--model_name', type=str, required=True,
                        choices=['SimpleConvNet', 'DeepConvNet', 'MobileNetLike', 'ResNetFromScratch', 'ResNetPretrained'],
                        help='Name of the model to train.')
    parser.add_argument('--base_dir', type=str, default="/users/eleves-b/2023/georgii.kuznetsov/CNN_nutrition/nutrition5k", # Example, adjust
                        help='Base directory for the dataset.')
    parser.add_argument('--output_dir', type=str, default="distributed_output_sides",
                        help='Directory to save models, history, and plots.')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs.')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate.')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size *per GPU*.')
    parser.add_argument('--num_workers', type=int, default=2, help='Number of workers for DataLoader *per GPU*.')
    parser.add_argument('--save_plots', action='store_true', help='Save plots to files.')
    parser.add_argument('--backend', type=str, default='nccl', choices=['nccl', 'gloo'], help='Distributed backend.')
    parser.add_argument('--enable_gpu_caching', action='store_true',
                        help='Enable caching of transformed images to GPU VRAM (best with num_workers=0).')

    # --- NEW: Arguments for side angles ---
    parser.add_argument('--include_side_angles', action='store_true',
                        help='Include side angle images in the dataset if available.')
    parser.add_argument('--num_side_angles_per_dish', type=int, default=20,
                        help='Max number of random side angle frames to use per dish if --include_side_angles is set. Set to 0 to use all available.')

    if any('jupyter' in arg for arg in sys.argv) or 'ipykernel_launcher.py' in sys.argv[0]:
        print("Running in interactive mode (e.g., Jupyter). Using default args for non-distributed test.")
        args = parser.parse_args([
            '--model_name', 'SimpleConvNet',
            '--epochs', '2',
            '--output_dir', 'interactive_test_output_ddp_sides', # Changed output dir
            '--include_side_angles', # For testing side angles
            '--num_side_angles_per_dish', '2', # For testing
            # '--enable_gpu_caching' # Optional for interactive
            # '--save_plots'
        ])
        current_rank, current_world_size, current_device, is_distributed_mode = 0, 1, torch.device("cuda" if torch.cuda.is_available() else "cpu"), False
        global PROCESS_DEVICE
        PROCESS_DEVICE = current_device
    else:
        args = parser.parse_args()
        current_rank, current_world_size, current_device, is_distributed_mode = setup_distributed(backend=args.backend)

    return args, current_rank, current_world_size, current_device, is_distributed_mode


# %% Model Definitions (Keep as is: SimpleConvNet, DeepConvNet, MobileNetLike, ResNetFromScratch)
class SimpleConvNet(nn.Module):
    def __init__(self, num_outputs=4):
        super(SimpleConvNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(256 * 14 * 14, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, num_outputs)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = self.pool(F.relu(self.bn4(self.conv4(x))))
        x = x.view(x.size(0), -1)
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.dropout(F.relu(self.fc2(x)))
        x = self.fc3(x)
        return x

class DeepConvNet(nn.Module): # Shortened for brevity
    def __init__(self, num_outputs=4):
        super(DeepConvNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d(3, stride=2, padding=1)
        self.res_block1 = self._make_residual_block(64, 128)
        self.res_block2 = self._make_residual_block(128, 256)
        self.res_block3 = self._make_residual_block(256, 512)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_outputs)
    def _make_residual_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride=2, padding=1), nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1), nn.BatchNorm2d(out_channels)
        )
    def forward(self, x):
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.res_block1(x); x = self.res_block2(x); x = self.res_block3(x)
        x = self.avgpool(x); x = x.view(x.size(0), -1); x = self.fc(x)
        return x

class MobileNetLike(nn.Module): # Shortened for brevity
    def __init__(self, num_outputs=4):
        super(MobileNetLike, self).__init__()
        def depthwise_separable_conv(in_c, out_c, s=1):
            return nn.Sequential(
                nn.Conv2d(in_c, in_c, 3, stride=s, padding=1, groups=in_c), nn.BatchNorm2d(in_c), nn.ReLU(inplace=True),
                nn.Conv2d(in_c, out_c, 1), nn.BatchNorm2d(out_c), nn.ReLU(inplace=True)
            )
        self.conv1 = nn.Conv2d(3, 32, 3, stride=2, padding=1); self.bn1 = nn.BatchNorm2d(32)
        self.dw_conv2 = depthwise_separable_conv(32, 64, s=2)
        self.dw_conv3 = depthwise_separable_conv(64, 128, s=2)
        self.dw_conv4 = depthwise_separable_conv(128, 256, s=2)
        self.dw_conv5 = depthwise_separable_conv(256, 512, s=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1)); self.fc = nn.Linear(512, num_outputs)
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.dw_conv2(x); x = self.dw_conv3(x); x = self.dw_conv4(x); x = self.dw_conv5(x)
        x = self.avgpool(x); x = x.view(x.size(0), -1); x = self.fc(x)
        return x

class ResNetFromScratch(nn.Module): # Shortened for brevity
    def __init__(self, num_outputs=4, use_pretrained=False):
        super(ResNetFromScratch, self).__init__()
        weights = models.ResNet34_Weights.IMAGENET1K_V1 if use_pretrained else None
        self.backbone = models.resnet34(weights=weights)
        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Linear(num_features, 256), nn.ReLU(), nn.Dropout(0.5), nn.Linear(256, num_outputs)
        )
    def forward(self, x): return self.backbone(x)


# %% Data Parsing (parse_nutrition_csv - same as your single-GPU version)
def parse_nutrition_csv(file_path):
    dishes = []
    with open(file_path, 'r') as f:
        for line in f:
            parts = line.strip().split(',')
            if not parts[0].startswith('dish_'): continue
            dish_id = parts[0]
            try:
                dish_calories = float(parts[1]); dish_weight = float(parts[2])
                dish_fat = float(parts[3]); dish_carbs = float(parts[4]); dish_protein = float(parts[5])
            except (ValueError, IndexError): continue # Skip malformed lines
            if dish_weight == 0: continue
            dishes.append({
                'dish_id': dish_id, 'calories': dish_calories, 'weight': dish_weight,
                'fat': dish_fat, 'carbs': dish_carbs, 'protein': dish_protein,
                'calories_per_100g': (dish_calories / dish_weight) * 100,
                'fat_per_100g': (dish_fat / dish_weight) * 100,
                'carbs_per_100g': (dish_carbs / dish_weight) * 100,
                'protein_per_100g': (dish_protein / dish_weight) * 100
            })
    return pd.DataFrame(dishes)


# %% Dataset Class (Modified for item_list with full paths and separate labels_array)
class NutritionDataset(Dataset):
    def __init__(self, items_list, all_labels_array_source, transform=None,
                 enable_gpu_cache=False, device=None,
                 process_rank_for_log=0, is_distributed_for_log=False):
        self.items_list = items_list # List of dicts {'dish_id', 'label_idx', 'image_type', 'image_path'}
        self.all_labels_array_source = all_labels_array_source # Full np.array of labels
        self.transform = transform
        self.device = device
        self.process_rank_for_log = process_rank_for_log
        self.is_distributed_for_log = is_distributed_for_log

        self.enable_gpu_cache = enable_gpu_cache and torch.cuda.is_available() and self.device is not None
        self.image_cache_gpu = {}

        if self.enable_gpu_cache:
            # Each rank caches the images relevant to its own dataset instance (which is the full train/val set)
            # The sampler then selects indices from this.
            # If num_workers > 0, caching happens on-the-fly in __getitem__.
            # If num_workers == 0 (recommended for GPU caching), pre-cache here.
            # This logic assumes num_workers will be forced to 0 if GPU caching is on.
            unique_image_paths_to_cache = sorted(list(set(item['image_path'] for item in self.items_list)))

            if self.process_rank_for_log == 0 or not self.is_distributed_for_log:
                print(f"Rank {self.process_rank_for_log}: GPU Caching ENABLED. Attempting to cache {len(unique_image_paths_to_cache)} unique images to {self.device}...")
                cache_iterator = tqdm(unique_image_paths_to_cache, desc=f"Rank {self.process_rank_for_log} Caching to GPU", disable=False)
            else: # Other ranks cache silently or with minimal logging
                print(f"Rank {self.process_rank_for_log}: GPU Caching ENABLED. Attempting to cache {len(unique_image_paths_to_cache)} unique images to {self.device}...")
                cache_iterator = unique_image_paths_to_cache

            num_cached_successfully = 0
            oom_occurred = False
            for image_path in cache_iterator:
                if oom_occurred: continue
                try:
                    self._load_and_cache_to_gpu(image_path)
                    num_cached_successfully += 1
                except RuntimeError as e:
                    if "out of memory" in str(e).lower():
                        print(f"WARNING (Rank {self.process_rank_for_log}): GPU OOM caching {image_path}. Stopping caching. Cached: {num_cached_successfully}.")
                        oom_occurred = True
                        if image_path in self.image_cache_gpu: del self.image_cache_gpu[image_path] # Clean up failed cache attempt
                    else: print(f"WARNING (Rank {self.process_rank_for_log}): RuntimeError caching {image_path}: {e}. Skipping.")
                except Exception as e:
                    print(f"WARNING (Rank {self.process_rank_for_log}): Unexpected error caching {image_path}: {e}. Skipping.")

            if self.process_rank_for_log == 0 or not self.is_distributed_for_log:
                print(f"Rank {self.process_rank_for_log}: GPU Caching Summary: {num_cached_successfully}/{len(unique_image_paths_to_cache)} unique images cached.")
            if oom_occurred:
                 print(f"Rank {self.process_rank_for_log}: Caching stopped due to OOM. Some images will be loaded on-demand if accessed.")

    def _load_image_pil(self, image_path_arg): # Takes full image_path
        try:
            image = Image.open(image_path_arg).convert("RGB")
            return image
        except FileNotFoundError:
            # Potentially reduce logging frequency for non-rank 0 if it becomes too verbose
            print(f"ERROR (Rank {self.process_rank_for_log}): Image not found at {image_path_arg}. Using dummy.")
            return Image.new("RGB", (224, 224), color=(0, 0, 0))
        except Exception as e:
            print(f"ERROR (Rank {self.process_rank_for_log}): Loading image {image_path_arg} failed: {e}. Using dummy.")
            return Image.new("RGB", (224, 224), color=(0, 0, 0))

    def _load_and_cache_to_gpu(self, image_path_to_cache):
        pil_image = self._load_image_pil(image_path_to_cache)
        if self.transform:
            image_tensor = self.transform(pil_image)
        else:
            image_tensor = transforms.ToTensor()(pil_image)

        if not isinstance(image_tensor, torch.Tensor):
            raise TypeError(f"Transform must output a torch.Tensor. Got {type(image_tensor)}")

        with torch.no_grad():
            self.image_cache_gpu[image_path_to_cache] = image_tensor.to(self.device, non_blocking=True)

    def __len__(self):
        return len(self.items_list)

    def __getitem__(self, idx):
        item_data = self.items_list[idx]
        image_path = item_data['image_path']
        label_idx = item_data['label_idx']

        label_tensor = torch.tensor(self.all_labels_array_source[label_idx], dtype=torch.float32)

        if self.enable_gpu_cache and image_path in self.image_cache_gpu:
            image_tensor = self.image_cache_gpu[image_path].clone() # Clone to avoid in-place op issues
        else: # Not cached or cache miss
            if self.enable_gpu_cache and image_path not in self.image_cache_gpu: # Cache miss, load and cache now
                 if self.process_rank_for_log == 0 or not self.is_distributed_for_log: # Log only for rank 0 or non-DDP
                    print(f"Rank {self.process_rank_for_log}: Cache miss for {image_path}. Loading on-the-fly & caching.")
                 try:
                    self._load_and_cache_to_gpu(image_path)
                    image_tensor = self.image_cache_gpu[image_path].clone()
                 except Exception as e: # Fallback if caching during __getitem__ fails
                    print(f"Rank {self.process_rank_for_log}: Failed to cache {image_path} during __getitem__: {e}. Loading without cache for this item.")
                    pil_image = self._load_image_pil(image_path)
                    image_tensor = self.transform(pil_image) if self.transform else transforms.ToTensor()(pil_image)
            else: # GPU caching disabled or not applicable
                pil_image = self._load_image_pil(image_path)
                image_tensor = self.transform(pil_image) if self.transform else transforms.ToTensor()(pil_image)
        return image_tensor, label_tensor

# %% Memory Check Function (Keep as is, but note how num_images is determined later)
def check_memory_for_gpu_caching(num_images, per_image_tensor_shape, current_device_index, rank_for_log=0, is_distributed_for_log=False):
    # ... (definition from your DDP script) ...
    try:
        gpu_caching_viable = False
        if torch.cuda.is_available():
            elements_per_tensor = np.prod(per_image_tensor_shape)
            bytes_per_tensor = elements_per_tensor * 4 # float32
            total_estimated_cache_bytes = num_images * bytes_per_tensor
            total_estimated_cache_mb = total_estimated_cache_bytes / (1024 * 1024)

            props = torch.cuda.get_device_properties(current_device_index)
            total_gpu_memory_mb = props.total_memory / (1024 * 1024)
            free_mem_bytes, _ = torch.cuda.mem_get_info(current_device_index)
            free_memory_mb = free_mem_bytes / (1024*1024)
            usable_free_memory_for_cache_mb = free_memory_mb * 0.80 # Target 80% of free

            gpu_caching_viable = usable_free_memory_for_cache_mb > total_estimated_cache_mb

            if rank_for_log == 0 or not is_distributed_for_log:
                print(f"--- GPU Caching Memory Check (Rank {rank_for_log}, Device {current_device_index}) ---")
                print(f"Unique images for this dataset instance: {num_images}")
                print(f"Per image tensor shape: {per_image_tensor_shape}")
                print(f"Estimated GPU cache size needed: {total_estimated_cache_mb:.2f} MB")
                print(f"Total VRAM: {total_gpu_memory_mb:.2f} MB, Free VRAM: {free_memory_mb:.2f} MB")
                print(f"Usable for cache: {usable_free_memory_for_cache_mb:.2f} MB")
                print(f"Cache Viable: {'YES' if gpu_caching_viable else 'NO'}")
            return gpu_caching_viable
        else:
            if rank_for_log == 0 or not is_distributed_for_log: print(f"Rank {rank_for_log}: CUDA not available. GPU caching disabled.")
            return False
    except Exception as e_mem:
        if rank_for_log == 0 or not is_distributed_for_log: print(f"Rank {rank_for_log}: Error checking memory for GPU caching: {e_mem}")
        return False


# %% Training and Validation Functions (Keep as is - train_epoch, validate)
def train_epoch(model, loader, criterion, optimizer, device, current_epoch, sampler, is_distributed, rank_process):
    # ... (definition from your DDP script) ...
    model.train()
    if is_distributed and sampler is not None:
        sampler.set_epoch(current_epoch)

    total_loss_epoch = 0; batch_losses_list = []
    pbar_disabled = rank_process != 0 if is_distributed else False
    pbar = tqdm(loader, desc=f'Epoch {current_epoch+1} Training', leave=False, disable=pbar_disabled)
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad(); outputs = model(images); loss = criterion(outputs, labels)
        loss.backward(); optimizer.step()
        batch_loss_item = loss.item()
        total_loss_epoch += batch_loss_item; batch_losses_list.append(batch_loss_item)
        if rank_process == 0: pbar.set_postfix({'loss': f'{batch_loss_item:.4f}', 'avg_loss': f'{np.mean(batch_losses_list):.4f}'})
    avg_loss_epoch = total_loss_epoch / len(loader) if len(loader) > 0 else 0.0
    if is_distributed:
        loss_tensor_device = torch.tensor(avg_loss_epoch).to(device)
        return reduce_tensor(loss_tensor_device, dist.get_world_size()).item()
    return avg_loss_epoch

def validate(model, loader, criterion, device, is_distributed, rank_process, world_size_val, target_columns_val):
    # ... (definition from your DDP script) ...
    model.eval()
    total_loss_val_epoch = 0; all_predictions_val_list = []; all_labels_val_list = []
    pbar_disabled = rank_process != 0 if is_distributed else False
    pbar = tqdm(loader, desc='Validating', leave=False, disable=pbar_disabled)
    with torch.no_grad():
        for images, labels_batch_val in pbar:
            images, labels_batch_val = images.to(device), labels_batch_val.to(device)
            outputs = model(images); loss_val = criterion(outputs, labels_batch_val)
            total_loss_val_epoch += loss_val.item()
            if is_distributed:
                outputs_cont = outputs.contiguous(); labels_batch_val_cont = labels_batch_val.contiguous()
                pred_list_gathered = [torch.zeros_like(outputs_cont) for _ in range(world_size_val)]
                label_list_gathered = [torch.zeros_like(labels_batch_val_cont) for _ in range(world_size_val)]
                dist.all_gather(pred_list_gathered, outputs_cont); dist.all_gather(label_list_gathered, labels_batch_val_cont)
                if rank_process == 0:
                    all_predictions_val_list.extend([p.cpu().numpy() for p in pred_list_gathered])
                    all_labels_val_list.extend([l.cpu().numpy() for l in label_list_gathered])
            else:
                all_predictions_val_list.append(outputs.cpu().numpy()); all_labels_val_list.append(labels_batch_val.cpu().numpy())
            if rank_process == 0: pbar.set_postfix({'loss': f'{loss_val.item():.4f}'})
    avg_loss_val_epoch = total_loss_val_epoch / len(loader) if len(loader) > 0 else 0.0
    percentage_errors_val_agg = {}; predictions_np_val = np.array([]); labels_np_val = np.array([])
    if rank_process == 0 and all_predictions_val_list:
        predictions_np_val = np.concatenate([arr for arr in all_predictions_val_list if arr.size > 0])
        labels_np_val = np.concatenate([arr for arr in all_labels_val_list if arr.size > 0])
        if labels_np_val.ndim == 2 and labels_np_val.shape[1] == len(target_columns_val):
            for i, col_name in enumerate(target_columns_val):
                mae_val = mean_absolute_error(labels_np_val[:, i], predictions_np_val[:, i])
                mean_true_val = labels_np_val[:, i].mean()
                percentage_errors_val_agg[col_name] = (mae_val / (mean_true_val + 1e-9)) * 100
        # ... (handle 1D case if needed) ...
    if is_distributed:
        loss_val_tensor_device = torch.tensor(avg_loss_val_epoch).to(device)
        return reduce_tensor(loss_val_tensor_device, world_size_val).item(), percentage_errors_val_agg, predictions_np_val, labels_np_val
    return avg_loss_val_epoch, percentage_errors_val_agg, predictions_np_val, labels_np_val

# %% Metrics and Plotting Helper Functions (from your single-GPU script, for Rank 0)
def calculate_metrics(df, target_cols_list, per_100g=True):
    metrics_list = []
    for nutrient_col in target_cols_list:
        true_col_suffix = '_true'; pred_col_suffix = '_pred'
        base_nutrient_name = nutrient_col # e.g. 'calories_per_100g'
        if not per_100g: # For absolute values
            # nutrient_col is like 'calories_per_100g', need 'calories'
            root_name = nutrient_col.replace('_per_100g', '')
            true_col_name = f'{root_name}_abs_true'
            pred_col_name = f'{root_name}_abs_pred'
        else:
            true_col_name = f'{base_nutrient_name}_true'
            pred_col_name = f'{base_nutrient_name}_pred'

        if true_col_name not in df.columns or pred_col_name not in df.columns:
            # print(f"Rank 0 Warning: Columns {true_col_name} or {pred_col_name} not found for {nutrient_col} in metrics.")
            continue
        
        valid_idx = ~ (df[true_col_name].isnull() | df[pred_col_name].isnull())
        true_vals = df.loc[valid_idx, true_col_name].values
        pred_vals = df.loc[valid_idx, pred_col_name].values

        if len(true_vals) == 0: continue
        mae = mean_absolute_error(true_vals, pred_vals)
        rmse = np.sqrt(mean_squared_error(true_vals, pred_vals))
        r2 = r2_score(true_vals, pred_vals)
        mean_true = np.mean(true_vals); mean_pred = np.mean(pred_vals)
        perc_err = (mae / (mean_true + 1e-9)) * 100 if mean_true != 0 else float('inf')
        metrics_list.append({
            'Nutrient': nutrient_col if per_100g else root_name + " (abs)",
            'MAE': mae, 'RMSE': rmse, 'R²': r2,
            '% Error': perc_err, 'Mean True': mean_true, 'Mean Pred': mean_pred
        })
    return pd.DataFrame(metrics_list)

def show_predictions_with_images_ddp(n_samples, results_df, items_list_for_images, # items_list_for_images is val_items_main_final
                                     output_dir, model_name_arg, target_columns_arg, rank_for_log=0):
    if rank_for_log != 0: return # Only rank 0 saves plots

    actual_n = min(n_samples, len(results_df))
    if actual_n == 0:
        print("Rank 0: No samples for image predictions plot."); return

    sample_indices_df = np.random.choice(len(results_df), actual_n, replace=False)
    ncols = 3; nrows = (actual_n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(15, 5 * nrows), squeeze=False)
    axes = axes.flatten()

    for i_ax, ax_plot in enumerate(axes):
        if i_ax >= actual_n: ax_plot.axis('off'); continue
        df_row_idx = sample_indices_df[i_ax] # Index for results_df

        # results_df.iloc[df_row_idx] corresponds to items_list_for_images[df_row_idx]
        item_info_for_plot = items_list_for_images[df_row_idx]
        dish_id_plot = item_info_for_plot['dish_id']
        image_path_plot = item_info_for_plot['image_path']

        if not os.path.exists(image_path_plot):
            ax_plot.text(0.5,0.5, f"Img not found:\n{os.path.basename(image_path_plot)}", ha='center'); ax_plot.axis('off'); continue
        
        try: img = Image.open(image_path_plot)
        except Exception as e_img:
            ax_plot.text(0.5,0.5, f"Error opening:\n{os.path.basename(image_path_plot)}\n{e_img}", ha='center'); ax_plot.axis('off'); continue

        ax_plot.imshow(img); ax_plot.axis('off')
        pred_txt, true_txt = "Pred:\n", "Actual (Err %):\n"
        for nutrient_col in target_columns_arg:
            pred_val = results_df.iloc[df_row_idx][f'{nutrient_col}_pred']
            true_val = results_df.iloc[df_row_idx][f'{nutrient_col}_true']
            err_pct = abs(pred_val - true_val) / (abs(true_val) + 1e-9) * 100 if true_val != 0 else float('inf')
            disp_name = nutrient_col.split('_')[0].capitalize()[:3]
            pred_txt += f"{disp_name}: {pred_val:.0f}\n"
            true_txt += f"{disp_name}: {true_val:.0f} ({err_pct:.1f}%)\n"
        
        ax_plot.text(0.02, 0.98, pred_txt, transform=ax_plot.transAxes, va='top', bbox=dict(boxstyle='round', fc='lightblue', alpha=0.8), fontsize=8)
        ax_plot.text(0.98, 0.98, true_txt, transform=ax_plot.transAxes, va='top', ha='right', bbox=dict(boxstyle='round', fc='lightgreen', alpha=0.8), fontsize=8)
        ax_plot.set_title(f"Dish: {dish_id_plot}\nView: {os.path.basename(image_path_plot)}", fontsize=9)

    for j in range(actual_n, len(axes)): fig.delaxes(axes[j])
    plt.tight_layout(rect=[0,0,1,0.96]); plt.suptitle(f'Sample Predictions ({model_name_arg})', fontsize=14)
    plot_path = os.path.join(output_dir, f"plot_sample_preds_{model_name_arg}.png")
    plt.savefig(plot_path); print(f"Rank 0: Sample predictions plot saved: {plot_path}"); plt.close(fig)


# %% Main Execution
if __name__ == "__main__":
    ARGS, RANK, WORLD_SIZE, DEVICE, IS_DISTRIBUTED = parse_arguments()

    if RANK == 0:
        os.makedirs(ARGS.output_dir, exist_ok=True)
        print(f"Running with arguments: {ARGS}")
        print(f"Process Rank: {RANK}, World Size: {WORLD_SIZE}, Device: {PROCESS_DEVICE}, Distributed: {IS_DISTRIBUTED}")

    # --- Configuration from ARGS ---
    LOCAL_BASE_DIR_MAIN = ARGS.base_dir
    IMAGERY_BASE_DIR_MAIN = os.path.join(LOCAL_BASE_DIR_MAIN, "imagery")
    OVERHEAD_IMAGERY_DIR_MAIN = os.path.join(IMAGERY_BASE_DIR_MAIN, "realsense_overhead")
    SIDE_ANGLES_IMAGERY_DIR_MAIN = os.path.join(IMAGERY_BASE_DIR_MAIN, "side_angles")
    SIDE_ANGLES_SUBDIR_NAME_MAIN = "extracted_frames"
    METADATA_FILE_CAFE1_MAIN = os.path.join(LOCAL_BASE_DIR_MAIN, "metadata/dish_metadata_cafe1.csv")
    METADATA_FILE_CAFE2_MAIN = os.path.join(LOCAL_BASE_DIR_MAIN, "metadata/dish_metadata_cafe2.csv")
    RGB_IMAGE_FILENAME_MAIN = "rgb.png" # For overhead
    TARGET_COLUMNS_MAIN = ['calories_per_100g', 'fat_per_100g', 'carbs_per_100g', 'protein_per_100g']

    # --- Data Loading and Preprocessing (Rank 0 prepares, then broadcasts) ---
    train_items_main_final, val_items_main_final, all_labels_main_array, filtered_metadata_for_eval_main = None, None, None, None

    if RANK == 0:
        print("Rank 0: Preparing and loading dataset metadata...")
        # Path checks
        paths_to_check = [LOCAL_BASE_DIR_MAIN, OVERHEAD_IMAGERY_DIR_MAIN, METADATA_FILE_CAFE1_MAIN, METADATA_FILE_CAFE2_MAIN]
        if ARGS.include_side_angles:
            paths_to_check.append(SIDE_ANGLES_IMAGERY_DIR_MAIN)
        for p_main in paths_to_check:
            if not os.path.exists(p_main):
                print(f"ERROR (Rank 0): Path not found: {p_main}. Exiting.")
                # Signal other processes to exit if in DDP
                if IS_DISTRIBUTED: dist.barrier(); cleanup_distributed()
                sys.exit(1)

        dish_df_cafe1_main = parse_nutrition_csv(METADATA_FILE_CAFE1_MAIN)
        dish_df_cafe2_main = parse_nutrition_csv(METADATA_FILE_CAFE2_MAIN)
        raw_dish_metadata_df = pd.concat([dish_df_cafe1_main, dish_df_cafe2_main], ignore_index=True)
        raw_dish_metadata_df = raw_dish_metadata_df.replace([np.inf, -np.inf], np.nan)
        all_dish_metadata_df_main = raw_dish_metadata_df.dropna(subset=TARGET_COLUMNS_MAIN + ['dish_id', 'weight'])
        all_dish_metadata_df_main = all_dish_metadata_df_main.set_index('dish_id')

        dataset_items_rank0 = []
        all_labels_list_rank0 = []
        dish_id_to_label_idx_rank0 = {}

        for dish_id, row_data in tqdm(all_dish_metadata_df_main.iterrows(), total=len(all_dish_metadata_df_main), desc="Rank 0: Processing dishes"):
            has_any_image_for_dish = False
            if dish_id not in dish_id_to_label_idx_rank0:
                dish_id_to_label_idx_rank0[dish_id] = len(all_labels_list_rank0)
                all_labels_list_rank0.append(row_data[TARGET_COLUMNS_MAIN].values.astype(np.float32))
            label_idx = dish_id_to_label_idx_rank0[dish_id]

            overhead_img_path = os.path.join(OVERHEAD_IMAGERY_DIR_MAIN, dish_id, RGB_IMAGE_FILENAME_MAIN)
            if os.path.exists(overhead_img_path):
                dataset_items_rank0.append({'dish_id': dish_id, 'label_idx': label_idx, 'image_type': 'overhead', 'image_path': overhead_img_path})
                has_any_image_for_dish = True
            
            if ARGS.include_side_angles:
                dish_side_angle_base = os.path.join(SIDE_ANGLES_IMAGERY_DIR_MAIN, dish_id, SIDE_ANGLES_SUBDIR_NAME_MAIN)
                if os.path.isdir(dish_side_angle_base):
                    available_frames = [os.path.join(dish_side_angle_base, f) for f in os.listdir(dish_side_angle_base) if f.startswith("camera_") and f.endswith(".png")]
                    if available_frames:
                        has_any_image_for_dish = True
                        if ARGS.num_side_angles_per_dish > 0 and len(available_frames) > ARGS.num_side_angles_per_dish:
                            selected_frames = np.random.choice(available_frames, ARGS.num_side_angles_per_dish, replace=False).tolist()
                        else:
                            selected_frames = available_frames
                        for frame_path in selected_frames:
                            dataset_items_rank0.append({'dish_id': dish_id, 'label_idx': label_idx, 'image_type': 'side_angle', 'image_path': frame_path})
            
            # If a dish has no images, its label might be unused. This is generally okay.
        
        if not dataset_items_rank0:
            print("ERROR (Rank 0): No dataset items found after scanning for images. Check paths and data. Exiting.")
            if IS_DISTRIBUTED: dist.barrier(); cleanup_distributed()
            sys.exit(1)

        all_labels_main_array = np.array(all_labels_list_rank0, dtype=np.float32)

        unique_dish_ids_in_dataset = sorted(list(set(item['dish_id'] for item in dataset_items_rank0)))
        if len(unique_dish_ids_in_dataset) < 2:
            print("ERROR (Rank 0): Not enough unique dishes for train/test split. Exiting.")
            if IS_DISTRIBUTED: dist.barrier(); cleanup_distributed()
            sys.exit(1)
            
        train_dish_ids_split, val_dish_ids_split = train_test_split(unique_dish_ids_in_dataset, test_size=0.2, random_state=42)
        
        train_items_main_final = [item for item in dataset_items_rank0 if item['dish_id'] in train_dish_ids_split]
        val_items_main_final = [item for item in dataset_items_rank0 if item['dish_id'] in val_dish_ids_split]

        # filtered_metadata_for_eval_main (for evaluation reports)
        final_dish_ids_in_eval_set = sorted(list(set(item['dish_id'] for item in val_items_main_final))) # Use val_items for eval metadata
        filtered_metadata_for_eval_main = all_dish_metadata_df_main.loc[all_dish_metadata_df_main.index.isin(final_dish_ids_in_eval_set)].copy()
        if filtered_metadata_for_eval_main.index.name == 'dish_id':
            filtered_metadata_for_eval_main.reset_index(inplace=True) # Ensure 'dish_id' is a column

        print(f"Rank 0: Total dataset items (views): {len(dataset_items_rank0)}")
        image_type_counts_rank0 = Counter(item['image_type'] for item in dataset_items_rank0)
        print(f"Rank 0: Image type distribution: {image_type_counts_rank0}")
        print(f"Rank 0: Train items: {len(train_items_main_final)}, Val items: {len(val_items_main_final)}")
        print(f"Rank 0: Unique labels: {len(all_labels_main_array)}")
        if not filtered_metadata_for_eval_main.empty:
            print(f"Rank 0: Metadata for eval contains {len(filtered_metadata_for_eval_main)} unique dishes from validation set.")

    # Broadcast data from Rank 0 to all other ranks
    broadcast_data_list = [None] * 4 # For train_items, val_items, all_labels, filtered_eval_meta
    if IS_DISTRIBUTED:
        if RANK == 0:
            broadcast_data_list[0] = train_items_main_final
            broadcast_data_list[1] = val_items_main_final
            broadcast_data_list[2] = all_labels_main_array
            broadcast_data_list[3] = filtered_metadata_for_eval_main
        
        dist.broadcast_object_list(broadcast_data_list, src=0)
        
        if RANK != 0: # Assign received data on other ranks
            train_items_main_final = broadcast_data_list[0]
            val_items_main_final = broadcast_data_list[1]
            all_labels_main_array = broadcast_data_list[2]
            filtered_metadata_for_eval_main = broadcast_data_list[3]
    
    # Verify data received on all ranks
    data_valid = True
    if train_items_main_final is None or val_items_main_final is None or \
       all_labels_main_array is None or (RANK == 0 and filtered_metadata_for_eval_main is None):
        data_valid = False
        print(f"Rank {RANK}: ERROR - Data not properly received/initialized after broadcast.")

    if IS_DISTRIBUTED: # Sync point to check validity across all ranks
        valid_tensor = torch.tensor([1.0 if data_valid else 0.0], device=DEVICE)
        dist.all_reduce(valid_tensor, op=dist.ReduceOp.MIN) # If any rank failed, all_reduce results in 0
        if valid_tensor.item() == 0.0:
            if RANK == 0: print("CRITICAL ERROR: Data loading/broadcasting failed on one or more ranks. Exiting.")
            cleanup_distributed()
            sys.exit(1)
    elif not data_valid: # Single process mode failure
         print("CRITICAL ERROR: Data loading failed in non-distributed mode. Exiting.")
         sys.exit(1)

    if RANK == 0 and not IS_DISTRIBUTED: # Print summary for non-DDP if it was rank 0 only
        print(f"Dataset items (views): {len(train_items_main_final) + len(val_items_main_final)}")
        print(f"Train items: {len(train_items_main_final)}, Val items: {len(val_items_main_final)}")


    # --- Transforms ---
    train_transform_main = transforms.Compose([
        transforms.Resize((256, 256)), transforms.RandomCrop(224), transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1), transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    val_transform_main = transforms.Compose([
        transforms.Resize((224, 224)), transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # --- GPU Caching Check & DataLoader Params ---
    attempt_gpu_cache_train, attempt_gpu_cache_val = False, False
    _dummy_pil_train = Image.new("RGB", (256, 256)); _tensor_shape_train = train_transform_main(_dummy_pil_train).shape
    _dummy_pil_val = Image.new("RGB", (224, 224));   _tensor_shape_val = val_transform_main(_dummy_pil_val).shape
    
    num_unique_train_images = len(set(item['image_path'] for item in train_items_main_final))
    num_unique_val_images = len(set(item['image_path'] for item in val_items_main_final))

    if ARGS.enable_gpu_caching:
        can_cache_train_rank = check_memory_for_gpu_caching(num_unique_train_images, _tensor_shape_train, DEVICE.index if DEVICE.type == 'cuda' else 0, RANK, IS_DISTRIBUTED)
        can_cache_val_rank = check_memory_for_gpu_caching(num_unique_val_images, _tensor_shape_val, DEVICE.index if DEVICE.type == 'cuda' else 0, RANK, IS_DISTRIBUTED)
        
        if IS_DISTRIBUTED:
            # All ranks must agree to cache (or not cache) for simplicity, based on min capability
            train_cache_tensor = torch.tensor(1.0 if can_cache_train_rank else 0.0, device=DEVICE)
            val_cache_tensor = torch.tensor(1.0 if can_cache_val_rank else 0.0, device=DEVICE)
            dist.all_reduce(train_cache_tensor, op=dist.ReduceOp.MIN)
            dist.all_reduce(val_cache_tensor, op=dist.ReduceOp.MIN)
            attempt_gpu_cache_train = train_cache_tensor.item() == 1.0
            attempt_gpu_cache_val = val_cache_tensor.item() == 1.0
        else:
            attempt_gpu_cache_train = can_cache_train_rank
            attempt_gpu_cache_val = can_cache_val_rank
    
    actual_num_workers_main = ARGS.num_workers
    # If any dataset is cached, force num_workers to 0 for all DataLoaders on this rank
    if (attempt_gpu_cache_train or attempt_gpu_cache_val):
        if ARGS.num_workers > 0 and (RANK == 0 or not IS_DISTRIBUTED):
            print(f"Rank {RANK}: GPU Caching active for at least one dataset. Forcing num_workers from {ARGS.num_workers} to 0.")
        actual_num_workers_main = 0
    
    pin_memory_main = not (attempt_gpu_cache_train or attempt_gpu_cache_val) # Pin if not caching to GPU directly

    if RANK == 0:
        print(f"Final DataLoader settings: GPU Caching Train: {attempt_gpu_cache_train}, Val: {attempt_gpu_cache_val}")
        print(f"Num Workers: {actual_num_workers_main}, Pin Memory: {pin_memory_main}")

    # --- Datasets and DataLoaders ---
    train_dataset_main = NutritionDataset(train_items_main_final, all_labels_main_array, train_transform_main,
                                          attempt_gpu_cache_train, DEVICE, RANK, IS_DISTRIBUTED)
    val_dataset_main = NutritionDataset(val_items_main_final, all_labels_main_array, val_transform_main,
                                        attempt_gpu_cache_val, DEVICE, RANK, IS_DISTRIBUTED)

    train_sampler_main = DistributedSampler(train_dataset_main, num_replicas=WORLD_SIZE, rank=RANK, shuffle=True) if IS_DISTRIBUTED else None
    val_sampler_main = DistributedSampler(val_dataset_main, num_replicas=WORLD_SIZE, rank=RANK, shuffle=False, drop_last=False) if IS_DISTRIBUTED else None # drop_last=False for full eval

    train_loader_main = DataLoader(train_dataset_main, batch_size=ARGS.batch_size, sampler=train_sampler_main,
                                   num_workers=actual_num_workers_main, shuffle=(train_sampler_main is None),
                                   pin_memory=pin_memory_main)
    val_loader_main = DataLoader(val_dataset_main, batch_size=ARGS.batch_size, sampler=val_sampler_main,
                                 num_workers=actual_num_workers_main, shuffle=False,
                                 pin_memory=pin_memory_main)
    if RANK == 0:
        print(f"Training views: {len(train_dataset_main)}, Validation views: {len(val_dataset_main)}")

    # --- Model, Criterion, Optimizer, Scheduler (largely same as DDP script) ---
    model_configs_main = {
        'SimpleConvNet': SimpleConvNet(num_outputs=len(TARGET_COLUMNS_MAIN)),
        'DeepConvNet': DeepConvNet(num_outputs=len(TARGET_COLUMNS_MAIN)),
        'MobileNetLike': MobileNetLike(num_outputs=len(TARGET_COLUMNS_MAIN)),
        'ResNetFromScratch': ResNetFromScratch(num_outputs=len(TARGET_COLUMNS_MAIN), use_pretrained=False),
        'ResNetPretrained': ResNetFromScratch(num_outputs=len(TARGET_COLUMNS_MAIN), use_pretrained=True)
    }
    model_main = model_configs_main[ARGS.model_name].to(DEVICE)
    if IS_DISTRIBUTED:
        model_main = DDP(model_main, device_ids=[DEVICE.index] if DEVICE.type == 'cuda' else None,
                         output_device=DEVICE.index if DEVICE.type == 'cuda' else None,
                         find_unused_parameters=False) # Set to True if you have conditional paths in forward not always used
    if RANK == 0: print(f"Model: {ARGS.model_name}, Total params: {sum(p.numel() for p in model_main.parameters()):,}")
    criterion_main = nn.L1Loss()
    optimizer_main = optim.Adam(model_main.parameters(), lr=ARGS.lr)
    scheduler_main = optim.lr_scheduler.ReduceLROnPlateau(optimizer_main, 'min', patience=10, factor=0.5, verbose=(RANK==0))


    # --- Training Loop (largely same as DDP script) ---
    best_val_loss_main = float('inf')
    history_main = {'train_loss': [], 'val_loss': [], 'percentage_errors': [], 'lr': []}
    MODEL_SAVE_PATH_MAIN = os.path.join(ARGS.output_dir, f'best_nutrition_model_{ARGS.model_name}.pth')
    HISTORY_SAVE_PATH_MAIN = os.path.join(ARGS.output_dir, f'training_history_{ARGS.model_name}.pkl')

    if train_loader_main and val_loader_main:
        # ... (training loop from your DDP script, ensure it uses main variables) ...
        loop_current_epoch = 0
        try:
            for epoch_iter in range(ARGS.epochs):
                loop_current_epoch = epoch_iter; epoch_start_time_main = time.time()
                current_lr_main = optimizer_main.param_groups[0]['lr']
                if RANK == 0: print(f"\nEPOCH {epoch_iter+1}/{ARGS.epochs} | LR: {current_lr_main:.6f}")

                train_loss_main = train_epoch(model_main, train_loader_main, criterion_main, optimizer_main, DEVICE, epoch_iter, train_sampler_main, IS_DISTRIBUTED, RANK)
                val_loss_main, percentage_errors_main, _, _ = validate(model_main, val_loader_main, criterion_main, DEVICE, IS_DISTRIBUTED, RANK, WORLD_SIZE, TARGET_COLUMNS_MAIN)
                scheduler_main.step(val_loss_main)

                if RANK == 0:
                    history_main['train_loss'].append(train_loss_main)
                    history_main['val_loss'].append(val_loss_main)
                    history_main['percentage_errors'].append(percentage_errors_main)
                    history_main['lr'].append(current_lr_main)
                    epoch_time_main = time.time() - epoch_start_time_main
                    print(f"Epoch {epoch_iter+1} Results (Rank 0): Train Loss: {train_loss_main:.4f}, Val Loss: {val_loss_main:.4f}, Time: {epoch_time_main:.2f}s")
                    if percentage_errors_main:
                        print("Val Percentage Errors (per 100g, Rank 0):")
                        for nutrient, error in percentage_errors_main.items(): print(f"  {nutrient}: {error:.2f}%")
                    if val_loss_main < best_val_loss_main:
                        best_val_loss_main = val_loss_main
                        torch.save({
                            'epoch': epoch_iter,
                            'model_state_dict': model_main.module.state_dict() if IS_DISTRIBUTED else model_main.state_dict(),
                            'optimizer_state_dict': optimizer_main.state_dict(), 'scheduler_state_dict': scheduler_main.state_dict(),
                            'best_val_loss': best_val_loss_main, 'model_name': ARGS.model_name, 'args': vars(ARGS),
                            'target_columns': TARGET_COLUMNS_MAIN # Save target columns
                        }, MODEL_SAVE_PATH_MAIN)
                        print(f"✓ NEW BEST MODEL SAVED to {MODEL_SAVE_PATH_MAIN}")
                if IS_DISTRIBUTED: dist.barrier()
        except KeyboardInterrupt:
            if RANK == 0: print(f"\nTraining interrupted. Completed {loop_current_epoch}/{ARGS.epochs} epochs.")
        except Exception as e_main:
            print(f"Rank {RANK} Error during training: {e_main}"); import traceback; traceback.print_exc()
        finally:
            if RANK == 0 and history_main['train_loss']:
                print(f"\nTraining Summary (Rank 0) - {ARGS.model_name}"); print(f"Best Val Loss: {best_val_loss_main:.4f}")
                with open(HISTORY_SAVE_PATH_MAIN, 'wb') as f_hist: pickle.dump(history_main, f_hist)
                print(f"Training history saved to: {HISTORY_SAVE_PATH_MAIN}")

    # --- Plotting and Final Evaluation (Rank 0 only) ---
    if RANK == 0:
        # Load history for plotting
        loaded_history_main = None
        if os.path.exists(HISTORY_SAVE_PATH_MAIN):
            try:
                with open(HISTORY_SAVE_PATH_MAIN, 'rb') as f: loaded_history_main = pickle.load(f)
            except Exception as e: print(f"Rank 0: Could not load history {HISTORY_SAVE_PATH_MAIN}: {e}")

        if loaded_history_main and loaded_history_main.get('train_loss') and ARGS.save_plots:
            # ... (Plotting loss/errors from your DDP script, adapted for loaded_history_main)
            fig_loss, (ax1_loss, ax2_loss) = plt.subplots(1, 2, figsize=(15, 5))
            ax1_loss.plot(loaded_history_main['train_loss'], label='Train Loss')
            ax1_loss.plot(loaded_history_main['val_loss'], label='Val Loss')
            ax1_loss.set_xlabel('Epoch'); ax1_loss.set_ylabel('Loss'); ax1_loss.set_title('Loss'); ax1_loss.legend(); ax1_loss.grid(True)
            if loaded_history_main.get('percentage_errors') and loaded_history_main['percentage_errors']:
                percentage_df_main = pd.DataFrame(loaded_history_main['percentage_errors'])
                if not percentage_df_main.empty:
                    for col_plot in percentage_df_main.columns: ax2_loss.plot(percentage_df_main[col_plot], label=col_plot)
                    ax2_loss.set_xlabel('Epoch'); ax2_loss.set_ylabel('Percentage Error (%)'); ax2_loss.set_title('Val Pct Errors'); ax2_loss.legend(); ax2_loss.grid(True)
            plt.tight_layout(); plt.savefig(os.path.join(ARGS.output_dir, f"plot_loss_errors_{ARGS.model_name}.png")); plt.close(fig_loss)
            print(f"Rank 0: Loss/Error plot saved for {ARGS.model_name}")


        # Evaluation with the best model
        results_df_eval_main = pd.DataFrame()
        if os.path.exists(MODEL_SAVE_PATH_MAIN) and val_loader_main is not None: # val_loader_main implies val_dataset_main exists
            print(f"\nRank 0: Evaluating best model from {MODEL_SAVE_PATH_MAIN}...")
            try:
                checkpoint_main = torch.load(MODEL_SAVE_PATH_MAIN, map_location=DEVICE)
                eval_model_main = model_configs_main[ARGS.model_name].to(DEVICE) # Re-init model structure
                
                state_dict = checkpoint_main['model_state_dict']
                # Adjust for DDP saved model if current mode is non-DDP for eval, or vice-versa
                # This logic assumes eval is done on a single GPU (non-DDP) on Rank 0
                if IS_DISTRIBUTED and not list(state_dict.keys())[0].startswith('module.'): # Model saved non-DDP, current DDP
                     #This case should not happen if saving from DDP correctly
                     pass
                elif (not IS_DISTRIBUTED or WORLD_SIZE == 1) and list(state_dict.keys())[0].startswith('module.'): # Model saved DDP, current non-DDP
                    from collections import OrderedDict
                    new_state_dict = OrderedDict()
                    for k, v in state_dict.items(): new_state_dict[k[7:]] = v # remove `module.`
                    state_dict = new_state_dict
                eval_model_main.load_state_dict(state_dict)
                eval_model_main.eval()

                # Create a new DataLoader for rank 0 specific evaluation
                # Uses val_items_main_final which was broadcasted, and all_labels_main_array
                val_dataset_eval_rank0 = NutritionDataset(
                    val_items_main_final, all_labels_main_array, val_transform_main,
                    attempt_gpu_cache_val, DEVICE, RANK, False # is_distributed=False for this specific instance
                )
                val_loader_eval_rank0 = DataLoader(val_dataset_eval_rank0, batch_size=ARGS.batch_size, shuffle=False,
                                                   num_workers=actual_num_workers_main, pin_memory=pin_memory_main)

                # Validate on rank 0 using the full validation set
                _, eval_perc_errors, eval_preds_np, eval_labels_np = validate(
                    eval_model_main, val_loader_eval_rank0, criterion_main, DEVICE,
                    False, RANK, 1, TARGET_COLUMNS_MAIN # is_distributed=False, world_size=1 for this eval
                )

                if eval_perc_errors:
                    print(f"Rank 0: Best Model ({ARGS.model_name}) Validation Percentage Errors:")
                    for n, e in eval_perc_errors.items(): print(f"  {n}: {e:.2f}%")

                if eval_preds_np.size > 0 :
                    num_eval_samples = len(eval_preds_np)
                    # Ensure val_items_main_final is sliced if drop_last was True during training val
                    # Here, val_loader_eval_rank0 ensures we get preds for all items in val_dataset_eval_rank0
                    eval_item_details_df = val_items_main_final[:num_eval_samples]

                    all_dish_ids_eval_df = [item['dish_id'] for item in eval_item_details_df]
                    
                    # Use filtered_metadata_for_eval_main (already broadcasted)
                    weight_map_eval_df = pd.Series(filtered_metadata_for_eval_main.weight.values, index=filtered_metadata_for_eval_main.dish_id).to_dict()
                    all_weights_eval_df = [weight_map_eval_df.get(did, np.nan) for did in all_dish_ids_eval_df]

                    results_df_eval_main = pd.DataFrame({
                        'dish_id': all_dish_ids_eval_df,
                        'weight': all_weights_eval_df
                    })
                    for i_col, col_name_target in enumerate(TARGET_COLUMNS_MAIN):
                        results_df_eval_main[f'{col_name_target}_pred'] = eval_preds_np[:, i_col]
                        results_df_eval_main[f'{col_name_target}_true'] = eval_labels_np[:, i_col]
                    
                    # Add absolute values
                    for nutrient_root in ['calories', 'fat', 'carbs', 'protein']:
                        nutrient_col_100g = next((tc for tc in TARGET_COLUMNS_MAIN if tc.startswith(nutrient_root)), None)
                        if nutrient_col_100g and f'{nutrient_col_100g}_pred' in results_df_eval_main.columns and 'weight' in results_df_eval_main.columns:
                            results_df_eval_main[f'{nutrient_root}_abs_pred'] = results_df_eval_main[f'{nutrient_col_100g}_pred'] * results_df_eval_main['weight'] / 100
                            results_df_eval_main[f'{nutrient_root}_abs_true'] = results_df_eval_main[f'{nutrient_col_100g}_true'] * results_df_eval_main['weight'] / 100
                    
                    print(f"Rank 0: Evaluation predictions completed for {len(results_df_eval_main)} views.")
                    
                    # --- Calculate and Print Metrics ---
                    print("\nRank 0:" + "="*30 + f" METRICS - {ARGS.model_name} (Per 100g) " + "="*30)
                    metrics_df_100g = calculate_metrics(results_df_eval_main, TARGET_COLUMNS_MAIN, per_100g=True)
                    if not metrics_df_100g.empty: print(metrics_df_100g.to_string(index=False, float_format='%.3f'))

                    print("\nRank 0:" + "="*25 + " METRICS - ABSOLUTE (using ground truth weight) " + "="*25)
                    metrics_df_abs = calculate_metrics(results_df_eval_main, TARGET_COLUMNS_MAIN, per_100g=False) # Uses root name logic in calculate_metrics
                    if not metrics_df_abs.empty: print(metrics_df_abs.to_string(index=False, float_format='%.3f'))


                    # --- Plotting Detailed Evaluation (Preds vs Actual, Error Dist) ---
                    if ARGS.save_plots and not results_df_eval_main.empty:
                        num_targets_plot = len(TARGET_COLUMNS_MAIN)
                        ncols_plot = 2
                        # Predictions vs Actual
                        nrows_pvsa = (num_targets_plot + ncols_plot - 1) // ncols_plot
                        fig_pvsa, axes_pvsa = plt.subplots(nrows_pvsa, ncols_plot, figsize=(7 * ncols_plot, 6 * nrows_pvsa), squeeze=False)
                        axes_pvsa = axes_pvsa.flatten()
                        for i, nutrient_key in enumerate(TARGET_COLUMNS_MAIN):
                            ax = axes_pvsa[i]
                            # ... (copy plotting logic for PvA from single-GPU, using results_df_eval_main, nutrient_key) ...
                            true_col, pred_col = f'{nutrient_key}_true', f'{nutrient_key}_pred'
                            if true_col not in results_df_eval_main.columns or pred_col not in results_df_eval_main.columns: continue
                            x_data, y_data = results_df_eval_main[true_col].values, results_df_eval_main[pred_col].values
                            valid = ~ (np.isnan(x_data) | np.isnan(y_data))
                            x_plot, y_plot = x_data[valid], y_data[valid]
                            if len(x_plot) == 0: continue
                            ax.scatter(x_plot, y_plot, alpha=0.5, s=30, edgecolors='k', linewidth=0.5)
                            min_val_plot, max_val_plot = min(x_plot.min(), y_plot.min()), max(x_plot.max(), y_plot.max())
                            ax.plot([min_val_plot, max_val_plot], [min_val_plot, max_val_plot], 'r--', lw=2)
                            r2_plot = r2_score(x_plot, y_plot)
                            display_name = nutrient_key.replace('_per_100g', '').replace('_', ' ').capitalize()
                            ax.set_xlabel(f'True {display_name}'); ax.set_ylabel(f'Pred {display_name}')
                            ax.set_title(f'{display_name} (R²={r2_plot:.3f})'); ax.grid(True, alpha=0.3)

                        for j_plot in range(num_targets_plot, len(axes_pvsa)): fig_pvsa.delaxes(axes_pvsa[j_plot])
                        plt.tight_layout(rect=[0,0,1,0.96]); fig_pvsa.suptitle(f'Predictions vs True ({ARGS.model_name}, per 100g)', fontsize=16)
                        pvsa_path = os.path.join(ARGS.output_dir, f"plot_preds_vs_actual_{ARGS.model_name}.png")
                        plt.savefig(pvsa_path); print(f"Rank 0: PvA plot saved: {pvsa_path}"); plt.close(fig_pvsa)

                        # Error Distribution Plot (copy from single GPU, adapt for results_df_eval_main)

                    # --- Plotting Sample Predictions with Images ---
                    if ARGS.save_plots and len(results_df_eval_main) >= 6:
                         show_predictions_with_images_ddp(6, results_df_eval_main, val_items_main_final,
                                                         ARGS.output_dir, ARGS.model_name, TARGET_COLUMNS_MAIN, RANK)

            except FileNotFoundError: print(f"Rank 0: Model checkpoint {MODEL_SAVE_PATH_MAIN} not found. Skipping final evaluation.")
            except Exception as e_eval: print(f"Rank 0: Error during final model evaluation: {e_eval}"); import traceback; traceback.print_exc()

    # --- Cleanup DDP ---
    if IS_DISTRIBUTED:
        cleanup_distributed()

    if RANK == 0:
        print("Script finished.")
# %%
