#!/usr/bin/env python
# %%
import argparse
import os
import sys
import pandas as pd
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
import torch.nn.functional as F
from torchvision import transforms, models
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from tqdm import tqdm
import time
import pickle
import concurrent.futures
import random
import gc # Add import gc at the top of your file

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

# %%
# Argument Parsing
# ----------------------------------------------------------------------------
def parse_arguments():
    parser = argparse.ArgumentParser(description="Train nutrition estimation models.")
    parser.add_argument('--model_name', type=str, required=True,
                        choices=['SimpleConvNet', 'DeepConvNet', 'MobileNetLike', 'ResNetFromScratch', 'ResNetPretrained'],
                        help='Name of the model to train.')
    parser.add_argument('--base_dir', type=str, default="/Data/nutrition5k", help='Base directory for the dataset.')
    parser.add_argument('--output_dir', type=str, default=".", help='Directory to save models, history, and plots.')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs.')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate.')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training and validation.')
    parser.add_argument('--num_workers', type=int, default=0, help='Number of workers for DataLoader. Forced to 0 if GPU caching is active.')
    parser.add_argument('--save_plots', action='store_true', help='Save plots to files instead of displaying them.')
    parser.add_argument('--gpu_caching', action='store_true', help='Enable caching of dataset images into GPU VRAM. Forces num_workers=0.')
    parser.add_argument('--include_side_angles', action='store_true', help='Include side angle images.')
    parser.add_argument('--num_side_angles_per_dish', type=int, default=20, help='Max random side angle frames per dish if included (0 for all).')
    parser.add_argument('--epoch_data_fraction', type=float, default=1.0, help='Fraction of training AND validation data to use per epoch (0.0 to 1.0). Requires >0.')
    parser.add_argument('--early_stopping_patience', type=int, default=10, help='Patience for early stopping (epochs). 0 to disable.')

    if any('jupyter' in arg for arg in sys.argv) or 'ipykernel_launcher.py' in sys.argv[0]:
        print("Running in interactive mode. Using default/test args.")
        default_args_list = [
            '--model_name', 'SimpleConvNet', '--epochs', '20', # Test with more epochs for early stopping
            '--output_dir', 'interactive_test_final_optimized', '--save_plots',
            '--include_side_angles', '--num_side_angles_per_dish', '2',
            '--gpu_caching', 
            '--epoch_data_fraction', '0.5', 
            '--early_stopping_patience', '5' # Test early stopping
        ]
        current_args_str = " ".join(sys.argv)
        if '--model_name' not in current_args_str: args = parser.parse_args(default_args_list)
        else:
            try: args = parser.parse_args()
            except SystemExit: args = parser.parse_args(default_args_list)
    else:
        print("Running as a script. Parsing command-line arguments.")
        args = parser.parse_args()

    if not (0.0 < args.epoch_data_fraction <= 1.0):
        parser.error("--epoch_data_fraction must be > 0.0 and <= 1.0.")
    if args.early_stopping_patience < 0:
        parser.error("--early_stopping_patience must be >= 0.")
    return args

args = parse_arguments()
os.makedirs(args.output_dir, exist_ok=True)
print(f"Running with arguments: {args}")

# %%
# Global Variables and Constants
# ----------------------------------------------------------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LOCAL_BASE_DIR=args.base_dir; BATCH_SIZE=args.batch_size; LEARNING_RATE=args.lr; NUM_EPOCHS=args.epochs
MODEL_NAME_ARG=args.model_name; NUM_WORKERS_ARG=args.num_workers; OUTPUT_DIR=args.output_dir
SAVE_PLOTS=args.save_plots; GPU_CACHING_REQUESTED_ARG=args.gpu_caching
INCLUDE_SIDE_ANGLES_ARG=args.include_side_angles; NUM_SIDE_ANGLES_PER_DISH_ARG=args.num_side_angles_per_dish
EPOCH_DATA_FRACTION_ARG=args.epoch_data_fraction; EARLY_STOPPING_PATIENCE_ARG=args.early_stopping_patience

IMAGERY_BASE_DIR=os.path.join(LOCAL_BASE_DIR,"imagery"); OVERHEAD_IMAGERY_DIR=os.path.join(IMAGERY_BASE_DIR,"realsense_overhead")
SIDE_ANGLES_IMAGERY_DIR=os.path.join(IMAGERY_BASE_DIR,"side_angles");
SIDE_ANGLES_SUBDIR_NAME="extracted_frames"
# SIDE_ANGLES_SUBDIR_NAME="extracted_framesss"
METADATA_FILE_CAFE1=os.path.join(LOCAL_BASE_DIR,"metadata/dish_metadata_cafe1.csv")
METADATA_FILE_CAFE2=os.path.join(LOCAL_BASE_DIR,"metadata/dish_metadata_cafe2.csv")


# Example outlier filtering (adjust thresholds as needed)
MAX_CAL_100G = 600  # Max reasonable calories per 100g
MAX_FAT_100G = 70   # Max reasonable fat per 100g (e.g. oil is 100, but few foods are pure oil)
MAX_CARBS_100G = 100 # Pure sugar/starch
MAX_PROT_100G = 50  # Dried protein powder might be higher, but whole foods rarely

assert os.path.exists(LOCAL_BASE_DIR), f"LOCAL_BASE_DIR not found: {LOCAL_BASE_DIR}"
assert os.path.exists(OVERHEAD_IMAGERY_DIR), f"OVERHEAD_IMAGERY_DIR not found: {OVERHEAD_IMAGERY_DIR}"
if INCLUDE_SIDE_ANGLES_ARG: assert os.path.exists(SIDE_ANGLES_IMAGERY_DIR), f"SIDE_ANGLES_IMAGERY_DIR not found: {SIDE_ANGLES_IMAGERY_DIR}"
assert os.path.exists(METADATA_FILE_CAFE1), f"METADATA_FILE_CAFE1 not found: {METADATA_FILE_CAFE1}"
assert os.path.exists(METADATA_FILE_CAFE2), f"METADATA_FILE_CAFE2 not found: {METADATA_FILE_CAFE2}"

# RGB_IMAGE_FILENAME="rgb.png"; TARGET_COLUMNS=['calories_per_100g','fat_per_100g','carbs_per_100g','protein_per_100g']
RGB_IMAGE_FILENAME="rgb.jpg"; TARGET_COLUMNS=['calories_per_100g','fat_per_100g','carbs_per_100g','protein_per_100g']
print(f"Device: {DEVICE}"); print(f"Selected Model: {MODEL_NAME_ARG}")
print(f"GPU Caching Requested: {GPU_CACHING_REQUESTED_ARG}"); print(f"Epoch Data Fraction: {EPOCH_DATA_FRACTION_ARG}")
print(f"Early Stopping Patience: {EARLY_STOPPING_PATIENCE_ARG if EARLY_STOPPING_PATIENCE_ARG > 0 else 'Disabled'}")

# %%
# Model Definitions
# ----------------------------------------------------------------------------
class SimpleConvNet(nn.Module):
    def __init__(self,num_outputs=4):
        super().__init__()
        self.conv1=nn.Conv2d(3,32,3,padding=1)
        self.bn1=nn.BatchNorm2d(32)
        self.conv2=nn.Conv2d(32,64,3,padding=1)
        self.bn2=nn.BatchNorm2d(64)
        self.conv3=nn.Conv2d(64,128,3,padding=1)
        self.bn3=nn.BatchNorm2d(128)
        self.conv4=nn.Conv2d(128,256,3,padding=1)
        self.bn4=nn.BatchNorm2d(256)
        self.pool=nn.MaxPool2d(2,2)
        self.dropout=nn.Dropout(0.5)
        self.fc1=nn.Linear(256*14*14,512)
        self.fc2=nn.Linear(512,256)
        self.fc3=nn.Linear(256,num_outputs)

    def forward(self,x):
        x=self.pool(F.relu(self.bn1(self.conv1(x))))
        x=self.pool(F.relu(self.bn2(self.conv2(x))))
        x=self.pool(F.relu(self.bn3(self.conv3(x))))
        x=self.pool(F.relu(self.bn4(self.conv4(x))))
        x=x.view(x.size(0),-1)
        x=self.dropout(F.relu(self.fc1(x)))
        x=self.dropout(F.relu(self.fc2(x)))
        return self.fc3(x)
    

class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += self.shortcut(identity) # Add skip connection
        out = self.relu(out)
        return out
    
class DeepConvNet(nn.Module):
    def __init__(self,num_outputs=4):
        super().__init__()
        self.conv1=nn.Conv2d(3,64,7,stride=2,padding=3)
        self.bn1=nn.BatchNorm2d(64)
        self.pool1=nn.MaxPool2d(3,stride=2,padding=1)
        self.res_block1 = BasicBlock(64, 128, stride=2)
        self.res_block2 = BasicBlock(128, 256, stride=2)
        self.res_block3 = BasicBlock(256, 512, stride=2)
        self.res_block4 = BasicBlock(512, 512, stride=2)
        # self.res_block1=self._make_residual_block(64,128,stride=2)
        # self.res_block2=self._make_residual_block(128,256,stride=2)
        # self.res_block3=self._make_residual_block(256,512,stride=2)
        self.avgpool=nn.AdaptiveAvgPool2d((1,1))
        self.dropout = nn.Dropout(0.5) 
        self.fc=nn.Linear(512,num_outputs)

    def _make_residual_block(self,in_channels,out_channels,stride=1):
        # This creates a sequence of layers, but a true ResNet block adds input to output.
        # If stride != 1 or in_channels != out_channels, a projection shortcut is needed.
        layers = []
        layers.append(nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False))
        layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False))
        layers.append(nn.BatchNorm2d(out_channels))
        
        # Shortcut connection for ResNet
        shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        # This needs to be integrated into a block class that does x = self.layers(x) + self.shortcut(x)
        # For simplicity with your current structure, this _make_residual_block is just a sequence.
        # To make it a true ResNet block, you'd need a custom nn.Module for the block.
        return nn.Sequential(*layers)    
    
    def forward(self,x):
        x=self.pool1(F.relu(self.bn1(self.conv1(x))))
        x=self.res_block1(x)
        x=self.res_block2(x)
        x=self.res_block3(x)
        x=self.res_block4(x)
        x=self.avgpool(x)
        x=x.view(x.size(0),-1)
        x=self.dropout(x)
        return self.fc(x)
    
class MobileNetLike(nn.Module):
    def __init__(self,num_outputs=4):super().__init__();ds_conv=lambda i,o,s=1:nn.Sequential(nn.Conv2d(i,i,3,s,1,groups=i,bias=False),nn.BatchNorm2d(i),nn.ReLU(True),nn.Conv2d(i,o,1,bias=False),nn.BatchNorm2d(o),nn.ReLU(True));self.conv1=nn.Conv2d(3,32,3,stride=2,padding=1,bias=False);self.bn1=nn.BatchNorm2d(32);self.dw2=ds_conv(32,64,2);self.dw3=ds_conv(64,128,2);self.dw4=ds_conv(128,256,2);self.dw5=ds_conv(256,512,2);self.avgpool=nn.AdaptiveAvgPool2d((1,1));self.fc=nn.Linear(512,num_outputs)
    def forward(self,x):x=F.relu(self.bn1(self.conv1(x)));x=self.dw2(x);x=self.dw3(x);x=self.dw4(x);x=self.dw5(x);x=self.avgpool(x);x=x.view(x.size(0),-1);return self.fc(x)
class ResNetFromScratch(nn.Module):
    def __init__(self,num_outputs=4,use_pretrained=False):super().__init__();weights=models.ResNet34_Weights.IMAGENET1K_V1 if use_pretrained else None;self.backbone=models.resnet34(weights=weights);n_feat=self.backbone.fc.in_features;self.backbone.fc=nn.Sequential(nn.Linear(n_feat,256),nn.ReLU(),nn.Dropout(0.5),nn.Linear(256,num_outputs))
    def forward(self,x):return self.backbone(x)

# %%

# %%
# Loss Function
# ----------------------------------------------------------------------------
class MultiTaskLoss(nn.Module): # Kept for potential future use, L1Loss used by default
    def __init__(self,task_weights=None):
        super().__init__()
        self.w=torch.ones(len(TARGET_COLUMNS))if task_weights is None else torch.tensor(task_weights,dtype=torch.float32)
    def forward(self,p,t):L=torch.abs(p-t);W=self.w.to(p.device)if self.w.device!=p.device else self.w;return(L*W).mean()

# %%
# Data Preprocessing (Metadata)
# ----------------------------------------------------------------------------
# %%
# Data Preprocessing (Metadata)
# ----------------------------------------------------------------------------
def parse_nutrition_csv(file_path):
    dishes=[]
    try:
        with open(file_path,'r')as f:
            for line in f:
                parts=line.strip().split(',')
                if not parts[0].startswith('dish_'):continue
                try:did,cal,wt,fat,car,pro=parts[0],float(parts[1]),float(parts[2]),float(parts[3]),float(parts[4]),float(parts[5])
                except(ValueError,IndexError):continue
                if wt <= 0: continue # Ensure weight is positive here
                dishes.append({'dish_id':did,'calories':cal,'weight':wt,'fat':fat,'carbs':car,'protein':pro,
                               'calories_per_100g':(cal/wt)*100,
                               'fat_per_100g':(fat/wt)*100,
                               'carbs_per_100g':(car/wt)*100,
                               'protein_per_100g':(pro/wt)*100})
    except FileNotFoundError: print(f"Error: Metadata file {file_path} not found.")
    return pd.DataFrame(dishes)

raw_dish_metadata_df=pd.concat([parse_nutrition_csv(METADATA_FILE_CAFE1),parse_nutrition_csv(METADATA_FILE_CAFE2)],ignore_index=True)

# Initial cleaning
all_dish_metadata_df_unfiltered = raw_dish_metadata_df.replace([np.inf,-np.inf],np.nan).dropna(
    subset=TARGET_COLUMNS+['dish_id','weight'] # Ensure these are present before filtering
).set_index('dish_id')

print(f"Unfiltered metadata rows: {len(all_dish_metadata_df_unfiltered)}")
if not all_dish_metadata_df_unfiltered.empty:
    print("Unfiltered metadata stats:\n", all_dish_metadata_df_unfiltered[TARGET_COLUMNS].describe())

# --- APPLY OUTLIER FILTERING HERE ---
MAX_CAL_100G = 600    # Example: Max reasonable calories per 100g
MIN_CAL_100G = 5      # Min reasonable calories per 100g
MAX_FAT_100G = 80     # Example: Max reasonable fat per 100g (oil is 100, but most foods less)
MIN_FAT_100G = 1
MAX_CARBS_100G = 100  # Pure sugar/starch
MIN_CARBS_100G = 1
MAX_PROT_100G = 60    # Dried protein powder might be higher, but whole foods often less
MIN_PROT_100G = 1

# Ensure that 'weight' is positive as well, although parse_nutrition_csv should handle it.
# This check is redundant if parse_nutrition_csv already ensures wt > 0.
# all_dish_metadata_df_unfiltered = all_dish_metadata_df_unfiltered[all_dish_metadata_df_unfiltered['weight'] > 0]


all_dish_metadata_df_filtered = all_dish_metadata_df_unfiltered[
    (all_dish_metadata_df_unfiltered['calories_per_100g'] >= MIN_CAL_100G) & (all_dish_metadata_df_unfiltered['calories_per_100g'] <= MAX_CAL_100G) &
    (all_dish_metadata_df_unfiltered['fat_per_100g'] >= MIN_FAT_100G) & (all_dish_metadata_df_unfiltered['fat_per_100g'] <= MAX_FAT_100G) &
    (all_dish_metadata_df_unfiltered['carbs_per_100g'] >= MIN_CARBS_100G) & (all_dish_metadata_df_unfiltered['carbs_per_100g'] <= MAX_CARBS_100G) &
    (all_dish_metadata_df_unfiltered['protein_per_100g'] >= MIN_PROT_100G) & (all_dish_metadata_df_unfiltered['protein_per_100g'] <= MAX_PROT_100G)
]
print(f"Filtered metadata rows: {len(all_dish_metadata_df_filtered)}")
if not all_dish_metadata_df_filtered.empty:
    print("Filtered metadata stats (this will be used for dataset construction):\n", all_dish_metadata_df_filtered[TARGET_COLUMNS].describe())
else:
    print("CRITICAL: All data was filtered out by outlier removal. Check filter thresholds and data.")
    sys.exit(1)

# Now, use all_dish_metadata_df_filtered to build dataset_items
dataset_items=[]
all_labels_list=[]
dish_id_to_label_idx={} # This will map dish_id to an index in all_labels_list

# --- REVISED LOGIC FOR ROBUST LABEL INDEXING WITH IMAGE VERIFICATION ---
print("Starting revised dataset item construction with image verification...")

candidate_image_info = [] # List of (dish_id, image_path, image_type) for images that EXIST
for dish_id, row_data in tqdm(all_dish_metadata_df_filtered.iterrows(), total=len(all_dish_metadata_df_filtered), desc="Phase 1: Identifying existing images"):
    overhead_img_path = os.path.join(OVERHEAD_IMAGERY_DIR, dish_id, RGB_IMAGE_FILENAME)
    if os.path.exists(overhead_img_path):
        candidate_image_info.append({'dish_id': dish_id, 'image_path': overhead_img_path, 'image_type': 'overhead'})
    
    if INCLUDE_SIDE_ANGLES_ARG:
        side_angle_base = os.path.join(SIDE_ANGLES_IMAGERY_DIR, dish_id, SIDE_ANGLES_SUBDIR_NAME)
        if os.path.isdir(side_angle_base):
            available_frames=[os.path.join(side_angle_base,f)for f in os.listdir(side_angle_base)if f.startswith("camera_")and f.endswith(".jpg")] # or .png
            if available_frames:
                n_to_select = NUM_SIDE_ANGLES_PER_DISH_ARG if NUM_SIDE_ANGLES_PER_DISH_ARG > 0 else len(available_frames)
                selected_frames = np.random.choice(available_frames, min(len(available_frames), n_to_select), replace=False).tolist()
                for frame_path in selected_frames:
                    candidate_image_info.append({'dish_id': dish_id, 'image_path': frame_path, 'image_type': 'side_angle'})

verified_image_info = []
corrupted_image_paths = []
print(f"Phase 2: Verifying {len(candidate_image_info)} potential image items...")
for item_info in tqdm(candidate_image_info, desc="Verifying image readability"):
    try:
        with Image.open(item_info['image_path']) as img:
            img.verify() 
        verified_image_info.append(item_info)
    except Exception as e: # Catching a broader range of PIL errors
        print(f"WARNING: Skipping corrupted/unreadable image: {item_info['image_path']} ({type(e).__name__}: {e})")
        corrupted_image_paths.append(item_info['image_path'])

if corrupted_image_paths:
    print(f"INFO: Skipped {len(corrupted_image_paths)} corrupted/unreadable images during dataset construction.")

if not verified_image_info:
    print("CRITICAL: No valid and readable images found. Exiting.")
    sys.exit(1)

# Now build labels ONLY for dishes that have at least one verified image
final_dish_ids_with_verified_images = sorted(list(set(item['dish_id'] for item in verified_image_info)))

all_labels_list = []
dish_id_to_label_idx = {}
for dish_id in final_dish_ids_with_verified_images:
    # It's guaranteed that dish_id is in all_dish_metadata_df_filtered if it's in verified_image_info
    row_data = all_dish_metadata_df_filtered.loc[dish_id] 
    dish_id_to_label_idx[dish_id] = len(all_labels_list)
    all_labels_list.append(row_data[TARGET_COLUMNS].values.astype(np.float32))
    
all_labels_array = np.array(all_labels_list, dtype=np.float32) if all_labels_list else np.array([])

dataset_items = []
for item_info in verified_image_info:
    dish_id = item_info['dish_id']
    if dish_id in dish_id_to_label_idx: # Ensure the dish made it into the final label mapping
        label_idx = dish_id_to_label_idx[dish_id]
        dataset_items.append({
            'dish_id': dish_id, 
            'label_idx': label_idx, 
            'image_type': item_info['image_type'], 
            'image_path': item_info['image_path']
        })
print(f"Total dataset items (views) after verification and label mapping: {len(dataset_items)}")


all_labels_array = np.array(all_labels_list, dtype=np.float32) if all_labels_list else np.array([])

# This `filtered_metadata_for_eval` is now mainly for reference if you need the original weights later.
# The actual labels come from `all_labels_array` which is built from filtered data.
final_dish_ids_in_dataset = sorted(list(set(item['dish_id'] for item in dataset_items))) # These are dish_ids from filtered data
# For evaluation, we might still want original weights, so get them from the initial unfiltered or partially cleaned df
# but ensure we only look at dishes that actually made it into our final dataset.
_temp_metadata_for_weights = all_dish_metadata_df_unfiltered.loc[all_dish_metadata_df_unfiltered.index.isin(final_dish_ids_in_dataset)].copy()
if _temp_metadata_for_weights.index.name == 'dish_id':
    _temp_metadata_for_weights.reset_index(inplace=True)
# This is used later in final evaluation to get 'weight' for absolute metrics.
# Ensure it's named clearly if you keep it, or integrate weight into dataset_items if preferred.
# For now, let's rename the variable used later in the eval section to avoid confusion.
# metadata_for_final_eval_weights = _temp_metadata_for_weights 


if not all_dish_metadata_df_filtered.empty: # This was the one used for creating dataset_items
    print("\nPer-100g stats for dishes IN THE FINAL DATASET (after filtering):\n", all_dish_metadata_df_filtered.loc[final_dish_ids_in_dataset][TARGET_COLUMNS].describe())

if not all_dish_metadata_df_unfiltered.empty and all_dish_metadata_df_unfiltered.index.name == 'dish_id':
    # Select only the dishes that are in our final dataset_items
    metadata_for_final_eval_weights = all_dish_metadata_df_unfiltered.loc[
        all_dish_metadata_df_unfiltered.index.isin(final_dish_ids_in_dataset)
    ].copy()
    # Ensure 'dish_id' is a column if it was the index, for easier use later if needed,
    # though pd.Series can work directly with the index.
    # metadata_for_final_eval_weights.reset_index(inplace=True) # Optional: reset if you prefer dish_id as column
else:
    print("Warning: Could not properly create metadata_for_final_eval_weights. Weight data might be affected.")
    # Create an empty DataFrame or handle appropriately
    metadata_for_final_eval_weights = pd.DataFrame(columns=['dish_id', 'weight']) 
    if 'dish_id' in all_dish_metadata_df_unfiltered.columns: # if dish_id was already a column
         metadata_for_final_eval_weights = all_dish_metadata_df_unfiltered.loc[
            all_dish_metadata_df_unfiltered['dish_id'].isin(final_dish_ids_in_dataset)
        ].copy()

# raw_dish_metadata_df=pd.concat([parse_nutrition_csv(METADATA_FILE_CAFE1),parse_nutrition_csv(METADATA_FILE_CAFE2)],ignore_index=True)
# all_dish_metadata_df=raw_dish_metadata_df.replace([np.inf,-np.inf],np.nan).dropna(subset=TARGET_COLUMNS+['dish_id','weight']).set_index('dish_id')
# print("Original metadata stats:\n", all_dish_metadata_df[TARGET_COLUMNS].describe())
# all_dish_metadata_df = all_dish_metadata_df[
#     (all_dish_metadata_df['calories_per_100g'] >= 0) & (all_dish_metadata_df['calories_per_100g'] <= MAX_CAL_100G) &
#     (all_dish_metadata_df['fat_per_100g'] >= 0) & (all_dish_metadata_df['fat_per_100g'] <= MAX_FAT_100G) &
#     (all_dish_metadata_df['carbs_per_100g'] >= 0) & (all_dish_metadata_df['carbs_per_100g'] <= MAX_CARBS_100G) &
#     (all_dish_metadata_df['protein_per_100g'] >= 0) & (all_dish_metadata_df['protein_per_100g'] <= MAX_PROT_100G)
# ]
# # Crucially, also ensure weight is positive if not already handled by parse_nutrition_csv
# all_dish_metadata_df = all_dish_metadata_df[all_dish_metadata_df['weight'] > 0]

# print("Filtered metadata stats:\n", all_dish_metadata_df[TARGET_COLUMNS].describe())
# # Now proceed with creating dataset_items from this filtered DataFrame
# dataset_items=[];all_labels_list=[];dish_id_to_label_idx={}
# print("Scanning for available images and preparing dataset items...")
# for dish_id,row_data in tqdm(all_dish_metadata_df.iterrows(),total=len(all_dish_metadata_df),desc="Processing dishes"):
#     if dish_id not in dish_id_to_label_idx:dish_id_to_label_idx[dish_id]=len(all_labels_list);all_labels_list.append(row_data[TARGET_COLUMNS].values.astype(np.float32))
#     label_idx=dish_id_to_label_idx[dish_id]
#     overhead_img_path=os.path.join(OVERHEAD_IMAGERY_DIR,dish_id,RGB_IMAGE_FILENAME)
#     if os.path.exists(overhead_img_path):dataset_items.append({'dish_id':dish_id,'label_idx':label_idx,'image_type':'overhead','image_path':overhead_img_path})
#     if INCLUDE_SIDE_ANGLES_ARG:
#         side_angle_base=os.path.join(SIDE_ANGLES_IMAGERY_DIR,dish_id,SIDE_ANGLES_SUBDIR_NAME)
#         if os.path.isdir(side_angle_base):
#             # available_frames=[os.path.join(side_angle_base,f)for f in os.listdir(side_angle_base)if f.startswith("camera_")and f.endswith(".png")]
#             available_frames=[os.path.join(side_angle_base,f)for f in os.listdir(side_angle_base)if f.startswith("camera_")and f.endswith(".jpg")]
#             if available_frames:
#                 n_to_select=NUM_SIDE_ANGLES_PER_DISH_ARG if NUM_SIDE_ANGLES_PER_DISH_ARG>0 else len(available_frames)
#                 selected_frames=np.random.choice(available_frames,min(len(available_frames),n_to_select),replace=False).tolist()
#                 for frame_path in selected_frames:dataset_items.append({'dish_id':dish_id,'label_idx':label_idx,'image_type':'side_angle','image_path':frame_path})
# if not dataset_items:print(f"CRITICAL: No dataset items found after scanning imagery directories like {OVERHEAD_IMAGERY_DIR}. Check paths and data structure.");sys.exit(1)
# print(f"Total dataset items (views): {len(dataset_items)}")
# all_labels_array=np.array(all_labels_list,dtype=np.float32)if all_labels_list else np.array([])
# final_dish_ids_in_dataset=sorted(list(set(item['dish_id']for item in dataset_items)))
# filtered_metadata_for_eval=all_dish_metadata_df.loc[all_dish_metadata_df.index.isin(final_dish_ids_in_dataset)].copy()
# if filtered_metadata_for_eval.index.name=='dish_id':filtered_metadata_for_eval.reset_index(inplace=True)
# if not filtered_metadata_for_eval.empty:print("\nPer-100g stats for final dataset dishes:\n",filtered_metadata_for_eval[TARGET_COLUMNS].describe())


# %%
# Dataset and DataLoader with Enhanced GPU Caching
# ----------------------------------------------------------------------------
def check_memory_for_caching(num_unique_images, image_size_wh=224, device=None, return_estimate=False, subtract_from_free_mb=0.0, context=""):
    if device is None or device.type!='cuda' or not torch.cuda.is_available():return(False,0.0)if return_estimate else False
    try:
        img_mem_mb=image_size_wh*image_size_wh*3*4/(1024*1024);total_est_mb_this_set=num_unique_images*img_mem_mb
        props=torch.cuda.get_device_properties(device);total_gpu_mem_mb=props.total_memory/(1024*1024)
        free_mem_bytes,_=torch.cuda.mem_get_info(device);current_free_gpu_mem_mb=free_mem_bytes/(1024*1024)
        effective_free_gpu_mem_mb=current_free_gpu_mem_mb-subtract_from_free_mb
        if effective_free_gpu_mem_mb<0:effective_free_gpu_mem_mb=0
        # This factor (0.5) is for deciding if a NEW SET of images can be cached in currently free VRAM.
        # It's different from the overall cache utilization cap.
        usable_cache_mb_this_set=effective_free_gpu_mem_mb*0.8
        if context: # Print only if context is provided
            print(f"MemCheck({context}): EstCacheThisSet={total_est_mb_this_set:.1f}MB({num_unique_images} imgs).")
            print(f"  CurrentFreeGPU={current_free_gpu_mem_mb:.1f}MB,SubtractedUsed={subtract_from_free_mb:.1f}MB,EffectiveFree={effective_free_gpu_mem_mb:.1f}MB.")
            print(f"  UsableEstForThisSet={usable_cache_mb_this_set:.1f}MB,TotalGPU={total_gpu_mem_mb:.1f}MB.")
        can_cache_this_set=False
        if total_est_mb_this_set==0:can_cache_this_set=True
        elif total_est_mb_this_set > total_gpu_mem_mb:
            if context: print(f"  ✗ EstCacheThisSet > TotalGPU. Cannot cache.")
        elif usable_cache_mb_this_set >= total_est_mb_this_set:
            if context: print("  ✓ Sufficient VRAM estimate for this set.");can_cache_this_set=True
        else:
            if context: print(f"  ✗ Insufficient VRAM estimate for this set (needs {total_est_mb_this_set:.1f}MB, usable {usable_cache_mb_this_set:.1f}MB).")
        return(can_cache_this_set,total_est_mb_this_set)if return_estimate else can_cache_this_set
    except Exception as e:
        if context: print(f"Err MemCheck({context}):{e}")
        return(False,0.0)if return_estimate else False

class NutritionDataset(Dataset):
    def __init__(self, items_list, all_labels_array_source, transform=None,
                 user_requests_gpu_caching=False, attempt_initial_full_cache_for_this_set=False,
                 max_cache_utilization_factor=0.3): # Cap cache at 60% of total GPU VRAM by default
        self.items_list = items_list
        self.all_labels_array_source = all_labels_array_source
        self.transform = transform
        self.image_cache = {} # Stores path: tensor_on_gpu
        self.device_to_cache_to = DEVICE # Assuming DEVICE is globally defined
        self.caching_is_active_for_retrieval = False
        
        self.total_gpu_memory_mb = 0
        self.max_cache_memory_mb = 0 # Max memory the cache should occupy
        self.max_cache_utilization_factor = max_cache_utilization_factor
        self.current_epoch_image_paths = set() # Image paths for the current epoch's subset

        # For multithreaded loading for _cache_image_paths_to_gpu
        self.num_cache_workers = min(8, os.cpu_count() // 2 if os.cpu_count() else 1, len(items_list) // 100 if len(items_list) > 100 else 1)
        # self.num_cache_workers = 16
        self.num_cache_workers = max(1, self.num_cache_workers)

        if user_requests_gpu_caching and self.device_to_cache_to.type == 'cuda':
            self.caching_is_active_for_retrieval = True
            try:
                props = torch.cuda.get_device_properties(self.device_to_cache_to)
                self.total_gpu_memory_mb = props.total_memory / (1024 * 1024)
                self.max_cache_memory_mb = self.total_gpu_memory_mb * self.max_cache_utilization_factor
                print(f"Dataset '{self.__class__.__name__}': Max GPU cache size target: {self.max_cache_memory_mb:.1f}MB ({self.max_cache_utilization_factor*100:.0f}% of total {self.total_gpu_memory_mb:.1f}MB VRAM).")
            except Exception as e:
                print(f"Warning: Could not get GPU properties to set max cache size: {e}. Disabling GPU caching for safety.")
                self.caching_is_active_for_retrieval = False

            if self.caching_is_active_for_retrieval and attempt_initial_full_cache_for_this_set:
                unique_paths = list(set(item['image_path'] for item in self.items_list))
                print(f"Attempting initial full GPU cache for {len(unique_paths)} unique images in this dataset instance ({self.__class__.__name__})...")
                
                # For initial load, all unique paths are considered "current" for purpose of not evicting them immediately if they fit.
                initial_load_paths_set = set(unique_paths)
                self._cache_image_paths_to_gpu(unique_paths, "Initial caching")
                # Manage capacity after the initial load attempt
                self._manage_cache_capacity(current_epoch_image_paths_set=initial_load_paths_set)
        
        elif user_requests_gpu_caching:
            print("GPU caching requested, but device not CUDA. Caching disabled.")

    def _calculate_cache_memory_usage_mb(self):
        if not self.image_cache:
            return 0.0
        current_cache_size_bytes = 0
        for tensor in self.image_cache.values():
            current_cache_size_bytes += tensor.nelement() * tensor.element_size()
        return current_cache_size_bytes / (1024 * 1024)

    def _manage_cache_capacity(self, current_epoch_image_paths_set):
        if not self.caching_is_active_for_retrieval or self.max_cache_memory_mb <= 0:
            return

        current_cache_usage_mb = self._calculate_cache_memory_usage_mb()

        if current_cache_usage_mb > self.max_cache_memory_mb:
            num_items_before_eviction = len(self.image_cache)
            print(f"Cache Info ({self.__class__.__name__}): Usage {current_cache_usage_mb:.1f}MB (Items: {num_items_before_eviction}) > Target {self.max_cache_memory_mb:.1f}MB. Evicting non-epoch images...")
            
            evictable_paths = [
                p for p in self.image_cache.keys() if p not in current_epoch_image_paths_set
            ]
            np.random.shuffle(evictable_paths) # Evict randomly among non-epoch items
            
            num_evicted = 0
            evicted_mem_mb = 0.0

            for path_to_evict in evictable_paths:
                if current_cache_usage_mb <= self.max_cache_memory_mb:
                    break # Target met

                if path_to_evict in self.image_cache:
                    tensor_to_evict = self.image_cache.pop(path_to_evict) # Remove and get tensor
                    tensor_size_bytes = tensor_to_evict.nelement() * tensor_to_evict.element_size()
                    del tensor_to_evict # Explicitly delete reference
                    
                    current_cache_usage_mb -= tensor_size_bytes / (1024 * 1024)
                    evicted_mem_mb += tensor_size_bytes / (1024 * 1024)
                    num_evicted += 1
                
            if num_evicted > 0:
                gc.collect()
                torch.cuda.empty_cache() # Try to free memory after deletions
                print(f"Cache Eviction ({self.__class__.__name__}): Evicted {num_evicted} non-epoch images (freed ~{evicted_mem_mb:.1f}MB). New cache: {current_cache_usage_mb:.1f}MB (Items: {len(self.image_cache)}).")
            
            if current_cache_usage_mb > self.max_cache_memory_mb:
                 # This implies all remaining images are for the current epoch, or no evictable images were found.
                print(f"Warning ({self.__class__.__name__}): Cache usage {current_cache_usage_mb:.1f}MB still > target {self.max_cache_memory_mb:.1f}MB. "
                      f"{len(self.image_cache)} items remain, potentially all for current epoch ({len(current_epoch_image_paths_set)} paths). "
                      f"Consider reducing epoch_data_fraction, batch_size, or max_cache_utilization_factor if OOMs persist.")
        # else:
            # print(f"Cache Info ({self.__class__.__name__}): Current usage {current_cache_usage_mb:.1f}MB (Items: {len(self.image_cache)}) is within target {self.max_cache_memory_mb:.1f}MB.")


    def _load_and_transform_image_cpu(self, image_path):
        # (This method remains the same as in your original code)
        try:
            img_pil = Image.open(image_path).convert("RGB")
            img_tensor_cpu = self.transform(img_pil) if self.transform else transforms.ToTensor()(img_pil)
            return image_path, img_tensor_cpu
        except FileNotFoundError:
            return image_path, None
        except Exception: # as e: print(f"W:Err CPU pre-load {image_path}:{e}.")
            return image_path, None
        
    def __len__(self):
        return len(self.items_list)

    def __getitem__(self, idx):
        item = self.items_list[idx]
        image_path = item['image_path']
        label_idx = item['label_idx']
        img_tensor = None
        
        if self.caching_is_active_for_retrieval and image_path in self.image_cache:
            img_tensor_gpu = self.image_cache[image_path] # This tensor is on self.device_to_cache_to (e.g., CUDA)
            
            # Defensive check: ensure it's on the intended cache device first
            if img_tensor_gpu.device != self.device_to_cache_to:
                img_tensor_gpu = img_tensor_gpu.to(self.device_to_cache_to)
            
            # Move to CPU before returning, so collate_fn gets consistent CPU tensors
            img_tensor = img_tensor_gpu.cpu() 
        else:
            # Fallback: Load from disk if not cached or caching disabled
            try:
                img_pil = Image.open(image_path).convert("RGB")
                img_tensor = self.transform(img_pil) if self.transform else transforms.ToTensor()(img_pil) # Already on CPU
            except Exception as e:
                # It's good practice to log which image failed
                print(f"WARNING: Error loading image {image_path} in __getitem__: {e}. Returning zero tensor.")
                img_tensor = torch.zeros((3, 224, 224), dtype=torch.float32) # Already on CPU
        
        # Labels are typically created on CPU and moved to GPU later with the batch
        label_tensor = torch.tensor(self.all_labels_array_source[label_idx], dtype=torch.float32) # Already on CPU
        
        return img_tensor, label_tensor
    
    def clear_cache_aggressively(self, reason=""): # Added reason for logging
        if not self.caching_is_active_for_retrieval: return
        num_items = len(self.image_cache)
        if num_items == 0: return

        print(f"Aggressively clearing image cache ({num_items} items, {self._calculate_cache_memory_usage_mb():.1f} MB) due to: {reason}...")
        
        paths_to_remove = list(self.image_cache.keys()) 
        for p in paths_to_remove:
            if p in self.image_cache:
                tensor = self.image_cache.pop(p)
                del tensor
        
        self.image_cache.clear() 

        print(f"Cache dictionary cleared. Python-level items: {len(self.image_cache)}")
        
        gc.collect()
        torch.cuda.empty_cache() 
        gc.collect()
        torch.cuda.empty_cache()

        free_after_clear, total_mem = torch.cuda.mem_get_info(self.device_to_cache_to)
        used_after_clear = (total_mem - free_after_clear) / (1024 * 1024)
        reserved_pytorch = torch.cuda.memory_reserved(self.device_to_cache_to) / (1024 * 1024)
        allocated_pytorch = torch.cuda.memory_allocated(self.device_to_cache_to) / (1024 * 1024)
        print(f"Aggressive clear: PyTorch Used: {used_after_clear:.1f}MB, Allocated: {allocated_pytorch:.1f}MB, Reserved: {reserved_pytorch:.1f}MB")
    
    def _cache_image_paths_to_gpu(self, image_paths_to_cache, desc_prefix="Caching"):
        if not self.caching_is_active_for_retrieval or not image_paths_to_cache: 
            return False # Return a status indicating if an OOM and clear occurred

        # --- Step 1: Determine paths that genuinely need caching (not already in self.image_cache) ---
        # This is important because image_paths_to_cache might be the full set for the current epoch.
        # If we cleared the cache, ALL of them will need caching.
        # If we didn't clear, only some might.
        paths_genuinely_needing_cache = [p for p in image_paths_to_cache if p not in self.image_cache]
        
        if not paths_genuinely_needing_cache:
            # print(f"Debug: All {len(image_paths_to_cache)} requested paths already in cache.")
            return False # No OOM, no new caching done

        num_newly_cached = 0
        oom_triggered_clear = False # Flag to indicate if OOM caused a full cache clear
        
        # --- Step 2: Parallel CPU load and transform for paths_genuinely_needing_cache ---
        cpu_processed_images = {}
        actual_workers = min(self.num_cache_workers, len(paths_genuinely_needing_cache))
        if actual_workers <= 0: actual_workers = 1

        with concurrent.futures.ThreadPoolExecutor(max_workers=actual_workers) as executor:
            future_to_path = {executor.submit(self._load_and_transform_image_cpu, path): path for path in paths_genuinely_needing_cache}
            iterator = concurrent.futures.as_completed(future_to_path)
            if len(paths_genuinely_needing_cache) > actual_workers * 5:
                 iterator = tqdm(iterator, total=len(paths_genuinely_needing_cache), 
                                 desc=f"{desc_prefix} CPU PreProc ({self.__class__.__name__})", 
                                 leave=False, mininterval=1.0)
            for future in iterator:
                path = future_to_path[future]
                try: 
                    _, img_tensor_cpu = future.result()
                    if img_tensor_cpu is not None:
                        cpu_processed_images[path] = img_tensor_cpu
                except Exception as exc:
                     print(f"W: Path {path} generated an exception during CPU pre-processing: {exc}")


        # --- Step 3: Sequential GPU transfer ---
        # Iterate over the original image_paths_to_cache because these are what the current epoch needs.
        # We will try to load them if they were successfully preprocessed on CPU.
        paths_to_attempt_gpu_transfer = [p for p in image_paths_to_cache if p in cpu_processed_images]

        gpu_transfer_pbar_desc = f"{desc_prefix} GPU Transfer ({self.__class__.__name__})"
        gpu_transfer_disable_pbar = len(paths_to_attempt_gpu_transfer) < 10
        
        for image_path in tqdm(paths_to_attempt_gpu_transfer, desc=gpu_transfer_pbar_desc, leave=False, mininterval=1.0, disable=gpu_transfer_disable_pbar):
            if oom_triggered_clear: # If a clear already happened in this call, stop trying to add more.
                                  # The caller (prime_cache_for_indices) will recall this function.
                break 

            if image_path not in cpu_processed_images: continue # Should not happen if logic is correct

            img_tensor_cpu = cpu_processed_images[image_path]
            
            # Pre-emptive check against overall cache limit
            # This check is crucial before attempting the .to(device)
            # if self.max_cache_memory_mb > 0:
            #     current_cache_usage_mb_before_add = self._calculate_cache_memory_usage_mb()
            #     new_img_mem_mb = (img_tensor_cpu.nelement() * img_tensor_cpu.element_size()) / (1024 * 1024)
                
            #     if current_cache_usage_mb_before_add + new_img_mem_mb > self.max_cache_memory_mb and len(self.image_cache) > 0 : # Check if cache is not empty
            #         print(f"W: Pre-emptive limit reached before caching {image_path}. "
            #               f"Current cache ({current_cache_usage_mb_before_add:.1f}MB) + new ({new_img_mem_mb:.1f}MB) "
            #               f"would exceed target ({self.max_cache_memory_mb:.1f}MB).")
            #         self.clear_cache_aggressively(reason="Pre-emptive limit in _cache_image_paths_to_gpu")
            #         oom_triggered_clear = True # Signal that a clear happened
            #         # After clearing, we need to restart the caching process for the current epoch's items.
            #         # So we break here, and the caller (prime_cache_for_indices) will handle re-calling.
            #         break 

            try:
                with torch.no_grad():
                    self.image_cache[image_path] = img_tensor_cpu.to(self.device_to_cache_to, non_blocking=True)
                num_newly_cached += 1
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    print(f"W: GPU OOM during {desc_prefix} for {image_path}. Newly cached in this attempt before OOM: {num_newly_cached}.")
                    if image_path in self.image_cache: del self.image_cache[image_path] # Clean up failed current attempt
                    # Don't call torch.cuda.empty_cache() yet, clear_cache_aggressively will do it more thoroughly
                    
                    self.clear_cache_aggressively(reason=f"OOM for {image_path} in _cache_image_paths_to_gpu")
                    oom_triggered_clear = True # Signal that a clear happened
                    break # Stop this GPU transfer loop. Caller will re-initiate.
                else:
                    print(f"W: RuntimeErr during GPU transfer for {image_path}: {e}.")
            except Exception as e:
                print(f"W: General Err during GPU transfer for {image_path}: {e}.")
        
        # --- Step 4: Summary Logging ---
        # This logging might be a bit complex if a clear happened mid-way.
        # The num_newly_cached would refer to items added *before* any clear in this specific call.
        if num_newly_cached > 0 or oom_triggered_clear or len(cpu_processed_images) < len(paths_genuinely_needing_cache):
            current_cache_total_mb = self._calculate_cache_memory_usage_mb()
            print(f"{desc_prefix} Summary ({self.__class__.__name__}): "
                  f"{len(cpu_processed_images)}/{len(paths_genuinely_needing_cache)} CPU pre-processed. "
                  f"{num_newly_cached} new to GPU (this attempt). " # Corrected wording
                  f"Total GPU cache now: {len(self.image_cache)} items ({current_cache_total_mb:.1f}MB).")
        
        if oom_triggered_clear:
            print("Caching process in _cache_image_paths_to_gpu was interrupted by OOM/limit, full cache clear performed.")
        
        return oom_triggered_clear # Return status

    def prime_cache_for_indices(self, indices):
        if not self.caching_is_active_for_retrieval:
            self.current_epoch_image_paths = set()
            return
        
        valid_indices = [i for i in indices if 0 <= i < len(self.items_list)]
        new_epoch_image_paths = set()
        if valid_indices:
            new_epoch_image_paths = set(self.items_list[i]['image_path'] for i in valid_indices)

        # Option 1: Evict based on previous epoch's paths first (your current logic)
        # This is less relevant if _cache_image_paths_to_gpu itself will clear on OOM.
        # self._manage_cache_capacity(current_epoch_image_paths_set=self.current_epoch_image_paths) 

        self.current_epoch_image_paths = new_epoch_image_paths # Update to current epoch's needs
        
        if self.current_epoch_image_paths:
            print(f"Priming cache for {len(self.current_epoch_image_paths)} unique paths for current epoch/subset...")
            # First attempt to cache
            oom_and_cleared = self._cache_image_paths_to_gpu(list(self.current_epoch_image_paths), desc_prefix="Epoch subset priming (Attempt 1)")
            
            if oom_and_cleared:
                # If an OOM occurred AND the cache was cleared from within _cache_image_paths_to_gpu,
                # we need to re-attempt caching the current epoch's items into the now (hopefully) emptier cache.
                print("Retrying cache population for current epoch after aggressive clear...")
                # The self.image_cache is now empty or much smaller.
                # self.current_epoch_image_paths still holds the paths we need for *this* epoch.
                self._cache_image_paths_to_gpu(list(self.current_epoch_image_paths), desc_prefix="Epoch subset priming (Attempt 2 Post-Clear)")
        
        # After all attempts, ensure the overall cache policy is still met.
        # This handles cases where the second attempt might still overfill if max_cache_memory_mb is extremely small.
        self._manage_cache_capacity(current_epoch_image_paths_set=self.current_epoch_image_paths)


train_transform=transforms.Compose([transforms.Resize((256,256)),transforms.RandomCrop(224),transforms.RandomHorizontalFlip(),transforms.ColorJitter(0.1,0.1,0.1),transforms.ToTensor(),transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
val_transform=transforms.Compose([transforms.Resize((224,224)),transforms.ToTensor(),transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])


train_loader,val_loader=None,None
train_dataset_instance,val_dataset_instance=None,None
actual_num_workers=NUM_WORKERS_ARG


if dataset_items and all_labels_array.size > 0:
    unique_dish_ids = sorted(list(set(item['dish_id'] for item in dataset_items)))
    train_dish_ids_split, val_dish_ids_split = (train_test_split(unique_dish_ids, test_size=0.2, random_state=42)
                                                if len(unique_dish_ids) >= 2 else (unique_dish_ids, []))
    train_items_final = [item for item in dataset_items if item['dish_id'] in train_dish_ids_split]
    val_items_final = [item for item in dataset_items if item['dish_id'] in val_dish_ids_split]

    # --- RESTRUCTURED LOGIC ---
    user_wants_caching_on_cuda = GPU_CACHING_REQUESTED_ARG and DEVICE.type == 'cuda'
    actual_num_workers = NUM_WORKERS_ARG  # Start with the argument value

    attempt_initial_cache_train, attempt_initial_cache_val = False, False
    train_cache_estimate_mb = 0.0

    if user_wants_caching_on_cuda:
        print("\n--- GPU Caching Pre-check for Initial Full Cache Attempts ---")
        if NUM_WORKERS_ARG > 0:
            print(f"INFO: GPU caching is active, forcing num_workers from {NUM_WORKERS_ARG} to 0.")
            actual_num_workers = 0 # Force to 0 if GPU caching is active

        if train_items_final:
            num_unique_train = len(set(i['image_path'] for i in train_items_final))
            if num_unique_train > 0:
                can_cache_train_ind, train_cache_est_actual = check_memory_for_caching(
                    num_unique_train, device=DEVICE, return_estimate=True, context="Train Set(Individual)")
                if can_cache_train_ind:
                    attempt_initial_cache_train = True
                    train_cache_estimate_mb = train_cache_est_actual
            else: # num_unique_train is 0
                attempt_initial_cache_train = True

        if attempt_initial_cache_train and val_items_final:
            num_unique_val = len(set(i['image_path'] for i in val_items_final))
            if num_unique_val > 0:
                attempt_initial_cache_val = check_memory_for_caching(
                    num_unique_val, device=DEVICE, return_estimate=False,
                    subtract_from_free_mb=train_cache_estimate_mb, context="Val Set(w/Train)")
            else: # num_unique_val is 0
                attempt_initial_cache_val = True
        # else: Val caching not attempted if train caching not attempted or no val items
        # attempt_initial_cache_val remains False if train caching failed/not attempted

        print(f"Initial cache decision: Train={attempt_initial_cache_train}, Val={attempt_initial_cache_val} (Val depends on Train fitting).")
        print("--- End GPU Caching Pre-check ---\n")
    elif GPU_CACHING_REQUESTED_ARG: # Caching requested but device not CUDA
        print("INFO: GPU caching requested, but DEVICE is not CUDA. All caching disabled.")
        # actual_num_workers remains NUM_WORKERS_ARG in this case

    # Now, determine pin_memory_setting based on the final actual_num_workers
    pin_memory_setting = (DEVICE.type == 'cuda' and actual_num_workers > 0)

    # Create Datasets and DataLoaders (this part is now outside the caching pre-check conditional)
    if train_items_final:
        train_dataset_instance = NutritionDataset(
            train_items_final, all_labels_array, train_transform,
            user_requests_gpu_caching=user_wants_caching_on_cuda,
            attempt_initial_full_cache_for_this_set=attempt_initial_cache_train,
            max_cache_utilization_factor=0.7 # Or make this an argparse parameter
        )
        
        # train_dataset_instance = NutritionDataset(
        #     train_items_final, all_labels_array, train_transform,
        #     user_requests_gpu_caching=user_wants_caching_on_cuda, # Pass the flag for dataset's internal logic
        #     attempt_initial_full_cache_for_this_set=attempt_initial_cache_train
        # )
        train_loader = DataLoader(
            train_dataset_instance, batch_size=BATCH_SIZE,
            shuffle=(EPOCH_DATA_FRACTION_ARG == 1.0),
            num_workers=actual_num_workers,
            pin_memory=pin_memory_setting
        )
        print(f"Base Train DataLoader created with {len(train_dataset_instance)} items. Workers: {actual_num_workers}, Pin memory: {pin_memory_setting}")

    if val_items_final:
        val_dataset_instance = NutritionDataset(
            val_items_final, all_labels_array, val_transform,
            user_requests_gpu_caching=user_wants_caching_on_cuda,
            attempt_initial_full_cache_for_this_set=attempt_initial_cache_val,
            max_cache_utilization_factor=0.7 # Or make this an argparse parameter
        )
        val_loader = DataLoader(
            val_dataset_instance, batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=actual_num_workers,
            pin_memory=pin_memory_setting
        )
        print(f"Base Val DataLoader created with {len(val_dataset_instance)} items. Workers: {actual_num_workers}, Pin memory: {pin_memory_setting}")

else:
    print("CRITICAL: Skipping DataLoader creation: no data items or labels.")
    # Consider sys.exit(1) here if DataLoaders are essential
    train_loader, val_loader = None, None # Ensure they are defined as None if not created
    train_dataset_instance, val_dataset_instance = None, None


# %%
# Model Selection and Optimizer
# ----------------------------------------------------------------------------
model_configs = {
    'SimpleConvNet':SimpleConvNet,
    'DeepConvNet':DeepConvNet,
    'MobileNetLike':MobileNetLike,
    'ResNetFromScratch': lambda num_outputs: ResNetFromScratch(num_outputs=num_outputs, use_pretrained=False),
    'ResNetPretrained': lambda num_outputs: ResNetFromScratch(num_outputs=num_outputs, use_pretrained=True),
    }
model=model_configs[MODEL_NAME_ARG](num_outputs=len(TARGET_COLUMNS)).to(DEVICE)
print(f"Selected model: {MODEL_NAME_ARG}, Trainable Params: {sum(p.numel()for p in model.parameters()if p.requires_grad):,}")
criterion=nn.L1Loss()
optimizer=optim.Adam(model.parameters(),lr=LEARNING_RATE);scheduler=optim.lr_scheduler.ReduceLROnPlateau(optimizer,'min',patience=max(1,EARLY_STOPPING_PATIENCE_ARG//2),factor=0.5,verbose=True) # Scheduler patience related to early stopping

# %%
# Training and Validation Loop
# ----------------------------------------------------------------------------
def train_epoch(model,loader,criterion,optimizer,device):
    model.train();total_loss,count=0,0;pbar=tqdm(loader,desc='Training',leave=False,disable=len(loader)==0,mininterval=0.5)
    for img,lbl in pbar:img,lbl=img.to(device),lbl.to(device);optimizer.zero_grad();out=model(img);loss=criterion(out,lbl);loss.backward();optimizer.step();total_loss+=loss.item();count+=1;pbar.set_postfix({'loss':f'{loss.item():.4f}'})
    return total_loss/count if count>0 else float('nan')
def validate(model,loader,criterion,device, desc="Validating"): # Added desc
    model.eval();total_loss,count=0,0;all_preds,all_lbls=[],[]
    if not loader or len(loader)==0:return float('nan'),{c:float('nan')for c in TARGET_COLUMNS},np.empty((0,len(TARGET_COLUMNS))),np.empty((0,len(TARGET_COLUMNS)))
    pbar=tqdm(loader,desc=desc,leave=False,disable=len(loader)==0,mininterval=0.5)
    with torch.no_grad():
        for img,lbl_b in pbar:img,lbl_b=img.to(device),lbl_b.to(device);out=model(img);loss=criterion(out,lbl_b);total_loss+=loss.item();count+=1;all_preds.append(out.cpu().numpy());all_lbls.append(lbl_b.cpu().numpy());pbar.set_postfix({'loss':f'{loss.item():.4f}'})
    avg_loss=total_loss/count if count>0 else float('nan')
    if not all_preds:return avg_loss,{c:float('nan')for c in TARGET_COLUMNS},np.empty((0,len(TARGET_COLUMNS))),np.empty((0,len(TARGET_COLUMNS)))
    preds_np,lbls_np=np.concatenate(all_preds),np.concatenate(all_lbls);errors={}
    for i,col in enumerate(TARGET_COLUMNS):
        if lbls_np.shape[0]>0:mae=mean_absolute_error(lbls_np[:,i],preds_np[:,i]);mean_val=np.mean(lbls_np[:,i]);errors[col]=(mae/(np.abs(mean_val)+1e-9))*100 if mean_val!=0 else(float('inf')if mae>1e-9 else 0.0)
        else:errors[col]=float('nan')
    return avg_loss,errors,preds_np,lbls_np

if train_loader and train_dataset_instance:
    best_val_loss=float('inf');epochs_no_improve=0;history={'train_loss':[],'val_loss':[],'percentage_errors':[],'lr':[]}
    MODEL_SAVE_PATH=os.path.join(OUTPUT_DIR,f'best_nutrition_model_{MODEL_NAME_ARG}.pth');HISTORY_SAVE_PATH=os.path.join(OUTPUT_DIR,f'training_history_{MODEL_NAME_ARG}.pkl')
    print("="*60+f"\nSTARTING TRAINING - {MODEL_NAME_ARG}\n"+"="*60)
    try:
        for epoch in range(NUM_EPOCHS):
            epoch_start_time=time.time();current_lr=optimizer.param_groups[0]['lr']
            print(f"\nEPOCH {epoch+1}/{NUM_EPOCHS} | LR: {current_lr:.6f} | Best Val Loss: {best_val_loss:.4f} | Epochs no improve: {epochs_no_improve}" + "-"*20)
            current_epoch_train_loader=train_loader
            # if epoch > 0 and epoch % 10 == 0: # Example: every 10 epochs
            #     if hasattr(train_dataset_instance, 'clear_cache_aggressively') and train_dataset_instance.caching_is_active_for_retrieval:
            #         print(f"\n--- Performing aggressive cache clear for train_dataset_instance at epoch {epoch+1} ---")
            #         train_dataset_instance.clear_cache_aggressively()
            #     if hasattr(val_dataset_instance, 'clear_cache_aggressively') and val_dataset_instance.caching_is_active_for_retrieval:
            #         print(f"\n--- Performing aggressive cache clear for val_dataset_instance at epoch {epoch+1} ---")
            #         val_dataset_instance.clear_cache_aggressively()
            #     # After clearing, you MUST re-prime if you expect items to be in cache for the current epoch
            #     # The existing priming logic in your loop should handle this.
            if EPOCH_DATA_FRACTION_ARG<1.0 and len(train_dataset_instance)>0:
                full_len=len(train_dataset_instance);samples_count=max(1,int(EPOCH_DATA_FRACTION_ARG*full_len))
                epoch_indices=torch.randperm(full_len).tolist()[:samples_count]
                if train_dataset_instance.caching_is_active_for_retrieval:train_dataset_instance.prime_cache_for_indices(epoch_indices)
                current_epoch_train_loader=DataLoader(train_dataset_instance,batch_size=BATCH_SIZE,sampler=SubsetRandomSampler(epoch_indices),num_workers=actual_num_workers,pin_memory=pin_memory_setting)
            elif train_dataset_instance.caching_is_active_for_retrieval:train_dataset_instance.prime_cache_for_indices(list(range(len(train_dataset_instance))))
            train_loss=train_epoch(model,current_epoch_train_loader,criterion,optimizer,DEVICE)

            current_epoch_val_loader=val_loader;val_loss,percentage_errors=float('nan'),{c:float('nan')for c in TARGET_COLUMNS}
            if val_dataset_instance and len(val_dataset_instance)>0:
                if EPOCH_DATA_FRACTION_ARG<1.0:
                    full_len_val=len(val_dataset_instance);samples_count_val=max(1,int(EPOCH_DATA_FRACTION_ARG*full_len_val))
                    epoch_indices_val=torch.randperm(full_len_val).tolist()[:samples_count_val]
                    if val_dataset_instance.caching_is_active_for_retrieval:val_dataset_instance.prime_cache_for_indices(epoch_indices_val)
                    current_epoch_val_loader=DataLoader(val_dataset_instance,batch_size=BATCH_SIZE,sampler=SubsetRandomSampler(epoch_indices_val),num_workers=actual_num_workers,pin_memory=pin_memory_setting)
                elif val_dataset_instance.caching_is_active_for_retrieval:val_dataset_instance.prime_cache_for_indices(list(range(len(val_dataset_instance))))
                if current_epoch_val_loader and len(current_epoch_val_loader)>0:val_loss,percentage_errors,_,_=validate(model,current_epoch_val_loader,criterion,DEVICE, desc="Validating (Epoch Subset)")
            elif val_loader and len(val_loader)>0:val_loss,percentage_errors,_,_=validate(model,val_loader,criterion,DEVICE, desc="Validating (Full Set)")
            
            if not np.isnan(val_loss):scheduler.step(val_loss)
            history['train_loss'].append(train_loss);history['val_loss'].append(val_loss);history['percentage_errors'].append(percentage_errors);history['lr'].append(current_lr)
            print(f"\n{'='*20} EPOCH {epoch+1} RESULTS {'='*20}")
            print(f"Train Loss: {train_loss:.4f} | Val Loss (epoch subset): {val_loss:.4f} | Time: {time.time()-epoch_start_time:.2f}s")
            print("Percentage Errors (on epoch val subset):");[print(f"  {n:20s}:{e_v:6.2f}%")for n,e_v in percentage_errors.items()]
            if not np.isnan(val_loss):
                if val_loss<best_val_loss:best_val_loss=val_loss;epochs_no_improve=0;torch.save({'epoch':epoch,'model_state_dict':model.state_dict(),'optimizer_state_dict':optimizer.state_dict(),'scheduler_state_dict':scheduler.state_dict(),'best_val_loss':best_val_loss,'model_name':MODEL_NAME_ARG,'history':history},MODEL_SAVE_PATH);print(f"✓ NEW BEST MODEL (epoch val) SAVED to {MODEL_SAVE_PATH}")
                else:epochs_no_improve+=1
            if EARLY_STOPPING_PATIENCE_ARG > 0 and epochs_no_improve >= EARLY_STOPPING_PATIENCE_ARG:print(f"\nEarly stopping triggered after {EARLY_STOPPING_PATIENCE_ARG} epochs without improvement on validation loss.");break
            print("="*60)
    except KeyboardInterrupt:print(f"\n⚠️ Training interrupted by user.")
    except Exception as e:print(f"\n❌ Error during training:{e}");import traceback;traceback.print_exc()
    finally:
        print("\n" + "=" * 20 + f" TRAINING SUMMARY - {MODEL_NAME_ARG} " + "=" * 20)
        if history['train_loss']:
            final_tl = history['train_loss'][-1]
            final_vl = history['val_loss'][-1] if history['val_loss'] else float('nan')
            bvl_disp = best_val_loss if best_val_loss != float('inf') else float('nan')
            print(f"Final Train Loss: {final_tl:.4f}")
            print(f"Final Val Loss (epoch subset): {final_vl:.4f}")
            print(f"Best Val Loss (epoch subset): {bvl_disp:.4f}")
            with open(HISTORY_SAVE_PATH, 'wb') as f:
                pickle.dump(history, f)
            print(f"History saved: {HISTORY_SAVE_PATH}")
else:print("Training skipped:train_loader or train_dataset_instance unavailable.")

# %%
# Plotting Training History
# ----------------------------------------------------------------------------
if 'history' in locals() and history['train_loss'] and SAVE_PLOTS:
    fig,(ax1,ax2)=plt.subplots(1,2,figsize=(16,6)) # Removed squeeze=False as it's 1x2

    # Plotting Training and Validation Loss on ax1
    ax1.plot(history['train_loss'],label='Train Loss',color='royalblue',linewidth=2)
    
    # Ensure val_loss has same length or is handled properly if shorter (e.g. early stopping)
    # Extract valid validation loss points for plotting
    val_loss_series = pd.Series(history['val_loss']).dropna()
    if not val_loss_series.empty:
        ax1.plot(val_loss_series.index, val_loss_series.values, 
                 label='Val Loss(Epoch Subset)',color='darkorange',linestyle='--',linewidth=2)

    ax1.set_xlabel('Epoch',fontsize=12)
    ax1.set_ylabel('Loss',fontsize=12)
    ax1.set_title(f'Training & Validation Loss ({MODEL_NAME_ARG})',fontsize=14)
    if ax1.has_data(): # Check if any lines were plotted before adding legend
        ax1.legend(fontsize=10)
    ax1.grid(True,linestyle=':',alpha=0.7)

    # Corrected plotting for percentage errors on ax2:
    if history['percentage_errors']: # Check if the key exists
        # Filter out epochs where percentage_errors might be None or not a dict
        valid_percentage_errors = [item for item in history['percentage_errors'] if isinstance(item, dict)]
        if valid_percentage_errors: # Proceed only if we have valid dicts
            df_errors = pd.DataFrame(valid_percentage_errors) 
            # No need for .dropna(how='all') here if we pre-filtered, 
            # but it's fine if you want to keep it as an extra safeguard for fully NaN rows.
            # df_errors = df_errors.dropna(how='all', axis=0) 

            if not df_errors.empty:
                something_plotted_on_ax2 = False
                for col in df_errors.columns:  # Iterate through each nutrient column
                    series_col = df_errors[col].dropna() # Get data for the current nutrient, drop NaNs
                    if not series_col.empty: # Check if there's any data left for this nutrient
                        valid_error_indices = series_col.index.tolist()
                        valid_error_values = series_col.values.tolist()
                        
                        ax2.plot(valid_error_indices, valid_error_values, 
                                 label=f"{col.split('_')[0]} %Err", alpha=0.8, linewidth=1.5)
                        something_plotted_on_ax2 = True
                
                ax2.set_xlabel('Epoch', fontsize=12)
                ax2.set_ylabel('Percentage Error(%)', fontsize=12)
                ax2.set_title(f'Nutrient % Errors(Epoch Val Subset)', fontsize=14)
                if something_plotted_on_ax2: # Add legend only if something was plotted
                    ax2.legend(fontsize=9)
                ax2.grid(True, linestyle=':', alpha=0.7)
                
                # Determine y_lim dynamically or use a robust fixed one
                if something_plotted_on_ax2 and any(line.get_ydata().size > 0 for line in ax2.lines):
                    current_ylim_top = ax2.get_ylim()[1]
                    ax2.set_ylim(bottom=0, top=min(200, current_ylim_top if current_ylim_top > 0 else 200))
                else:
                    ax2.set_ylim(bottom=0, top=200) # Default if no data
            else:
                ax2.set_title('Nutrient % Errors - No Data')
                ax2.axis('off')
        else:
            ax2.set_title('Nutrient % Errors - No Valid Error Data')
            ax2.axis('off')
    else: # history['percentage_errors'] key doesn't exist or is empty
        ax2.set_title('Nutrient % Errors - History Key Missing')
        ax2.axis('off')

    plt.tight_layout()
    plot_path=os.path.join(OUTPUT_DIR,f"plot_training_history_{MODEL_NAME_ARG}.png")
    plt.savefig(plot_path)
    print(f"Plot saved:{plot_path}")
    plt.close(fig)

elif 'history'in locals()and history['train_loss']:print("SAVE_PLOTS False or history empty.Skip training history plot.")
# %%
# Evaluation on FULL Validation Set
# ----------------------------------------------------------------------------
MODEL_SAVE_PATH_EVAL = os.path.join(OUTPUT_DIR, f'best_nutrition_model_{MODEL_NAME_ARG}.pth')
results_df = pd.DataFrame()
print("\n" + "="*30 + f" FINAL EVALUATION ON FULL VALIDATION SET - {MODEL_NAME_ARG} " + "="*30)
if os.path.exists(MODEL_SAVE_PATH_EVAL) and val_loader and val_dataset_instance and len(val_loader) > 0:
    print(f"Loading best model from {MODEL_SAVE_PATH_EVAL} for final evaluation...")
    checkpoint = torch.load(MODEL_SAVE_PATH_EVAL, map_location=DEVICE, weights_only=False)
    state_dict = checkpoint.get('model_state_dict', checkpoint)
    if all(k.startswith('module.') for k in state_dict.keys()):
        state_dict = {k[7:]: v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    model.eval()
    print(f"Model {MODEL_NAME_ARG} (epoch {checkpoint.get('epoch', 'N/A')}, best val loss {checkpoint.get('best_val_loss', 'N/A')}) loaded.")
    if val_dataset_instance.caching_is_active_for_retrieval:
        print("Priming cache for full validation set...")
        val_dataset_instance.prime_cache_for_indices(list(range(len(val_dataset_instance))))
    _, _, predictions_np_eval, labels_np_eval = validate(model, val_loader, criterion, DEVICE, desc="Validating (FULL SET)")
    if predictions_np_eval.shape[0] > 0:
        num_eval_samples = len(predictions_np_eval)
        eval_item_details = val_dataset_instance.items_list[:num_eval_samples]
        all_dish_ids_eval = [item['dish_id'] for item in eval_item_details]
        weight_map = metadata_for_final_eval_weights['weight'].to_dict()
        all_weights_eval = [weight_map.get(did, np.nan) for did in all_dish_ids_eval]
        results_df_data = {'dish_id': all_dish_ids_eval, 'weight': all_weights_eval}
        for i, col_name in enumerate(TARGET_COLUMNS):
            results_df_data[f'{col_name}_pred'] = predictions_np_eval[:, i]
            results_df_data[f'{col_name}_true'] = labels_np_eval[:, i]
        results_df = pd.DataFrame(results_df_data)
        for nutr_root in ['calories', 'fat', 'carbs', 'protein']:
            nutr_col = next((tc for tc in TARGET_COLUMNS if tc.startswith(nutr_root)), None)
            if nutr_col and f'{nutr_col}_pred' in results_df.columns and 'weight' in results_df.columns:
                results_df[f'{nutr_root}_abs_pred'] = results_df[f'{nutr_col}_pred'] * results_df['weight'] / 100
                results_df[f'{nutr_root}_abs_true'] = results_df[f'{nutr_col}_true'] * results_df['weight'] / 100
        print(f"Final evaluation predictions completed for {len(results_df)} views on FULL validation set.")
        
        # Save the results table to CSV
        results_file = os.path.join(OUTPUT_DIR, f"evaluation_results_{MODEL_NAME_ARG}.csv")
        results_df.to_csv(results_file, index=False, float_format='%.3f')
        print(f"Results saved to {results_file}")
    else:
        print("Final evaluation skipped: No predictions from validate on full validation set.")
else:
    if not os.path.exists(MODEL_SAVE_PATH_EVAL):
        print(f"Skip final eval: Model {MODEL_SAVE_PATH_EVAL} not found.")
    elif not (val_loader and val_dataset_instance and len(val_loader) > 0):
        print("Skip final eval: Full val_loader unavailable/empty.")

# Metrics Calculation (on full validation set results)
if not results_df.empty:
    def calculate_metrics(df, target_cols_list, per_100g=True):
        metrics_list = []
        for ncol_key in target_cols_list:
            true_c, pred_c = f'{ncol_key}_true', f'{ncol_key}_pred'
            if not per_100g:
                base_n = ncol_key.split('_per_100g')[0]
                true_c, pred_c = f'{base_n}_abs_true', f'{base_n}_abs_pred'
            if true_c not in df.columns or pred_c not in df.columns:
                continue
            valid_idx = df[[true_c, pred_c]].dropna().index
            if len(valid_idx) == 0:
                continue
            true_v, pred_v = df.loc[valid_idx, true_c].values, df.loc[valid_idx, pred_c].values
            R2_val = r2_score(true_v, pred_v) if len(true_v) >= 2 else np.nan
            mae = mean_absolute_error(true_v, pred_v)
            rmse = np.sqrt(mean_squared_error(true_v, pred_v))
            mean_t = np.mean(true_v)
            perc_err = (mae / (np.abs(mean_t) + 1e-9)) * 100 if mean_t != 0 else (float('inf') if mae > 1e-9 else 0.0)
            metrics_list.append({
                'Nutrient': ncol_key if per_100g else base_n.capitalize(),
                'MAE': mae,
                'RMSE': rmse,
                'R²': R2_val,
                '% Err': perc_err,
                'Mean True': mean_t,
                'Mean Pred': np.mean(pred_v)
            })
        return pd.DataFrame(metrics_list)
    
    print("\nMetrics on FULL Validation Set (Per 100g):")
    metrics_df_100g = calculate_metrics(results_df, TARGET_COLUMNS, per_100g=True)
    if not metrics_df_100g.empty:
        print(metrics_df_100g.to_string(index=False, float_format='%.3f'))
        # Save metrics per 100g to CSV
        metrics_100g_file = os.path.join(OUTPUT_DIR, f"metrics_per_100g_{MODEL_NAME_ARG}.csv")
        metrics_df_100g.to_csv(metrics_100g_file, index=False, float_format='%.3f')
        print(f"Metrics (per 100g) saved to {metrics_100g_file}")
    
    print("\nMetrics on FULL Validation Set (Absolute):")
    metrics_df_abs = calculate_metrics(results_df, TARGET_COLUMNS, per_100g=False)
    if not metrics_df_abs.empty:
        print(metrics_df_abs.to_string(index=False, float_format='%.3f'))
        # Save absolute metrics to CSV
        metrics_abs_file = os.path.join(OUTPUT_DIR, f"metrics_absolute_{MODEL_NAME_ARG}.csv")
        metrics_df_abs.to_csv(metrics_abs_file, index=False, float_format='%.3f')
        print(f"Metrics (absolute) saved to {metrics_abs_file}")
else:
    print("Skipping metrics: results_df from final evaluation empty.")

# %%
# Metrics Calculation (on full validation set results)
# ----------------------------------------------------------------------------
if not results_df.empty:
    def calculate_metrics(df,target_cols_list,per_100g=True):
        metrics_list=[];
        for ncol_key in target_cols_list:
            true_c,pred_c=f'{ncol_key}_true',f'{ncol_key}_pred'
            if not per_100g:base_n=ncol_key.split('_per_100g')[0];true_c,pred_c=f'{base_n}_abs_true',f'{base_n}_abs_pred'
            if true_c not in df.columns or pred_c not in df.columns:continue
            valid_idx=df[[true_c,pred_c]].dropna().index;
            if len(valid_idx)==0:continue
            true_v,pred_v=df.loc[valid_idx,true_c].values,df.loc[valid_idx,pred_c].values
            R2_val=r2_score(true_v,pred_v)if len(true_v)>=2 else np.nan
            mae=mean_absolute_error(true_v,pred_v);rmse=np.sqrt(mean_squared_error(true_v,pred_v));mean_t=np.mean(true_v);perc_err=(mae/(np.abs(mean_t)+1e-9))*100
            metrics_list.append({'Nutrient':ncol_key if per_100g else base_n.capitalize(),'MAE':mae,'RMSE':rmse,'R²':R2_val,'% Err':perc_err,'Mean True':mean_t,'Mean Pred':np.mean(pred_v)})
        return pd.DataFrame(metrics_list)
    print("\nMetrics on FULL Validation Set(Per 100g):");metrics_df_100g=calculate_metrics(results_df,TARGET_COLUMNS,per_100g=True)
    if not metrics_df_100g.empty:print(metrics_df_100g.to_string(index=False,float_format='%.3f'))
    print("\nMetrics on FULL Validation Set(Absolute):");metrics_df_abs=calculate_metrics(results_df,TARGET_COLUMNS,per_100g=False)
    if not metrics_df_abs.empty:print(metrics_df_abs.to_string(index=False,float_format='%.3f'))
else:print("Skipping metrics:results_df from final evaluation empty.")

# %%
# Plotting Predictions vs Actual & Error Distributions (on full validation set results)
# ----------------------------------------------------------------------------
if not results_df.empty and SAVE_PLOTS:
    # Predictions vs True Values plots
    num_targets=len(TARGET_COLUMNS);ncols=2
    nrows_pvsa=(num_targets+ncols-1)//ncols
    fig_pvsa,axes_pvsa=plt.subplots(nrows_pvsa,ncols,figsize=(7*ncols,6*nrows_pvsa),squeeze=False)
    axes_pvsa=axes_pvsa.flatten()
    for i,nutr_key in enumerate(TARGET_COLUMNS):
        ax=axes_pvsa[i]
        true_col,pred_col=f'{nutr_key}_true',f'{nutr_key}_pred'
        if true_col not in results_df.columns or pred_col not in results_df.columns:
            ax.set_title(f'{nutr_key.replace("_per_100g","").capitalize()} - No Data') # Indicate missing data
            ax.axis('off') # Turn off axis if no data
            continue
        plot_df=results_df[[true_col,pred_col]].dropna()
        if plot_df.empty:
            ax.set_title(f'{nutr_key.replace("_per_100g","").capitalize()} - No Valid Data') # Indicate no valid data after dropna
            ax.axis('off')
            continue
        x_p,y_p=plot_df[true_col].values,plot_df[pred_col].values
        ax.scatter(x_p,y_p,alpha=0.6,s=35,edgecolors='k',linewidth=0.5,color='cornflowerblue')
        min_v_data,max_v_data=min(x_p.min(),y_p.min()),max(x_p.max(),y_p.max()) # Renamed to avoid conflict with potential plt.min/max
        # Ensure min_v and max_v define a valid range for the identity line
        if np.isfinite(min_v_data) and np.isfinite(max_v_data) and min_v_data <= max_v_data:
             ax.plot([min_v_data,max_v_data],[min_v_data,max_v_data],'r--',lw=2.5)
        else: # Handle cases with NaN or single point after dropna (though plot_df.empty should catch most)
            # Fallback: plot identity line based on some reasonable default or just skip it
            print(f"Warning: Could not determine valid range for identity line for {nutr_key}.")

        r2_val=r2_score(x_p,y_p)if len(x_p)>1 else float('nan')
        disp_n=nutr_key.replace('_per_100g','').replace('_',' ').capitalize()
        ax.set_xlabel(f'True {disp_n}',fontsize=11)
        ax.set_ylabel(f'Pred {disp_n}',fontsize=11)
        ax.set_title(f'{disp_n} (R²={r2_val:.3f})',fontsize=13)
        ax.grid(True,alpha=0.4,linestyle='--')
    for j in range(num_targets,len(axes_pvsa)): # Delete unused subplots
        fig_pvsa.delaxes(axes_pvsa[j])
    plt.tight_layout(rect=[0,0,1,0.96])
    fig_pvsa.suptitle(f'Predictions vs True Values (Full Val Set, {MODEL_NAME_ARG})',fontsize=16)
    plt.savefig(os.path.join(OUTPUT_DIR,f"plot_final_preds_vs_actual_{MODEL_NAME_ARG}.png"))
    plt.close(fig_pvsa)
    print(f"Plot saved: plot_final_preds_vs_actual_{MODEL_NAME_ARG}.png")

    # Error Distribution plots
    nrows_err=(num_targets+ncols-1)//ncols
    fig_err,axes_err=plt.subplots(nrows_err,ncols,figsize=(9*ncols,5*nrows_err),squeeze=False)
    axes_err=axes_err.flatten()
    for i,nutr_key in enumerate(TARGET_COLUMNS):
        ax=axes_err[i]
        true_col,pred_col=f'{nutr_key}_true',f'{nutr_key}_pred'
        if true_col not in results_df.columns or pred_col not in results_df.columns:
            ax.set_title(f'{nutr_key.replace("_per_100g","").capitalize()} Error - No Data')
            ax.axis('off')
            continue
        err_df=results_df[[true_col,pred_col]].dropna()
        if err_df.empty:
            ax.set_title(f'{nutr_key.replace("_per_100g","").capitalize()} Error - No Valid Data')
            ax.axis('off')
            continue
        
        errors_abs=err_df[pred_col]-err_df[true_col] # Absolute errors
        # Relative errors, handle potential division by zero or near-zero in true values
        rel_errors_raw = np.zeros_like(errors_abs, dtype=float)
        valid_denom_mask = np.abs(err_df[true_col].values) > 1e-9 # Mask for valid denominators
        rel_errors_raw[valid_denom_mask] = (errors_abs[valid_denom_mask] / err_df[true_col].values[valid_denom_mask]) * 100
        rel_errors_raw[~valid_denom_mask] = np.nan # Set to NaN if denominator is too small
        
        rel_err_clean=pd.Series(rel_errors_raw).dropna() # Convert to Series to use .quantile() and handle NaNs
        rel_err_clean=rel_err_clean[np.isfinite(rel_err_clean)] # Ensure finite values

        if len(rel_err_clean)==0:
            ax.set_title(f'{nutr_key.replace("_per_100g","").capitalize()} Error - No Finite Rel. Errors')
            ax.axis('off')
            continue
            
        # Clip outliers for histogram visualization (robust range)
        q_low,q_high=rel_err_clean.quantile(0.02),rel_err_clean.quantile(0.98)
        # Ensure q_low <= q_high, can happen if data is very sparse or mostly identical
        if q_low > q_high: q_low, q_high = q_high, q_low 
        filt_errs=rel_err_clean[(rel_err_clean>=q_low)&(rel_err_clean<=q_high)]
        
        if len(filt_errs)==0: # If filtering removes all data (e.g., all values are identical outside quantiles)
            filt_errs=rel_err_clean # Use original cleaned errors

        if len(filt_errs) > 0: # Check again if there's anything to plot
            ax.hist(filt_errs,bins=40,alpha=0.75,color='mediumseagreen',edgecolor='black')
            ax.axvline(x=0,color='red',linestyle='--',lw=2,label='Zero Err')
            ax.axvline(x=filt_errs.mean(),color='darkblue',linestyle='-',lw=2,label=f'Mean:{filt_errs.mean():.1f}%')
            ax.legend()
        else:
            ax.text(0.5, 0.5, "No data for histogram after filtering", ha='center', va='center')


        disp_n=nutr_key.replace('_per_100g','').replace('_',' ').capitalize()
        ax.set_xlabel('Relative Error (%)',fontsize=11)
        ax.set_ylabel('Frequency',fontsize=11)
        ax.set_title(f'{disp_n} Error Distribution',fontsize=13)
        ax.grid(True,alpha=0.4,linestyle='--')
    for j in range(num_targets,len(axes_err)): # Delete unused subplots
        fig_err.delaxes(axes_err[j])
    plt.tight_layout(rect=[0,0,1,0.96])
    fig_err.suptitle(f'Error Distribution (Full Val Set, {MODEL_NAME_ARG})',fontsize=16)
    plt.savefig(os.path.join(OUTPUT_DIR,f"plot_final_error_dist_{MODEL_NAME_ARG}.png"))
    plt.close(fig_err)
    print(f"Plot saved: plot_final_error_dist_{MODEL_NAME_ARG}.png")
# %%
# Plotting Sample Predictions with Images (on full validation set results)
# ----------------------------------------------------------------------------
if not results_df.empty and SAVE_PLOTS and val_dataset_instance and len(results_df)>=1:
    def show_predictions_with_images(n_samples=6):
        actual_n=min(n_samples,len(results_df));
        if actual_n==0:print("No samples for image predictions plot.");return
        sample_indices_df=np.random.choice(len(results_df),actual_n,replace=False);ncols=min(3,actual_n);nrows=(actual_n+ncols-1)//ncols
        fig,axes=plt.subplots(nrows,ncols,figsize=(6*ncols,6*nrows),squeeze=False);axes=axes.flatten()
        for i_ax,ax_plot in enumerate(axes):
            if i_ax>=actual_n:ax_plot.axis('off');continue
            df_row_idx=sample_indices_df[i_ax];item_info=val_dataset_instance.items_list[df_row_idx];dish_id_p,img_path_p=item_info['dish_id'],item_info['image_path']
            if not os.path.exists(img_path_p):ax_plot.text(0.5,0.5,f"ImgNF:\n{os.path.basename(img_path_p)}",ha='center',va='center',fontsize=9);ax_plot.axis('off');continue
            try:img=Image.open(img_path_p)
            except Exception as e_img:ax_plot.text(0.5,0.5,f"ErrLoadImg:\n{os.path.basename(img_path_p)}\n{e_img}",ha='center',va='center',fontsize=9);ax_plot.axis('off');continue
            ax_plot.imshow(img);ax_plot.axis('off');pred_txt,true_txt="Pred:\n","Actual(Err%):\n";row=results_df.iloc[df_row_idx]
            for nutr_col in TARGET_COLUMNS:pred_v,true_v=row.get(f'{nutr_col}_pred',np.nan),row.get(f'{nutr_col}_true',np.nan);err_pct=abs(pred_v-true_v)/(abs(true_v)+1e-9)*100 if not(np.isnan(pred_v)or np.isnan(true_v))else np.nan;disp_n=nutr_col.split('_')[0].capitalize()[:3];pred_txt+=f"{disp_n}:{pred_v:.0f}\n";true_txt+=f"{disp_n}:{true_v:.0f}({err_pct:.1f}%)\n"
            ax_plot.text(0.02,0.98,pred_txt,transform=ax_plot.transAxes,va='top',ha='left',fontsize=9,bbox=dict(boxstyle='round,pad=0.4',fc='powderblue',alpha=0.85));ax_plot.text(0.98,0.98,true_txt,transform=ax_plot.transAxes,va='top',ha='right',fontsize=9,bbox=dict(boxstyle='round,pad=0.4',fc='lightgoldenrodyellow',alpha=0.85));ax_plot.set_title(f"Dish:{dish_id_p}\n{os.path.basename(img_path_p)}",fontsize=10,y=1.02) # Adjust title y
        for j in range(actual_n,len(axes)):fig.delaxes(axes[j])
        plt.tight_layout(rect=[0,0,1,0.95]);plt.suptitle(f'Sample Predictions (Full Val Set, {MODEL_NAME_ARG})',fontsize=16,y=0.99);sample_plot_path=os.path.join(OUTPUT_DIR,f"plot_final_sample_preds_{MODEL_NAME_ARG}.png");plt.savefig(sample_plot_path);plt.close(fig);print(f"Plot saved:{sample_plot_path}")
    show_predictions_with_images(min(6,len(results_df)))
elif not results_df.empty:print("SAVE_PLOTS False.Skip final sample predictions plot.")
elif results_df.empty and SAVE_PLOTS:print("Results_df empty.Skip final sample predictions plot.")

print("\n--- Script Finished ---")