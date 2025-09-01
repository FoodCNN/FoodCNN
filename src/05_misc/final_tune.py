#!/usr/bin/env python
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
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from torchvision import transforms, models
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from tqdm import tqdm
import time

# --- Configuration ---
BASE_DIR = "/Data/nutrition5k"
YOUR_MODEL_PATH = "/users/eleves-b/2023/georgii.kuznetsov/CNN_nutrition/results_full/best_nutrition_model_ResNetPretrained.pth"
FRIEND_MODEL_PATH = "/users/eleves-b/2023/georgii.kuznetsov/CNN_nutrition/results_full/big_data_deepweightcnn_best.pth"
OUTPUT_DIR = "/users/eleves-b/2023/georgii.kuznetsov/CNN_nutrition/results_finetuned"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- Fine-Tuning Configuration ---
FINETUNED_MODEL_PATH = os.path.join(OUTPUT_DIR, "best_finetuned_combined_model.pth")
NUM_EPOCHS = 60
LEARNING_RATE = 1e-5  # CRITICAL: Use a very low learning rate for fine-tuning
WEIGHT_LOSS_FACTOR = 0.5 # How much to weigh the weight prediction loss vs nutrition loss

# Global Variables and Constants
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 32
TARGET_COLUMNS = ['calories_per_100g', 'fat_per_100g', 'carbs_per_100g', 'protein_per_100g']
RGB_IMAGE_FILENAME_OVERHEAD = "rgb.png"
IMAGERY_BASE_DIR = os.path.join(BASE_DIR, "imagery")
OVERHEAD_IMAGERY_DIR = os.path.join(IMAGERY_BASE_DIR, "realsense_overhead")
METADATA_FILE_CAFE1 = os.path.join(BASE_DIR, "metadata/dish_metadata_cafe1.csv")
METADATA_FILE_CAFE2 = os.path.join(BASE_DIR, "metadata/dish_metadata_cafe2.csv")
MIN_CAL_100G, MAX_CAL_100G = 5, 600
MIN_FAT_100G, MAX_FAT_100G = 1, 80
MIN_CARBS_100G, MAX_CARBS_100G = 1, 100
MIN_PROT_100G, MAX_PROT_100G = 1, 60

print(f"Device: {DEVICE}")
print(f"Starting fine-tuning for {NUM_EPOCHS} epochs with LR={LEARNING_RATE}")
print(f"Fine-tuned model will be saved to: {FINETUNED_MODEL_PATH}")

# --- Model Definitions ---
class ResNetFromScratch(nn.Module):
    # (Same as your original code)
    def __init__(self, num_outputs=4, use_pretrained=False):
        super().__init__()
        weights = models.ResNet34_Weights.IMAGENET1K_V1 if use_pretrained else None
        self.backbone = models.resnet34(weights=weights)
        n_feat = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(nn.Linear(n_feat, 256), nn.ReLU(), nn.Dropout(0.5), nn.Linear(256, num_outputs))
    def forward(self, x): return self.backbone(x)

class DeepWeightCNN(nn.Module):
    # (Same as your original code)
    def __init__(self, num_outputs):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1), nn.BatchNorm2d(256), nn.ReLU(), nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d((1,1)),
        )
        self.regressor = nn.Sequential(nn.Flatten(), nn.Linear(256, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.3),
                                       nn.Linear(128, 64), nn.BatchNorm1d(64), nn.ReLU(), nn.Dropout(0.2), nn.Linear(64, num_outputs))
    def forward(self, x): return self.regressor(self.features(x))

# --- NEW: Combined Model for Fine-Tuning ---
class CombinedSystem(nn.Module):
    def __init__(self, nutrition_model, weight_model):
        super().__init__()
        self.nutrition_model = nutrition_model
        self.weight_model = weight_model

    def forward(self, image_base, image_normalized):
        pred_nutrition_100g = self.nutrition_model(image_normalized)
        pred_weight = self.weight_model(image_base).squeeze(-1)
        pred_absolute_nutrition = pred_nutrition_100g * (pred_weight.unsqueeze(1) / 100.0)
        return pred_absolute_nutrition, pred_weight

# --- Data Loading and Preprocessing (Condensed for brevity) ---
# ... (Same as your original data loading part)
def parse_nutrition_csv(file_path):
    dishes = []
    try:
        with open(file_path, 'r') as f:
            for line in f:
                parts = line.strip().split(',')
                if not parts[0].startswith('dish_'): continue
                try:
                    did, cal, wt, fat, car, pro = parts[0], float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4]), float(parts[5])
                    if wt > 0:
                        dishes.append({'dish_id': did, 'calories': cal, 'weight': wt, 'fat': fat, 'carbs': car, 'protein': pro,
                                       'calories_per_100g': (cal / wt) * 100, 'fat_per_100g': (fat / wt) * 100,
                                       'carbs_per_100g': (car / wt) * 100, 'protein_per_100g': (pro / wt) * 100})
                except (ValueError, IndexError): continue
    except FileNotFoundError: print(f"Error: Metadata file {file_path} not found.")
    return pd.DataFrame(dishes)
# --- Your existing data loading code ---
raw_dish_metadata_df = pd.concat([parse_nutrition_csv(METADATA_FILE_CAFE1), parse_nutrition_csv(METADATA_FILE_CAFE2)], ignore_index=True)
df = raw_dish_metadata_df.replace([np.inf, -np.inf], np.nan).dropna(subset=TARGET_COLUMNS + ['dish_id', 'weight']).set_index('dish_id')

# --- MODIFIED FILTERING LOGIC ---

# 1. Define your original filter conditions
original_conditions = (
    (df['calories_per_100g'] >= MIN_CAL_100G) & (df['calories_per_100g'] <= MAX_CAL_100G) & 
    (df['fat_per_100g'] >= MIN_FAT_100G) & (df['fat_per_100g'] <= MAX_FAT_100G) & 
    (df['carbs_per_100g'] >= MIN_CARBS_100G) & (df['carbs_per_100g'] <= MAX_CARBS_100G) & 
    (df['protein_per_100g'] >= MIN_PROT_100G) & (df['protein_per_100g'] <= MAX_PROT_100G)
)

# 2. Define the ID of the dish you want to add
test_dish_id = 'dish_1560456814'

# 3. Add the test dish to the filter using the OR `|` operator
#    This says "keep rows that meet the original conditions OR have this specific index"
df_with_test_dish = df[original_conditions | (df.index == test_dish_id)]

print(OVERHEAD_IMAGERY_DIR, os.path.exists(os.path.join(OVERHEAD_IMAGERY_DIR, 'dish_1560456814')))

print(f"Shape of df after combined filtering: {df_with_test_dish.shape}")
# You can now use `df_with_test_dish` for the rest of your script

all_dish_ids = [dish_id for dish_id in df_with_test_dish.index if os.path.exists(os.path.join(OVERHEAD_IMAGERY_DIR, dish_id, RGB_IMAGE_FILENAME_OVERHEAD))]
# print([dish_id for dish_id in df_with_test_dish.index])
final_dish_ids_with_verified_images = sorted(list(set(all_dish_ids)))
print(final_dish_ids_with_verified_images.__len__())
print(raw_dish_metadata_df.shape)
labels_dict = {dish_id: list(df_with_test_dish.loc[dish_id][TARGET_COLUMNS].values.astype(np.float32)) + [float(df_with_test_dish.loc[dish_id]['weight'])] for dish_id in final_dish_ids_with_verified_images}

class FineTuningDataset(Dataset):
    def __init__(self, dish_ids, labels_dict, transform_base):
        self.dish_ids = dish_ids
        self.labels_dict = labels_dict
        self.transform_base = transform_base
    def __len__(self): return len(self.dish_ids)
    def __getitem__(self, idx):
        dish_id = self.dish_ids[idx]
        image_path = os.path.join(OVERHEAD_IMAGERY_DIR, dish_id, RGB_IMAGE_FILENAME_OVERHEAD)
        try:
            img_pil = Image.open(image_path).convert("RGB")
            img_tensor_base = self.transform_base(img_pil)
        except Exception: img_tensor_base = torch.zeros((3, 224, 224), dtype=torch.float32)
        label_data = self.labels_dict[dish_id]
        return img_tensor_base, torch.tensor(label_data[:-1]), torch.tensor(label_data[-1]), dish_id

transform_base = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
imagenet_normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

# --- NEW: Create Train/Validation Split ---
train_dish_ids, val_dish_ids = train_test_split(final_dish_ids_with_verified_images, test_size=0.2, random_state=42)
train_dataset = FineTuningDataset(train_dish_ids, labels_dict, transform_base)
val_dataset = FineTuningDataset(val_dish_ids, labels_dict, transform_base)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
print(f"Data loaded: {len(train_dataset)} training samples, {len(val_dataset)} validation samples.")

# --- Load Pre-trained Models ---
nutrition_model = ResNetFromScratch(num_outputs=len(TARGET_COLUMNS), use_pretrained=True)
# Add weights_only=False to tell PyTorch you trust this file
checkpoint_nutrition = torch.load(YOUR_MODEL_PATH, map_location=DEVICE, weights_only=False)
state_dict_nutrition = checkpoint_nutrition.get('model_state_dict', checkpoint_nutrition) # More robust way to get state_dict
nutrition_model.load_state_dict({k.replace('module.',''): v for k, v in state_dict_nutrition.items()})

weight_model = DeepWeightCNN(num_outputs=1)
# Also add weights_only=False here
state_dict_weight = torch.load(FRIEND_MODEL_PATH, map_location=DEVICE, weights_only=False)
weight_model.load_state_dict({k.replace('module.',''): v for k, v in state_dict_weight.items()})

# --- Setup for Fine-Tuning ---
model = CombinedSystem(nutrition_model, weight_model).to(DEVICE)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
criterion_abs_nutr = nn.L1Loss() # MAE Loss for absolute nutrition
criterion_weight = nn.L1Loss() # MAE Loss for weight

# --- Fine-Tuning Loop ---
best_val_loss = float('inf')
print("\n--- Starting Fine-Tuning ---")
for epoch in range(NUM_EPOCHS):
    model.train()
    train_loss = 0.0
    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} [T]", leave=False)
    for images_base, true_100g, true_weight, _ in progress_bar:
        images_base_dev = images_base.to(DEVICE)
        images_norm_dev = imagenet_normalize(images_base_dev)
        true_100g_dev = true_100g.to(DEVICE)
        true_weight_dev = true_weight.to(DEVICE)
        
        optimizer.zero_grad()
        pred_abs, pred_weight = model(images_base_dev, images_norm_dev)
        
        true_abs = true_100g_dev * (true_weight_dev.unsqueeze(1) / 100.0)
        
        loss_abs = criterion_abs_nutr(pred_abs, true_abs)
        loss_w = criterion_weight(pred_weight, true_weight_dev)
        total_loss = loss_abs + WEIGHT_LOSS_FACTOR * loss_w
        
        total_loss.backward()
        optimizer.step()
        train_loss += total_loss.item()
        progress_bar.set_postfix(loss=total_loss.item())

    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for images_base, true_100g, true_weight, _ in tqdm(val_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} [V]", leave=False):
            images_base_dev = images_base.to(DEVICE)
            images_norm_dev = imagenet_normalize(images_base_dev)
            true_100g_dev = true_100g.to(DEVICE)
            true_weight_dev = true_weight.to(DEVICE)
            
            pred_abs, pred_weight = model(images_base_dev, images_norm_dev)
            true_abs = true_100g_dev * (true_weight_dev.unsqueeze(1) / 100.0)
            
            loss_abs = criterion_abs_nutr(pred_abs, true_abs)
            loss_w = criterion_weight(pred_weight, true_weight_dev)
            total_loss = loss_abs + WEIGHT_LOSS_FACTOR * loss_w
            val_loss += total_loss.item()
            
    avg_train_loss = train_loss / len(train_loader)
    avg_val_loss = val_loss / len(val_loader)
    
    print(f"Epoch {epoch+1}/{NUM_EPOCHS} -> Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
    
    if avg_val_loss < best_val_loss:
        print(f"Validation loss improved ({best_val_loss:.4f} -> {avg_val_loss:.4f}). Saving model...")
        best_val_loss = avg_val_loss
        torch.save(model.state_dict(), FINETUNED_MODEL_PATH)

# --- Final Evaluation using the BEST Fine-Tuned Model ---
print("\n--- Starting Final Evaluation with Best Fine-Tuned Model ---")
# Load the best model state
final_model = CombinedSystem(ResNetFromScratch(4, True), DeepWeightCNN(1)).to(DEVICE)
final_model.load_state_dict(torch.load(FINETUNED_MODEL_PATH, map_location=DEVICE))
final_model.eval()

all_results_data = []
with torch.no_grad():
    for images_base, true_100g, true_weight, dish_ids in tqdm(val_loader, desc="Final Evaluation"):
        images_base_dev = images_base.to(DEVICE)
        images_norm_dev = imagenet_normalize(images_base_dev)
        
        pred_abs_batch, pred_weight_batch = final_model(images_base_dev, images_norm_dev)
        true_abs_batch = true_100g * (true_weight.unsqueeze(1) / 100.0)

        pred_abs_np, true_abs_np = pred_abs_batch.cpu().numpy(), true_abs_batch.cpu().numpy()
        pred_weight_np, true_weight_np = pred_weight_batch.cpu().numpy(), true_weight.cpu().numpy()

        for i in range(len(dish_ids)):
            entry = {'dish_id': dish_ids[i], 'pred_weight_g': pred_weight_np[i], 'true_weight_g': true_weight_np[i]}
            for j, name in enumerate(['calories', 'fat', 'carbs', 'protein']):
                entry[f'{name}_abs_pred'] = pred_abs_np[i, j]
                entry[f'{name}_abs_true'] = true_abs_np[i, j]
            all_results_data.append(entry)

results_df = pd.DataFrame(all_results_data)
results_file = os.path.join(OUTPUT_DIR, "finetuned_evaluation_detailed_results.csv")
results_df.to_csv(results_file, index=False, float_format='%.3f')
print(f"\nDetailed fine-tuned results saved to {results_file}")

# --- Metrics Calculation Function ---
def calculate_combined_metrics(df):
    metrics_list = []
    nutrient_bases = ['calories', 'fat', 'carbs', 'protein']
    for base_name in nutrient_bases + ['weight']:
        if base_name == 'weight':
            true_c, pred_c, d_name = 'true_weight_g', 'pred_weight_g', 'Weight (g)'
        else:
            true_c, pred_c, d_name = f'{base_name}_abs_true', f'{base_name}_abs_pred', f'{base_name.capitalize()} (abs)'
        
        true_v, pred_v = df[true_c].values, df[pred_c].values
        mae = mean_absolute_error(true_v, pred_v)
        rmse = np.sqrt(mean_squared_error(true_v, pred_v))
        r2 = r2_score(true_v, pred_v)
        mean_t = np.mean(true_v)
        perc_err = (mae / (np.abs(mean_t) + 1e-9)) * 100
        metrics_list.append({'Nutrient': d_name, 'MAE': mae, 'RMSE': rmse, 'R²': r2, '% Err': perc_err})
    return pd.DataFrame(metrics_list)

print("\n--- Fine-Tuned Model Absolute Nutrition Metrics (on Validation Set) ---")
absolute_metrics_df = calculate_combined_metrics(results_df)
print(absolute_metrics_df.to_string(index=False, float_format='%.3f'))
metrics_file = os.path.join(OUTPUT_DIR, "finetuned_model_absolute_metrics.csv")
absolute_metrics_df.to_csv(metrics_file, index=False, float_format='%.3f')
print(f"Fine-tuned metrics saved to {metrics_file}")

print("\n--- Fine-Tuning and Evaluation Script Finished ---")