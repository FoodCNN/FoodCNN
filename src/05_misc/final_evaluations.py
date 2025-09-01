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
import pickle
import random # For selecting side angle images

# --- Configuration ---
BASE_DIR = "/Data/nutrition5k" # Path to nutrition5k dataset
YOUR_MODEL_PATH = "/users/eleves-b/2023/georgii.kuznetsov/CNN_nutrition/results_full/best_nutrition_model_ResNetPretrained.pth"
FRIEND_MODEL_PATH = "/users/eleves-b/2023/georgii.kuznetsov/CNN_nutrition/results_full/big_data_deepweightcnn_best.pth"
OUTPUT_DIR_COMBINED = "/users/eleves-b/2023/georgii.kuznetsov/CNN_nutrition/results_full_combined"

# --- NEW: Configuration for Image Results ---
IMAGE_RESULTS_DIR = os.path.join(OUTPUT_DIR_COMBINED, "image_results_best")
# Increase the number of images to check and save the best ones
NUM_RESULT_IMAGES_TO_SAVE = 100 

os.makedirs(OUTPUT_DIR_COMBINED, exist_ok=True)
os.makedirs(IMAGE_RESULTS_DIR, exist_ok=True) # Create the new directory

# Global Variables and Constants
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 32
TARGET_COLUMNS = ['calories_per_100g', 'fat_per_100g', 'carbs_per_100g', 'protein_per_100g']
# Increase the validation set size to get a better sample pool
VALIDATION_SET_SIZE = 0.4 

# Image sources
RGB_IMAGE_FILENAME_OVERHEAD = "rgb.jpg" # For overhead
IMAGERY_BASE_DIR = os.path.join(BASE_DIR, "imagery")
OVERHEAD_IMAGERY_DIR = os.path.join(IMAGERY_BASE_DIR, "realsense_overhead")
SIDE_ANGLES_IMAGERY_DIR = os.path.join(IMAGERY_BASE_DIR, "side_angles")
SIDE_ANGLES_SUBDIR_NAME = "extracted_frames"
NUM_SIDE_ANGLES_PER_DISH_ARG = 0 # Max random side angle frames per dish (0 for all, similar to your script)

METADATA_FILE_CAFE1 = os.path.join(BASE_DIR, "metadata/dish_metadata_cafe1.csv")
METADATA_FILE_CAFE2 = os.path.join(BASE_DIR, "metadata/dish_metadata_cafe2.csv")

# Outlier filtering
MIN_CAL_100G = 5; MAX_CAL_100G = 600
MIN_FAT_100G = 1; MAX_FAT_100G = 80
MIN_CARBS_100G = 1; MAX_CARBS_100G = 100
MIN_PROT_100G = 1; MAX_PROT_100G = 60

print(f"Device: {DEVICE}")
print(f"Using your nutrition model: {YOUR_MODEL_PATH}")
print(f"Using friend's weight model: {FRIEND_MODEL_PATH}")
print(f"Saving combined results to: {OUTPUT_DIR_COMBINED}")
print(f"Saving {NUM_RESULT_IMAGES_TO_SAVE} best result images to: {IMAGE_RESULTS_DIR}")
print(f"Using {VALIDATION_SET_SIZE*100:.0f}% of data for validation.")

# --- Model Definitions ---
class ResNetFromScratch(nn.Module):
    def __init__(self, num_outputs=4, use_pretrained=False):
        super().__init__()
        weights = models.ResNet34_Weights.IMAGENET1K_V1 if use_pretrained else None
        self.backbone = models.resnet34(weights=weights)
        n_feat = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Linear(n_feat, 256), nn.ReLU(), nn.Dropout(0.5), nn.Linear(256, num_outputs)
        )
    def forward(self, x): return self.backbone(x)

class DeepWeightCNN(nn.Module):
    def __init__(self, num_outputs):
        super().__init__()
        self.name = "big_data_DeepWeightCNN"
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1), nn.BatchNorm2d(256), nn.ReLU(), nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d((1,1)),
        )
        self.regressor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, 64), nn.BatchNorm1d(64), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(64, num_outputs),
        )
    def forward(self, x): x = self.features(x); return self.regressor(x)

# --- Data Preprocessing (Metadata) ---
def parse_nutrition_csv(file_path):
    dishes = []
    try:
        with open(file_path, 'r') as f:
            for line in f:
                parts = line.strip().split(',')
                if not parts[0].startswith('dish_'): continue
                try:
                    did, cal, wt, fat, car, pro = parts[0], float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4]), float(parts[5])
                except (ValueError, IndexError): continue
                if wt <= 0: continue
                dishes.append({
                    'dish_id': did, 'calories': cal, 'weight': wt, 'fat': fat, 'carbs': car, 'protein': pro,
                    'calories_per_100g': (cal / wt) * 100, 'fat_per_100g': (fat / wt) * 100,
                    'carbs_per_100g': (car / wt) * 100, 'protein_per_100g': (pro / wt) * 100
                })
    except FileNotFoundError: print(f"Error: Metadata file {file_path} not found.")
    return pd.DataFrame(dishes)

raw_dish_metadata_df = pd.concat([parse_nutrition_csv(METADATA_FILE_CAFE1), parse_nutrition_csv(METADATA_FILE_CAFE2)], ignore_index=True)
all_dish_metadata_df_unfiltered = raw_dish_metadata_df.replace([np.inf, -np.inf], np.nan).dropna(
    subset=TARGET_COLUMNS + ['dish_id', 'weight']
).set_index('dish_id')

all_dish_metadata_df_filtered = all_dish_metadata_df_unfiltered[
    (all_dish_metadata_df_unfiltered['calories_per_100g'] >= MIN_CAL_100G) & (all_dish_metadata_df_unfiltered['calories_per_100g'] <= MAX_CAL_100G) &
    (all_dish_metadata_df_unfiltered['fat_per_100g'] >= MIN_FAT_100G) & (all_dish_metadata_df_unfiltered['fat_per_100g'] <= MAX_FAT_100G) &
    (all_dish_metadata_df_unfiltered['carbs_per_100g'] >= MIN_CARBS_100G) & (all_dish_metadata_df_unfiltered['carbs_per_100g'] <= MAX_CARBS_100G) &
    (all_dish_metadata_df_unfiltered['protein_per_100g'] >= MIN_PROT_100G) & (all_dish_metadata_df_unfiltered['protein_per_100g'] <= MAX_PROT_100G)
]
print(f"Filtered metadata rows: {len(all_dish_metadata_df_filtered)}")
if all_dish_metadata_df_filtered.empty: print("CRITICAL: All data filtered out."); sys.exit(1)

# --- Collect Image Paths ---
# ... (This part of the code is unchanged)
candidate_image_info = []
for dish_id, row_data in tqdm(all_dish_metadata_df_filtered.iterrows(), total=len(all_dish_metadata_df_filtered), desc="Identifying existing images"):
    overhead_img_path = os.path.join(OVERHEAD_IMAGERY_DIR, dish_id, RGB_IMAGE_FILENAME_OVERHEAD)
    if os.path.exists(overhead_img_path):
        candidate_image_info.append({'dish_id': dish_id, 'image_path': overhead_img_path, 'image_type': 'overhead'})
    side_angle_dish_dir = os.path.join(SIDE_ANGLES_IMAGERY_DIR, dish_id, SIDE_ANGLES_SUBDIR_NAME)
    if os.path.isdir(side_angle_dish_dir):
        available_frames = [os.path.join(side_angle_dish_dir, f) for f in os.listdir(side_angle_dish_dir) if f.startswith("camera_") and f.endswith(".jpg")]
        if available_frames:
            n_to_select = NUM_SIDE_ANGLES_PER_DISH_ARG if NUM_SIDE_ANGLES_PER_DISH_ARG > 0 else len(available_frames)
            selected_frames = random.sample(available_frames, min(len(available_frames), n_to_select))
            for frame_path in selected_frames:
                candidate_image_info.append({'dish_id': dish_id, 'image_path': frame_path, 'image_type': 'side_angle'})
verified_image_info = []
for item_info in tqdm(candidate_image_info, desc="Verifying image readability"):
    try:
        with Image.open(item_info['image_path']) as img: img.verify()
        verified_image_info.append(item_info)
    except Exception: pass
if not verified_image_info: print("CRITICAL: No valid images found."); sys.exit(1)

# --- Final Dataset Construction ---
# ... (This part of the code is unchanged)
final_dish_ids_with_verified_images = sorted(list(set(item['dish_id'] for item in verified_image_info)))
all_labels_list = []
dish_id_to_label_idx = {}
for dish_id in final_dish_ids_with_verified_images:
    row_data = all_dish_metadata_df_filtered.loc[dish_id]
    dish_id_to_label_idx[dish_id] = len(all_labels_list)
    all_labels_list.append(list(row_data[TARGET_COLUMNS].values.astype(np.float32)) + [float(row_data['weight'])])
all_labels_array = np.array(all_labels_list, dtype=np.float32) if all_labels_list else np.array([])
dataset_items = []
for item_info in verified_image_info:
    dish_id = item_info['dish_id']
    if dish_id in dish_id_to_label_idx:
        label_idx = dish_id_to_label_idx[dish_id]
        dataset_items.append({'dish_id': dish_id, 'label_idx': label_idx, 'image_type': item_info['image_type'], 'image_path': item_info['image_path']})
print(f"Total dataset items (views) after verification: {len(dataset_items)}")

# --- Dataset and DataLoader ---
class CombinedEvaluationDataset(Dataset):
    def __init__(self, items_list, all_labels_array_source, transform_base=None):
        self.items_list = items_list
        self.all_labels_array_source = all_labels_array_source
        self.transform_base = transform_base
    def __len__(self): return len(self.items_list)
    def __getitem__(self, idx):
        item = self.items_list[idx]
        image_path = item['image_path']
        label_idx = item['label_idx']
        try:
            img_pil = Image.open(image_path).convert("RGB")
            img_tensor_base = self.transform_base(img_pil) if self.transform_base else transforms.ToTensor()(img_pil)
        except Exception as e:
            print(f"WARNING: Error loading image {image_path}: {e}. Returning zero tensor.")
            img_tensor_base = torch.zeros((3, 224, 224), dtype=torch.float32)
        label_data = self.all_labels_array_source[label_idx]
        nutrition_per_100g_true = torch.tensor(label_data[:-1], dtype=torch.float32)
        weight_true = torch.tensor(label_data[-1], dtype=torch.float32)
        return img_tensor_base, nutrition_per_100g_true, weight_true, item['dish_id'], item['image_type']

transform_base_for_dataset = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
imagenet_normalize_transform = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

if dataset_items and all_labels_array.size > 0:
    unique_dish_ids = sorted(list(set(item['dish_id'] for item in dataset_items)))
    if len(unique_dish_ids) >=5 :
        # Use the new validation set size
        _, val_dish_ids_split = train_test_split(unique_dish_ids, test_size=VALIDATION_SET_SIZE, random_state=42)
    else: val_dish_ids_split = unique_dish_ids
    val_items_final = [item for item in dataset_items if item['dish_id'] in val_dish_ids_split]
    if val_items_final:
        val_dataset_combined = CombinedEvaluationDataset(val_items_final, all_labels_array, transform_base_for_dataset)
        val_loader_combined = DataLoader(val_dataset_combined, batch_size=BATCH_SIZE, shuffle=False, num_workers=0) # Use shuffle=False for predictable results
        print(f"Combined evaluation DataLoader created with {len(val_dataset_combined)} items.")
    else: print("CRITICAL: No items for combined validation set."); val_loader_combined = None
else: print("CRITICAL: Skipping DataLoader creation."); val_loader_combined = None

# --- Load Models ---
# ... (This part of the code is unchanged)
nutrition_model = ResNetFromScratch(num_outputs=len(TARGET_COLUMNS), use_pretrained=True).to(DEVICE)
try:
    checkpoint_nutrition = torch.load(YOUR_MODEL_PATH, map_location=DEVICE, weights_only=False)
    state_dict_nutrition = checkpoint_nutrition.get('model_state_dict', checkpoint_nutrition)
    if all(k.startswith('module.') for k in state_dict_nutrition.keys()):
        state_dict_nutrition = {k[7:]: v for k, v in state_dict_nutrition.items()}
    nutrition_model.load_state_dict(state_dict_nutrition)
    nutrition_model.eval()
    print(f"Nutrition model loaded successfully from {YOUR_MODEL_PATH}")
except Exception as e: print(f"Error loading nutrition model: {e}"); sys.exit(1)
weight_model = DeepWeightCNN(num_outputs=1).to(DEVICE)
try:
    state_dict_weight = torch.load(FRIEND_MODEL_PATH, map_location=DEVICE)
    if all(k.startswith('module.') for k in state_dict_weight.keys()):
         state_dict_weight = {k[7:]: v for k, v in state_dict_weight.items()}
    weight_model.load_state_dict(state_dict_weight)
    weight_model.eval()
    print(f"Weight model loaded successfully from {FRIEND_MODEL_PATH}")
except Exception as e: print(f"Error loading weight model: {e}"); sys.exit(1)

# --- Function to Generate Result Images with Error ---
def create_result_image(image_tensor, pred_data, true_data, dish_id, image_type, output_path):
    """Generates and saves an image comparing predicted and true values with percentage error."""
    img_np = image_tensor.cpu().permute(1, 2, 0).numpy()
    fig, axs = plt.subplots(1, 2, figsize=(14, 6), gridspec_kw={'width_ratios': [1, 1]})
    
    axs[0].imshow(img_np)
    axs[0].axis('off')
    axs[0].set_title(f"Input Image ({image_type.replace('_', ' ').title()})", fontsize=12)
    axs[1].axis('off')
    
    # Helper to calculate percentage error
    def get_err(pred, true):
        return abs(pred - true) / (abs(true) + 1e-9) * 100

    pred_text = (
        f"Calories: {pred_data['calories_abs']:.1f} kcal (Err: {get_err(pred_data['calories_abs'], true_data['calories_abs']):.1f}%)\n"
        f"Fat:      {pred_data['fat_abs']:.1f} g    (Err: {get_err(pred_data['fat_abs'], true_data['fat_abs']):.1f}%)\n"
        f"Carbs:    {pred_data['carbs_abs']:.1f} g    (Err: {get_err(pred_data['carbs_abs'], true_data['carbs_abs']):.1f}%)\n"
        f"Protein:  {pred_data['protein_abs']:.1f} g    (Err: {get_err(pred_data['protein_abs'], true_data['protein_abs']):.1f}%)\n\n"
        f"Weight:   {pred_data['weight_g']:.1f} g    (Err: {get_err(pred_data['weight_g'], true_data['weight_g']):.1f}%)"
    )
    
    true_text = (
        f"Calories: {true_data['calories_abs']:.1f} kcal\n"
        f"Fat:      {true_data['fat_abs']:.1f} g\n"
        f"Carbs:    {true_data['carbs_abs']:.1f} g\n"
        f"Protein:  {true_data['protein_abs']:.1f} g\n\n"
        f"Weight:   {true_data['weight_g']:.1f} g"
    )

    y_pos = 0.9
    axs[1].text(0.05, y_pos, "Prediction", fontsize=14, fontweight='bold', va='top')
    axs[1].text(0.05, y_pos - 0.08, pred_text, fontsize=12, va='top', fontfamily='monospace', linespacing=1.5)
    
    y_pos = 0.4
    axs[1].text(0.05, y_pos, "Ground Truth", fontsize=14, fontweight='bold', va='top')
    axs[1].text(0.05, y_pos - 0.08, true_text, fontsize=12, va='top', fontfamily='monospace', linespacing=1.5)

    fig.suptitle(f"Result for Dish ID: {dish_id}", fontsize=16, fontweight='bold')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    try:
        plt.savefig(output_path, bbox_inches='tight', dpi=150)
    except Exception as e:
        print(f"Warning: Could not save result image {output_path}. Error: {e}")
    plt.close(fig)

# --- Individual Model Sanity Check ---
# ... (This section is unchanged and will now use the larger validation set)
if val_loader_combined:
    print("\n--- Starting Individual Model Sanity Checks ---")
    all_true_nutrition_100g_list, all_pred_nutrition_100g_list = [], []
    all_true_weight_list, all_pred_weight_list = [], []
    with torch.no_grad():
        for images_base, true_nutrition_100g_batch, true_weight_batch, _, _ in tqdm(val_loader_combined, desc="Individual Model Predictions"):
            images_base_dev = images_base.to(DEVICE)
            images_norm_dev = imagenet_normalize_transform(images_base_dev)
            pred_nutrition_100g_batch = nutrition_model(images_norm_dev)
            all_pred_nutrition_100g_list.append(pred_nutrition_100g_batch.cpu())
            all_true_nutrition_100g_list.append(true_nutrition_100g_batch.cpu())
            pred_weight_batch = weight_model(images_base_dev)
            all_pred_weight_list.append(pred_weight_batch.cpu())
            all_true_weight_list.append(true_weight_batch.cpu().unsqueeze(-1))
    # Helper for individual metrics (unchanged from your original)
    def calculate_individual_model_metrics(true_values_array, pred_values_array, target_names_list, model_name_str):
        metrics_summary = []
        for i in range(true_values_array.shape[1]):
            target_name, true_v, pred_v = target_names_list[i], true_values_array[:, i], pred_values_array[:, i]
            valid_idx = ~np.isnan(true_v) & ~np.isnan(pred_v)
            true_v, pred_v = true_v[valid_idx], pred_v[valid_idx]
            if len(true_v) == 0: continue
            R2_val, mae, rmse = (r2_score(true_v, pred_v), mean_absolute_error(true_v, pred_v), np.sqrt(mean_squared_error(true_v, pred_v)))
            mean_t = np.mean(true_v)
            perc_err = (mae / (np.abs(mean_t) + 1e-9)) * 100
            metrics_summary.append({'Target': target_name, 'MAE': mae, 'RMSE': rmse, 'R²': R2_val, '% Err': perc_err, 'Mean True': mean_t, 'Mean Pred': np.mean(pred_v)})
        metrics_df = pd.DataFrame(metrics_summary)
        print(f"\n--- {model_name_str} Individual Performance Metrics ---"); print(metrics_df.to_string(index=False, float_format='%.3f'))
    if all_pred_nutrition_100g_list:
        calculate_individual_model_metrics(torch.cat(all_true_nutrition_100g_list).numpy(), torch.cat(all_pred_nutrition_100g_list).numpy(), [t.replace('_per_100g', '/100g').capitalize() for t in TARGET_COLUMNS], "Nutrition Model (per 100g)")
    if all_pred_weight_list:
        calculate_individual_model_metrics(torch.cat(all_true_weight_list).numpy(), torch.cat(all_pred_weight_list).numpy(), ["Weight (g)"], "Weight Model")
else: print("Skipping individual model sanity checks: val_loader_combined not available.")


# --- Combined Evaluation Loop (Collect, then Sort, then Generate) ---
all_results_data = []
evaluation_samples_for_imaging = [] # Store all data needed for image generation

if val_loader_combined:
    print("\n--- Phase 1: Running Full Evaluation to Collect All Results ---")
    with torch.no_grad():
        for images_base, true_nutrition_100g_batch, true_weight_batch, dish_ids_batch, image_types_batch in tqdm(val_loader_combined, desc="Evaluating All Samples"):
            images_base_dev = images_base.to(DEVICE)
            images_norm_dev = imagenet_normalize_transform(images_base_dev)
            
            pred_nutrition_100g_batch = nutrition_model(images_norm_dev)
            pred_weight_batch = weight_model(images_base_dev).squeeze(-1)

            pred_absolute_nutrition_batch = pred_nutrition_100g_batch * (pred_weight_batch.unsqueeze(1) / 100.0)
            true_absolute_nutrition_batch = true_nutrition_100g_batch * (true_weight_batch.unsqueeze(1) / 100.0)
            
            pred_abs_np, true_abs_np = pred_absolute_nutrition_batch.cpu().numpy(), true_absolute_nutrition_batch.cpu().numpy()
            pred_weight_np, true_weight_np = pred_weight_batch.cpu().numpy(), true_weight_batch.cpu().numpy()

            for i in range(images_base.size(0)):
                # Store data for CSV results
                entry = {'dish_id': dish_ids_batch[i], 'pred_weight_g': pred_weight_np[i], 'true_weight_g': true_weight_np[i]}
                for j, nutr_base_name in enumerate(['calories', 'fat', 'carbs', 'protein']):
                    entry[f'{nutr_base_name}_abs_pred'] = pred_abs_np[i, j]
                    entry[f'{nutr_base_name}_abs_true'] = true_abs_np[i, j]
                all_results_data.append(entry)

                # Store all necessary data for later image generation
                pred_data_dict = {
                    'calories_abs': pred_abs_np[i, 0], 'fat_abs': pred_abs_np[i, 1],
                    'carbs_abs': pred_abs_np[i, 2], 'protein_abs': pred_abs_np[i, 3],
                    'weight_g': pred_weight_np[i]
                }
                true_data_dict = {
                    'calories_abs': true_abs_np[i, 0], 'fat_abs': true_abs_np[i, 1],
                    'carbs_abs': true_abs_np[i, 2], 'protein_abs': true_abs_np[i, 3],
                    'weight_g': true_weight_np[i]
                }
                
                # Calculate overall MAPE for this sample
                errors = [abs(pred_data_dict[k] - true_data_dict[k]) / (abs(true_data_dict[k]) + 1e-9) for k in true_data_dict]
                mape_score = np.mean(errors) * 100

                evaluation_samples_for_imaging.append({
                    'image_tensor': images_base[i].cpu(), # Store on CPU to save GPU memory
                    'pred_data': pred_data_dict,
                    'true_data': true_data_dict,
                    'dish_id': dish_ids_batch[i],
                    'image_type': image_types_batch[i],
                    'mape_score': mape_score
                })

# --- Phase 2: Sort Results and Generate Best Images ---
if evaluation_samples_for_imaging:
    print("\n--- Phase 2: Sorting Results to Find Best Performing Images ---")
    evaluation_samples_for_imaging.sort(key=lambda x: x['mape_score'])
    
    num_to_save = min(NUM_RESULT_IMAGES_TO_SAVE, len(evaluation_samples_for_imaging))
    print(f"Generating the top {num_to_save} result images...")

    for i, sample in enumerate(tqdm(evaluation_samples_for_imaging[:num_to_save], desc="Generating Best Images")):
        safe_dish_id = sample['dish_id'].replace('/', '_')
        # Create a more informative filename
        image_filename = f"rank_{i+1:03d}_err_{sample['mape_score']:.1f}pct_{safe_dish_id}_{sample['image_type']}.png"
        output_path = os.path.join(IMAGE_RESULTS_DIR, image_filename)
        
        create_result_image(
            image_tensor=sample['image_tensor'],
            pred_data=sample['pred_data'],
            true_data=sample['true_data'],
            dish_id=sample['dish_id'],
            image_type=sample['image_type'],
            output_path=output_path
        )
    print(f"Finished generating {num_to_save} result images.")


# --- Save Detailed CSV and Calculate Final Metrics ---
results_df = pd.DataFrame(all_results_data)
if not results_df.empty:
    results_file = os.path.join(OUTPUT_DIR_COMBINED, "combined_evaluation_detailed_results.csv")
    results_df.to_csv(results_file, index=False, float_format='%.3f')
    print(f"\nDetailed combined results saved to {results_file}")

    def calculate_combined_metrics(df, nutrient_bases=['calories', 'fat', 'carbs', 'protein']):
        metrics_list = []
        for base_name in nutrient_bases + ['weight']: 
            if base_name == 'weight': true_c, pred_c, display_name = 'true_weight_g', 'pred_weight_g', 'Weight (g)'
            else: true_c, pred_c, display_name = f'{base_name}_abs_true', f'{base_name}_abs_pred', f'{base_name.capitalize()} (abs)'
            valid_idx = df[[true_c, pred_c]].dropna().index
            if len(valid_idx) == 0: continue
            true_v, pred_v = df.loc[valid_idx, true_c].values, df.loc[valid_idx, pred_c].values
            R2_val, mae, rmse = r2_score(true_v, pred_v), mean_absolute_error(true_v, pred_v), np.sqrt(mean_squared_error(true_v, pred_v))
            mean_t = np.mean(true_v)
            perc_err = (mae / (np.abs(mean_t) + 1e-9)) * 100
            metrics_list.append({'Nutrient': display_name, 'MAE': mae, 'RMSE': rmse, 'R²': R2_val, '% Err': perc_err, 'Mean True': mean_t, 'Mean Pred': np.mean(pred_v)})
        return pd.DataFrame(metrics_list)

    print("\n--- Combined Model Absolute Nutrition Metrics (on Full Validation Set) ---")
    absolute_metrics_df = calculate_combined_metrics(results_df)
    if not absolute_metrics_df.empty:
        print(absolute_metrics_df.to_string(index=False, float_format='%.3f'))
        metrics_file = os.path.join(OUTPUT_DIR_COMBINED, "combined_model_absolute_metrics.csv")
        absolute_metrics_df.to_csv(metrics_file, index=False, float_format='%.3f')
        print(f"Absolute metrics saved to {metrics_file}")
    else: print("Could not calculate absolute metrics.")
else: print("Skipping metrics calculation: results_df empty.")

print("\n--- Combined Evaluation Script Finished ---")