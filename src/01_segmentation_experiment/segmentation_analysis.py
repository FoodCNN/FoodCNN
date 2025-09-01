# %% [markdown]
# # Calorie Prediction on Nutrition5k with segmentation
# This notebook implements a pipeline for calorie/macronutrient prediction from Nutrition5k dish images.
# It uses U-2-Net for food segmentation before training a
# ResNet-based CNN for calorie/macronutrient regression.

# %% [markdown]
# ## 1. Imports and Setup

# %%
import os
import logging
import pandas as pd
import numpy as np
from PIL import Image, ImageOps
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import multiprocessing
import cv2
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional, Any

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import resnet50, ResNet50_Weights
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
# Attempt to import U-2-Net model structure
# You need to ensure 'u2net_model.py' is in your PYTHONPATH or the same directory.
# This is a placeholder import. You'll need the actual U2NET model definition.
try:
    from u2net_model import U2NET # Or U2NETP, depending on the model used
    logger.info("Successfully imported U2NET from u2net_model.py")
except ImportError:
    logger.warning("Could not import U2NET from u2net_model.py. Segmentation will not work.")
    logger.warning("Please ensure u2net_model.py (containing the U2NET class definition) is in your project directory or Python path.")
    U2NET = None # Placeholder if import fails



# Ensure PyTorch multiprocessing works correctly
if multiprocessing.get_start_method(allow_none=True) != 'spawn':
    try:
        multiprocessing.set_start_method('spawn', force=True)
        logger.info("Set multiprocessing start method to 'spawn'.")
    except RuntimeError as e:
        logger.warning(f"Could not set multiprocessing start method to 'spawn': {e}")


def parse_nutrition_csv(file_path: str) -> pd.DataFrame:
    """
    Parses the Nutrition5k dish metadata CSV file which has an irregular structure.
    Extracts dish_id, calories, weight, fat, carbs, and protein.
    """
    dishes_data = []
    try:
        with open(file_path, "r") as f:
            for line in f:
                parts = line.strip().split(",")
                if not parts or not parts[0].startswith("dish_"): # Ensure parts is not empty and starts with "dish_"
                    continue
                try:
                    dish_id = parts[0]
                    # Ensure there are enough parts for all expected values
                    if len(parts) > 5:
                        dishes_data.append({
                            "dish_id": dish_id,
                            "calories": float(parts[1]),
                            "weight": float(parts[2]), # Mass
                            "fat": float(parts[3]),
                            "carbs": float(parts[4]),
                            "protein": float(parts[5]),
                        })
                    else:
                        logger.warning(f"Skipping malformed 'dish_' line (not enough parts) in {file_path}: {line.strip()}")
                except ValueError as e:
                    logger.warning(f"ValueError for 'dish_' line in {file_path}: {line.strip()} - {e}")
                except IndexError as e:
                    logger.warning(f"IndexError for 'dish_' line in {file_path}: {line.strip()} - {e}")
    except FileNotFoundError:
        logger.error(f"Metadata file not found: {file_path}")
        return pd.DataFrame() # Return empty DataFrame if file not found
    except Exception as e:
        logger.error(f"An unexpected error occurred while parsing {file_path}: {e}", exc_info=True)
        return pd.DataFrame()

    return pd.DataFrame(dishes_data)

# %% [markdown]
#  ## 2. Configuration
# %%
@dataclass
class Config:
    """Configuration class for the training pipeline."""
    # Paths
    BASE_DIR: str = "./datasets/nutrition5k" # UPDATE THIS PATH
    IMAGERY_DIR: str = field(init=False)
    METADATA_FILE_CAFE1: str = field(init=False)
    METADATA_FILE_CAFE2: str = field(init=False)
    RGB_IMAGE_FILENAME: str = "rgb.png"

    # Segmentation Model (U-2-Net)
    # IMPORTANT: Ensure u2net_model.py is present and U2NET_WEIGHTS_PATH points to the .pth file
    U2NET_MODEL_DIR: str = "./models/" # Directory expected to contain u2net_model.py
    U2NET_WEIGHTS_PATH: str = "./models/u2net.pth" # UPDATE if using a different name or path for U-2-Net weights
    SEGMENTATION_INPUT_SIZE: Tuple[int, int] = (320, 320) # U-2-Net typical input size

    # Processed Data Paths
    PROCESSED_SEGMENTED_DIR: str = field(init=False) # For saving/loading segmented images
    PRECOMPUTE_SEGMENTATION: bool = True # Master switch to use and save segmentation
    OVERWRITE_EXISTING_SEGMENTATION: bool = False # If True, re-segments even if file exists

    # CNN Model & Training
    DEVICE: torch.device = field(default_factory=lambda: torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    NUM_WORKERS: int = min(4, os.cpu_count() // 2 if os.cpu_count() else 1)
    PIN_MEMORY: bool = field(default_factory=lambda: torch.cuda.is_available())
    BATCH_SIZE: int = 16
    LEARNING_RATE: float = 1e-3
    NUM_EPOCHS: int = 100
    PATIENCE_EARLY_STOPPING: int = 10
    TARGET_COLUMN: str = "calories"
    RANDOM_STATE: int = 42
    CNN_INPUT_IMG_SIZE: Tuple[int, int] = (224, 224)
    
    # GPU Caching
    USE_GPU_CACHING: bool = True # <--- ADD THIS
    GPU_CACHE_SAFETY_MARGIN: float = 0.8 # Use 80% of free VRAM for caching estimate

    def __post_init__(self):
        self.IMAGERY_DIR = os.path.join(self.BASE_DIR, "imagery/realsense_overhead")
        self.PROCESSED_SEGMENTED_DIR = os.path.join(self.BASE_DIR, "imagery/u2net_segmented_food")
        self.METADATA_FILE_CAFE1 = os.path.join(self.BASE_DIR, "metadata/dish_metadata_cafe1.csv")
        self.METADATA_FILE_CAFE2 = os.path.join(self.BASE_DIR, "metadata/dish_metadata_cafe2.csv")

        if self.DEVICE.type == "cuda":
            torch.backends.cudnn.benchmark = True
        logger.info(f"Configuration initialized. Device set to: {self.DEVICE}")
        if self.PRECOMPUTE_SEGMENTATION:
            logger.info(f"U-2-Net segmentation enabled. Processed images stored in: {self.PROCESSED_SEGMENTED_DIR}")
            os.makedirs(self.PROCESSED_SEGMENTED_DIR, exist_ok=True)
        if self.USE_GPU_CACHING and self.DEVICE.type == 'cuda':
            logger.info(f"GPU Caching for datasets is ENABLED.")
        elif self.USE_GPU_CACHING and self.DEVICE.type == 'cpu':
            logger.warning("GPU Caching requested but device is CPU. Caching will not occur.")
            self.USE_GPU_CACHING = False

config = Config()

# %% [markdown]
# ## 3. U-2-Net Segmentation Module
#
# %%

def load_u2net_model(model_path: str, device: torch.device) -> Optional[nn.Module]:
    """Loads the U-2-Net model and its pretrained weights."""
    if U2NET is None:
        logger.error("U2NET class is not available (failed import). Cannot load model.")
        return None
    try:
        # Assuming U2NET class takes in_ch=3 (RGB) and out_ch=1 (grayscale mask)
        model = U2NET(in_ch=3, out_ch=1)
        if torch.cuda.is_available():
            model.load_state_dict(torch.load(model_path, map_location=device))
        else:
            model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        model.to(device)
        model.eval()
        logger.info(f"U-2-Net model loaded successfully from {model_path} to {device}.")
        return model
    except FileNotFoundError:
        logger.error(f"U-2-Net weights file not found at {model_path}.")
        return None
    except Exception as e:
        logger.error(f"Error loading U-2-Net model: {e}", exc_info=True)
        return None

def get_segmentation_mask_u2net(image_pil: Image.Image,
                                u2net_model: nn.Module,
                                input_size: Tuple[int, int] = (320, 320),
                                device: torch.device = torch.device("cpu")) -> Optional[Image.Image]:
    """
    Generates a food segmentation mask using a preloaded U-2-Net model.
    Returns a PIL Image (grayscale mask).
    """
    if u2net_model is None:
        return None

    original_size = image_pil.size # (width, height)

    # Prepare image for U-2-Net
    img_tensor = transforms.Compose([
        transforms.Resize(input_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]), # Typical ImageNet normalization
    ])(image_pil).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = u2net_model(img_tensor)
    
    # U-2-Net typically returns multiple outputs (d0, d1, ... d6)
    # d0 is usually the finest, full-resolution mask
    pred_mask_tensor = outputs[0].squeeze().cpu().detach() # Get d0, remove batch, to CPU

    # Normalize mask to 0-1 range (it's usually already close after sigmoid in model)
    pred_mask_tensor = (pred_mask_tensor - torch.min(pred_mask_tensor)) / \
                       (torch.max(pred_mask_tensor) - torch.min(pred_mask_tensor) + 1e-8)

    mask_pil = transforms.ToPILImage()(pred_mask_tensor.unsqueeze(0)) # Add channel dim back for ToPILImage
    
    # Resize mask to original image size
    mask_pil = mask_pil.resize(original_size, Image.BILINEAR)
    return mask_pil.convert("L") # Convert to grayscale

def apply_mask_and_crop(original_image_pil: Image.Image, mask_pil: Image.Image, threshold: float = 0.5) -> Image.Image:
    """
    Applies a binary mask to an image and crops to the bounding box of the masked area.
    """
    binary_mask_np = (np.array(mask_pil) / 255.0 > threshold).astype(np.uint8)
    
    # Find contours to get bounding box
    contours, _ = cv2.findContours(binary_mask_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        logger.warning("No contours found in mask. Returning original image (possibly uncropped).")
        # Fallback: create a black image or return a centrally cropped original
        # For simplicity here, we'll make it so the whole image is kept if no contour
        # which means it will just be resized later by the CNN's transforms.
        # A better fallback might be a central crop of the original.
        return original_image_pil

    # Get bounding box of the largest contour (assuming it's the main food item)
    largest_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest_contour)

    if w == 0 or h == 0: # Should not happen if contours were found
        return original_image_pil


    # Create a 3-channel version of the binary mask for applying to RGB image
    mask_rgb_np = np.stack([binary_mask_np*255]*3, axis=-1) # PIL expects 0-255 for mask
    mask_for_pil = Image.fromarray(mask_rgb_np.astype(np.uint8)).convert("1") # Convert to binary PIL mask


    # Apply mask (make background black)
    # Create a black background image
    black_background = Image.new("RGB", original_image_pil.size, (0,0,0))
    # Composite original image onto black background using the mask
    masked_image_pil = Image.composite(original_image_pil, black_background, mask_for_pil)


    # Crop to bounding box
    cropped_image_pil = masked_image_pil.crop((x, y, x + w, y + h))
    
    if cropped_image_pil.width == 0 or cropped_image_pil.height == 0:
        logger.warning("Cropped image has zero dimension. Returning original masked (uncropped) image.")
        return masked_image_pil # Fallback to uncropped masked image

    return cropped_image_pil


# %% [markdown]
#  ## 4. Nutrition5k Dataset Class
# %%
class Nutrition5kDataset(Dataset):
    def __init__(self,
                 metadata_df: pd.DataFrame,
                 imagery_dir: str,
                 processed_dir: str,
                 rgb_filename: str,
                 transform: transforms.Compose,
                 cnn_input_size: Tuple[int, int],
                 segmentation_model: Optional[nn.Module] = None,
                 segmentation_input_size: Optional[Tuple[int, int]] = None,
                 precompute_segmentation: bool = True,
                 overwrite_segmentation: bool = False,
                 target_column: str = "calories",
                 device: torch.device = torch.device("cpu")):
        self.metadata_df = metadata_df
        self.imagery_dir = imagery_dir
        self.processed_dir = processed_dir # Where segmented images are saved/loaded
        self.rgb_filename = rgb_filename
        self.transform = transform # For CNN input
        self.cnn_input_size = cnn_input_size

        self.segmentation_model = segmentation_model
        self.segmentation_input_size = segmentation_input_size
        self.precompute_segmentation = precompute_segmentation
        self.overwrite_segmentation = overwrite_segmentation
        self.target_column = target_column
        self.device = device # Device for segmentation model

        if self.precompute_segmentation and not os.path.exists(self.processed_dir):
            os.makedirs(self.processed_dir, exist_ok=True)

    def __len__(self) -> int:
        return len(self.metadata_df)

    def _get_segmented_image_path(self, dish_id: str) -> str:
        dish_folder = os.path.join(self.processed_dir, dish_id)
        return os.path.join(dish_folder, self.rgb_filename) # Using same filename for segmented

    def _load_and_segment_image(self, dish_id: str) -> Optional[Image.Image]:
        """Loads original, segments, saves if needed, and returns segmented PIL image."""
        original_img_path = os.path.join(self.imagery_dir, dish_id, self.rgb_filename)
        if not os.path.exists(original_img_path):
            logger.warning(f"Original image not found for {dish_id} at {original_img_path}")
            return None
        
        try:
            original_pil = Image.open(original_img_path).convert("RGB")
        except Exception as e:
            logger.error(f"Error loading original image {original_img_path}: {e}")
            return None

        # if not self.segmentation_model or not self.segmentation_input_size:
        #     logger.debug(f"No segmentation model provided or configured for {dish_id}. Using original.")
        #     # Fallback: central square crop if no segmentation
        #     w, h = original_pil.size
        #     side = min(w,h)
        #     return original_pil.crop(((w-side)//2, (h-side)//2, (w+side)//2, (h+side)//2))
        
        # mask_pil = get_segmentation_mask_u2net(original_pil, self.segmentation_model,
        #                                        self.segmentation_input_size, self.device)
        # if mask_pil is None:
        #     logger.warning(f"Segmentation failed for {dish_id}. Using original (center-cropped).")
        #     w, h = original_pil.size; side = min(w,h)
        #     return original_pil.crop(((w-side)//2, (h-side)//2, (w+side)//2, (h+side)//2))

        if not self.segmentation_model or not self.segmentation_input_size:
            logger.debug(f"No segmentation model provided or configured for {dish_id}. Using original (resized).")
            # Fallback: resize the original image instead of cropping
            return original_pil.resize(self.cnn_input_size, Image.BILINEAR)

        mask_pil = get_segmentation_mask_u2net(original_pil, self.segmentation_model,
                               self.segmentation_input_size, self.device)
        if mask_pil is None:
            logger.warning(f"Segmentation failed for {dish_id}. Using original (resized).")
            return original_pil.resize(self.cnn_input_size, Image.BILINEAR)


        final_segmented_pil = apply_mask_and_crop(original_pil, mask_pil)

        if self.precompute_segmentation: # Save the final segmented & cropped image
            segmented_path = self._get_segmented_image_path(dish_id)
            os.makedirs(os.path.dirname(segmented_path), exist_ok=True)
            try:
                final_segmented_pil.save(segmented_path)
                logger.debug(f"Saved segmented image for {dish_id} to {segmented_path}")
            except Exception as e:
                logger.error(f"Error saving segmented image {segmented_path}: {e}")
        
        return final_segmented_pil


    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        dish_info = self.metadata_df.iloc[idx]
        dish_id = dish_info["dish_id"]
        label = torch.tensor([dish_info[self.target_column]], dtype=torch.float32)

        processed_image_pil: Optional[Image.Image] = None

        if self.precompute_segmentation and not self.overwrite_segmentation:
            segmented_path = self._get_segmented_image_path(dish_id)
            if os.path.exists(segmented_path):
                try:
                    processed_image_pil = Image.open(segmented_path).convert("RGB")
                    logger.debug(f"Loaded pre-segmented image for {dish_id} from {segmented_path}")
                except Exception as e:
                    logger.warning(f"Error loading pre-segmented image {segmented_path}, re-segmenting: {e}")
                    processed_image_pil = None # Force re-segmentation

        if processed_image_pil is None: # Not found, or overwrite, or precompute is off but model exists
            processed_image_pil = self._load_and_segment_image(dish_id)

        if processed_image_pil is None: # If all attempts failed
            logger.error(f"Failed to load or segment image for {dish_id}. Returning zeros.")
            # Return a dummy tensor that matches expected output dimensions
            dummy_image = torch.zeros(3, *self.cnn_input_size)
            return dummy_image, label # Still return label to avoid dataloader issues

        # Apply CNN transforms
        image_tensor = self.transform(processed_image_pil)
        
        return image_tensor, label

def check_gpu_memory_for_caching(num_samples: int, sample_image_tensor_shape: Tuple, 
                                 sample_label_tensor_shape: Tuple, 
                                 device: torch.device,
                                 safety_margin: float = 0.8) -> bool:
    """Estimates if there's enough GPU VRAM for caching."""
    if device.type != 'cuda' or not torch.cuda.is_available():
        logger.info("Not on CUDA device or CUDA not available. GPU caching check skipped.")
        return False
    if num_samples == 0:
        return True # Nothing to cache

    try:
        # Estimate memory per sample
        bytes_per_image = np.prod(sample_image_tensor_shape) * 4 # Assuming float32 (4 bytes)
        bytes_per_label = np.prod(sample_label_tensor_shape) * 4 # Assuming float32
        total_bytes_per_sample = bytes_per_image + bytes_per_label
        estimated_vram_needed_mb = (num_samples * total_bytes_per_sample) / (1024**2)

        # Get free GPU memory
        torch.cuda.empty_cache() # Try to free up cached memory
        free_vram_bytes, _ = torch.cuda.mem_get_info(device)
        free_vram_mb = free_vram_bytes / (1024**2)
        usable_vram_mb = free_vram_mb * safety_margin

        logger.info(f"Estimated VRAM for caching: {estimated_vram_needed_mb:.2f} MB")
        logger.info(f"Available (usable with {safety_margin*100}%) VRAM: {usable_vram_mb:.2f} MB (Total Free: {free_vram_mb:.2f} MB)")

        return usable_vram_mb > estimated_vram_needed_mb
    except Exception as e:
        logger.error(f"Error checking GPU memory: {e}", exc_info=True)
        return False

class CachedDataset(Dataset):
    """
    A Dataset wrapper that caches all data from a base_dataset to a specified device.
    The base_dataset is expected to return fully processed tensors.
    """
    def __init__(self, base_dataset: Dataset, device: torch.device, dataset_name: str = "Dataset"):
        self.base_dataset = base_dataset
        self.device = device
        self.images: List[torch.Tensor] = []
        self.labels: List[torch.Tensor] = []
        
        if len(self.base_dataset) == 0:
            logger.warning(f"Base {dataset_name} for caching is empty. CachedDataset will also be empty.")
            return

        logger.info(f"Caching {len(self.base_dataset)} samples from {dataset_name} to {self.device}...")
        for i in tqdm(range(len(self.base_dataset)), desc=f"Caching {dataset_name} to {self.device}", leave=False):
            try:
                img_tensor, label_tensor = self.base_dataset[i] # __getitem__ from Nutrition5kDataset
                self.images.append(img_tensor.to(self.device))
                self.labels.append(label_tensor.to(self.device))
            except Exception as e:
                logger.error(f"Error caching sample {i} from {dataset_name}: {e}. Skipping this sample in cache.")
        
        if not self.images:
             logger.warning(f"No samples were successfully cached for {dataset_name}. CachedDataset is effectively empty.")
        else:
            logger.info(f"Caching for {dataset_name} complete. {len(self.images)} samples cached to {self.device}.")
            
    def __len__(self) -> int:
        return len(self.images)
        
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.images[idx], self.labels[idx]


# %% [markdown]
#  ## 5. Model Architecture (ResNet-based Regressor)
# %%
class CaloriePredictorCNN(nn.Module):
    def __init__(self, num_outputs: int = 1, pretrained: bool = True):
        super(CaloriePredictorCNN, self).__init__()
        self.name = "ResNet50CaloriePredictor"
        if pretrained:
            weights = ResNet50_Weights.IMAGENET1K_V2 # Or V1
            self.base_model = resnet50(weights=weights)
            logger.info("Loaded ResNet50 with ImageNet pretrained weights.")
        else:
            self.base_model = resnet50(weights=None)
            logger.info("Loaded ResNet50 without pretrained weights.")

        num_ftrs = self.base_model.fc.in_features
        # Replace the final fully connected layer for regression
        self.base_model.fc = nn.Sequential(
            nn.Linear(num_ftrs, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_outputs) # Predicting calories (1 output)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base_model(x)

# %% [markdown]
#  ## 6. Training and Evaluation Framework
# %%
class ModelTrainer:
    def __init__(self, train_loader: DataLoader, val_loader: Optional[DataLoader],
                 config_params: Config, model_name: str = "model"):
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config_params
        self.device = config_params.DEVICE
        self.model_name = model_name
        self.results: Dict[str, Any] = {}
        self.use_non_blocking = self.train_loader.pin_memory and self.device.type == 'cuda' if self.train_loader else False


    def _compute_metrics(self, predictions: np.ndarray, ground_truth: np.ndarray) -> Dict[str, float]:
        # Ensure inputs are 1D for regression metrics
        predictions = predictions.flatten()
        ground_truth = ground_truth.flatten()
        
        mae = mean_absolute_error(ground_truth, predictions)
        rmse = np.sqrt(mean_squared_error(ground_truth, predictions))
        r2 = r2_score(ground_truth, predictions)
        return {'mae': mae, 'rmse': rmse, 'r2': r2}

    def _run_epoch(self, model: nn.Module, data_loader: DataLoader, criterion: nn.Module,
                   optimizer: Optional[optim.Optimizer] = None, epoch_num: int = 0, phase: str = "Train") -> float:
        is_train = phase == "Train"
        if is_train:
            model.train()
            if optimizer is None:
                raise ValueError("Optimizer must be provided for training phase.")
        else:
            model.eval()

        total_loss = 0.0
        num_samples = 0
        
        pbar_desc = f"Epoch {epoch_num+1}/{self.config.NUM_EPOCHS} [{phase}]"
        pbar = tqdm(data_loader, desc=pbar_desc, leave=False, dynamic_ncols=True)

        for inputs, labels in pbar:
            inputs = inputs.to(self.device, non_blocking=self.use_non_blocking if is_train else False)
            labels = labels.to(self.device, non_blocking=self.use_non_blocking if is_train else False)

            if is_train:
                optimizer.zero_grad(set_to_none=True)
            
            with torch.set_grad_enabled(is_train):
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                if is_train:
                    loss.backward()
                    optimizer.step()
            
            total_loss += loss.item() * inputs.size(0)
            num_samples += inputs.size(0)
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})
        
        if num_samples == 0: return 0.0 # Avoid division by zero if dataloader is empty
        return total_loss / num_samples


    def train_and_evaluate_model(self, model: nn.Module) -> Dict[str, Any]:
        logger.info(f"Training {self.model_name} on {self.device}...")
        model.to(self.device)
        
        criterion = nn.MSELoss()
        optimizer = optim.AdamW(model.parameters(), lr=self.config.LEARNING_RATE, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.2, 
                                                         patience=self.config.PATIENCE_EARLY_STOPPING // 2, verbose=True)

        best_val_loss = float("inf")
        epochs_no_improve = 0
        train_losses, val_losses = [], []
        model_save_path = f"{self.model_name}_best_state.pth"

        for epoch in range(self.config.NUM_EPOCHS):
            train_loss = self._run_epoch(model, self.train_loader, criterion, optimizer, epoch, phase="Train")
            train_losses.append(train_loss)
            
            val_loss = float("inf")
            if self.val_loader and len(self.val_loader.dataset) > 0:
                val_loss = self._run_epoch(model, self.val_loader, criterion, epoch_num=epoch, phase="Valid")
                val_losses.append(val_loss)
            
            current_lr = optimizer.param_groups[0]['lr']
            log_msg = f"Epoch {epoch+1}/{self.config.NUM_EPOCHS} | Train Loss: {train_loss:.4f} | LR: {current_lr:.1e}"
            if self.val_loader and len(self.val_loader.dataset) > 0:
                 log_msg += f" | Val Loss: {val_loss:.4f}"
            logger.info(log_msg)

            # Early stopping and model saving logic
            current_metric_for_stopping = val_loss if (self.val_loader and len(self.val_loader.dataset) > 0) else train_loss

            if current_metric_for_stopping < best_val_loss:
                best_val_loss = current_metric_for_stopping
                epochs_no_improve = 0
                torch.save(model.state_dict(), model_save_path)
                logger.info(f"  New best {'validation' if self.val_loader else 'training'} loss: {best_val_loss:.4f}. Model state saved to {model_save_path}")
            else:
                epochs_no_improve += 1
                logger.info(f"  {'Validation' if self.val_loader else 'Training'} loss did not improve for {epochs_no_improve} epoch(s).")

            if epochs_no_improve >= self.config.PATIENCE_EARLY_STOPPING:
                logger.info(f"Early stopping triggered after {epoch+1} epochs.")
                break
            
            if self.val_loader and len(self.val_loader.dataset) > 0:
                 scheduler.step(val_loss)
            else:
                 scheduler.step(train_loss)
        
        logger.info(f"Loading best model state from {model_save_path} (Loss: {best_val_loss:.4f}).")
        try:
            model.load_state_dict(torch.load(model_save_path, map_location=self.device))
        except FileNotFoundError:
            logger.warning(f"Best model state file {model_save_path} not found. Using current model state.")

        # Final evaluation
        test_preds_list, test_gt_list = [], []
        model.eval()
        with torch.no_grad():
            for inputs, labels in tqdm(self.val_loader if self.val_loader and len(self.val_loader.dataset)>0 else self.train_loader, desc="Final Eval"): # Use val or train if val empty
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                outputs = model(inputs)
                test_preds_list.append(outputs.cpu().numpy())
                test_gt_list.append(labels.cpu().numpy())
        
        if not test_preds_list: # If no data was evaluated
            logger.warning("No data evaluated in the final step.")
            self.results = {"model": model, "train_losses": train_losses, "val_losses": val_losses}
            return self.results

        test_predictions = np.vstack(test_preds_list)
        test_ground_truth = np.vstack(test_gt_list)
        final_metrics = self._compute_metrics(test_predictions, test_ground_truth)

        self.results = {
            "model": model, "train_losses": train_losses, "val_losses": val_losses,
            "final_metrics": final_metrics,
            "final_predictions": test_predictions, "final_ground_truth": test_ground_truth,
        }
        return self.results

    def plot_training_history(self):
        if not self.results or not self.results.get("train_losses"): return
        plt.figure(figsize=(10, 5))
        plt.plot(self.results["train_losses"], label="Training Loss")
        if self.results.get("val_losses") and self.results["val_losses"]:
            plt.plot(self.results["val_losses"], label="Validation Loss")
        plt.xlabel("Epoch"); plt.ylabel("Loss (MSE)")
        plt.yscale("log")
        plt.title(f"Training and Validation Loss for {self.model_name}")
        plt.legend(); plt.grid(True)
        plt.savefig(f"{self.model_name}_loss_curve.png")
        plt.show(); plt.close()

    def display_final_metrics(self):
        if not self.results or not self.results.get("final_metrics"): return
        metrics = self.results["final_metrics"]
        logger.info(f"\n--- Final Evaluation Metrics for {self.model_name} (on validation/train set) ---")
        logger.info(f"  MAE: {metrics['mae']:.2f}")
        logger.info(f"  RMSE: {metrics['rmse']:.2f}")
        logger.info(f"  R²: {metrics['r2']:.3f}")

# %% [markdown]
#  ## 7. Main Training Pipeline
# %%
def evaluate_model_on_test_set(model: nn.Module, 
                               test_loader: DataLoader, 
                               device: torch.device,
                               criterion: Optional[nn.Module] = None) -> Dict[str, Any]:
    """Evaluates the model on the test set and returns predictions, ground truth, and metrics."""
    model.eval()
    all_predictions = []
    all_ground_truth = []
    total_loss = 0.0
    num_samples = 0

    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc="Evaluating on Test Set"):
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs = model(inputs)
            
            if criterion:
                loss = criterion(outputs, labels)
                total_loss += loss.item() * inputs.size(0)
            
            all_predictions.append(outputs.cpu().numpy())
            all_ground_truth.append(labels.cpu().numpy())
            num_samples += inputs.size(0)

    predictions_np = np.vstack(all_predictions).flatten()
    ground_truth_np = np.vstack(all_ground_truth).flatten()

    metrics = {
        'mae': mean_absolute_error(ground_truth_np, predictions_np),
        'rmse': np.sqrt(mean_squared_error(ground_truth_np, predictions_np)),
        'r2': r2_score(ground_truth_np, predictions_np)
    }
    if criterion and num_samples > 0:
        metrics['loss'] = total_loss / num_samples
    
    return {
        "predictions": predictions_np,
        "ground_truth": ground_truth_np,
        "metrics": metrics
    }

def visualize_test_predictions(model: nn.Module,
                               test_dataset: Dataset, # Pass the dataset directly to get specific items
                               device: torch.device,
                               num_samples: int = 5,
                               target_column_name: str = "Calories",
                               config_params: Config = None): # Pass config for CNN_INPUT_IMG_SIZE
    """Visualizes predictions for a few samples from the test set in a single figure."""
    if len(test_dataset) == 0:
        logger.info("Test dataset is empty. Skipping visualization.")
        return

    model.eval()
    
    # Define denormalization transform (assuming ImageNet stats were used)
    denormalize_transform = transforms.Compose([
        transforms.Normalize(mean=[0., 0., 0.], std=[1/0.229, 1/0.224, 1/0.225]),
        transforms.Normalize(mean=[-0.485, -0.456, -0.406], std=[1., 1., 1.]),
        transforms.ToPILImage() # Convert tensor to PIL Image
    ])

    # Select random samples
    indices = np.random.choice(len(test_dataset), min(num_samples, len(test_dataset)), replace=False)
    
    # Create a grid of subplots
    rows = int(np.ceil(num_samples / 3))
    cols = min(3, num_samples)
    fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows))
    
    # Handle case where we only have one sample
    if num_samples == 1:
        axes = np.array([axes])
    
    # Flatten axes for easy indexing if we have multiple rows
    axes = axes.flatten() if num_samples > 1 else [axes]

    for i, idx in enumerate(indices):
        image_tensor, label_tensor = test_dataset[idx] # Get a single sample
        image_tensor_unsqueezed = image_tensor.unsqueeze(0).to(device) # Add batch dim and move to device
        
        with torch.no_grad():
            prediction = model(image_tensor_unsqueezed).cpu().item() # Get single prediction value

        true_value = label_tensor.item()
        
        # Denormalize the image tensor for display
        img_for_plot_pil = denormalize_transform(image_tensor.cpu())

        # Plot in the corresponding subplot
        axes[i].imshow(img_for_plot_pil)
        title_text = (f"Sample {i+1}\n"
                      f"True {target_column_name}: {true_value:.2f}\n"
                      f"Predicted {target_column_name}: {prediction:.2f}\n"
                      f"Abs Error: {abs(true_value - prediction):.2f}")
        axes[i].set_title(title_text)
        axes[i].axis("off")
    
    # Hide any unused subplots
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')
        axes[j].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(f"test_predictions_{target_column_name.lower()}.png")
    plt.show()
    plt.close()

def plot_predictions_vs_truth_scatter(predictions: np.ndarray, 
                                      ground_truth: np.ndarray, 
                                      target_column_name: str = "Calories"):
    """Displays a scatter plot of predicted values vs. true values."""
    plt.figure(figsize=(8, 8))
    plt.scatter(ground_truth, predictions, alpha=0.5, label="Predictions")
    
    # Plot a diagonal line (perfect prediction)
    min_val = min(ground_truth.min(), predictions.min())
    max_val = max(ground_truth.max(), predictions.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label="Perfect Prediction")
    
    plt.xlabel(f"True {target_column_name}")
    plt.ylabel(f"Predicted {target_column_name}")
    plt.title(f"True vs. Predicted {target_column_name} (Test Set)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    plt.close()
    
# %%
def main_pipeline():
    global config # Use the global config instance
    global U2NET # Make sure U2NET class is accessible if loaded dynamically

    torch.manual_seed(config.RANDOM_STATE)
    np.random.seed(config.RANDOM_STATE)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.RANDOM_STATE)

    # --- Load Segmentation Model (U-2-Net) ---
    u2net_segmentation_model = None
    # ... (your existing U-2-Net loading logic) ...
    if config.PRECOMPUTE_SEGMENTATION:
        if not os.path.exists(config.U2NET_WEIGHTS_PATH):
            logger.error(f"U-2-Net weights not found at {config.U2NET_WEIGHTS_PATH}. Segmentation will be disabled.")
            config.PRECOMPUTE_SEGMENTATION = False # Disable if weights are missing
        else:
            import sys
            if config.U2NET_MODEL_DIR not in sys.path:
                 sys.path.append(config.U2NET_MODEL_DIR)
            try:
                if U2NET is None : # Attempt to import if not already available
                    from u2net_model import U2NET as U2NET_class
                    U2NET = U2NET_class # Assign to global U2NET

                if U2NET is not None:
                    u2net_segmentation_model = load_u2net_model(config.U2NET_WEIGHTS_PATH, config.DEVICE)
                    if u2net_segmentation_model is None:
                        config.PRECOMPUTE_SEGMENTATION = False
                else:
                    logger.error("U2NET class definition not found. Segmentation disabled.")
                    config.PRECOMPUTE_SEGMENTATION = False
            except ImportError:
                 logger.error("Failed to import U2NET from u2net_model.py. Segmentation disabled.")
                 config.PRECOMPUTE_SEGMENTATION = False
            except Exception as e:
                 logger.error(f"An error occurred during U2NET setup: {e}. Segmentation disabled.", exc_info=True)
                 config.PRECOMPUTE_SEGMENTATION = False

    # --- Load Metadata using custom parser ---
    logger.info(f"Parsing metadata file: {config.METADATA_FILE_CAFE1}")
    meta_cafe1 = parse_nutrition_csv(config.METADATA_FILE_CAFE1)
    logger.info(f"Parsing metadata file: {config.METADATA_FILE_CAFE2}")
    meta_cafe2 = parse_nutrition_csv(config.METADATA_FILE_CAFE2)

    if meta_cafe1.empty and meta_cafe2.empty:
        logger.error("Both metadata files are empty or failed to parse. Exiting.")
        return
        
    dish_metadata_df = pd.concat([meta_cafe1, meta_cafe2], ignore_index=True)
    
    if config.TARGET_COLUMN not in dish_metadata_df.columns or "dish_id" not in dish_metadata_df.columns:
        logger.error(f"Required columns ('dish_id', '{config.TARGET_COLUMN}') not found in parsed metadata. Check parse_nutrition_csv.")
        return

    dish_metadata_df = dish_metadata_df[["dish_id", config.TARGET_COLUMN]].copy()
    dish_metadata_df.dropna(subset=[config.TARGET_COLUMN], inplace=True)
    dish_metadata_df[config.TARGET_COLUMN] = pd.to_numeric(dish_metadata_df[config.TARGET_COLUMN], errors='coerce')
    dish_metadata_df.dropna(subset=[config.TARGET_COLUMN], inplace=True)
    logger.info(f"Loaded {len(dish_metadata_df)} total metadata entries after custom parsing.")

    valid_dish_ids = []
    for dish_id in tqdm(dish_metadata_df['dish_id'].unique(), desc="Checking image existence"):
        if os.path.exists(os.path.join(config.IMAGERY_DIR, dish_id, config.RGB_IMAGE_FILENAME)):
            valid_dish_ids.append(dish_id)
    dish_metadata_df = dish_metadata_df[dish_metadata_df['dish_id'].isin(valid_dish_ids)]
    logger.info(f"Filtered to {len(dish_metadata_df)} entries with existing images.")

    if len(dish_metadata_df) < 20: # Increased minimum for robust splitting
        logger.error(f"Not enough data ({len(dish_metadata_df)} samples) for splitting and training. Need at least 20. Exiting.")
        return

    # --- Split Data ---
    # Splitting into train, validation, and test sets
    train_val_df, test_df = train_test_split(dish_metadata_df, test_size=0.2, random_state=config.RANDOM_STATE)
    train_df, val_df = train_test_split(train_val_df, test_size=0.2, random_state=config.RANDOM_STATE) # 0.2 of (0.8) = 0.16 for val

    logger.info(f"Dataset split: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")
    
    # --- Define CNN Transforms ---
    normalize_transform = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    train_transforms = transforms.Compose([
        transforms.Resize(config.CNN_INPUT_IMG_SIZE),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(20),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        normalize_transform
    ])
    val_test_transforms = transforms.Compose([
        transforms.Resize(config.CNN_INPUT_IMG_SIZE),
        transforms.ToTensor(),
        normalize_transform
    ])

    # before caching
    # # --- Create Datasets ---
    # dataset_args = {
    #     "imagery_dir": config.IMAGERY_DIR,
    #     "processed_dir": config.PROCESSED_SEGMENTED_DIR,
    #     "rgb_filename": config.RGB_IMAGE_FILENAME,
    #     "cnn_input_size": config.CNN_INPUT_IMG_SIZE,
    #     "segmentation_model": u2net_segmentation_model if config.PRECOMPUTE_SEGMENTATION else None,
    #     "segmentation_input_size": config.SEGMENTATION_INPUT_SIZE if config.PRECOMPUTE_SEGMENTATION else None,
    #     "precompute_segmentation": config.PRECOMPUTE_SEGMENTATION,
    #     "overwrite_segmentation": config.OVERWRITE_EXISTING_SEGMENTATION,
    #     "target_column": config.TARGET_COLUMN,
    #     "device": config.DEVICE
    # }

    # train_dataset = Nutrition5kDataset(metadata_df=train_df.reset_index(drop=True), transform=train_transforms, **dataset_args)
    # val_dataset = Nutrition5kDataset(metadata_df=val_df.reset_index(drop=True), transform=val_test_transforms, **dataset_args)
    # test_dataset = Nutrition5kDataset(metadata_df=test_df.reset_index(drop=True), transform=val_test_transforms, **dataset_args) # Create test_dataset

    # # --- Create DataLoaders ---
    # train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True,
    #                           num_workers=config.NUM_WORKERS, pin_memory=config.PIN_MEMORY, persistent_workers=(config.NUM_WORKERS > 0),
    #                           drop_last=True)
    # val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False,
    #                         num_workers=config.NUM_WORKERS, pin_memory=config.PIN_MEMORY, persistent_workers=(config.NUM_WORKERS > 0))
    # test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False, 
    #                          num_workers=config.NUM_WORKERS, pin_memory=config.PIN_MEMORY, persistent_workers=(config.NUM_WORKERS > 0)) # Create test_loader

    # --- Create Base Datasets ---
    dataset_args = {
        "imagery_dir": config.IMAGERY_DIR,
        "processed_dir": config.PROCESSED_SEGMENTED_DIR,
        "rgb_filename": config.RGB_IMAGE_FILENAME,
        "cnn_input_size": config.CNN_INPUT_IMG_SIZE,
        "segmentation_model": u2net_segmentation_model if config.PRECOMPUTE_SEGMENTATION else None,
        "segmentation_input_size": config.SEGMENTATION_INPUT_SIZE if config.PRECOMPUTE_SEGMENTATION else None,
        "precompute_segmentation": config.PRECOMPUTE_SEGMENTATION,
        "overwrite_segmentation": config.OVERWRITE_EXISTING_SEGMENTATION,
        "target_column": config.TARGET_COLUMN,
        "device": config.DEVICE # Pass main device, seg model uses this. Original dataset still produces CPU tensors first.
    }

    base_train_dataset = Nutrition5kDataset(metadata_df=train_df.reset_index(drop=True), transform=train_transforms, **dataset_args)
    base_val_dataset = Nutrition5kDataset(metadata_df=val_df.reset_index(drop=True), transform=val_test_transforms, **dataset_args)
    base_test_dataset = Nutrition5kDataset(metadata_df=test_df.reset_index(drop=True), transform=val_test_transforms, **dataset_args)

    # --- GPU Caching Logic ---
    train_dataset, val_dataset, test_dataset = base_train_dataset, base_val_dataset, base_test_dataset
    can_cache_all = False
    if config.USE_GPU_CACHING and config.DEVICE.type == 'cuda':
        # Get sample tensor shapes for memory estimation (assumes datasets are not empty)
        sample_img_shape, sample_label_shape = (3, *config.CNN_INPUT_IMG_SIZE), (1,) # Default if datasets are empty
        if len(base_train_dataset) > 0:
            sample_img, sample_label = base_train_dataset[0]
            sample_img_shape = sample_img.shape
            sample_label_shape = sample_label.shape
        
        total_samples_to_cache = len(base_train_dataset) + len(base_val_dataset) + len(base_test_dataset)
        can_cache_all = check_gpu_memory_for_caching(total_samples_to_cache, 
                                                     sample_img_shape, sample_label_shape,
                                                     config.DEVICE, config.GPU_CACHE_SAFETY_MARGIN)
        if can_cache_all:
            logger.info("Attempting to cache all datasets to GPU.")
            if len(base_train_dataset) > 0:
                train_dataset = CachedDataset(base_train_dataset, config.DEVICE, "TrainSet")
            if len(base_val_dataset) > 0:
                val_dataset = CachedDataset(base_val_dataset, config.DEVICE, "ValSet")
            else: val_dataset = None # Ensure val_dataset can be None
            if len(base_test_dataset) > 0:
                test_dataset = CachedDataset(base_test_dataset, config.DEVICE, "TestSet")
        else:
            logger.warning("Insufficient GPU VRAM to cache all datasets. Proceeding without GPU caching.")
            config.USE_GPU_CACHING = False # Explicitly turn off if check fails
    
    # Adjust DataLoader parameters if caching is active and successful
    loader_num_workers = config.NUM_WORKERS
    loader_pin_memory = config.PIN_MEMORY
    if config.USE_GPU_CACHING and can_cache_all: # only if caching was successful
        loader_num_workers = 0  # Data is already on GPU, no parallel loading from CPU needed
        loader_pin_memory = False # Not needed when data is on GPU
        logger.info("Using 0 num_workers and pin_memory=False for DataLoaders due to GPU caching.")


    # --- Create DataLoaders ---
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True,
                              num_workers=loader_num_workers, pin_memory=loader_pin_memory, 
                              persistent_workers=(loader_num_workers > 0),
                              drop_last=True)
    
    val_loader = None
    if val_dataset and len(val_dataset) > 0: # Check if val_dataset is not None and not empty
        val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False,
                                num_workers=loader_num_workers, pin_memory=loader_pin_memory, 
                                persistent_workers=(loader_num_workers > 0))
    
    test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False, 
                             num_workers=loader_num_workers, pin_memory=loader_pin_memory, 
                             persistent_workers=(loader_num_workers > 0))


    # --- Initialize Model and Trainer ---
    calorie_cnn_model = CaloriePredictorCNN(num_outputs=1, pretrained=True) # This is the model variable we need
    trainer = ModelTrainer(train_loader, val_loader, config_params=config, model_name=calorie_cnn_model.name)
    
    # --- Train and Evaluate (on validation set during training) ---
    training_results = trainer.train_and_evaluate_model(calorie_cnn_model) # calorie_cnn_model is updated in-place
    trainer.plot_training_history()
    trainer.display_final_metrics() # Displays metrics on validation set (or train if val empty)

    logger.info("Training pipeline finished.")
    
    # --- Final Evaluation and Visualization on the TEST SET ---
    logger.info("\n--- Evaluating on dedicated Test Set ---")
    if test_loader and len(test_dataset) > 0:
        # The model (calorie_cnn_model) has been trained and best weights loaded by ModelTrainer
        test_eval_results = evaluate_model_on_test_set(calorie_cnn_model, test_loader, config.DEVICE, criterion=nn.MSELoss())
        
        logger.info("Test Set Metrics:")
        for metric_name, value in test_eval_results["metrics"].items():
            logger.info(f"  {metric_name.upper()}: {value:.3f}")

        logger.info("\n--- Visualizing some Test Set Predictions ---")
        visualize_test_predictions(calorie_cnn_model, 
                                   test_dataset, # Pass the dataset for easy sample access
                                   config.DEVICE, 
                                   num_samples=5, 
                                   target_column_name=config.TARGET_COLUMN.capitalize(),
                                   config_params=config)

        logger.info("\n--- Plotting Test Set Predictions vs. Truth ---")
        plot_predictions_vs_truth_scatter(test_eval_results["predictions"],
                                          test_eval_results["ground_truth"],
                                          target_column_name=config.TARGET_COLUMN.capitalize())
    else:
        logger.info("Test set is empty or test_loader not created. Skipping final test set evaluation and visualization.")

    # --- Visualize samples with highest prediction errors ---
    if test_loader and len(test_dataset) > 0:
        logger.info("\n--- Visualizing samples with highest prediction errors ---")
        
        # Get all test predictions with their dish IDs
        all_preds = []
        all_labels = []
        all_indices = []
        
        model = calorie_cnn_model.eval()
        with torch.no_grad():
            for i in range(len(test_dataset)):
                img, label = test_dataset[i]
                img_tensor = img.unsqueeze(0).to(config.DEVICE)
                pred = model(img_tensor).cpu().item()
                all_preds.append(pred)
                all_labels.append(label.item())
                all_indices.append(i)
        
        # Calculate errors and get dish_ids
        dish_ids = test_df['dish_id'].values
        errors = np.abs(np.array(all_preds) - np.array(all_labels))
        error_data = list(zip(all_indices, errors, all_preds, all_labels, dish_ids))
        
        # Sort by error (descending) and get top 5
        error_data.sort(key=lambda x: x[1], reverse=True)
        top_error_samples = error_data[:5]
        
        # Create visualization
        fig, axes = plt.subplots(1, 5, figsize=(20, 6))
        
        denormalize = transforms.Compose([
            transforms.Normalize(mean=[0., 0., 0.], std=[1/0.229, 1/0.224, 1/0.225]),
            transforms.Normalize(mean=[-0.485, -0.456, -0.406], std=[1., 1., 1.]),
            transforms.ToPILImage()
        ])
        
        for i, (idx, error, pred, true, dish_id) in enumerate(top_error_samples):
            img, _ = test_dataset[idx]
            img_pil = denormalize(img.cpu())
            
            axes[i].imshow(img_pil)
            axes[i].set_title(f"Dish ID: {dish_id}\nTrue: {true:.1f}\nPred: {pred:.1f}\nError: {error:.1f}")
            axes[i].axis('off')
        
        plt.tight_layout()
        plt.savefig("highest_error_samples.png")
        plt.show()
        plt.close()
    # done outlier viz

    logger.info("--- End of Pipeline ---")

# %%
if __name__ == "__main__":
    main_pipeline()