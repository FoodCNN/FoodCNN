#!/usr/bin/env python
import argparse
import os
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image

# --- Configuration ---
# IMPORTANT: Update this path to where your fine-tuned model is saved
FINETUNED_MODEL_PATH = "./best_finetuned_combined_model.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Model Definitions (Must match the architecture of the saved model) ---

class ResNetFromScratch(nn.Module):
    """ The nutrition-per-100g estimation model. """
    def __init__(self, num_outputs=4, use_pretrained=True): # Ensure use_pretrained matches training
        super().__init__()
        weights = models.ResNet34_Weights.IMAGENET1K_V1 if use_pretrained else None
        self.backbone = models.resnet34(weights=weights)
        n_feat = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Linear(n_feat, 256), nn.ReLU(), nn.Dropout(0.5), nn.Linear(256, num_outputs)
        )
    def forward(self, x): return self.backbone(x)

class DeepWeightCNN(nn.Module):
    """ The weight estimation model. """
    def __init__(self, num_outputs=1):
        super().__init__()
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
    def forward(self, x): return self.regressor(self.features(x))

class CombinedSystem(nn.Module):
    """ The fine-tuned combined system. """
    def __init__(self, nutrition_model, weight_model):
        super().__init__()
        self.nutrition_model = nutrition_model
        self.weight_model = weight_model

    def forward(self, image_base, image_normalized):
        # Predict nutritional density (per 100g)
        pred_nutrition_100g = self.nutrition_model(image_normalized)
        # Predict weight
        pred_weight = self.weight_model(image_base).squeeze(-1)
        # Calculate final absolute nutrition values
        pred_absolute_nutrition = pred_nutrition_100g * (pred_weight.unsqueeze(1) / 100.0)
        return pred_absolute_nutrition, pred_weight

# --- Prediction Function ---

def predict_nutrition(model, image_path, device):
    """
    Loads an image, preprocesses it, and returns the predicted nutrition facts.
    """
    # 1. Define the exact same transformations used during training
    transform_base = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    imagenet_normalize = transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )

    # 2. Load and process the image
    try:
        img = Image.open(image_path).convert("RGB")
    except FileNotFoundError:
        print(f"Error: Image file not found at {image_path}")
        return None

    # Create the two versions of the image tensor the model needs
    image_base = transform_base(img)
    image_normalized = imagenet_normalize(image_base)

    # Add a batch dimension (models expect N, C, H, W)
    image_base_batch = image_base.unsqueeze(0).to(device)
    image_normalized_batch = image_normalized.unsqueeze(0).to(device)

    # 3. Set model to evaluation mode and predict
    model.eval()
    with torch.no_grad():
        pred_abs, pred_weight = model(image_base_batch, image_normalized_batch)

    # 4. Extract results from tensors
    pred_abs_values = pred_abs.cpu().numpy().flatten()
    pred_weight_value = pred_weight.item()

    # 5. Format the results into a dictionary
    results = {
        'Calories (kcal)': pred_abs_values[0],
        'Fat (g)': pred_abs_values[1],
        'Carbohydrates (g)': pred_abs_values[2],
        'Protein (g)': pred_abs_values[3],
        'Estimated Weight (g)': pred_weight_value
    }

    return results


def main():
    # --- Setup Command-Line Argument Parsing ---
    parser = argparse.ArgumentParser(description="Predict nutrition facts for a food image.")
    parser.add_argument("--image", type=str, required=True, help="Path to the input image file.")
    args = parser.parse_args()

    if not os.path.exists(args.image):
        print(f"Error: The file '{args.image}' does not exist.")
        return

    # --- Load the Fine-Tuned Model ---
    print(f"Loading fine-tuned model from {FINETUNED_MODEL_PATH}...")
    try:
        # First, create an instance of the model with the correct architecture
        nutrition_branch = ResNetFromScratch(num_outputs=4, use_pretrained=True)
        weight_branch = DeepWeightCNN(num_outputs=1)
        model = CombinedSystem(nutrition_branch, weight_branch)

        # Then, load the saved state dictionary
        model.load_state_dict(torch.load(FINETUNED_MODEL_PATH, map_location=DEVICE))
        model.to(DEVICE)
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # --- Perform Prediction ---
    print(f"\nAnalyzing image: {args.image}")
    predictions = predict_nutrition(model, args.image, DEVICE)

    # --- Display Results ---
    if predictions:
        print("\n--- Estimated Nutrition Facts ---")
        for key, value in predictions.items():
            print(f"{key:<22}: {value:.2f}")
        print("---------------------------------")


if __name__ == "__main__":
    main()
