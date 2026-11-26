"""
Project 4: AI Semantic Segmentation Pipeline for Drone Orthophotos
------------------------------------------------------------------
This script takes a massive georeferenced drone orthophoto (GeoTIFF),
tiles it into 512x512 patches, runs a PyTorch U-Net to predict
land-cover classes for each tile, and stitches the results back
into a single, georeferenced mask GeoTIFF.
"""

import rasterio
from rasterio.windows import Window
import numpy as np
import torch
import os
import shutil
import segmentation_models_pytorch as smp

# --- CONFIGURATION ---
# Update these paths to match your local or Colab environment
ORTHO_LOCAL = '/content/odm_orthophoto.tif'
MASK_LOCAL  = '/content/mask.tif'
MODEL_PATH  = '/content/drive/MyDrive/visdrone/project4/quick_unet.pt'
MASK_DRIVE  = '/content/drive/MyDrive/visdrone/project4/mask.tif'

device = 'cuda' if torch.cuda.is_available() else 'cpu'
PATCH_SIZE = 512

# --- 1. LOAD MODEL ---
# We use a ResNet-34 U-Net (same architecture as Project 3)
model = smp.Unet(encoder_name='resnet34', encoder_weights=None, in_channels=3, classes=23).to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

print("Opening local orthophoto...")
# --- 2. READ GEOTIFF & TILE ---
with rasterio.open(ORTHO_LOCAL) as src:
    meta = src.meta.copy()
    width, height = src.width, src.height
    print(f"Image size: {width} x {height} px")
    
    # Update metadata for the output mask (1 band, uint8)
    meta.update({
        "count": 1,
        "dtype": "uint8",
        "compress": "lzw"
    })
    
    # Open the output file
    with rasterio.open(MASK_LOCAL, 'w', **meta) as dst:
        print("Tiling and running AI inference...")
        
        # Iterate over the image in 512x512 blocks
        for y in range(0, height, PATCH_SIZE):
            for x in range(0, width, PATCH_SIZE):
                h = min(PATCH_SIZE, height - y)
                w = min(PATCH_SIZE, width - x)
                
                # Read tile and keep ONLY the first 3 bands (drop Alpha)
                tile = src.read(window=Window(x, y, w, h))[:3, :, :]
                
                # Normalize (ImageNet stats)
                tile_float = tile.astype(np.float32) / 255.0
                mean = np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)
                std = np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1)
                tile_norm = (tile_float - mean) / std
                
                # Pad edges (if the tile is smaller than 512x512)
                tile_padded = np.zeros((3, PATCH_SIZE, PATCH_SIZE), dtype=np.float32)
                tile_padded[:, :h, :w] = tile_norm
                
                # Run Inference
                with torch.no_grad():
                    input_tensor = torch.from_numpy(tile_padded).unsqueeze(0).to(device)
                    logits = model(input_tensor)
                    pred = logits.argmax(1).squeeze(0).cpu().numpy().astype(np.uint8)
                
                # Crop prediction back to original tile size and write to mask
                pred_cropped = pred[:h, :w]
                dst.write(pred_cropped, 1, window=Window(x, y, w, h))

print("Inference complete!")
print("Copying mask back to Google Drive...")
shutil.copy(MASK_LOCAL, MASK_DRIVE)
print(f"Success! Saved to {MASK_DRIVE}")
