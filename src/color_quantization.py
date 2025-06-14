import os
import numpy as np
from PIL import Image
from skimage.color import rgb2lab

# === User Configuration ===
# Set your input directory of PNG images and output directory for results:
INPUT_DIR = "../image_maps/image_raw"
OUTPUT_DIR = "../image_maps/output"

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Define pen palette including white
PALETTE = {
    "green": (0, 128, 0),
    "orange": (255, 165, 0),
    "blue": (0, 0, 255),
    "red": (255, 0, 0),
    "yellow": (255, 255, 0),
    "turquoise": (64, 224, 208),
    "black": (0, 0, 0),
    "purple": (128, 0, 128),
    "light_green": (144, 238, 144),
    "white": (255, 255, 255),
}

# Precompute Lab palette for distance calculation
palette_rgb = np.array(list(PALETTE.values()), dtype=np.float64) / 255.0
# rgb2lab expects an image array, so reshape to (1, N, 3)
palette_lab = rgb2lab(palette_rgb.reshape(1, -1, 3)).reshape(-1, 3)

for filename in os.listdir(INPUT_DIR):
    if not filename.lower().endswith(".png"):
        continue
    basename = os.path.splitext(filename)[0]
    # Load and convert image to float RGB
    img = Image.open(os.path.join(INPUT_DIR, filename)).convert("RGB")
    arr = np.array(img, dtype=np.float64) / 255.0
    h, w, _ = arr.shape
    # Convert image to Lab space
    arr_lab = rgb2lab(arr)

    # Compute squared distances to each palette color
    # arr_lab: (h, w, 3), palette_lab: (N, 3)
    diff = arr_lab[:, :, np.newaxis, :] - palette_lab[np.newaxis, np.newaxis, :, :]
    dist2 = np.sum(diff ** 2, axis=3)
    # Nearest palette index per pixel
    idx_map = np.argmin(dist2, axis=2)

    # Build quantized RGB image
    quant_rgb = (palette_rgb * 255).astype(np.uint8)
    quant_flat = quant_rgb[idx_map.flatten()]
    quant_img = quant_flat.reshape(h, w, 3)
    # Save full quantized image
    Image.fromarray(quant_img).save(os.path.join(OUTPUT_DIR, f"{basename}_quantized.png"))

    # Generate per-color RGBA layers
    for i, (color_name, rgb) in enumerate(PALETTE.items()):
        mask = (idx_map == i)
        # Create RGBA array
        layer = np.zeros((h, w, 4), dtype=np.uint8)
        layer[..., 0] = rgb[0]
        layer[..., 1] = rgb[1]
        layer[..., 2] = rgb[2]
        layer[..., 3] = (mask * 255).astype(np.uint8)
        Image.fromarray(layer, mode="RGBA").save(
            os.path.join(OUTPUT_DIR, f"{basename}_{color_name}.png")
        )
