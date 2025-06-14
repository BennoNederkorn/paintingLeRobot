import os
import numpy as np
from PIL import Image
from skimage.color import rgb2lab
from typing import List, Tuple, Union, Dict
from config import PALETTE

# === User Configuration ===
INPUT_DIR = "../image_maps/image_raw"
OUTPUT_DIR = "../image_maps/output"
SIZE = (26, 38)    # width × height in dots for A6
RESAMPLE_FILTER = Image.NEAREST  # blocky downsampling for pen output

def process_and_downsample(
    input_dir: str,
    output_dir: str,
    size: Tuple[int, int] = SIZE,
    palette: Dict[str, Tuple[int, int, int]] = PALETTE,
    resample: int = RESAMPLE_FILTER
) -> None:
    """
    Downsample all PNGs in `input_dir` to `size`, quantize to pen palette, 
    and save both the resized and per-color RGBA layers in `output_dir`.

    Args:
        input_dir: Path to folder with source PNGs.
        output_dir: Path to save outputs; created if missing.
        size: (width, height) for downsampling.
        palette: Mapping from color names to RGB tuples.
        resample: PIL resampling filter (e.g. Image.NEAREST).
    """
    os.makedirs(output_dir, exist_ok=True)

    # Precompute normalized RGB and Lab palette arrays
    rgb_vals = np.array(list(palette.values()), dtype=np.float64) / 255.0
    lab_vals = rgb2lab(rgb_vals.reshape(1, -1, 3)).reshape(-1, 3)

    for fname in os.listdir(input_dir):
        if not fname.lower().endswith('.png'):
            continue
        name, _ = os.path.splitext(fname)
        in_path = os.path.join(input_dir, fname)

        # Load and downsample
        img = Image.open(in_path).convert('RGB')
        small = img.resize(size, resample)
        small_path = os.path.join(output_dir, f"{name}_small.png")
        small.save(small_path)

        # Quantize to palette
        arr = np.array(small, dtype=np.float64) / 255.0
        h, w, _ = arr.shape
        lab_img = rgb2lab(arr)

        # Compute nearest palette index per pixel
        diff = lab_img[:, :, np.newaxis, :] - lab_vals[np.newaxis, np.newaxis, :, :]
        idx_map = np.argmin((diff ** 2).sum(axis=3), axis=2)

        # Build quantized RGB image
        quant_rgb = (rgb_vals * 255).astype(np.uint8)
        flat = quant_rgb[idx_map.ravel()]
        quant = flat.reshape(h, w, 3)
        quant_path = os.path.join(output_dir, f"{name}_quantized.png")
        Image.fromarray(quant).save(quant_path)

        # Save per-color RGBA layers
        for i, (color, rgb) in enumerate(palette.items()):
            mask = (idx_map == i)
            layer = np.zeros((h, w, 4), dtype=np.uint8)
            layer[..., :3] = rgb
            layer[..., 3] = (mask * 255).astype(np.uint8)
            layer_path = os.path.join(output_dir, f"{name}_{color}.png")
            Image.fromarray(layer, mode='RGBA').save(layer_path)

    print(f"Processed {len(os.listdir(input_dir))} images into '{output_dir}'")


if __name__ == '__main__':
    process_and_downsample(INPUT_DIR, OUTPUT_DIR)