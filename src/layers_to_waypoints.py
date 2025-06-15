import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Tuple, List
from config import PALETTE

def layers_to_trajectories(
    layer_dir: str,
    palette: Dict[str, Tuple[int, int, int]],
    paper_size_mm: Tuple[float, float],
    grid_size_px: Tuple[int, int],
    origin_mm: Tuple[float, float] = (0.0, 0.0),
) -> Dict[str, List[List[Tuple[float, float]]]]:
    """
    For each color in `palette`, reads all layer PNGs named '*_<color>.png' in `layer_dir`,
    labels connected regions, snake‐fills each region, and returns:
      
      {
        color_name: [
          [ (x_mm, y_mm), (x_mm, y_mm), … ],   # first region’s waypoint sequence
          [ … ],                              # second region
          …
        ],
        …
      }
    """
    dx = paper_size_mm[0] / grid_size_px[0]
    dy = paper_size_mm[1] / grid_size_px[1]
    ox, oy = origin_mm

    trajectories: Dict[str, List[List[Tuple[float, float]]]] = {}

    for color in palette:
        trajs_for_color: List[List[Tuple[float, float]]] = []
        for fname in os.listdir(layer_dir):
            if not fname.endswith(f"_{color}.png"):
                continue

            rgba = cv2.imread(os.path.join(layer_dir, fname), cv2.IMREAD_UNCHANGED)
            if rgba is None or rgba.shape[2] < 4:
                continue

            # binary mask of this layer’s fill
            alpha = rgba[..., 3]
            _, bw = cv2.threshold(alpha, 127, 255, cv2.THRESH_BINARY)

            # label connected regions
            num_labels, lbl = cv2.connectedComponents(bw)
            for label in range(1, num_labels):
                mask = (lbl == label)
                ys, xs = np.where(mask)
                if ys.size == 0:
                    continue

                min_y, max_y = ys.min(), ys.max()
                path: List[Tuple[float, float]] = []

                # snake-fill row by row
                for row in range(min_y, max_y + 1):
                    cols = xs[ys == row]
                    if cols.size == 0:
                        continue
                    cols = np.sort(cols)
                    if (row - min_y) % 2:
                        cols = cols[::-1]
                    for col in cols:
                        x_world = ox + col * dx
                        y_world = oy + row * dy
                        path.append((x_world, y_world))

                trajs_for_color.append(path)

        trajectories[color] = trajs_for_color

    return trajectories

def generate_fill_with_overlay(
    layer_dir: str,
    small_dir: str,
    output_dir: str,
    palette: Dict[str, Tuple[int, int, int]],
    paper_size_mm: Tuple[float, float],
    grid_size_px: Tuple[int, int],
    origin_mm: Tuple[float, float],
) -> Dict[str, List[List[Tuple[float, float]]]]:
    """
    Computes fill trajectories for each color, saves fill plots,
    and overlays those paths onto the downsampled small image.

    Args:
        layer_dir: contains *_<color>.png RGBA layers
        small_dir: contains *_small.png downsampled originals
        output_dir: where to save fill and overlay images
        palette: color name → RGB tuple for pens
        paper_size_mm: real drawing area in mm (width, height)
        grid_size_px: grid resolution (width, height)
        origin_mm: world coordinates of pixel (0,0)

    Returns:
        dict mapping color to list of waypoint lists (in mm)
    """
    os.makedirs(output_dir, exist_ok=True)
    dx = paper_size_mm[0] / grid_size_px[0]
    dy = paper_size_mm[1] / grid_size_px[1]

    trajectories: Dict[str, List[List[Tuple[float, float]]]] = {}

    for color in palette:
        trajectories[color] = []
        # 1) check for small image
        small_path = None
        base_names = [os.path.splitext(f)[0][:-6] for f in os.listdir(small_dir) if f.endswith('_small.png')]
        if base_names:
            small_path = os.path.join(small_dir, base_names[0] + '_quantized.png')
        # 2) gather layer files
        layer_files = [f for f in os.listdir(layer_dir) if f.endswith(f"_{color}.png")]
        all_paths = []

        for fname in layer_files:
            img_path = os.path.join(layer_dir, fname)
            rgba = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
            if rgba is None or rgba.shape[2] < 4:
                continue
            alpha = rgba[..., 3]
            _, bw = cv2.threshold(alpha, 127, 255, cv2.THRESH_BINARY)
            num_labels, label_map = cv2.connectedComponents(bw)
            for label in range(1, num_labels):
                mask = (label_map == label)
                ys, xs = np.where(mask)
                if ys.size == 0:
                    continue
                path_mm: List[Tuple[float,float]] = []
                min_y, max_y = ys.min(), ys.max()
                for row in range(min_y, max_y+1):
                    cols = xs[ys == row]
                    if cols.size == 0:
                        continue
                    ordered = np.sort(cols) if (row-min_y)%2==0 else np.sort(cols)[::-1]
                    for col in ordered:
                        x = origin_mm[0] + col*dx
                        y = origin_mm[1] + row*dy
                        path_mm.append((x,y))
                all_paths.append(path_mm)
        trajectories[color] = all_paths

        # 3) Save fill trajectory plot
        if all_paths:
            plt.figure()
            for path in all_paths:
                xs, ys = zip(*path)
                plt.plot(xs, ys, linewidth=1, color='black')  # draw in white
            plt.gca().invert_yaxis()
            plt.axis('equal')
            plt.title(f"Fill Trajectory: {color}")
            out_fill = os.path.join(output_dir, f"fill_traj_{color}.png")
            plt.savefig(out_fill, dpi=150, bbox_inches='tight')
            plt.close()

        # 4) Overlay on small image
        if small_path and os.path.exists(small_path) and all_paths:
            small_img = cv2.imread(small_path)
            h, w = grid_size_px[1], grid_size_px[0]
            # upsample for clearer overlay
            display = cv2.resize(small_img, (w*10, h*10), interpolation=cv2.INTER_NEAREST)
            scale = 10.0
            plt.figure()
            plt.imshow(cv2.cvtColor(display, cv2.COLOR_BGR2RGB), interpolation='nearest')
            for path in all_paths:
                xs = [p[0]/dx*scale for p in path]
                ys = [p[1]/dy*scale for p in path]
                plt.plot(xs, ys, linewidth=1, color='black')  # black for contrast
            plt.axis('off')
            out_ov = os.path.join(output_dir, f"overlay_{color}.png")
            plt.savefig(out_ov, dpi=150, bbox_inches='tight', pad_inches=0)
            plt.close()
            print(f"Saved overlay for '{color}' → {out_ov}")

    return trajectories


if __name__ == '__main__':
    generate_fill_with_overlay(
        layer_dir="../image_maps/output",
        small_dir="../image_maps/output",
        output_dir="../image_maps/filled_regions_overlay",
        palette=PALETTE,
        paper_size_mm=(105.0,148.0),
        grid_size_px=(26,38),
        origin_mm=(0.0,0.0),
    )
