#!/usr/bin/env python3
"""
Test script for the aspect ratio functionality in image_generation module.
This demonstrates how to generate images with specific dimensions using Vertex AI.
"""

import sys
import os

# Add the src directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from image_generation import (
    calculate_target_aspect_ratio,
    get_optimal_aspect_ratio_for_target,
    TARGET_WIDTH,
    TARGET_HEIGHT
)

def demonstrate_aspect_ratios():
    """Demonstrate the aspect ratio calculations."""
    print("🎯 Target Dimensions Analysis")
    print("=" * 40)
    print(f"Target dimensions: {TARGET_WIDTH} x {TARGET_HEIGHT}")
    print(f"Target ratio: {TARGET_WIDTH/TARGET_HEIGHT:.3f}")
    print()
    
    print("📐 Available Aspect Ratios:")
    aspect_ratios = {
        "1:1": 1.0,      # Square
        "3:4": 0.75,     # Portrait (closest to our target)
        "4:3": 1.333,    # Landscape
        "9:16": 0.5625,  # Tall portrait (mobile)
        "16:9": 1.778    # Wide landscape
    }
    
    target_ratio = TARGET_WIDTH / TARGET_HEIGHT
    
    for ratio_name, ratio_value in aspect_ratios.items():
        difference = abs(ratio_value - target_ratio)
        closest = " ← CLOSEST MATCH" if ratio_value == 0.75 else ""
        print(f"  {ratio_name:5} = {ratio_value:.3f} (diff: {difference:.3f}){closest}")
    
    print()
    print("🎯 Recommendations:")
    calculated = calculate_target_aspect_ratio()
    optimal = get_optimal_aspect_ratio_for_target()
    print(f"  Calculated best match: {calculated}")
    print(f"  Recommended for target: {optimal}")
    
    print()
    print("💡 Usage Examples:")
    print("  # Generate with optimal aspect ratio:")
    print("  generate_image_with_optimal_aspect(prompt, 'output.png')")
    print()
    print("  # Generate with specific aspect ratio:")
    print("  generate_image(prompt, 'output.png', aspect_ratio='3:4')")
    print()
    print("  # Generate with default aspect ratio (will be resized):")
    print("  generate_image(prompt, 'output.png')")

if __name__ == "__main__":
    demonstrate_aspect_ratios()
