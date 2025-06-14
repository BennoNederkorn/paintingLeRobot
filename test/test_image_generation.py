#!/usr/bin/env python3
"""
Test script for the updated image_generation module.
This demonstrates the new numpy array functionality and size validation.
"""

import numpy as np
import sys
import os

# Add the src directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from image_generation import (
    validate_image_size, 
    resize_image_to_target, 
    check_and_fix_image_size,
    TARGET_WIDTH,
    TARGET_HEIGHT
)

def test_size_validation():
    """Test the size validation functions."""
    print("🧪 Testing size validation functions...")
    
    # Test 1: Correct size image
    correct_image = np.random.randint(0, 256, (TARGET_HEIGHT, TARGET_WIDTH, 3), dtype=np.uint8)
    print(f"Test 1 - Correct size image {correct_image.shape}: {validate_image_size(correct_image)}")
    
    # Test 2: Wrong size image
    wrong_image = np.random.randint(0, 256, (200, 200, 3), dtype=np.uint8)
    print(f"Test 2 - Wrong size image {wrong_image.shape}: {validate_image_size(wrong_image)}")
    
    # Test 3: Grayscale image with correct size
    gray_correct = np.random.randint(0, 256, (TARGET_HEIGHT, TARGET_WIDTH), dtype=np.uint8)
    print(f"Test 3 - Grayscale correct size {gray_correct.shape}: {validate_image_size(gray_correct)}")
    
    # Test 4: Resize functionality
    print("\n🔧 Testing resize functionality...")
    large_image = np.random.randint(0, 256, (500, 400, 3), dtype=np.uint8)
    resized = resize_image_to_target(large_image)
    print(f"Original: {large_image.shape} -> Resized: {resized.shape}")
    print(f"Resized image size validation: {validate_image_size(resized)}")
    
    # Test 5: Check and fix functionality
    print("\n🛠️ Testing check_and_fix functionality...")
    test_image = np.random.randint(0, 256, (300, 250, 3), dtype=np.uint8)
    fixed_image = check_and_fix_image_size(test_image, "Test image")
    print(f"Fixed image size validation: {validate_image_size(fixed_image)}")

def test_image_data_conversion():
    """Test converting image data to numpy arrays."""
    print("\n📸 Testing image data as numpy arrays...")
    
    # Create a sample image array
    sample_image = np.random.randint(0, 256, (TARGET_HEIGHT, TARGET_WIDTH, 3), dtype=np.uint8)
    
    print(f"Sample image shape: {sample_image.shape}")
    print(f"Sample image dtype: {sample_image.dtype}")
    print(f"Sample image size validation: {validate_image_size(sample_image)}")
    
    # Test some basic operations
    print(f"Image min value: {sample_image.min()}")
    print(f"Image max value: {sample_image.max()}")
    print(f"Image mean value: {sample_image.mean():.2f}")

if __name__ == "__main__":
    print(f"🎯 Target image dimensions: {TARGET_HEIGHT}x{TARGET_WIDTH}")
    print("=" * 50)
    
    test_size_validation()
    test_image_data_conversion()
    
    print("\n✅ All tests completed!")
