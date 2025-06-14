#!/usr/bin/env python3
"""
Test the aspect ratio functionality with error handling.
"""

import sys
import os

# Add the src directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from image_generation import generate_image

def test_aspect_ratio():
    """Test aspect ratio generation with error handling."""
    print("🧪 Testing Aspect Ratio Functionality")
    print("=" * 40)
    
    test_prompt = "a beautiful sunset over mountains"
    output_path = "test_aspect_ratio.png"
    
    # Test with 3:4 aspect ratio
    print("\n📐 Testing 3:4 aspect ratio...")
    try:
        image = generate_image(test_prompt, output_path, aspect_ratio="3:4")
        print(f"✅ Success! Generated image shape: {image.shape}")
        return True
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

if __name__ == "__main__":
    success = test_aspect_ratio()
    if success:
        print("\n🎉 Aspect ratio test completed successfully!")
    else:
        print("\n💥 Aspect ratio test failed. Check the error messages above.")
