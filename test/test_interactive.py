#!/usr/bin/env python3
"""
Interactive test for aspect ratio selection.
"""

import sys
import os

# Add the src directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from image_generation import generate_image_interactive

def main():
    print("🤖 Interactive Image Generation Test")
    print("=" * 40)
    
    try:
        # Test interactive generation
        image = generate_image_interactive("a robot painting on canvas")
        print(f"\n✅ Generated image shape: {image.shape}")
        print("🎉 Interactive test completed!")
        
    except KeyboardInterrupt:
        print("\n🛑 Test cancelled by user.")
    except Exception as e:
        print(f"\n❌ Error: {e}")

if __name__ == "__main__":
    main()
