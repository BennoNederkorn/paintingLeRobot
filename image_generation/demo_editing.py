#!/usr/bin/env python3
"""
Demo script to test the image editing functionality with a sample image.
This script demonstrates the complete workflow without user interaction.
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from imagegen import detect_objects_in_image, create_mask_from_bbox, edit_image_with_inpainting
import cv2

def demo_image_editing():
    """
    Demonstrates the image editing workflow with automated selections.
    """
    print("🎨 AI Image Editing Demo")
    print("=" * 40)
    
    # Use an existing image from the project
    script_dir = os.path.dirname(os.path.abspath(__file__))
    sample_image = os.path.join(script_dir, "five_computer_students_acing_robotics_hackathon.png")
    
    if not os.path.exists(sample_image):
        print("❌ Sample image not found. Please ensure the demo image exists.")
        return
    
    print(f"📸 Using sample image: {os.path.basename(sample_image)}")
    
    # Step 1: Detect objects
    print("\n🔍 Step 1: Detecting objects...")
    image_with_boxes, detections = detect_objects_in_image(sample_image)
    
    if not detections:
        print("❌ No objects detected in the sample image.")
        return
    
    # Display detected objects
    print(f"\n🎯 Found {len(detections)} objects:")
    for i, detection in enumerate(detections):
        print(f"  {i}: {detection['class_name']} (confidence: {detection['confidence']:.2f})")
    
    # Save detection reference
    if image_with_boxes is not None:
        detection_ref_path = os.path.join(script_dir, "demo_detections.png")
        cv2.imwrite(detection_ref_path, image_with_boxes)
        print(f"📸 Detection reference saved to: {os.path.basename(detection_ref_path)}")
    
    # Automatically select the first detected object for demo
    if detections:
        selected_detection = detections[0]
        print(f"\n✅ Demo: Automatically selecting '{selected_detection['class_name']}'")
        
        # Step 2: Create mask
        print("\n🎭 Step 2: Creating mask...")
        original_image = cv2.imread(sample_image)
        mask = create_mask_from_bbox(original_image.shape, selected_detection['bbox'])
        
        # Step 3: Perform inpainting (demo with a simple prompt)
        demo_prompt = f"a colorful rainbow {selected_detection['class_name']}"
        print(f"\n🎨 Step 3: Editing with prompt: '{demo_prompt}'")
        
        output_filename = "demo_edited_image.png"
        output_path = os.path.join(script_dir, output_filename)
        
        print("⏳ This may take a few moments for model loading and inference...")
        success = edit_image_with_inpainting(sample_image, mask, demo_prompt, output_path)
        
        if success:
            print(f"\n🎉 Demo completed successfully!")
            print(f"📁 Original: {os.path.basename(sample_image)}")
            print(f"📁 Edited: {output_filename}")
            print(f"📁 Detection reference: demo_detections.png")
        else:
            print("\n❌ Demo failed during image editing.")
    
    print("\n💡 To run the interactive version, use: python imagegen.py")

if __name__ == "__main__":
    demo_image_editing()
