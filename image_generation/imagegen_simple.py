import vertexai
from vertexai.preview.vision_models import ImageGenerationModel, Image as VertexImage
import base64
import os
import uuid
import re
from datetime import datetime
from PIL import Image, ImageDraw
import json

# --- Configuration ---
# Your Google Cloud project ID
PROJECT_ID = "image-generation-462903"  

# The Google Cloud region to use
LOCATION = "us-central1"

# The output image format
OUTPUT_FORMAT = "png"
# ---------------------

def generate_image(prompt, num_images=1, seed=None):
    """
    Generate an image using Vertex AI based on a text prompt.
    
    Args:
        prompt (str): The text prompt describing the image to generate
        num_images (int): Number of images to generate (default: 1)
        seed (int): Random seed for reproducible results (optional)
    
    Returns:
        tuple: (image_paths, used_seed) where image_paths is a list of generated image file paths
    """
    
    print(f"🎨 Generating image for the prompt: '{prompt}'")
    
    try:
        # Initialize Vertex AI
        vertexai.init(project=PROJECT_ID, location=LOCATION)
        
        # Load the image generation model - using a working model
        model = ImageGenerationModel.from_pretrained("imagen-3.0-generate-001")
        
        # Generate the image(s)
        if seed is not None:
            response = model.generate_images(
                prompt=prompt,
                number_of_images=num_images,
                seed=seed
            )
            used_seed = seed
        else:
            response = model.generate_images(
                prompt=prompt,
                number_of_images=num_images
            )
            # Generate a random seed for future reference
            used_seed = uuid.uuid4().int % (2**32)
        
        # Save the generated images
        image_paths = []
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        for i, image in enumerate(response.images):
            # Create a unique filename
            unique_id = str(uuid.uuid4()).replace('-', '')[:8]
            filename = f"generated_image_{timestamp}_{unique_id}.{OUTPUT_FORMAT}"
            filepath = os.path.join(os.getcwd(), filename)
            
            # Save the image
            image.save(location=filepath, include_generation_parameters=False)
            image_paths.append(filepath)
            
            print(f"✅ Image {i+1} saved as: {filename}")
        
        print(f"🌱 Seed used: {used_seed}")
        print(f"📝 Prompt: {prompt}")
        
        return image_paths, used_seed
        
    except Exception as e:
        print(f"❌ Error generating image: {e}")
        return [], None

def create_simple_mask(image_path, mask_type="eyes"):
    """
    Create a simple mask for image editing.
    For now, this creates a basic circular mask in the center-upper area (eyes region).
    
    Args:
        image_path (str): Path to the original image
        mask_type (str): Type of mask to create (currently only supports "eyes")
    
    Returns:
        str: Path to the created mask file
    """
    
    try:
        # Open the original image to get dimensions
        with Image.open(image_path) as img:
            width, height = img.size
        
        # Create a black image (mask background)
        mask = Image.new('RGB', (width, height), 'black')
        draw = ImageDraw.Draw(mask)
        
        if mask_type == "eyes":
            # Create two circular white areas for eyes
            # Approximate eye positions (adjust as needed)
            left_eye_x = int(width * 0.35)
            right_eye_x = int(width * 0.65)
            eye_y = int(height * 0.4)
            eye_radius = int(min(width, height) * 0.05)
            
            # Draw white circles for eye areas
            draw.ellipse([
                left_eye_x - eye_radius, eye_y - eye_radius,
                left_eye_x + eye_radius, eye_y + eye_radius
            ], fill='white')
            
            draw.ellipse([
                right_eye_x - eye_radius, eye_y - eye_radius,
                right_eye_x + eye_radius, eye_y + eye_radius
            ], fill='white')
        
        # Save the mask
        mask_filename = f"mask_{mask_type}_{os.path.basename(image_path)}"
        mask_path = os.path.join(os.path.dirname(image_path), mask_filename)
        mask.save(mask_path)
        
        print(f"🎭 Mask created: {mask_filename}")
        return mask_path
        
    except Exception as e:
        print(f"❌ Error creating mask: {e}")
        return None

def edit_image_with_mask(original_image_path, mask_path, new_prompt, seed=None):
    """
    Edit an image using inpainting with a mask.
    
    Args:
        original_image_path (str): Path to the original image
        mask_path (str): Path to the mask image
        new_prompt (str): New prompt describing the desired changes
        seed (int): Random seed for reproducible results (optional)
    
    Returns:
        tuple: (edited_image_path, used_seed)
    """
    
    print(f"🎨 Editing image with prompt: '{new_prompt}'")
    print(f"🖼️ Original: {os.path.basename(original_image_path)}")
    print(f"🎭 Mask: {os.path.basename(mask_path)}")
    
    try:
        # Initialize Vertex AI
        vertexai.init(project=PROJECT_ID, location=LOCATION)
        
        # Load the image generation model
        model = ImageGenerationModel.from_pretrained("imagen-3.0-generate-001")
        
        # Load the original image as Vertex AI Image
        original_image = VertexImage.load_from_file(original_image_path)
        
        # Load the mask as Vertex AI Image  
        mask_image = VertexImage.load_from_file(mask_path)

        # Generate the edited image using edit_image
        if seed is not None:
            response = model.edit_image(
                prompt=new_prompt,
                base_image=original_image,
                mask=mask_image,
                number_of_images=1,
                seed=seed
            )
            used_seed = seed
        else:
            response = model.edit_image(
                prompt=new_prompt,
                base_image=original_image,
                mask=mask_image,
                number_of_images=1
            )
            used_seed = uuid.uuid4().int % (2**32)

        # Save the edited image
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        unique_id = str(uuid.uuid4()).replace('-', '')[:8]
        edited_filename = f"edited_image_{timestamp}_{unique_id}.{OUTPUT_FORMAT}"
        edited_filepath = os.path.join(os.path.dirname(original_image_path), edited_filename)
        
        response.images[0].save(location=edited_filepath, include_generation_parameters=False)
        
        print(f"✅ Edited image saved as: {edited_filename}")
        print(f"🌱 Seed used: {used_seed}")
        
        return edited_filepath, used_seed
        
    except Exception as e:
        print(f"❌ Error editing image: {e}")
        return None, None

def interactive_workflow():
    """
    Interactive workflow for generating and editing images.
    """
    
    print("🎨 Welcome to the AI Image Generator with Selective Editing!")
    print("=" * 60)
    
    while True:
        print("\nChoose an option:")
        print("1. Generate a new image")
        print("2. Edit an existing image (selective editing)")
        print("3. Exit")
        
        choice = input("\nEnter your choice (1-3): ").strip()
        
        if choice == "1":
            # Generate new image
            prompt = input("\nEnter your image prompt: ").strip()
            if not prompt:
                print("❌ Please enter a valid prompt.")
                continue
            
            seed_input = input("Enter seed (optional, press Enter to skip): ").strip()
            seed = None
            if seed_input:
                try:
                    seed = int(seed_input)
                except ValueError:
                    print("❌ Invalid seed. Using random seed.")
            
            image_paths, used_seed = generate_image(prompt, seed=seed)
            
            if image_paths:
                print(f"\n🎉 Success! Generated {len(image_paths)} image(s)")
                for path in image_paths:
                    print(f"📁 {path}")
                print(f"🌱 Seed: {used_seed} (save this for reproducible results)")
        
        elif choice == "2":
            # Edit existing image
            image_path = input("\nEnter the path to your image: ").strip()
            if not os.path.exists(image_path):
                print("❌ Image file not found.")
                continue
            
            print("\nAvailable mask types:")
            print("- eyes (default): Creates masks for both eyes")
            mask_type = input("Enter mask type (or press Enter for 'eyes'): ").strip() or "eyes"
            
            # Create mask
            print(f"\n🎭 Creating {mask_type} mask...")
            mask_path = create_simple_mask(image_path, mask_type)
            
            if not mask_path:
                print("❌ Failed to create mask.")
                continue
            
            # Get new prompt for editing
            new_prompt = input("\nEnter your editing prompt (e.g., 'striking blue eyes'): ").strip()
            if not new_prompt:
                print("❌ Please enter a valid editing prompt.")
                continue
            
            seed_input = input("Enter seed (optional, press Enter to skip): ").strip()
            seed = None
            if seed_input:
                try:
                    seed = int(seed_input)
                except ValueError:
                    print("❌ Invalid seed. Using random seed.")
            
            # Edit the image
            edited_path, used_seed = edit_image_with_mask(image_path, mask_path, new_prompt, seed)
            
            if edited_path:
                print(f"\n🎉 Success! Edited image saved as:")
                print(f"📁 {edited_path}")
                print(f"🌱 Seed: {used_seed}")
            
        elif choice == "3":
            print("\n👋 Goodbye!")
            break
        
        else:
            print("❌ Invalid choice. Please enter 1, 2, or 3.")

if __name__ == "__main__":
    interactive_workflow()
