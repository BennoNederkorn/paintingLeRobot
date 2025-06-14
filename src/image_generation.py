import numpy as np
import config
import vertexai
from vertexai.preview.vision_models import ImageGenerationModel
from vertexai.generative_models import GenerativeModel
from PIL import Image
import io
import os
import re
from datetime import datetime
import requests
import base64
import json
from google.auth import default
from google.auth.transport.requests import Request


# --- Configuration ---
# Your Google Cloud project ID
PROJECT_ID = "image-generation-462903"  

# The Google Cloud region to use
LOCATION = "us-central1"

# The output image format
OUTPUT_FORMAT = "png"

# Target image dimensions
TARGET_WIDTH = 105
TARGET_HEIGHT = 148
# ---------------------


# -------------------------------------
# Helper functions for image generation
# -------------------------------------
def create_creative_filename(prompt: str, max_length: int = 50) -> str:
    """
    Creates a creative and descriptive filename from the user's prompt.
    
    Args:
        prompt: The user's input prompt
        max_length: Maximum length for the filename (excluding extension)
    
    Returns:
        A creative filename based on the prompt
    """
    # Convert to lowercase and remove extra whitespace
    clean_prompt = prompt.lower().strip()
    
    # Remove common stop words that don't add meaning to filenames
    stop_words = {'a', 'an', 'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'must', 'can', 'shall'}
    
    # Split into words and filter out stop words and short words
    words = [word for word in re.findall(r'\b\w+\b', clean_prompt) 
             if word not in stop_words and len(word) > 2]
    
    # Take the most descriptive words (up to 6-8 words for creativity)
    key_words = words[:8]
    
    # Join with underscores and ensure it's filesystem safe
    filename_base = '_'.join(key_words)
    
    # Remove any remaining problematic characters
    filename_base = re.sub(r'[^\w\-_]', '', filename_base)
    
    # Truncate if too long, but keep it meaningful
    if len(filename_base) > max_length:
        # Try to keep complete words when truncating
        truncated = filename_base[:max_length]
        last_underscore = truncated.rfind('_')
        if last_underscore > max_length * 0.7:  # Keep if we're not losing too much
            filename_base = truncated[:last_underscore]
        else:
            filename_base = truncated
    
    # If we ended up with a very short name, add a timestamp for uniqueness
    if len(filename_base) < 10:
        timestamp = datetime.now().strftime("%m%d_%H%M")
        filename_base = f"{filename_base}_{timestamp}"
      # Ensure we have something meaningful
    if not filename_base:
        filename_base = "creative_image"
    
    return filename_base

def validate_image_size(image_array: np.ndarray) -> bool:
    """
    Check if the image array has the correct dimensions (105x148).
    
    Args:
        image_array: numpy array representing the image
        
    Returns:
        bool: True if dimensions match 105x148, False otherwise
    """
    if len(image_array.shape) == 3:  # Color image (H, W, C)
        height, width = image_array.shape[:2]
    elif len(image_array.shape) == 2:  # Grayscale image (H, W)
        height, width = image_array.shape
    else:
        return False
    
    return height == TARGET_HEIGHT and width == TARGET_WIDTH

def resize_image_to_target(image_array: np.ndarray) -> np.ndarray:
    """
    Resize the image array to the target dimensions (105x148).
    
    Args:
        image_array: numpy array representing the image
        
    Returns:
        np.ndarray: Resized image array with dimensions 105x148
    """
    # Convert numpy array to PIL Image
    if len(image_array.shape) == 3:
        # Color image
        pil_image = Image.fromarray(image_array.astype('uint8'))
    else:
        # Grayscale image
        pil_image = Image.fromarray(image_array.astype('uint8'), mode='L')
    
    # Resize to target dimensions
    resized_pil = pil_image.resize((TARGET_WIDTH, TARGET_HEIGHT), Image.Resampling.LANCZOS)
    
    # Convert back to numpy array
    return np.array(resized_pil)

def bytes_to_numpy_array(image_bytes: bytes) -> np.ndarray:
    """
    Convert image bytes to numpy array.
    
    Args:
        image_bytes: Raw image bytes
        
    Returns:
        np.ndarray: Image as numpy array
    """
    # Create PIL Image from bytes
    pil_image = Image.open(io.BytesIO(image_bytes))
    
    # Convert to RGB if necessary
    if pil_image.mode != 'RGB':
        pil_image = pil_image.convert('RGB')
    
    # Convert to numpy array
    return np.array(pil_image)

def create_color_maps(segmentated_image : np.ndarray) -> list[np.ndarray]:
    """
    This function uses the segmentated_image to create len(config.colors) color maps
    """
    # TODO use config.colors for the number of colors/pens
    return []

def generate_image_with_rest_api(prompt: str, output_file_path: str, aspect_ratio: str | None = None) -> np.ndarray:
    """
    Alternative implementation using REST API directly for better aspect ratio support.
    
    Args:
        prompt: Text prompt for image generation
        output_file_path: Path to save the generated image
        aspect_ratio: Aspect ratio ("1:1", "3:4", "4:3", "9:16", "16:9")
        
    Returns:
        np.ndarray: Generated image as numpy array (105x148)
    """
    print(f"🎨 Generating image via REST API for prompt: '{prompt}'")
    
    try:
        # Get authentication
        credentials, project = default()
        credentials.refresh(Request())
        
        # Prepare the request
        url = f"https://{LOCATION}-aiplatform.googleapis.com/v1/projects/{PROJECT_ID}/locations/{LOCATION}/publishers/google/models/imagen-3.0-generate-001:predict"
        
        headers = {
            "Authorization": f"Bearer {credentials.token}",
            "Content-Type": "application/json"
        }
        
        # Prepare request body
        request_body = {
            "instances": [{"prompt": prompt}],
            "parameters": {"sampleCount": 1}
        }
        
        # Add aspect ratio if specified
        if aspect_ratio is not None:
            supported_ratios = ["1:1", "3:4", "4:3", "9:16", "16:9"]
            if aspect_ratio in supported_ratios:
                print(f"🎯 Setting aspect ratio to: {aspect_ratio}")
                request_body["parameters"]["aspectRatio"] = aspect_ratio
            else:
                print(f"⚠️ Unsupported aspect ratio '{aspect_ratio}'. Using default.")
        
        # Make the request
        response = requests.post(url, headers=headers, json=request_body)
        
        if response.status_code == 200:
            result = response.json()
            if "predictions" in result and len(result["predictions"]) > 0:
                # Decode the base64 image
                image_b64 = result["predictions"][0]["bytesBase64Encoded"]
                image_bytes = base64.b64decode(image_b64)
                
                # Convert to numpy array
                image_array = bytes_to_numpy_array(image_bytes)
                
                # Resize to target dimensions
                if not validate_image_size(image_array):
                    print(f"⚠️ Resizing from {image_array.shape} to {TARGET_HEIGHT}x{TARGET_WIDTH}")
                    image_array = resize_image_to_target(image_array)
                
                # Save the image
                im = Image.fromarray(image_array)
                im.save(output_file_path)
                print(f"✅ Image saved successfully to: {output_file_path}")
                
                return image_array
            else:
                print("❌ No image was generated in the response.")
                return np.zeros((TARGET_HEIGHT, TARGET_WIDTH, 3), dtype=np.uint8)
        else:
            print(f"❌ REST API error {response.status_code}: {response.text}")
            return np.zeros((TARGET_HEIGHT, TARGET_WIDTH, 3), dtype=np.uint8)
            
    except Exception as e:
        print(f"❌ REST API error: {e}")
        return np.zeros((TARGET_HEIGHT, TARGET_WIDTH, 3), dtype=np.uint8)

# -------------------------------
# Main image generation function
# -------------------------------

def generate_image(prompt : str, output_file_path: str, image : np.ndarray | None = None, aspect_ratio: str | None = None) -> np.ndarray:
    """This function generates a image with dimensions 105x148 using Google's Imagen model. 
    
    Args:
        prompt: string which is the instruction for the image
        output_file_path: path to save the generated image.
        image: can be ignored for now. Will be needed if we want to update the old image during drawing.
        aspect_ratio: optional aspect ratio for the generated image. 
                     Supported values: "1:1", "3:4", "4:3", "9:16", "16:9"
                     If None, will generate at default aspect ratio and resize to target dimensions.

    Returns:
        generated_image: numpy array of the generated image (105x148)
    """

    print(f"🎨 Generating image for the prompt: '{prompt}'")

    try:
        # Initialize Vertex AI
        vertexai.init(project=PROJECT_ID, location=LOCATION)        # Load the image generation model
        model = ImageGenerationModel.from_pretrained("imagegeneration@006")  # Use newer version that supports aspect ratio
        
        # Prepare generation parameters
        generation_params = {
            "prompt": prompt,
            "number_of_images": 1,  # Generate a single image
        }
        
        # Add aspect ratio if specified
        if aspect_ratio is not None:
            # Supported aspect ratios: "1:1", "3:4", "4:3", "9:16", "16:9"
            supported_ratios = ["1:1", "3:4", "4:3", "9:16", "16:9"]
            if aspect_ratio in supported_ratios:
                print(f"🎯 Setting aspect ratio to: {aspect_ratio}")
                # Try different parameter names that might work with the Python SDK
                generation_params["aspectRatio"] = aspect_ratio  # camelCase
            else:
                print(f"⚠️ Unsupported aspect ratio '{aspect_ratio}'. Using default. Supported: {supported_ratios}")
          # Generate the image
        try:
            response = model.generate_images(**generation_params)
        except Exception as e:
            if ("aspect" in str(e).lower() or "Invalid aspect ratio" in str(e)) and aspect_ratio is not None:
                print(f"⚠️ Python SDK doesn't support aspect ratio. Trying REST API...")
                # Fallback to REST API which has better aspect ratio support
                return generate_image_with_rest_api(prompt, output_file_path, aspect_ratio)
            else:
                raise e

        # The API returns a list of generated images. We are only getting one.
        if response:
            image_data = response[0]._image_bytes

            # Convert image bytes to numpy array
            image_array = bytes_to_numpy_array(image_data)
            
            # Check if the image size matches target dimensions
            if validate_image_size(image_array):
                print(f"✅ Image already has correct dimensions: {TARGET_HEIGHT}x{TARGET_WIDTH}")
                # Save the image to the specified file
                im = Image.fromarray(image_array)
                im.save(output_file_path)
                print(f"✅ Image saved successfully to: {output_file_path}")

                return image_array
            else:
                print(f"⚠️ Image size mismatch. Resizing from {image_array.shape} to {TARGET_HEIGHT}x{TARGET_WIDTH}")
                resized_image = resize_image_to_target(image_array)

                # Save the image to the specified file
                im = Image.fromarray(resized_image)
                im.save(output_file_path)
                print(f"✅ Image saved successfully to: {output_file_path}")

                return resized_image

        else:
            print("❌ No image was generated. Please check your prompt and configuration.")
            # Return a default image array with target dimensions
            return np.zeros((TARGET_HEIGHT, TARGET_WIDTH, 3), dtype=np.uint8)

    except Exception as e:
        print(f"An error occurred: {e}")
        # Return a default image array with target dimensions
        return np.zeros((TARGET_HEIGHT, TARGET_WIDTH, 3), dtype=np.uint8)

def create_segmentated_image(image : np.ndarray) -> np.ndarray:
    """
    This function creates an image with the same size but only with len(config.colors) colors.
    return a image with len(config.colors) colors with the same size.
    You can also scale the image down
    """
    # Validate input image size
    if not validate_image_size(image):
        print(f"⚠️ Input image size mismatch. Expected {TARGET_HEIGHT}x{TARGET_WIDTH}, got {image.shape}")
        image = resize_image_to_target(image)
        print(f"✅ Image resized to {TARGET_HEIGHT}x{TARGET_WIDTH}")
    
    # TODO use config.colors for the number of colors/pens
    # For now, return a random image with the correct dimensions
    return np.random.randint(0, 256, (TARGET_HEIGHT, TARGET_WIDTH, 3), dtype=np.uint8)

# ---------------------
if __name__ == "__main__":
    print("🤖 AI Robotic Artist")
    print("=" * 40)
    try:
        # Step 1: Get user prompt for image generation
        user_prompt = input("Enter a description for the image you want to create: ").strip()
        
        if not user_prompt:
            print("❌ No prompt provided. Exiting.")
        
        # Save the prompt in local buffer
        prompt_buffer = user_prompt
        print(f"💾 Prompt saved: '{prompt_buffer}'")
        
        # Step 2: Generate the image
        # Generate a creative filename based on the user's prompt
        creative_filename = create_creative_filename(user_prompt)
        
        # Get the directory of the current script
        script_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Check if file already exists and add suffix if needed
        full_filename = f"{creative_filename}.{OUTPUT_FORMAT}"
        generated_image_path = os.path.join(script_dir, full_filename)
        
        counter = 1
        while os.path.exists(generated_image_path):
            full_filename = f"{creative_filename}_v{counter}.{OUTPUT_FORMAT}"
            generated_image_path = os.path.join(script_dir, full_filename)
            counter += 1
    
        print(f"💡 Saving as: {full_filename}")
    
        # Generate the image
        generate_image(prompt=user_prompt, output_file_path=generated_image_path, aspect_ratio="3:4")
        
        # Check if image was actually created
        if not os.path.exists(generated_image_path):
            print("❌ Image generation failed. Cannot proceed with editing.")
        

    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
    except Exception as e:
        print(f"\n❌ An unexpected error occurred: {e}")
        print("💡 Please try again or check your configuration.")