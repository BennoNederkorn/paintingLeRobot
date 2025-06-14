import vertexai
from vertexai.preview.vision_models import ImageGenerationModel
from vertexai.generative_models import GenerativeModel
import base64
import os
import uuid
import re
from datetime import datetime
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO
from diffusers import StableDiffusionInpaintPipeline
import torch

# --- Configuration ---
# Your Google Cloud project ID
PROJECT_ID = "image-generation-462903"  

# The Google Cloud region to use
LOCATION = "us-central1"

# The output image format
OUTPUT_FORMAT = "png"
# ---------------------

def generate_image_from_text(prompt: str, output_file_path: str):
    """
    Generates an image from a text prompt using Google's Imagen model.

    Args:
        prompt: The text description of the image to generate.
        output_file_path: The path to save the generated image.
    """
    print(f"🎨 Generating image for the prompt: '{prompt}'")

    try:
        # Initialize Vertex AI
        vertexai.init(project=PROJECT_ID, location=LOCATION)

        # Load the image generation model
        model = ImageGenerationModel.from_pretrained("imagegeneration@005")

        # Generate the image
        response = model.generate_images(
            prompt=prompt,
            number_of_images=1,  # Generate a single image
        )

        # The API returns a list of generated images. We are only getting one.
        if response:
            image_data = response[0]._image_bytes

            # Save the image to the specified file
            with open(output_file_path, "wb") as f:
                f.write(image_data)
            print(f"✅ Image saved successfully to: {output_file_path}")
        else:
            print("❌ No image was generated. Please check your prompt and configuration.")

    except Exception as e:
        print(f"An error occurred: {e}")

# ---------------------

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

def detect_objects_in_image(image_path: str):
    """
    Detects objects in an image using YOLO and returns bounding boxes.
    
    Args:
        image_path: Path to the input image
    
    Returns:
        tuple: (image_with_boxes, detections) where detections is a list of detection info
    """
    print(f"🔍 Detecting objects in image: {image_path}")
    
    try:
        # Load YOLO model
        model = YOLO('yolov8n.pt')  # Using nano version for speed
        
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image from {image_path}")
        
        # Run inference
        results = model(image)
        
        # Process results
        detections = []
        image_with_boxes = image.copy()
        
        for r in results:
            boxes = r.boxes
            if boxes is not None:
                for i, box in enumerate(boxes):
                    # Get box coordinates
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    confidence = box.conf[0].cpu().numpy()
                    class_id = int(box.cls[0].cpu().numpy())
                    class_name = model.names[class_id]
                    
                    # Only include detections with reasonable confidence
                    if confidence > 0.5:
                        detection_info = {
                            'id': i,
                            'bbox': (int(x1), int(y1), int(x2), int(y2)),
                            'confidence': float(confidence),
                            'class_name': class_name,
                            'class_id': class_id
                        }
                        detections.append(detection_info)
                        
                        # Draw bounding box
                        color = (0, 255, 0)  # Green
                        cv2.rectangle(image_with_boxes, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                        
                        # Add label
                        label = f"{i}: {class_name} ({confidence:.2f})"
                        cv2.putText(image_with_boxes, label, (int(x1), int(y1-10)), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        print(f"✅ Found {len(detections)} objects")
        return image_with_boxes, detections
        
    except Exception as e:
        print(f"❌ Error during object detection: {e}")
        return None, []

def create_mask_from_bbox(image_shape, bbox):
    """
    Creates a binary mask from a bounding box.
    
    Args:
        image_shape: Shape of the original image (height, width, channels)
        bbox: Bounding box coordinates (x1, y1, x2, y2)
    
    Returns:
        PIL Image: Binary mask
    """
    height, width = image_shape[:2]
    x1, y1, x2, y2 = bbox
    
    # Create a black image
    mask = np.zeros((height, width), dtype=np.uint8)
    
    # Fill the bounding box area with white
    mask[y1:y2, x1:x2] = 255
    
    # Convert to PIL Image
    mask_pil = Image.fromarray(mask)
    return mask_pil

def clear_huggingface_cache():
    """Clear Hugging Face cache to force re-download"""
    import shutil
    cache_dirs = [
        os.path.expanduser("~/.cache/huggingface"),
        os.path.expanduser("~/.cache/torch"),
        "C:\\Users\\zhuol\\.cache\\huggingface"  # Windows specific path from error
    ]
    
    for cache_dir in cache_dirs:
        if os.path.exists(cache_dir):
            try:
                print(f"🧹 Clearing cache: {cache_dir}")
                shutil.rmtree(cache_dir)
                print(f"✅ Cleared cache")
                return True
            except Exception as e:
                print(f"⚠️ Could not clear {cache_dir}: {e}")
    return False

def create_inpainting_pipeline_safe():
    """Create inpainting pipeline with multiple fallback options"""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Method 1: Try the original model
    try:
        print("🎨 Loading Stable Diffusion inpainting pipeline...")
        pipe = StableDiffusionInpaintPipeline.from_pretrained(
            "runwayml/stable-diffusion-inpainting",
            torch_dtype=torch.float32,  # Use float32 for compatibility
            safety_checker=None,
            requires_safety_checker=False
        )
        pipe = pipe.to(device)
        print("✅ Successfully loaded original inpainting model")
        return pipe
    except Exception as e1:
        print(f"⚠️ Original model failed: {e1}")
        
    # Method 2: Try with cache clearing
    try:
        print("🔄 Clearing cache and trying again...")
        clear_huggingface_cache()
        pipe = StableDiffusionInpaintPipeline.from_pretrained(
            "runwayml/stable-diffusion-inpainting",
            torch_dtype=torch.float32,
            safety_checker=None,
            requires_safety_checker=False,
            force_download=True
        )
        pipe = pipe.to(device)
        print("✅ Successfully loaded with fresh download")
        return pipe
    except Exception as e2:
        print(f"⚠️ Fresh download failed: {e2}")
        
    # Method 3: Try alternative model
    try:
        print("🔄 Trying alternative inpainting model...")
        pipe = StableDiffusionInpaintPipeline.from_pretrained(
            "stabilityai/stable-diffusion-2-inpainting",
            torch_dtype=torch.float32,
            safety_checker=None,
            requires_safety_checker=False
        )
        pipe = pipe.to(device)
        print("✅ Successfully loaded alternative model")
        return pipe
    except Exception as e3:
        print(f"⚠️ Alternative model failed: {e3}")
        
    return None

def simple_image_edit_fallback(image_path: str, mask: Image.Image, prompt: str, output_path: str):
    """Simple fallback image editing when AI inpainting fails"""
    try:
        print("🔄 Using fallback image editing method...")
        
        # Load original image
        original_cv = cv2.imread(image_path)
        original_pil = Image.open(image_path).convert("RGB")
        
        # Convert mask to numpy array
        mask_np = np.array(mask.convert("L"))
        
        # Determine color based on prompt keywords
        color_bgr = (255, 0, 0)  # Default blue in BGR
        if 'red' in prompt.lower():
            color_bgr = (0, 0, 255)
        elif 'green' in prompt.lower():
            color_bgr = (0, 255, 0)
        elif 'blue' in prompt.lower():
            color_bgr = (255, 0, 0)
        elif 'yellow' in prompt.lower():
            color_bgr = (0, 255, 255)
        elif 'purple' in prompt.lower():
            color_bgr = (255, 0, 255)
        elif 'orange' in prompt.lower():
            color_bgr = (0, 165, 255)
        
        # Create colored overlay
        overlay = original_cv.copy()
        overlay[mask_np > 128] = color_bgr
        
        # Blend with original (30% overlay, 70% original)
        alpha = 0.3
        result = cv2.addWeighted(original_cv, 1-alpha, overlay, alpha, 0)
        
        # Save result
        cv2.imwrite(output_path, result)
        
        print(f"✅ Fallback edit completed: {output_path}")
        print("ℹ️ Note: This is a simple color overlay. For AI-powered editing, please run 'python fix_huggingface.py'")
        return True
        
    except Exception as e:
        print(f"❌ Even fallback method failed: {e}")
        return False

def edit_image_with_inpainting(image_path: str, mask: Image.Image, prompt: str, output_path: str):
    """
    Edits an image using inpainting with Stable Diffusion, with fallback options.
    
    Args:
        image_path: Path to the original image
        mask: PIL Image mask indicating areas to edit
        prompt: Text description of desired changes
        output_path: Path to save the edited image
    """
    print(f"🎨 Editing image with prompt: '{prompt}'")
    
    try:
        # Try to create the inpainting pipeline
        pipe = create_inpainting_pipeline_safe()
        
        if pipe is None:
            print("❌ Could not load any AI inpainting model. Using fallback method...")
            return simple_image_edit_fallback(image_path, mask, prompt, output_path)
        
        # Load and prepare the original image
        original_image = Image.open(image_path).convert("RGB")
        
        # Ensure mask and image have the same size
        mask_resized = mask.resize(original_image.size)
        
        print("🎨 Generating AI-edited image...")
          # Generate the edited image with error handling
        try:
            result_obj = pipe(
                prompt=prompt,
                image=original_image,
                mask_image=mask_resized,
                num_inference_steps=20,
                guidance_scale=7.5,
                height=512,
                width=512
            )
            
            # Handle different return types from diffusers
            if hasattr(result_obj, 'images'):
                result = result_obj.images[0]
            else:
                result = result_obj[0] if isinstance(result_obj, (list, tuple)) else result_obj
            
            # Resize result back to original size if needed
            if result.size != original_image.size:
                result = result.resize(original_image.size, Image.Resampling.LANCZOS)
            
            # Save the result
            result.save(output_path)
            print(f"✅ AI-edited image saved to: {output_path}")
            return True
            
        except Exception as generation_error:
            print(f"❌ AI generation failed: {generation_error}")
            print("🔄 Falling back to simple editing...")
            return simple_image_edit_fallback(image_path, mask, prompt, output_path)
        
    except Exception as e:
        print(f"❌ Error during image editing setup: {e}")
        print("🔄 Using fallback method...")
        return simple_image_edit_fallback(image_path, mask, prompt, output_path)

def edit_existing_image():
    """
    Main function to handle the image editing workflow.
    """
    print("\n🖼️  IMAGE EDITING MODE")
    print("=" * 50)
    
    # Step 1: Get image path from user
    while True:
        image_path = input("Enter the path to the image you want to edit: ").strip().strip('"')
        if os.path.exists(image_path):
            break
        else:
            print(f"❌ File not found: {image_path}")
            print("Please enter a valid file path.")
    
    # Step 2: Detect objects in the image
    image_with_boxes, detections = detect_objects_in_image(image_path)
    
    if not detections:
        print("❌ No objects detected in the image. Cannot proceed with editing.")
        return
    
    # Save image with bounding boxes for user reference
    script_dir = os.path.dirname(os.path.abspath(__file__))
    boxes_image_path = os.path.join(script_dir, "temp_objects_detected.png")
    if image_with_boxes is not None:
        cv2.imwrite(boxes_image_path, image_with_boxes)
        print(f"📸 Image with detected objects saved to: {boxes_image_path}")
    else:
        print("❌ Could not save image with bounding boxes")
    
    # Step 3: Display detected objects and let user choose
    print("\n🎯 DETECTED OBJECTS:")
    print("-" * 30)
    for detection in detections:
        print(f"ID {detection['id']}: {detection['class_name']} (confidence: {detection['confidence']:.2f})")
    
    # Get user's choice
    while True:
        try:
            choice = int(input(f"\nEnter the ID of the object you want to edit (0-{len(detections)-1}): "))
            if 0 <= choice < len(detections):
                selected_detection = detections[choice]
                break
            else:
                print(f"❌ Please enter a number between 0 and {len(detections)-1}")
        except ValueError:
            print("❌ Please enter a valid number")
    
    print(f"✅ You selected: {selected_detection['class_name']}")
    
    # Step 4: Get editing prompt from user
    edit_prompt = input(f"\nEnter your description for how you want to modify the {selected_detection['class_name']}: ").strip()
    
    if not edit_prompt:
        print("❌ No editing prompt provided. Exiting.")
        return
    
    # Step 5: Create mask and perform inpainting
    # Load original image to get shape
    original_image = cv2.imread(image_path)
    mask = create_mask_from_bbox(original_image.shape, selected_detection['bbox'])
    
    # Generate output filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename_base = os.path.splitext(os.path.basename(image_path))[0]
    output_filename = f"edited_{filename_base}_{timestamp}_{uuid.uuid4().hex[:8]}.png"
    output_path = os.path.join(script_dir, output_filename)
    
    print(f"💡 Saving edited image as: {output_filename}")
      # Perform the inpainting
    success = edit_image_with_inpainting(image_path, mask, edit_prompt, output_path)
    
    if success:
        print(f"🎉 Image editing completed successfully!")
        print(f"📁 Original image: {image_path}")
        print(f"📁 Edited image: {output_path}")
        print(f"📁 Objects detection reference: {boxes_image_path}")
    else:
        print("❌ Image editing failed. Please try again.")
        print("💡 If you're getting AI model errors, try running: python fix_huggingface.py")
    
    # Clean up temporary file
    try:
        os.remove(boxes_image_path)
    except:
        pass

def identify_objects_from_prompt(prompt: str) -> list:
    """
    Uses LLM to identify objects mentioned in the text prompt.
    
    Args:
        prompt: The original text prompt used for image generation
    
    Returns:
        list: List of objects identified in the prompt
    """
    print(f"🧠 Analyzing prompt to identify objects: '{prompt}'")
    
    try:
        # Initialize Vertex AI
        vertexai.init(project=PROJECT_ID, location=LOCATION)
        
        # Load the generative model
        model = GenerativeModel("gemini-pro")
        
        # Create a prompt for object identification
        analysis_prompt = f"""
        Analyze the following image generation prompt and list all the specific objects, animals, or things mentioned in it.
        Only list concrete, visible objects that would appear in an image. Do not include abstract concepts, colors, or descriptive words.
        Return the objects as a simple comma-separated list.
        
        Prompt: "{prompt}"
        
        Objects:
        """
        
        response = model.generate_content(analysis_prompt)
        
        if response and response.text:
            # Parse the response to extract objects
            objects_text = response.text.strip()
            # Split by comma and clean up
            objects = [obj.strip().lower() for obj in objects_text.split(',') if obj.strip()]
            # Remove empty strings and duplicates
            objects = list(set([obj for obj in objects if obj]))
            
            print(f"✅ Identified objects: {objects}")
            return objects
        else:
            print("❌ No objects identified from prompt")
            return []
            
    except Exception as e:
        print(f"❌ Error analyzing prompt: {e}")        # Enhanced fallback: comprehensive keyword extraction with better parsing
        print("🔄 Using enhanced fallback object detection...")
        return extract_objects_from_text(prompt)

def get_object_detection_for_editing(image_path: str, target_object: str):
    """
    Detects specific objects in an image for editing purposes.
    
    Args:
        image_path: Path to the input image
        target_object: The specific object to look for
    
    Returns:
        tuple: (image_with_boxes, matching_detections)
    """
    print(f"🎯 Looking for '{target_object}' in the image...")
    
    try:
        # Load YOLO model
        model = YOLO('yolov8n.pt')
        
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image from {image_path}")
        
        # Run inference
        results = model(image)
        
        # Process results and find matching objects
        matching_detections = []
        image_with_boxes = image.copy()
        
        for r in results:
            boxes = r.boxes
            if boxes is not None:
                for i, box in enumerate(boxes):
                    # Get box coordinates
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    confidence = box.conf[0].cpu().numpy()
                    class_id = int(box.cls[0].cpu().numpy())
                    class_name = model.names[class_id]
                    
                    # Check if this object matches our target (with some flexibility)
                    if confidence > 0.3 and is_object_match(class_name, target_object):
                        detection_info = {
                            'id': len(matching_detections),
                            'bbox': (int(x1), int(y1), int(x2), int(y2)),
                            'confidence': float(confidence),
                            'class_name': class_name,
                            'class_id': class_id
                        }
                        matching_detections.append(detection_info)
                        
                        # Draw bounding box
                        color = (0, 255, 255)  # Yellow for target objects
                        cv2.rectangle(image_with_boxes, (int(x1), int(y1)), (int(x2), int(y2)), color, 3)
                        
                        # Add label
                        label = f"{len(matching_detections)-1}: {class_name} ({confidence:.2f})"
                        cv2.putText(image_with_boxes, label, (int(x1), int(y1-10)), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        print(f"✅ Found {len(matching_detections)} instances of '{target_object}'")
        return image_with_boxes, matching_detections
        
    except Exception as e:
        print(f"❌ Error during object detection: {e}")
        return None, []

def is_object_match(detected_class: str, target_object: str) -> bool:
    """
    Check if a detected object class matches the target object with enhanced flexibility.
    
    Args:
        detected_class: The class name from YOLO detection
        target_object: The object we're looking for
    
    Returns:
        bool: True if they match
    """
    detected_lower = detected_class.lower()
    target_lower = target_object.lower()
    
    # Direct match
    if detected_lower == target_lower:
        return True
    
    # Check if target is contained in detected or vice versa
    if target_lower in detected_lower or detected_lower in target_lower:
        return True
    
    # Enhanced mappings with more comprehensive categories
    mappings = {
        # People
        'person': ['people', 'man', 'woman', 'human', 'individual', 'child', 'baby', 'boy', 'girl', 'student'],
        'people': ['person', 'man', 'woman', 'human', 'individual'],
        
        # Vehicles
        'car': ['vehicle', 'automobile', 'sedan', 'suv'],
        'truck': ['lorry', 'pickup', 'vehicle'],
        'bus': ['coach', 'vehicle'],
        'bicycle': ['bike', 'cycle'],
        'motorcycle': ['motorbike', 'bike'],
        'boat': ['ship', 'vessel', 'watercraft'],
        
        # Animals
        'dog': ['puppy', 'canine', 'hound'],
        'cat': ['kitten', 'feline'],
        'bird': ['eagle', 'crow', 'pigeon', 'seagull', 'hawk', 'pelican', 'chicken', 'duck'],
        'fish': ['salmon', 'trout', 'tuna'],
        'whale': ['mammal', 'cetacean'],
        'horse': ['pony', 'stallion', 'mare'],
        'cow': ['cattle', 'bull'],
        'sheep': ['lamb'],
        
        # Objects
        'bottle': ['container', 'flask', 'jar'],
        'cup': ['mug', 'glass', 'tumbler'],
        'chair': ['seat', 'stool'],
        'table': ['desk', 'surface'],
        'bag': ['backpack', 'purse', 'suitcase', 'handbag'],
        'phone': ['mobile', 'smartphone', 'cellphone'],
        'computer': ['laptop', 'pc', 'monitor'],
        'book': ['novel', 'magazine', 'journal'],
        
        # Sports equipment
        'ball': ['soccer ball', 'football', 'basketball', 'tennis ball', 'baseball'],
        'racket': ['tennis racket', 'badminton racket'],
        
        # Kitchen items
        'knife': ['blade', 'cutlery'],
        'fork': ['cutlery'],
        'spoon': ['cutlery'],
        'plate': ['dish'],
        'bowl': ['dish'],
        
        # Nature (YOLO might detect some of these as other classes)
        'tree': ['plant', 'vegetation'],
        'flower': ['plant', 'rose', 'daisy'],
        'grass': ['plant', 'vegetation'],
        
        # Note: YOLO typically doesn't detect natural features like rivers, sky, etc.
        # These would need specialized models or semantic segmentation
    }
    
    # Check mappings in both directions
    for main_object, synonyms in mappings.items():
        # If detected is main and target is synonym
        if detected_lower == main_object and target_lower in synonyms:
            return True
        # If target is main and detected is synonym
        if target_lower == main_object and detected_lower in synonyms:
            return True
        # If both are synonyms of the same main object
        if detected_lower in synonyms and target_lower in synonyms:
            return True
        # If detected is synonym and target is main
        if detected_lower in synonyms and target_lower == main_object:
            return True
        # If target is synonym and detected is main
        if target_lower in synonyms and detected_lower == main_object:
            return True
    
    # Special handling for partial word matches (be careful with this)
    # Only for longer words to avoid false positives
    if len(target_lower) > 4 and len(detected_lower) > 4:
        if target_lower[:4] == detected_lower[:4]:  # First 4 characters match
            return True
    
    return False

# ---------------------
def handle_undetectable_objects(image_path: str, target_object: str, prompt: str):
    """
    Handle objects that YOLO typically can't detect (like rivers, sky, etc.)
    by creating a smart mask based on image analysis or user guidance.
    
    Args:
        image_path: Path to the image
        target_object: The object we're looking for
        prompt: Original prompt for context
    
    Returns:
        tuple: (image_with_indication, synthetic_detection_info) or (None, None)
    """
    # Objects that YOLO typically cannot detect well
    undetectable_objects = [
        'river', 'stream', 'water', 'ocean', 'sea', 'lake', 'pond',
        'sky', 'cloud', 'sun', 'moon', 'star', 'rainbow',
        'grass', 'field', 'meadow', 'forest', 'ground', 'earth',
        'mountain', 'hill', 'landscape', 'background', 'foreground',
        'light', 'shadow', 'reflection', 'mist', 'fog'
    ]
    
    if target_object.lower() not in undetectable_objects:
        return None, None
    
    print(f"🎯 '{target_object}' is typically not detectable by object detection models.")
    print("💡 Using intelligent region suggestion...")
    
    try:
        # Load the image
        image = cv2.imread(image_path)
        if image is None:
            return None, None
        
        height, width = image.shape[:2]
        
        # Create region suggestions based on object type and common locations
        synthetic_detections = []
        
        if target_object.lower() in ['sky', 'cloud', 'sun', 'moon']:
            # Sky is typically in the upper portion
            detection = {
                'id': 0,
                'bbox': (0, 0, width, height // 2),
                'confidence': 0.95,
                'class_name': f'{target_object} (estimated region)',
                'class_id': -1
            }
            synthetic_detections.append(detection)
            
        elif target_object.lower() in ['river', 'stream', 'water', 'lake']:
            # Water could be in various locations, suggest horizontal bands
            # Middle horizontal strip
            detection1 = {
                'id': 0,
                'bbox': (0, height // 3, width, 2 * height // 3),
                'confidence': 0.90,
                'class_name': f'{target_object} (estimated region - middle)',
                'class_id': -1
            }
            # Lower horizontal strip
            detection2 = {
                'id': 1,
                'bbox': (0, 2 * height // 3, width, height),
                'confidence': 0.85,
                'class_name': f'{target_object} (estimated region - lower)',
                'class_id': -1
            }
            synthetic_detections.extend([detection1, detection2])
            
        elif target_object.lower() in ['grass', 'ground', 'meadow', 'field']:
            # Ground/grass is typically in the lower portion
            detection = {
                'id': 0,
                'bbox': (0, height // 2, width, height),
                'confidence': 0.90,
                'class_name': f'{target_object} (estimated region)',
                'class_id': -1
            }
            synthetic_detections.append(detection)
            
        elif target_object.lower() in ['mountain', 'hill']:
            # Mountains/hills could be in background, suggest middle to upper area
            detection = {
                'id': 0,
                'bbox': (0, height // 4, width, 3 * height // 4),
                'confidence': 0.85,
                'class_name': f'{target_object} (estimated region)',
                'class_id': -1
            }
            synthetic_detections.append(detection)
            
        else:
            # Generic fallback - offer full image regions
            detection1 = {
                'id': 0,
                'bbox': (0, 0, width // 2, height // 2),
                'confidence': 0.75,
                'class_name': f'{target_object} (top-left region)',
                'class_id': -1
            }
            detection2 = {
                'id': 1,
                'bbox': (width // 2, 0, width, height // 2),
                'confidence': 0.75,
                'class_name': f'{target_object} (top-right region)',
                'class_id': -1
            }
            detection3 = {
                'id': 2,
                'bbox': (0, height // 2, width // 2, height),
                'confidence': 0.75,
                'class_name': f'{target_object} (bottom-left region)',
                'class_id': -1
            }
            detection4 = {
                'id': 3,
                'bbox': (width // 2, height // 2, width, height),
                'confidence': 0.75,
                'class_name': f'{target_object} (bottom-right region)',
                'class_id': -1
            }
            synthetic_detections.extend([detection1, detection2, detection3, detection4])
        
        # Draw the suggested regions
        image_with_boxes = image.copy()
        for detection in synthetic_detections:
            x1, y1, x2, y2 = detection['bbox']
            color = (255, 165, 0)  # Orange for synthetic detections
            cv2.rectangle(image_with_boxes, (x1, y1), (x2, y2), color, 2)
            
            label = f"{detection['id']}: {detection['class_name']}"
            cv2.putText(image_with_boxes, label, (x1, y1-10), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        print(f"✅ Generated {len(synthetic_detections)} region suggestions for '{target_object}'")
        return image_with_boxes, synthetic_detections
        
    except Exception as e:
        print(f"❌ Error creating region suggestions: {e}")
        return None, None

# ---------------------
def main_workflow():
    """
    Main workflow that handles image generation and optional editing.
    """
    print("🤖 AI Robotic Artist")
    print("=" * 40)
    
    # Step 1: Get user prompt for image generation
    print("\n🎨 IMAGE GENERATION")
    print("=" * 30)
    user_prompt = input("Enter a description for the image you want to create: ").strip()
    
    if not user_prompt:
        print("❌ No prompt provided. Exiting.")
        return
    
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
    generate_image_from_text(user_prompt, generated_image_path)
    
    # Check if image was actually created
    if not os.path.exists(generated_image_path):
        print("❌ Image generation failed. Cannot proceed with editing.")
        return
    
    print(f"✅ Image generated successfully: {generated_image_path}")
    
    # Step 3: Ask if user wants to edit the generated image
    print("\n�️  EDIT GENERATED IMAGE?")
    print("=" * 30)
    
    while True:
        edit_choice = input("Do you want to edit the generated image? (y/n): ").strip().lower()
        if edit_choice in ['y', 'yes', 'n', 'no']:
            break
        print("❌ Please enter 'y' for yes or 'n' for no")
    
    if edit_choice in ['n', 'no']:
        print("✅ Task complete! Your generated image is ready.")
        print(f"📁 Image location: {generated_image_path}")
        return
    
    # Step 4: Use LLM to identify objects from the saved prompt
    print("\n🧠 ANALYZING PROMPT FOR OBJECTS")
    print("=" * 35)
    
    identified_objects = identify_objects_from_prompt(prompt_buffer)
    
    if not identified_objects:
        print("❌ No objects could be identified from your prompt.")
        print("💡 Try using the manual editing mode by running the script with option 2.")
        return
    
    # Step 5: Ask user which object they want to change
    print("\n🎯 SELECT OBJECT TO EDIT")
    print("=" * 25)
    print("Objects identified in your prompt:")
    for i, obj in enumerate(identified_objects):
        print(f"{i + 1}. {obj}")
    
    while True:
        try:
            obj_choice = int(input(f"\nEnter the number of the object you want to edit (1-{len(identified_objects)}): "))
            if 1 <= obj_choice <= len(identified_objects):
                selected_object = identified_objects[obj_choice - 1]
                break
            else:
                print(f"❌ Please enter a number between 1 and {len(identified_objects)}")
        except ValueError:
            print("❌ Please enter a valid number")
    
    print(f"✅ You selected: {selected_object}")
    
    # Step 6: Ask how they want to change it
    edit_description = input(f"\nHow do you want to change the {selected_object}? ").strip()
    
    if not edit_description:
        print("❌ No edit description provided. Exiting.")
        return
    
    # Step 7: Run object detection for the selected object
    print(f"\n🔍 DETECTING '{selected_object.upper()}' IN IMAGE")
    print("=" * 40)
    
    image_with_boxes, matching_detections = get_object_detection_for_editing(generated_image_path, selected_object)
    
    if not matching_detections:
        print(f"⚠️ YOLO could not detect '{selected_object}' in the generated image.")
        print("🔄 Trying intelligent region suggestions...")
        
        # Try the fallback for undetectable objects
        image_with_boxes, matching_detections = handle_undetectable_objects(
            generated_image_path, selected_object, prompt_buffer
        )
        
        if not matching_detections:
            print(f"❌ Could not provide region suggestions for '{selected_object}'.")
            print("💡 This might be because:")
            print("   - The object is not clearly visible in the image")
            print("   - The object is abstract or not a physical entity")
            print("   - The generated image doesn't contain this object")
            print("💡 Try using the manual editing mode by running the script with option 2.")
            return
        else:
            print(f"📋 Note: Using intelligent region suggestions for '{selected_object}'")
            print("💡 Orange boxes show estimated regions where this object might be located.")
    
    # Save image with detected objects for user reference
    boxes_image_path = os.path.join(script_dir, f"detected_{selected_object}_{uuid.uuid4().hex[:8]}.png")
    if image_with_boxes is not None:
        cv2.imwrite(boxes_image_path, image_with_boxes)
        print(f"📸 Image with detected {selected_object} saved to: {boxes_image_path}")
    
    # Step 8: If multiple instances found, let user choose
    selected_detection = None
    if len(matching_detections) == 1:
        selected_detection = matching_detections[0]
        print(f"✅ Found 1 instance of '{selected_object}'. Proceeding with editing.")
    else:
        print(f"\n🎯 MULTIPLE INSTANCES FOUND")
        print("=" * 30)
        for detection in matching_detections:
            print(f"ID {detection['id']}: {detection['class_name']} (confidence: {detection['confidence']:.2f})")
        
        while True:
            try:
                choice = int(input(f"\nEnter the ID of the {selected_object} you want to edit (0-{len(matching_detections)-1}): "))
                if 0 <= choice < len(matching_detections):
                    selected_detection = matching_detections[choice]
                    break
                else:
                    print(f"❌ Please enter a number between 0 and {len(matching_detections)-1}")
            except ValueError:
                print("❌ Please enter a valid number")
    
    # Step 9: Create mask and perform inpainting
    print(f"\n🎨 EDITING {selected_object.upper()}")
    print("=" * 30)
    
    # Load original image to get shape
    original_image = cv2.imread(generated_image_path)
    mask = create_mask_from_bbox(original_image.shape, selected_detection['bbox'])
    
    # Generate output filename for edited image
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    edited_filename = f"edited_{creative_filename}_{timestamp}_{uuid.uuid4().hex[:8]}.png"
    edited_image_path = os.path.join(script_dir, edited_filename)
    
    print(f"💡 Saving edited image as: {edited_filename}")
    
    # Perform the inpainting
    success = edit_image_with_inpainting(generated_image_path, mask, edit_description, edited_image_path)
    
    if success:
        print(f"\n🎉 IMAGE EDITING COMPLETED!")
        print("=" * 30)
        print(f"📁 Original generated image: {generated_image_path}")
        print(f"📁 Edited image: {edited_image_path}")
        print(f"📁 Detection reference: {boxes_image_path}")
        print("✅ Task complete!")
    else:
        print("❌ Image editing failed. Please try again.")
        print("💡 If you're getting AI model errors, try running: python fix_huggingface.py")
    
    # Clean up temporary detection image
    try:
        os.remove(boxes_image_path)
    except:
        pass

def extract_objects_from_text(text: str) -> list:
    """
    Enhanced object extraction from text using multiple techniques.
    
    Args:
        text: The input text to analyze
    
    Returns:
        list: List of objects found in the text
    """
    text_lower = text.lower()
    found_objects = set()  # Use set to avoid duplicates
    
    # Comprehensive object dictionary with categories
    object_categories = {
        # Animals and creatures
        'animals': [
            'whale', 'dolphin', 'shark', 'fish', 'salmon', 'trout',
            'bird', 'pelican', 'eagle', 'crow', 'pigeon', 'seagull', 'hawk',
            'dog', 'cat', 'horse', 'cow', 'sheep', 'pig', 'goat',
            'lion', 'tiger', 'elephant', 'bear', 'wolf', 'fox',
            'rabbit', 'deer', 'squirrel', 'mouse', 'butterfly', 'bee'
        ],
        
        # People
        'people': [
            'person', 'people', 'man', 'woman', 'child', 'baby', 'boy', 'girl',
            'human', 'individual', 'figure', 'character', 'student', 'teacher'
        ],
        
        # Natural elements
        'nature': [
            'tree', 'forest', 'flower', 'plant', 'grass', 'leaf', 'branch',
            'mountain', 'hill', 'rock', 'stone', 'cliff',
            'river', 'stream', 'lake', 'pond', 'ocean', 'sea', 'water',
            'beach', 'shore', 'sand', 'meadow', 'field', 'garden',
            'sun', 'moon', 'star', 'cloud', 'sky', 'rainbow'
        ],
        
        # Vehicles
        'vehicles': [
            'car', 'truck', 'bus', 'van', 'motorcycle', 'bicycle', 'bike',
            'boat', 'ship', 'plane', 'airplane', 'helicopter', 'train'
        ],
        
        # Buildings and structures
        'structures': [
            'house', 'building', 'tower', 'bridge', 'road', 'path', 'street',
            'church', 'castle', 'barn', 'shed', 'fence', 'gate', 'door', 'window'
        ],
        
        # Objects and items
        'objects': [
            'table', 'chair', 'sofa', 'bed', 'desk', 'bench',
            'cup', 'glass', 'bottle', 'plate', 'bowl', 'spoon', 'fork', 'knife',
            'ball', 'toy', 'book', 'computer', 'phone', 'camera',
            'lamp', 'clock', 'mirror', 'painting', 'picture', 'vase',
            'hat', 'bag', 'umbrella', 'bicycle', 'sword', 'shield'
        ],
        
        # Weather and natural phenomena
        'weather': [
            'rain', 'snow', 'wind', 'storm', 'lightning', 'thunder',
            'mist', 'fog', 'ice', 'fire', 'flame', 'smoke'
        ]
    }
    
    # Method 1: Direct word matching with word boundaries
    all_objects = []
    for category, objects in object_categories.items():
        all_objects.extend(objects)
      # Sort by length (longest first) to catch compound words first
    all_objects.sort(key=len, reverse=True)
    
    import re  # Import here to ensure it's available
    
    for obj in all_objects:
        # Use word boundaries to avoid partial matches
        import re
        pattern = r'\b' + re.escape(obj) + r'\b'
        if re.search(pattern, text_lower):
            found_objects.add(obj)
    
    # Method 2: Handle plural forms and variations
    plural_mappings = {
        'trees': 'tree', 'flowers': 'flower', 'plants': 'plant',
        'birds': 'bird', 'whales': 'whale', 'fish': 'fish',
        'people': 'person', 'children': 'child', 'babies': 'baby',
        'cars': 'car', 'trucks': 'truck', 'boats': 'boat',
        'houses': 'house', 'buildings': 'building', 'roads': 'road',
        'clouds': 'cloud', 'mountains': 'mountain', 'rivers': 'river',
        'tables': 'table', 'chairs': 'chair', 'books': 'book',
        'bottles': 'bottle', 'cups': 'cup', 'balls': 'ball'
    }
    
    for plural, singular in plural_mappings.items():
        pattern = r'\b' + re.escape(plural) + r'\b'
        if re.search(pattern, text_lower):
            found_objects.add(singular)
    
    # Method 3: Handle common compound descriptions
    compound_mappings = {
        'blue whale': 'whale',
        'red car': 'car',
        'green tree': 'tree',
        'flying bird': 'bird',
        'jumping fish': 'fish',
        'running horse': 'horse',
        'bright sun': 'sun',
        'blue sky': 'sky',
        'flowing river': 'river',
        'tall building': 'building',
        'old tree': 'tree'
    }
    
    for compound, base_obj in compound_mappings.items():
        if compound in text_lower:
            found_objects.add(base_obj)
    
    # Method 4: Context-based detection
    context_patterns = {
        r'swimming|diving|jumping.*water': 'fish',
        r'flying|soaring|perched': 'bird',
        r'driving|parking|racing': 'car',
        r'flowing|rushing|meandering': 'river',
        r'growing|blooming|planted': 'plant',
        r'shining|bright.*sky': 'sun'
    }
    
    for pattern, obj in context_patterns.items():
        if re.search(pattern, text_lower):
            found_objects.add(obj)
    
    # Convert to sorted list and filter out very generic terms if we have specific ones
    result = list(found_objects)
    
    # Remove generic terms if we have more specific ones
    if 'animal' in result and any(animal in result for animal in object_categories['animals']):
        result.remove('animal')
    if 'vehicle' in result and any(vehicle in result for vehicle in object_categories['vehicles']):
        result.remove('vehicle')
    
    result.sort()  # Sort alphabetically for consistency
    
    print(f"✅ Enhanced extraction found objects: {result}")
    return result

# ---------------------
if __name__ == "__main__":
    print("🤖 AI Robotic Artist")
    print("=" * 40)
    print("Choose an option:")
    print("1. Generate image and optionally edit it (New Workflow)")
    print("2. Edit an existing image (Manual Mode)")
    
    while True:
        try:
            choice = input("\nEnter your choice (1 or 2): ").strip()
            if choice in ['1', '2']:
                break
            else:
                print("❌ Please enter either 1 or 2")
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            exit()
    
    try:
        if choice == '1':
            main_workflow()
        elif choice == '2':
            edit_existing_image()
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
    except Exception as e:
        print(f"\n❌ An unexpected error occurred: {e}")
        print("💡 Please try again or check your configuration.")