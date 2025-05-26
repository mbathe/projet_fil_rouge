import time
import numpy as np
import matplotlib.pyplot as plt
from DepthAnythingV2.metric_depth.depth_anything_v2.dpt import DepthAnythingV2
from PIL import Image
import torch
import cv2
import sys
import os
import shutil  # Added import for directory operations
from dotenv import load_dotenv
from pathlib import Path

load_dotenv()
# Peut être 'small', 'base' ou 'large'
DEPTH_ANYTHING_TYPE = os.getenv("DEPTH_ANYTHING_TYPE", "base")

sys.path.append(os.path.abspath("src"))
checkpoints_dir = Path(__file__).parent.resolve() / \
    "DepthAnythingV2" / "checkpoints"
model_file = checkpoints_dir / "depth_anything_v2_vitb.pth" if DEPTH_ANYTHING_TYPE == "base" else (
    checkpoints_dir / "depth_anything_v2_vits.pth" if DEPTH_ANYTHING_TYPE == "small" else checkpoints_dir /
    "depth_anything_v2_vitl.pth"
)


# Load model once globally for efficiency
model = DepthAnythingV2(encoder='vits', features=64,
                        out_channels=[48, 96, 192, 384])
model.load_state_dict(torch.load(model_file,
                      map_location='cpu',  weights_only=True))
model.eval()
device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)
model.to(device)


def generate_depth_maps(image_folder, output_folder, image_extensions=(".png", ".jpg", ".jpeg")):
    """
    Generate depth maps for all images in the specified folder and save them as TIFF files.
    
    Args:
        image_folder (str): Path to folder containing input images
        output_folder (str): Path to save output depth maps
        image_extensions (tuple): File extensions to process (default: PNG, JPG, JPEG)
        
    Returns:
        dict: Performance statistics including number of images processed, total time, and average time
    """
    # Clear output directory if it exists, then create it
    if os.path.exists(output_folder):
        print(f"Clearing existing output directory: {output_folder}")
        # Remove all files in the directory
        for item in os.listdir(output_folder):
            item_path = os.path.join(output_folder, item)
            if os.path.isfile(item_path):
                os.remove(item_path)
            elif os.path.isdir(item_path):
                shutil.rmtree(item_path)
        print(f"Output directory cleared")
    else:
        # Create output directory if it doesn't exist
        os.makedirs(output_folder)
        print(f"Created output directory: {output_folder}")
    
    # Get all image files with specified extensions
    image_files = []
    for extension in image_extensions:
        image_files.extend([
            os.path.join(image_folder, f) 
            for f in os.listdir(image_folder) 
            if f.lower().endswith(extension)
        ])
    image_files = sorted(image_files)
    
    start_time = time.time()
    
    # Process each image to generate depth maps
    for idx, image_path in enumerate(image_files):
        # Load image
        raw_img = cv2.imread(image_path)
        raw_img = cv2.cvtColor(raw_img, cv2.COLOR_BGR2RGB)  # Convert BGR → RGB
        
        # Convert to tensor
        image_tensor = model.image2tensor(raw_img, input_size=518)[0]
        image_tensor = image_tensor.to(device)
        
        # Inference
        with torch.no_grad():
            depth = model.forward(image_tensor)
        
        heatmap = depth.squeeze().cpu().numpy()
        # Resize to original dimensions
        heatmap_resized = cv2.resize(heatmap, (raw_img.shape[1], raw_img.shape[0]))
        
        # Save output
        original_filename = os.path.splitext(os.path.basename(image_path))[0]
        output_path = os.path.join(output_folder, f"{original_filename}.tiff")
        cv2.imwrite(output_path, heatmap_resized)
        
        if (idx + 1) % 10 == 0 or idx == len(image_files) - 1:
            print(f"Processed {idx + 1}/{len(image_files)} images")
    
    end_time = time.time()
    
    # Performance calculations
    total_time = end_time - start_time
    num_images = len(image_files)
    average_time_per_image = total_time / num_images if num_images > 0 else 0
    
    # Create results dictionary
    results = {
        "num_images": num_images,
        "total_time": total_time,
        "average_time_per_image": average_time_per_image
    }
    
    # Display results
    print("Processing completed")
    print(f"Total images processed: {num_images}")
    print(f"Total execution time: {total_time:.2f} seconds")
    print(f"Average time per image: {average_time_per_image:.4f} seconds")
    
    return results


# Example usage
if __name__ == "__main__":
    # These can be replaced with command line arguments if needed
    input_folder = "../deer_walk/cam0/data/"
    output_folder = "../dear_walk_DAV2_metrice/"
    
    generate_depth_maps(input_folder, output_folder)