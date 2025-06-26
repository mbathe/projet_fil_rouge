# fmt: off
# isort: skip_file
import shutil
import numpy as np
import time
import matplotlib.pyplot as plt
import cv2
import torch
from PIL import Image
import os
from .DepthAnythingV2.metric_depth.depth_anything_v2.dpt import DepthAnythingV2
# fmt: on
from dotenv import load_dotenv
from pathlib import Path

load_dotenv()
# Peut être 'small', 'base' ou 'large'
DEPTH_ANYTHING_TYPE = os.getenv("DEPTH_ANYTHING_TYPE", "small")

DEPTH_ANYTHING_BASE_NAME = os.getenv(
    "DEPTH_ANYTHING_BASE_NAME", "depth_anything_v2_metric_hypersim_vit")


checkpoints_dir = Path(__file__).parent.resolve() / \
    "DepthAnythingV2" / "checkpoints"


end_name = "b" if DEPTH_ANYTHING_TYPE == "base" else (
    "s" if DEPTH_ANYTHING_TYPE == "small" else "l")

model_file = checkpoints_dir / f"{DEPTH_ANYTHING_BASE_NAME}{end_name}.pth"


device = 'cuda' if torch.cuda.is_available(
) else 'mps' if torch.backends.mps.is_available() else 'cpu'


model_configs = {
    'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
    'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
    'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
    'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
}

encoder = "vitb" if DEPTH_ANYTHING_TYPE == "base" else (
    "vits" if DEPTH_ANYTHING_TYPE == "small" else "vitl"
)

depth_anything = DepthAnythingV2(**model_configs[encoder])
depth_anything.load_state_dict(torch.load(
    f'{model_file}', map_location='cpu', weights_only=True))
model = depth_anything.to(device).eval()


def generate_depth_maps(image_folder, output_folder, image_extensions=(".png", ".jpg", ".jpeg"), output_extension=".tiff"):
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
        heatmap_resized = cv2.resize(
            heatmap, (raw_img.shape[1], raw_img.shape[0]))

        # Save output
        original_filename = os.path.splitext(os.path.basename(image_path))[0]
        output_path = os.path.join(
            output_folder, f"{original_filename}.{output_extension.lstrip('.').lower()}")
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

    # Chemin du dossier à créer

    script_dir = Path(__file__).parent.resolve()

    script_dir = os.path.dirname(os.path.abspath(__file__))

    output_dir = os.path.join(os.path.dirname(
        os.path.dirname(script_dir)), "output")

    data_dir = os.path.join(os.path.dirname(
        os.path.dirname(script_dir)), "data")

    input_dir = os.path.join(os.path.join(os.path.join(
        data_dir, 'dataset'), "deer_walk"), "images")

    output_dir = os.path.join(os.path.join(
        output_dir, 'depth'), "images")

    os.makedirs(output_dir, exist_ok=True)

    print(output_dir)

    generate_depth_maps(input_dir, output_dir)
