import os
import glob
import pandas as pd
import numpy as np
import subprocess
import shutil
import sys
from tols import load_config, config_to_args
from dotenv import load_dotenv
import os

# Charge les variables depuis le fichier .env
load_dotenv()

# Default paths - configurable via parameters



RTABMAB_DOCKER_ROOT =os.getenv("RTABMAB_DOCKER_ROOT")
RGB_PATH = f"{RTABMAB_DOCKER_ROOT}/rgb_sync"
DEPTH_PATH = f"{RTABMAB_DOCKER_ROOT}/depth_sync"
IMG_TIMESTAMPS = "img_timestamps.csv"
DEPTH_TIMESTAMPS = "depth_timestamps.csv"
EXPORT_PARAMS_FILES = f"{RTABMAB_DOCKER_ROOT}/rtabmap_params/export_params.json"
GENERATE_DB_PARAMS_FILES = f"{RTABMAB_DOCKER_ROOT}/rtabmap_params/generate_db_params.json"
REPROCESS_PARAMS_FILES = f"{RTABMAB_DOCKER_ROOT}/rtabmap_params/reprocess_params.json"


def convert_to_timestamps(img_path=None, depth_path=None, img_timestamps_path=None, depth_timestamps_path=None):
    """
    Convert image filenames to timestamp-based names while preserving original extensions.
    Handles PNG, JPG, and TIFF formats.
    
    Args:
        img_path: Path to RGB images
        depth_path: Path to depth images
        img_timestamps_path: Path to RGB timestamps CSV file
        depth_timestamps_path: Path to depth timestamps CSV file
    """
    def process_files(directory, timestamps_df, label="image"):
        used_timestamps = set()
        
        # Get all supported image files in the directory
        files = []
        for ext in ['*.png', '*.jpg', '*.jpeg', '*.tiff', '*.tif']:
            files.extend(glob.glob(os.path.join(directory, ext)))
        
        if not files:
            print(f"[WARN] No {label} files found in {directory}")
            return
            
        for file in files:
            basename = os.path.basename(file)
            name_without_ext, extension = os.path.splitext(basename)
            extension = extension.lower()  # Normalize extension
            
            # Search for timestamp in DataFrame - match by filename with or without extension
            match = timestamps_df[timestamps_df["filename"] == basename]
            if match.empty:
                # Try matching without extension
                match = timestamps_df[timestamps_df["filename"] == name_without_ext]
                if match.empty:
                    print(f"[WARN] {label} - No timestamp found for file {basename}, skipping.")
                    continue
            
            timestamp = match["timestamp"].values[0]
            
            # Avoid zero or duplicate timestamps
            while timestamp == 0 or timestamp in used_timestamps:
                timestamp += np.random.uniform(0.001, 0.005)
                print(f"[INFO] {label} - Adjusted duplicate/zero timestamp for {basename}: {timestamp:.6f}")
            
            used_timestamps.add(timestamp)
            
            new_name = os.path.join(os.path.dirname(file), f"{timestamp:.6f}{extension}")
            os.rename(file, new_name)
            print(f"[OK] {label} - Renamed {basename} → {os.path.basename(new_name)}")

    # Process RGB images
    if img_path and img_timestamps_path:
        img_timestamps_df = pd.read_csv(img_timestamps_path)
        process_files(img_path, img_timestamps_df, label="RGB")
    
    if depth_path and depth_timestamps_path:
        depth_timestamps_df = pd.read_csv(depth_timestamps_path)
        process_files(depth_path, depth_timestamps_df, label="Depth")


def validate_csv_columns(csv_path):
    """
    Verify that the CSV file contains the required columns.
    
    Args:
        csv_path: Path to CSV file
    
    Returns:
        bool: True if valid, False otherwise
    """
    try:
        df = pd.read_csv(csv_path, nrows=0)
        if "timestamp" not in df.columns or "filename" not in df.columns:
            print(f"[ERROR] File '{csv_path}' must contain 'timestamp' and 'filename' columns.")
            return False
        return True
    except Exception as e:
        print(f"[ERROR] Unable to read CSV file: {csv_path}\n{e}")
        return False


def prepare_dataset(rgb_dir, depth_dir, calib_file, rgb_timestamps, depth_timestamps):
    """
    Prepare the dataset by copying files to target directories.
    
    Args:
        rgb_dir: Source directory for RGB images
        depth_dir: Source directory for depth images
        calib_file: Calibration file path
        rgb_timestamps: RGB timestamps CSV file path
        depth_timestamps: Depth timestamps CSV file path
    """
    root_dir = os.getcwd()
    rgb_sync_dir = os.path.join(root_dir, "/rtabmap_ws/rgb_sync")
    depth_sync_dir = os.path.join(root_dir, "/rtabmap_ws/depth_sync")
    calib_target_path = os.path.join(root_dir, "/rtabmap_ws/rtabmap_calib.yaml")
    rgb_timestamps_target_path = os.path.join(root_dir, "/rtabmap_ws/img_timestamps.csv")
    depth_timestamps_target_path = os.path.join(root_dir, "/rtabmap_ws/depth_timestamps.csv")

    # Verify CSV files
    if not validate_csv_columns(rgb_timestamps) or not validate_csv_columns(depth_timestamps):
        return

    # Create target directories if they don't exist
    for sync_dir in [rgb_sync_dir, depth_sync_dir]:
        os.makedirs(sync_dir, exist_ok=True)
        print(f"[INFO] Directory ready: {sync_dir}")

    # Copy RGB files (all supported formats)
    copied_rgb_count = 0
    for ext in ['*.png', '*.jpg', '*.jpeg', '*.tiff', '*.tif']:
        for src in glob.glob(os.path.join(rgb_dir, ext)):
            filename = os.path.basename(src)
            dst = os.path.join(rgb_sync_dir, filename)
            shutil.copy2(src, dst)
            copied_rgb_count += 1
            print(f"[OK] Copied RGB: {filename}")
    
    if copied_rgb_count == 0:
        print(f"[WARN] No supported image files found in RGB directory: {rgb_dir}")

    # Copy depth files (all supported formats)
    copied_depth_count = 0
    
    for ext in ['*.png', '*.jpg', '*.jpeg', '*.tiff', '*.tif']:
        for src in glob.glob(os.path.join(depth_dir, ext)):
            filename = os.path.basename(src)
            dst = os.path.join(depth_sync_dir, filename)
            shutil.copy2(src, dst)
            copied_depth_count += 1
            print(f"[OK] Copied Depth: {filename}")
    
    
    if copied_depth_count == 0:
        print(f"[WARN] No supported image files found in depth directory: {depth_dir}")






def execute_command(config_file, start_command=[], end_command=[]):
    """Exécute la commande avec les paramètres de configuration"""
    # Charger la configuration
    config = load_config(config_file)
    
    config_args = config_to_args(config)
    
    full_command = start_command + config_args +end_command
    
    print("Commande exécutée:")
    print(" ".join(full_command))
    
    try:
        result = subprocess.run(
        full_command,
        check=True,
        stdout=sys.stdout,
        stderr=sys.stderr,
        text=True)
        print("Succès!")
        print(result.stdout)
    except subprocess.CalledProcessError as e:
        print(f"Erreur: {e}")
        print(f"Sortie d'erreur: {e.stderr}")








def generate_db():
    """
    Generate the RTAB-Map database using the specified parameters.
    
    Args:
        db_path: Path to the RTAB-Map database
        rgb_path: Path to RGB images
        depth_path: Path to depth images
        calib_file: Calibration file path
    """
    # Command to generate the database
         
    execute_command(GENERATE_DB_PARAMS_FILES, start_command=["rtabmap-rgbd_dataset"], end_command=["--output_path", "/rtabmap_ws"])
    
   

    
    


def reprocess():
    """
    Preprocess the RTAB-Map database using the specified parameters.
    Args:
        db_path: Path to the RTAB-Map database
    """
    execute_command(REPROCESS_PARAMS_FILES, start_command=["rtabmap-reprocess"], end_command=["/rtabmap_ws/rtabmap.db", "output_optimized.db",])




def export_point_cloud():
    """
    Export the point cloud from the RTAB-Map database.
    Args:
        db_path: Path to the RTAB-Map database
        output_type: Type of output (e.g., mesh, cloud)
    """
    output_type = "--cloud"  # Change to "--cloud" for point cloud export
    start_command = [
        "rtabmap-export"
    ]
    
    execute_command(EXPORT_PARAMS_FILES, start_command=start_command, end_command=[f"{output_type}","--output", "point", "/rtabmap_ws/output_optimized.db"])
    


def main():
    """Run RTAB-Map processing on the dataset."""
    
    output_type = "--mesh" # Exporter un mesh vous pouvez changer en "--cloud" pour exporter un nuage de points
    generate_db()
    reprocess()
    export_point_cloud()

   

if __name__ == "__main__":
    if len(sys.argv) != 1:
        print("Usage: python script.py")
        print("All paths are currently hardcoded in the script.")
        sys.exit(1)
    else:
        rgb_path = "/rtabmap_ws/rgb_sync"
        depth_path = "/rtabmap_ws/depth_sync"
        
        rgb_path_from = "/rtabmap_ws/rgb_sync_docker"
        depth_path_from = "/rtabmap_ws/depth_sync_docker"
        
        calib_path = "rtabmap_calib.yaml"
        rgb_timestamps = "img_timestamps.csv"
        depth_timestamps = "depth_timestamps.csv"
        
        print("===== Preparing Dataset =====")
        prepare_dataset(rgb_path_from, depth_path_from, calib_path, rgb_timestamps, depth_timestamps)
        
        print("\n===== Converting to Timestamps =====")
        
        
        convert_to_timestamps(
            img_path=rgb_path, 
            depth_path=DEPTH_PATH,  
            img_timestamps_path=rgb_timestamps, 
            depth_timestamps_path=depth_timestamps
        )
        
        from pathlib import Path

        chemin = Path("/rtabmap_ws")
        repertoires = [p.name for p in chemin.iterdir() if p.is_dir()]

        print(repertoires)
                
        print("\n===== Processing with RTAB-Map =====")
        main()
        
        print("\n===== Copying Results =====")
        os.makedirs("/rtabmap_ws/output", exist_ok=True)
        shutil.copy2("/rtabmap_ws/output_optimized.db", "/rtabmap_ws/output/rtabmap.db")
        shutil.copy2("/rtabmap_ws/point_cloud.ply", "/rtabmap_ws/output/point_cloud.ply")
        print("Processing complete! Results saved to /rtabmap_ws/output/")


