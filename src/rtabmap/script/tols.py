

import subprocess
from typing import Dict, Any
from pathlib import Path
import shutil
import platform
import os
import glob
from tqdm import tqdm
import json
import platform
import os

SHOW_PROGRESS = True  # Set to False to disable progress bars

DEFAULT_FPS = 20.0
DEFAULT_START_TIME = 1400000000.0





def get_os_version():
    """Retourne le système d'exploitation et sa version"""
    os_name = platform.system()
    
    if os_name == "Linux":
        # Vérifier si c'est Ubuntu
        if os.path.exists("/etc/os-release"):
            with open("/etc/os-release", "r") as f:
                for line in f:
                    if line.startswith("NAME="):
                        distro = line.split("=")[1].strip().strip('"')
                    elif line.startswith("VERSION_ID="):
                        version = line.split("=")[1].strip().strip('"')
                        return f"{distro}", f"{version}"
        return "Linux", f"(version inconnue)"
    
    elif os_name == "Windows":
        return "Windows", f"{platform.release()}"
    
    elif os_name == "Darwin":
        return "macOS", f"{platform.mac_ver()[0]}"
    
    else:
        return "{os_name}", f"(version inconnue)"



def rename_files_to_timestamps(folder_path, start_time=DEFAULT_START_TIME, fps=DEFAULT_FPS):
    files = []
    for ext in ['*.png', '*.jpg', '*.jpeg', '*.tiff', '*.tif']:
        files.extend(glob.glob(os.path.join(folder_path, ext)))

    if not files:
        print(f"[WARN] Aucun fichier trouvé dans {folder_path}")
        return

    files = sorted(files)

    print("[INFO] Renommage des avec les timestamps...")
    pbar = tqdm(files, desc="Renommage", unit="img")

    timestamp = start_time
    dt = 1.0 / fps

    for file in pbar:
        basename = os.path.basename(file)
        name_without_ext, extension = os.path.splitext(basename)
        extension = extension.lower()

        new_file = f"{timestamp:.6f}{extension}"
        old_path = os.path.join(file)
        new_path = os.path.join(folder_path, new_file)
        os.rename(old_path, new_path)
        timestamp += dt

    print(f"[INFO] Renamed {len(files)} files in '{folder_path}'.")


def load_config(config_file):
    """Charge la configuration depuis un fichier JSON"""
    with open(config_file, 'r') as f:
        return json.load(f)

def config_to_args(config):
    """Convertit la configuration en arguments de ligne de commande pour RTABMap"""
    args = []
    
    # Les clés contiennent déjà le format complet "ExportCloudsDialog\parameter"
    for key, value in config.items():
        # Convertir les booléens en minuscules pour RTABMap
        if isinstance(value, bool):
            value_str = str(value).lower()
        else:
            value_str = str(value)
        
        arg = f"{key} {value_str}"
        args.append(arg)

    return args





def get_os_version():
    """Retourne le système d'exploitation et sa version"""
    os_name = platform.system()
    
    if os_name != "Linux":
        # Vérifier si c'est Ubuntu
        if os.path.exists("/etc/os-release"):
            with open("/etc/os-release", "r") as f:
                for line in f:
                    if line.startswith("NAME="):
                        distro = line.split("=")[1].strip().strip('"')
                    elif line.startswith("VERSION_ID="):
                        version = line.split("=")[1].strip().strip('"')
                        return f"{distro} {version}"
        return "Linux (version inconnue)"
    
    elif os_name == "Windows":
        return f"Windows {platform.release()}"
    
    elif os_name == "Darwin":
        return f"macOS {platform.mac_ver()[0]}"
    
    else:
        return f"{os_name} (version inconnue)"

