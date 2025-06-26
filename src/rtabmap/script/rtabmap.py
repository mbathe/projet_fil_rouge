import os
import glob
import pandas as pd
import numpy as np
import subprocess
import shutil
import sys
from .tols import load_config, config_to_args, get_os_version
from dotenv import load_dotenv
import os
import re
from tqdm import tqdm
from pathlib import Path

# Charge les variables depuis le fichier .env
load_dotenv()

# Default paths - configurable via parameters
script_dir = os.path.dirname(os.path.abspath(__file__))
RTABMAB_DOCKER_ROOT = Path(script_dir)/"rtabmap_ws"
SHOW_PROGRESS = False
RGB_PATH = f"{RTABMAB_DOCKER_ROOT}/rgb_sync"
DEPTH_PATH = f"{RTABMAB_DOCKER_ROOT}/depth_sync"

RGB_PATH_FROM = f"{RTABMAB_DOCKER_ROOT}/rgb_sync_docker"
DEPTH_PATH_FROM = f"{RTABMAB_DOCKER_ROOT}/depth_sync_docker"


LOG_DIR = Path(RTABMAB_DOCKER_ROOT)/"logs"
EXPORT_PARAMS_FILES = f"{RTABMAB_DOCKER_ROOT}/export_params.json"
GENERATE_DB_PARAMS_FILES = f"{RTABMAB_DOCKER_ROOT}/db_params.json"
REPROCESS_PARAMS_FILES = f"{RTABMAB_DOCKER_ROOT}/reprocess_params.json"


calib_path = RTABMAB_DOCKER_ROOT/"rtabmap_calib.yaml"

OS= get_os_version()[0]

RUN_TO_DOCKER  = not OS != "Ubuntu"
if  RUN_TO_DOCKER:
    LOG_DIR = Path(__file__).resolve().parent.parent.parent.parent / "logs"
    print(LOG_DIR)





   


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


def prepare_dataset(rgb_dir, depth_dir, root_dir =RTABMAB_DOCKER_ROOT):
    """
    Prepare the dataset by copying files to target directories.
    
    Args:
        rgb_dir: Source directory for RGB images
        depth_dir: Source directory for depth images
        calib_file: Calibration file path
        
    """
    
    rgb_sync_dir = root_dir /"rgb_sync"
    depth_sync_dir =root_dir /"depth_sync"
   

    

    # Create target directories if they don't exist
    for sync_dir in [rgb_sync_dir, depth_sync_dir]:
        os.makedirs(sync_dir, exist_ok=True)
        print(f"[INFO] Directory ready: {sync_dir}")

    # Collecte de tous les fichiers RGB à copier
    rgb_files = []
    for ext in ['*.png', '*.jpg', '*.jpeg', '*.tiff', '*.tif']:
        rgb_files.extend(glob.glob(os.path.join(rgb_dir, ext)))

    # Copie des fichiers RGB avec barre de progression
    if rgb_files:
        print(f"[INFO] Copie de {len(rgb_files)} images RGB...")
        for src in tqdm(rgb_files, desc="Copie RGB", unit="img"):
            filename = os.path.basename(src)
            dst = os.path.join(rgb_sync_dir, filename)
            shutil.copy2(src, dst)
        print(f"[OK] {len(rgb_files)} images RGB copiées avec succès")
    else:
        print(f"[WARN] Aucune image trouvée dans le répertoire RGB: {rgb_dir}")

    # Collecte de tous les fichiers de profondeur à copier
    depth_files = []
    for ext in ['*.png', '*.jpg', '*.jpeg', '*.tiff', '*.tif']:
        depth_files.extend(glob.glob(os.path.join(depth_dir, ext)))

    if depth_files:
        print(f"[INFO] Copie de {len(depth_files)} images de profondeur...")
        for src in tqdm(depth_files, desc="Copie Depth", unit="img"):
            filename = os.path.basename(src)
            dst = os.path.join(depth_sync_dir, filename)
            shutil.copy2(src, dst)
        print(
            f"[OK] {len(depth_files)} images de profondeur copiées avec succès")
    else:
        print(
            f"[WARN] Aucune image trouvée dans le répertoire de profondeur: {depth_dir}")


def execute_command(config_file, start_command=[], end_command=[], show_progress=False, sud_dir_log="generate_db"):
    """Exécute la commande avec les paramètres de configuration"""
    # Charger la configuration
    config = load_config(config_file)
    
    config_args = config_to_args(config)
    
    full_command = start_command + config_args + end_command

    # Créer le fichier de log avec la date courante
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    LOG_SUB_PATH_DIR = LOG_DIR / f"{sud_dir_log}"
    LOG_SUB_PATH_DIR.mkdir(parents=True, exist_ok=True)
    log_file = LOG_DIR / f"{sud_dir_log}" / f"{sud_dir_log}_{timestamp}.log"

    if not show_progress:
        # Méthode standard d'exécution avec redirection vers le log ET console
        try:
            with open(log_file, 'w') as f:
                process = subprocess.Popen(
                    full_command,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                    universal_newlines=True
                )

                for line in process.stdout:
                    # Écrire dans le fichier de log
                    f.write(line)
                    f.flush()
                    # Afficher aussi dans la console
                    print(line, end='')

                process.wait()

            if process.returncode != 0:
                raise subprocess.CalledProcessError(
                    process.returncode, full_command)

            print(f"\nSuccès! Log sauvegardé dans: {log_file}")
        except subprocess.CalledProcessError as e:
            print(f"Erreur: {e}")
            # Afficher les dernières lignes du log en cas d'erreur
            try:
                with open(log_file, 'r') as f:
                    last_lines = f.readlines()[-10:]
                    print("Dernières lignes du log:")
                    for line in last_lines:
                        print(line.strip())
            except:
                pass
    else:
        try:
            # Démarrer le processus avec redirection complète vers le fichier de log
            process = subprocess.Popen(
                full_command,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                universal_newlines=True
            )

            pbar = tqdm(desc="Initialisation...", unit="iter", ncols=100)

            iter_pattern = re.compile(r'(?:Iteration|Processed) (\d+)/(\d+)')

            total_iters = None
            last_iter = 0

            # Lire et enregistrer la sortie ligne par ligne
            with open(log_file, 'w') as f:
                for line in process.stdout:
                    # Enregistrer la ligne dans le fichier de log
                    f.write(line)
                    f.flush()

                    # Afficher aussi dans la console (sous la barre de progression)
                    tqdm.write(line.rstrip())

                    # Analyser la ligne pour la barre de progression
                    match = iter_pattern.search(line)
                    if match:
                        current_iter = int(match.group(1))

                        if total_iters is None:
                            total_iters = int(match.group(2))
                            pbar.reset(total=total_iters)
                            pbar.set_description(f"Traitement RTAB-Map")

                        # Extraire les informations supplémentaires de la ligne
                        info_match = re.search(
                            r'(?:Iteration|Processed) \d+/\d+[:\s]*(.+)', line.strip())
                        if info_match:
                            last_info = info_match.group(1).strip()
                            if last_info:
                                pbar.set_description(f"RTAB-Map [{last_info}]")

                        if current_iter > last_iter:
                            pbar.update(current_iter - last_iter)
                            last_iter = current_iter

            # Attendre la fin du processus
            process.wait()
            pbar.close()

            if process.returncode != 0:
                print(
                    f"La commande a échoué avec le code de retour {process.returncode}")
                try:
                    with open(log_file, 'r') as f:
                        last_lines = f.readlines()[-20:]
                        print("Dernières lignes de la sortie:")
                        for line in last_lines:
                            print(line.strip())
                except:
                    pass
            else:
                print(
                    f"Traitement RTAB-Map terminé avec succès! Log sauvegardé dans: {log_file}")

        except Exception as e:
            print(f"Erreur lors de l'exécution: {e}")

        # Note: Le fichier de log n'est plus supprimé pour conserver l'historique


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
    print("\n===== Génération de la base de données RTAB-Map =====")
    execute_command(GENERATE_DB_PARAMS_FILES,
                    start_command=["rtabmap-rgbd_dataset",
                                   "--calib ", f"{RTABMAB_DOCKER_ROOT}/rtabmap_calib.yaml"],
                    end_command=["--output_path", f"{RTABMAB_DOCKER_ROOT}"],
                    show_progress=SHOW_PROGRESS)
    
   

    
    


def reprocess():
    """
    Preprocess the RTAB-Map database using the specified parameters.
    Args:
        db_path: Path to the RTAB-Map database
    """
    print("\n===== Retraitement de la base de données RTAB-Map =====")
    execute_command(REPROCESS_PARAMS_FILES,
                    start_command=["rtabmap-reprocess"],
                    end_command=[{RTABMAB_DOCKER_ROOT}/"rtabmap.db",
                                 "output_optimized.db"],
                    show_progress=SHOW_PROGRESS, sud_dir_log="reprocess")


def export_point_cloud(output_type="--cloud"):
    """
    Export the point cloud from the RTAB-Map database.
    Args:
        db_path: Path to the RTAB-Map database
        output_type: Type of output (e.g., mesh, cloud)
    """
    print("\n===== Exportation du nuage de points =====")
    
    print("TYPE DE SORTIE:", output_type)
    start_command = ["rtabmap-export"]
    
    execute_command(EXPORT_PARAMS_FILES,
                    start_command=start_command,
                    end_command=[f"{output_type}",  "--output", "point",
                                 RTABMAB_DOCKER_ROOT/"rtabmap.db"],
                    show_progress=SHOW_PROGRESS, sud_dir_log="export")






def generate_3db_map(config):

   

    """Run RTAB-Map processing on the dataset."""
    
    output_type = "--mesh" # Exporter un mesh vous pouvez changer en "--cloud" pour exporter un nuage de points
    generate_db()
    if config.get("reprocess", True):
        # reprocess()
        print("test...")
    export_point_cloud(output_type=config.get("export_format", output_type))

   






def main(output_path=Path("/rtabmap_ws/output/rtabmap")):
    config = load_config(f"{RTABMAB_DOCKER_ROOT}/config.json")
    extension = "cloud" if config.get("export_format", "--cloud")=="--cloud" else "mesh"

    prepare_dataset(RGB_PATH_FROM, DEPTH_PATH_FROM)

    
    generate_3db_map(config)
    file_name = f"point_{extension}.ply"
    print("\n===== Copying Results =====")
    os.makedirs(output_path, exist_ok=True)
    shutil.copy2(RTABMAB_DOCKER_ROOT/"rtabmap.db",output_path/"rtabmap.db")
    shutil.copy2(RTABMAB_DOCKER_ROOT/file_name,output_path/file_name)
    print(f"Processing complete! Results saved to {output_path}")







if __name__ == "__main__":
    main()
    


