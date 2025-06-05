import os
import glob
import pandas as pd
import numpy as np
import subprocess
import shutil
import sys
from .tools import load_config, config_to_args
from dotenv import load_dotenv
import re
from tqdm import tqdm

# Charge les variables depuis le fichier .env
load_dotenv()

SHOW_PROGRESS = True  # Set to False to disable progress bars

# Chemins locaux (plus de Docker)
WORKSPACE_ROOT = os.getcwd()
RGB_PATH = os.path.join(WORKSPACE_ROOT, "rtabmap_ws", "rgb_sync")
DEPTH_PATH = os.path.join(WORKSPACE_ROOT, "rtabmap_ws", "depth_sync")
IMG_TIMESTAMPS = os.path.join(
    WORKSPACE_ROOT, "rtabmap_ws", "img_timestamps.csv")
DEPTH_TIMESTAMPS = os.path.join(
    WORKSPACE_ROOT, "rtabmap_ws", "depth_timestamps.csv")
EXPORT_PARAMS_FILES = os.path.join(
    WORKSPACE_ROOT, "rtabmap_ws", "export_params.json")
GENERATE_DB_PARAMS_FILES = os.path.join(
    WORKSPACE_ROOT, "rtabmap_ws", "db_params.json")
REPROCESS_PARAMS_FILES = os.path.join(
    WORKSPACE_ROOT, "rtabmap_ws", "reprocess_params.json")


def convert_to_timestamps(img_path=None, depth_path=None, img_timestamps_path=None, depth_timestamps_path=None):
    """
    Convert image filenames to timestamp-based names while preserving original extensions.
    Handles PNG, JPG, and TIFF formats.
    """
    def process_files(directory, timestamps_df, label="image"):
        used_timestamps = set()
        adjusted_count = 0

        files = []
        for ext in ['*.png', '*.jpg', '*.jpeg', '*.tiff', '*.tif']:
            files.extend(glob.glob(os.path.join(directory, ext)))

        if not files:
            print(f"[WARN] Aucun fichier {label} trouvé dans {directory}")
            return

        print(
            f"[INFO] Renommage de {len(files)} images {label} avec les timestamps...")
        pbar = tqdm(files, desc=f"Renommage {label}", unit="img")

        for file in pbar:
            basename = os.path.basename(file)
            name_without_ext, extension = os.path.splitext(basename)
            extension = extension.lower()
            match = timestamps_df[timestamps_df["filename"] == basename]
            if match.empty:
                match = timestamps_df[timestamps_df["filename"]
                                      == name_without_ext]
                if match.empty:
                    pbar.set_description(f"❌ Pas de timestamp pour {basename}")
                    continue

            timestamp = match["timestamp"].values[0]
            if timestamp == 0 or timestamp in used_timestamps:
                old_timestamp = timestamp
                timestamp += np.random.uniform(0.001, 0.005)
                adjusted_count += 1
                pbar.set_description(
                    f"⚠️ Ajustement {basename}: {old_timestamp:.6f} → {timestamp:.6f}")
            else:
                pbar.set_description(f"✓ Renommage {basename}")

            used_timestamps.add(timestamp)
            new_name = os.path.join(os.path.dirname(
                file), f"{timestamp:.6f}{extension}")
            os.rename(file, new_name)

        print(
            f"[OK] {len(files)} fichiers {label} renommés ({adjusted_count} timestamps ajustés)")

    if img_path and img_timestamps_path:
        img_timestamps_df = pd.read_csv(img_timestamps_path)
        process_files(img_path, img_timestamps_df, label="RGB")

    if depth_path and depth_timestamps_path:
        depth_timestamps_df = pd.read_csv(depth_timestamps_path)
        process_files(depth_path, depth_timestamps_df, label="Depth")


def validate_csv_columns(csv_path):
    """
    Verify that the CSV file contains the required columns.
    """
    try:
        df = pd.read_csv(csv_path, nrows=0)
        if "timestamp" not in df.columns or "filename" not in df.columns:
            print(
                f"[ERROR] File '{csv_path}' must contain 'timestamp' and 'filename' columns.")
            return False
        return True
    except Exception as e:
        print(f"[ERROR] Unable to read CSV file: {csv_path}\n{e}")
        return False


def prepare_dataset(rgb_dir, depth_dir, calib_file, rgb_timestamps, depth_timestamps):
    """
    Prepare the dataset by copying files to target directories.
    """
    root_dir = os.getcwd()
    rgb_sync_dir = os.path.join(root_dir, "rtabmap_ws", "rgb_sync")
    depth_sync_dir = os.path.join(root_dir, "rtabmap_ws", "depth_sync")
    calib_target_path = os.path.join(
        root_dir, "rtabmap_ws", "rtabmap_calib.yaml")
    rgb_timestamps_target_path = os.path.join(
        root_dir, "rtabmap_ws", "img_timestamps.csv")
    depth_timestamps_target_path = os.path.join(
        root_dir, "rtabmap_ws", "depth_timestamps.csv")

    # Verify CSV files
    if not validate_csv_columns(rgb_timestamps) or not validate_csv_columns(depth_timestamps):
        return

    for sync_dir in [rgb_sync_dir, depth_sync_dir]:
        os.makedirs(sync_dir, exist_ok=True)
        print(f"[INFO] Directory ready: {sync_dir}")

    rgb_files = []
    for ext in ['*.png', '*.jpg', '*.jpeg', '*.tiff', '*.tif']:
        rgb_files.extend(glob.glob(os.path.join(rgb_dir, ext)))

    if rgb_files:
        print(f"[INFO] Copie de {len(rgb_files)} images RGB...")
        for src in tqdm(rgb_files, desc="Copie RGB", unit="img"):
            filename = os.path.basename(src)
            dst = os.path.join(rgb_sync_dir, filename)
            shutil.copy2(src, dst)
        print(f"[OK] {len(rgb_files)} images RGB copiées avec succès")
    else:
        print(f"[WARN] Aucune image trouvée dans le répertoire RGB: {rgb_dir}")

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

    # Copy calibration and timestamp files
    shutil.copy2(calib_file, calib_target_path)
    shutil.copy2(rgb_timestamps, rgb_timestamps_target_path)
    shutil.copy2(depth_timestamps, depth_timestamps_target_path)
    print("[OK] Fichiers de calibration et timestamps copiés.")


def execute_command(config_file, start_command=[], end_command=[], show_progress=False):
    """Exécute la commande avec les paramètres de configuration"""
    config = load_config(config_file)
    config_args = config_to_args(config)
    full_command = start_command + config_args + end_command

    if not show_progress:
        try:
            result = subprocess.run(
                full_command,
                check=True,
                stdout=sys.stdout,
                stderr=sys.stderr,
                text=True)
            print("Succès!")
        except subprocess.CalledProcessError as e:
            print(f"Erreur: {e}")
            print(f"Sortie d'erreur: {e.stderr}")
    else:
        try:
            temp_output_file = os.path.join(
                WORKSPACE_ROOT, "rtabmap_output.txt")
            with open(temp_output_file, 'w') as f:
                process = subprocess.Popen(
                    full_command,
                    stdout=f,
                    stderr=subprocess.STDOUT,
                    text=True
                )

            pbar = tqdm(desc="Initialisation...", unit="iter", ncols=100)
            iter_pattern = re.compile(r'(?:Iteration|Processed) (\d+)/(\d+)')
            total_iters = None
            last_iter = 0

            while process.poll() is None:
                try:
                    with open(temp_output_file, 'r') as f:
                        content = f.read()

                    matches = list(iter_pattern.finditer(content))
                    if matches:
                        latest_match = matches[-1]
                        current_iter = int(latest_match.group(1))

                        if total_iters is None and len(matches) > 0:
                            total_iters = int(latest_match.group(2))
                            pbar.reset(total=total_iters)
                            pbar.set_description(f"Traitement RTAB-Map")

                        last_line = content.splitlines()[-1] if content else ""
                        info_match = re.search(
                            r'Iteration|Processed \d+/\d+: (.+)', last_line)
                        if info_match:
                            last_info = info_match.group(1)
                            pbar.set_description(f"RTAB-Map [{last_info}]")

                        if current_iter > last_iter:
                            pbar.update(current_iter - last_iter)
                            last_iter = current_iter
                except Exception:
                    pass

                import time
                time.sleep(0.1)

            pbar.close()

            if process.returncode != 0:
                print(
                    f"La commande a échoué avec le code de retour {process.returncode}")
                try:
                    with open(temp_output_file, 'r') as f:
                        last_lines = f.readlines()[-20:]
                        print("Dernières lignes de la sortie:")
                        for line in last_lines:
                            print(line.strip())
                except:
                    pass
            else:
                print("Traitement RTAB-Map terminé avec succès!")

            try:
                os.remove(temp_output_file)
            except:
                pass

        except Exception as e:
            print(f"Erreur lors de l'exécution: {e}")


def generate_db():
    """
    Generate the RTAB-Map database using the specified parameters.
    """
    print("\n===== Génération de la base de données RTAB-Map =====")
    execute_command(
        GENERATE_DB_PARAMS_FILES,
        start_command=["rtabmap-rgbd_dataset"],
        end_command=["--output_path",
                     os.path.join(WORKSPACE_ROOT, "rtabmap_ws")],
        show_progress=SHOW_PROGRESS
    )


def reprocess():
    """
    Preprocess the RTAB-Map database using the specified parameters.
    """
    print("\n===== Retraitement de la base de données RTAB-Map =====")
    execute_command(
        REPROCESS_PARAMS_FILES,
        start_command=["rtabmap-reprocess"],
        end_command=[
            os.path.join(WORKSPACE_ROOT, "rtabmap_ws", "rtabmap.db"),
            "output_optimized.db"
        ],
        show_progress=SHOW_PROGRESS
    )


def export_point_cloud(output_type="--cloud"):
    """
    Export the point cloud from the RTAB-Map database.
    """
    print("\n===== Exportation du nuage de points =====")
    print("TYPE DE SORTIE:", output_type)
    start_command = ["rtabmap-export"]
    execute_command(
        EXPORT_PARAMS_FILES,
        start_command=start_command,
        end_command=[
            output_type,
            "--output", "point",
            os.path.join(WORKSPACE_ROOT, "rtabmap_ws", "output_optimized.db")
        ],
        show_progress=SHOW_PROGRESS
    )


def main(config):
    """Run RTAB-Map processing on the dataset."""
    output_type = config.get("export_format", "--mesh")
    generate_db()
    if config.get("reprocess", True):
        reprocess()
    export_point_cloud(output_type=output_type)


if __name__ == "__main__":
    print(sys.argv)

    config = load_config(os.path.join(
        WORKSPACE_ROOT, "rtabmap_ws", "config.json"))
    extension = "cloud" if config.get(
        "export_format", "--cloud") == "--cloud" else "mesh"
    rgb_path = os.path.join(WORKSPACE_ROOT, "rtabmap_ws", "rgb_sync")
    depth_path = os.path.join(WORKSPACE_ROOT, "rtabmap_ws", "depth_sync")

    rgb_path_from = os.path.join(
        WORKSPACE_ROOT, "rtabmap_ws", "rgb_sync_docker")
    depth_path_from = os.path.join(
        WORKSPACE_ROOT, "rtabmap_ws", "depth_sync_docker")

    calib_path = os.path.join(
        WORKSPACE_ROOT, "rtabmap_ws", "rtabmap_calib.yaml")
    rgb_timestamps = os.path.join(
        WORKSPACE_ROOT, "rtabmap_ws", "img_timestamps.csv")
    depth_timestamps = os.path.join(
        WORKSPACE_ROOT, "rtabmap_ws", "depth_timestamps.csv")

    print("===== Preparing Dataset =====")
    prepare_dataset(rgb_path_from, depth_path_from, calib_path,
                    rgb_timestamps, depth_timestamps)

    print("\n===== Converting to Timestamps =====")
    convert_to_timestamps(
        img_path=rgb_path,
        depth_path=depth_path,
        img_timestamps_path=rgb_timestamps,
        depth_timestamps_path=depth_timestamps
    )
    main(config)

    print("\n===== Copying Results =====")
    output_dir = os.path.join(
        WORKSPACE_ROOT, "rtabmap_ws", "output", "rtabmap")
    os.makedirs(output_dir, exist_ok=True)
    shutil.copy2(
        os.path.join(WORKSPACE_ROOT, "rtabmap_ws", "output_optimized.db"),
        os.path.join(output_dir, "rtabmap.db")
    )
    shutil.copy2(
        os.path.join(WORKSPACE_ROOT, "rtabmap_ws", f"point_{extension}.ply"),
        os.path.join(output_dir, f"point_{extension}.ply")
    )
    print(f"Processing complete! Results saved to {output_dir}/")
