import os
import sys
import glob
from tqdm import tqdm
import subprocess
import shutil
import json
import re
import argparse
from PIL import Image 

SHOW_PROGRESS = True  # Set to False to disable progress bars

WORKSPACE_ROOT = os.getcwd()
DEFAULT_DIR_RTABMAP = "rtabmap_ws"
DEFAULT_DIR_IMAGE = "rgb_sync"
DEFAULT_DIR_DEPTH = "depth_sync"
DEFAULT_FILE_CALIB = "rtabmap_calib.yaml"
DEFAULT_FPS=20.0
DEFAULT_START_TIME=1400000000.0

JSON_PARAM_DB = "db_params_test.json"
JSON_PARAM_DETECT_MORE_LOOP_CLOSURE = "detectMoreLoopClosures_params.json"
JSON_PARAM_EXPORT = "export_params.json"

PATH_RTABMAP = os.path.join(WORKSPACE_ROOT, DEFAULT_DIR_RTABMAP)
PATH_IMG = os.path.join(WORKSPACE_ROOT, DEFAULT_DIR_RTABMAP, DEFAULT_DIR_IMAGE)
PATH_DEPTH = os.path.join(WORKSPACE_ROOT, DEFAULT_DIR_RTABMAP, DEFAULT_DIR_DEPTH)
PATH_CALIB = os.path.join(WORKSPACE_ROOT, DEFAULT_DIR_RTABMAP, DEFAULT_FILE_CALIB)
PATH_PARAM_DB = os.path.join(WORKSPACE_ROOT, DEFAULT_DIR_RTABMAP, JSON_PARAM_DB)
PATH_PARAM_DETECT_MORE_LOOP_CLOSURE = os.path.join(WORKSPACE_ROOT, DEFAULT_DIR_RTABMAP, JSON_PARAM_DETECT_MORE_LOOP_CLOSURE)
PATH_PARAM_EXPORT = os.path.join(WORKSPACE_ROOT, DEFAULT_DIR_RTABMAP, JSON_PARAM_EXPORT)

# TODO début: Implémenter la création des dossiers et rtabmap_ws, depth_sync et rgb_sync, la copie des json, etc. (dans ton script main)
def create_dir(dir_name):
    if os.path.exists(dir_name):
        print(os.path.exists(dir_name))
        shutil.rmtree(dir_name)

    os.makedirs(dir_name)

create_dir(PATH_RTABMAP)
create_dir(DEFAULT_DIR_IMAGE)
create_dir(DEFAULT_DIR_DEPTH)

shutil.copy2(WORKSPACE_ROOT+'/src/rtabmap/params/'+JSON_PARAM_DB, PATH_RTABMAP)
shutil.copy2(WORKSPACE_ROOT+'/src/rtabmap/params/'+JSON_PARAM_DETECT_MORE_LOOP_CLOSURE, PATH_RTABMAP)
shutil.copy2(WORKSPACE_ROOT+'/src/rtabmap/params/'+JSON_PARAM_EXPORT, PATH_RTABMAP)
# TODO fin.


def load_config(config_file):
    """Charge la configuration depuis un fichier JSON"""
    with open(config_file, 'r') as f:
        return json.load(f)


def config_to_args(config):
    """Convertit la configuration en arguments de ligne de commande pour RTABMap"""
    args = []
    for key, value in config.items():
        if isinstance(value, bool):
            value_str = str(value).lower()
        else:
            value_str = str(value)
        # Split key and value for subprocess.run
        args.extend([str(key), value_str])
    return args

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

def cp_files(dir, dest, desc):

    files = []
    for ext in ['*.png', '*.jpg', '*.jpeg', '*.tiff', '*.tif']:
        files.extend(glob.glob(os.path.join(dir, ext)))

    if files:
        print(f"[INFO] Copie de {len(files)} fichiers...")
        for src in tqdm(files, desc=desc, unit="img"):
            filename = os.path.basename(src)
            dst = os.path.join(dest, filename)
            shutil.copy2(src, dst)
        print(f"[OK] {len(files)} fichiers copiées avec succès")
    else:
        print(f"[WARN] Aucun fichiers trouvée dans le répertoire {files}")
    

def prepare_dataset(img_dir, depth_dir, calib_file):
    """
    Prepare the dataset by copying files to target directories.
    """

    for dirs in [PATH_IMG, PATH_DEPTH]:
        os.makedirs(dirs, exist_ok=True)
        print(f"[INFO] Directory ready: {dirs}")

    cp_files(img_dir, PATH_IMG, "Copie img")
    cp_files(depth_dir, PATH_DEPTH, "Copie depth")

    # Copy calibration and timestamp files
    shutil.copy2(calib_file, PATH_CALIB)
    print("[OK] Fichiers de calibration copiés.")

    # exts = {os.path.splitext(f)[1].lower() for f in os.listdir(PATH_DEPTH) if os.path.isfile(os.path.join(PATH_DEPTH, f))}
    # if '.tiff' not in exts:
    #    for f in os.listdir(PATH_DEPTH):
    #     chemin_fichier = os.path.join(PATH_DEPTH, f)
    #     extension = os.path.splitext(f)[1].lower()

    #     if os.path.isfile(chemin_fichier) and extension not in {'.tiff', '.tif'}:
    #         try:
    #             with Image.open(chemin_fichier) as img:
    #                 nouveau_chemin = os.path.join(PATH_DEPTH, os.path.splitext(f)[0] + '.tiff')
    #                 img.save(nouveau_chemin, format='TIFF')
    #             os.remove(chemin_fichier)  # Supprime l’original
    #             print(f"Converti et supprimé : {f} → {os.path.basename(nouveau_chemin)}")
    #         except Exception as e:
    #             print(f"Erreur avec {f} : {e}")
    # else:
    #     print("Tous les fichiers sont déjà en TIFF.")

def execute_command(config_file, start_command=[], end_command=[], show_progress=SHOW_PROGRESS):
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

def cmd(config_file, start_command=[], end_command=[]):
    config = load_config(config_file)
    config_args = config_to_args(config)
    full_command = start_command + config_args + end_command
    print(full_command)

    subprocess.run(
                full_command,
                check=True,
                stdout=sys.stdout,
                stderr=sys.stderr,
                text=True)


def generate_db():
    """
    Generate the RTAB-Map database using the specified parameters.
    """
    print("\n===== Génération de la base de données RTAB-Map =====")
    execute_command(
        PATH_PARAM_DB,
        start_command=["rtabmap-rgbd_dataset"],
        end_command=[PATH_RTABMAP],
        show_progress=False
    )


def post_processing():
    """
    Preprocess the RTAB-Map database using the specified parameters.
    """
    print("\n===== Retraitement de la base de données RTAB-Map =====")
    execute_command(
        PATH_PARAM_DETECT_MORE_LOOP_CLOSURE,
        start_command=["rtabmap-detectMoreLoopClosures"],
        end_command=[
            os.path.join(WORKSPACE_ROOT, "rtabmap_ws", "rtabmap.db")
        ],
        show_progress=SHOW_PROGRESS
    )

    # print("\n===== Retraitement de la base de données RTAB-Map =====")
    # execute_command(
    #     '/Users/loux/Desktop/CODE/PROJET_FILROUGE/github/projet_fil_rouge/src/rtabmap/params/empty_params.json',
    #     start_command=["rtabmap-globalBundleAdjustment"],
    #     end_command=[
    #         os.path.join(WORKSPACE_ROOT, "rtabmap_ws", "rtabmap.db")
    #     ],
    #     show_progress=SHOW_PROGRESS
    # )


def export_point_cloud(output_type="--cloud"):
    """
    Export the point cloud from the RTAB-Map database.
    """
    print("\n===== Exportation du nuage de points =====")
    print("TYPE DE SORTIE:", output_type)
    start_command = ["rtabmap-export"]
    execute_command(
        "/Users/loux/Desktop/CODE/PROJET_FILROUGE/github/projet_fil_rouge/src/rtabmap/params/empty_params.json",
        start_command=start_command,
        end_command=[
            "--cloud",
            "--ba",
            "--texture_size", "16384",
            "--max_range", "0",
            "decimation", "1",
            "--color_radius", "0",
            os.path.join(WORKSPACE_ROOT, "rtabmap_ws", "rtabmap.db")
        ],
        show_progress=False
    )


def main():
    """Run RTAB-Map processing on the dataset."""
    generate_db()
    #post_processing()
    export_point_cloud()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="script rtabmap.")
    parser.add_argument("--img_dir", type=str, required=True, help="Path to images folder.")
    parser.add_argument("--depth_dir", type=str, required=True, help="Path to depths folder.")
    parser.add_argument("--calib_file", type=str, required=True, help="Path to calibration file")

    parser.add_argument("--fps", type=float, default=DEFAULT_FPS, help="Number of FPS in the video.")

    args = parser.parse_args()

    prepare_dataset(args.img_dir, args.depth_dir, args.calib_file)

    rename_files_to_timestamps(PATH_IMG, fps=args.fps)
    rename_files_to_timestamps(PATH_DEPTH, fps=args.fps)
    
    # cmd(
    #     PATH_PARAM_DB,
    #     start_command=["rtabmap-rgbd_dataset"],
    #     end_command=[PATH_RTABMAP]
    # )
    main()