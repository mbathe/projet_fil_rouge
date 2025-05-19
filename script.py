import os
import glob
import pandas as pd
import pandas as pd
import numpy as np
import subprocess
import os
import shutil
import sys

rgb_path = "/home/pmbathe/fin/my_map/rgb_sync"
rgb_depth_path = "/home/pmbathe/fin/my_map/depth_sync"
img_timestamps = "img_timestamps.csv"
depth_timestamps = "depth_timestamps.csv"

def convert_to_timestamps(img_path=None, depth_path=None, img_timestamps_path=None, depth_timestamps_path=None):
    def process_files(files, timestamps_df, output_path, label="image"):
        used_timestamps = set()

        for file in files:
            basename = os.path.basename(file)
            original_name = basename  # e.g. "frame_001.png"
            name_without_ext = os.path.splitext(basename)[0]

            # Chercher le timestamp correspondant dans le DataFrame
            match = timestamps_df[timestamps_df["filename"] == basename]
            if match.empty:
                print(f"[WARN] {label} - No timestamp found for file {basename}, skipping.")
                continue

            timestamp = match["timestamp"].values[0]

            # Éviter timestamp 0 ou déjà utilisé
            while timestamp == 0 or timestamp in used_timestamps:
                timestamp += np.random.uniform(0.001, 0.005)
                print(f"[INFO] {label} - Adjusted duplicate/zero timestamp for {basename}: {timestamp:.6f}")

            used_timestamps.add(timestamp)

            new_name = os.path.join(output_path, f"{timestamp:.6f}.png")
            os.rename(file, new_name)
            print(f"[OK] {label} - Renamed {basename} → {timestamp:.6f}.png")

    # Images RGB
    img_files = glob.glob(os.path.join(img_path, "*.png"))
    img_timestamps_df = pd.read_csv(img_timestamps_path)
    process_files(img_files, img_timestamps_df, img_path, label="RGB")

    # Images Depth
    depth_files = glob.glob(os.path.join(depth_path, "*.png"))
    depth_timestamps_df = pd.read_csv(depth_timestamps_path)
    process_files(depth_files, depth_timestamps_df, depth_path, label="Depth")



#convert_to_timestamps(img_path=rgb_path, depth_path=rgb_depth_path, img_timestamps_path=img_timestamps, depth_timestamps_path=depth_timestamps)



def main():
    db_command = [
    "rtabmap-rgbd_dataset",
    "--Rtabmap/DetectionRate", "2"
    "--RGBD/LinearUpdate", "0.1"
    "--RGBD/AngularUpdate", "0.1",
    "--Mem/STMSize", "30",
    "--Rtabmap/TimeThr","700",
    "--Vis/MinInliers", "12",
    "--Kp/MaxFeatures", "400",
    "--Kp/DetectorStrategy", "0",
    "--Rtabmap/LoopThr", "0.3",
    "--RGBD/OptimizeMaxError", "4.0",
    "--Optimizer/Strategy", "1",
    "--Rtabmap/PublishRAMUsage", "true",
    "--output_path", "/home/pmbathe/fin"
    ]

    export_command  = [
        "rtabmap-export", 
        "--output", "pointcloud.ply" 
        "--save",
        "--clouds", 
        "./my_map/rtabmap.db"
    ]

    print("Running command:")
    print(" ".join(db_command))

    try:
        subprocess.run(db_command, check=True)
        subprocess.run(export_command, check=True)
        print("RTAB-Map process completed.")
    except subprocess.CalledProcessError as e:
        print("Execution failed:", e)



def validate_csv_columns(csv_path):
    """Vérifie que le fichier CSV contient les colonnes nécessaires."""
    try:
        df = pd.read_csv(csv_path, nrows=0)
        if "timestamp" not in df.columns or "filename" not in df.columns:
            print(f"[ERREUR] Le fichier '{csv_path}' doit contenir les colonnes 'timestamp' et 'filename' permettant de faire la conversion.")
            return False
        return True
    except Exception as e:
        print(f"[ERREUR] Impossible de lire le fichier CSV : {csv_path}\n{e}")
        return False




def prepare_dataset(rgb_dir, depth_dir, calib_file, rgb_timestamps, depth_timestamps):
    root_dir = os.getcwd()
    rgb_sync_dir = os.path.join(root_dir, "rgb_sync")
    depth_sync_dir = os.path.join(root_dir, "depth_sync")
    calib_target_path = os.path.join(root_dir, "rtabmap_calib.yaml")
    rgb_timestamps_target_path = os.path.join(root_dir, "img_timestamps.csv")
    detph_timestamps_target_path = os.path.join(root_dir, "depth_timestamps.csv")



    # Vérification des CSV
    if not validate_csv_columns(rgb_timestamps) or not validate_csv_columns(depth_timestamps):
        return


    # Recréer les dossiers en supprimant leur contenu s'ils existent
    for sync_dir in [rgb_sync_dir, depth_sync_dir]:
        if os.path.exists(sync_dir):
            shutil.rmtree(sync_dir)
        os.makedirs(sync_dir)
        print(f"[RESET] Dossier vidé et recréé : {sync_dir}")

    # Supprimer le fichier de calibration existant
    if os.path.exists(calib_target_path):
        os.remove(calib_target_path)
        print(f"[DELETE] Ancien fichier de calibration supprimé : {calib_target_path}")
        
    if os.path.exists(rgb_timestamps_target_path):
        os.remove(rgb_timestamps_target_path)
        print(f"[DELETE] Ancien fichier de calibration supprimé : {rgb_timestamps_target_path}")

    if os.path.exists(detph_timestamps_target_path):
        os.remove(detph_timestamps_target_path)
        print(f"[DELETE] Ancien fichier de calibration supprimé : {detph_timestamps_target_path}")

    # Copier les fichiers RGB
    for file in os.listdir(rgb_dir):
        src = os.path.join(rgb_dir, file)
        dst = os.path.join(rgb_sync_dir, file)
        if os.path.isfile(src):
            shutil.copy2(src, dst)
            print(f"[OK] Copié RGB : {file}")

    # Copier les fichiers Depth
    for file in os.listdir(depth_dir):
        src = os.path.join(depth_dir, file)
        dst = os.path.join(depth_sync_dir, file)
        if os.path.isfile(src):
            shutil.copy2(src, dst)
            print(f"[OK] Copié Depth : {file}")

    # Copier le fichier de calibration
    shutil.copy2(calib_file, calib_target_path)
    shutil.copy2(rgb_timestamps, rgb_timestamps_target_path)
    shutil.copy2(depth_timestamps, detph_timestamps_target_path)
    print(f"[OK] Nouveau fichier de calibration copié vers : {calib_target_path}")
    print(f"[OK] Nouveau fichier de calibration copié vers : {rgb_timestamps_target_path}")
    print(f"[OK] Nouveau fichier de calibration copié vers : {detph_timestamps_target_path}")

# Exemple d'utilisation
if __name__ == "__main__":
    if len(sys.argv) != 6:
        print("Utilisation : python script.py <chemin_rgb> <chemin_depth> <fichier_calibration>")
    else:
        rgb_path = sys.argv[1]
        depth_path = sys.argv[2]
        calib_path = sys.argv[3]
        rgb_timestamps = sys.argv[4]
        depth_timestamps = sys.argv[5]
        prepare_dataset(rgb_path, depth_path, calib_path, rgb_timestamps, depth_timestamps)
        convert_to_timestamps(img_path=rgb_path, depth_path=rgb_depth_path, img_timestamps_path=img_timestamps, depth_timestamps_path=depth_timestamps)
        main()
