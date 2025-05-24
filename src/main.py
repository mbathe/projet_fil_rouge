import subprocess
from depth.generate_depth import generate_depth_maps
import sys
import argparse
parser = argparse.ArgumentParser(description="Mon application en ligne de commande")


def build_3d_map(image_folder, depth_folder, calibration_file, rgb_timestamps, depth_timestamps, output_folder, source, frequence):

    command = [
        "sudo", "docker", "run", "--rm", "-it",
        "-v", f"{image_folder}:/rtabmap_ws/rgb_sync_docker",
        "-v", f"{depth_folder}:/rtabmap_ws/depth_sync_docker",
        "-v", f"{rgb_timestamps}:/rtabmap_ws/img_timestamps.csv",
        "-v", f"{depth_timestamps}:/rtabmap_ws/depth_timestamps.csv",
        "-v", f"{output_folder}:/rtabmap_ws/output",
        "-v", f"{calibration_file}:/rtabmap_ws/rtabmap_calib.yaml",
        "rtabmap_ubuntu20"
    ]

    subprocess.run(command)


def main():

    parser.add_argument("--image_folder", type=str,
                        default="../data/images", help="Dossier contenant les images")
    parser.add_argument("--depth_folder", type=str, default="../data/depth",
                        help="Dossier contenant les images avec profondeur")
    parser.add_argument("--calibration_file", type=str, default="../data/rtabmap_calib.yaml",
                        help="Chemin vers le fichier de calibration")
    parser.add_argument("--rgb_timestamps", type=str, default="../data/img_timestamps.csv",
                        help="Chemin vers le fichier de timestamps RGB")
    parser.add_argument("--depth_timestamps", type=str, default="../data/depth_timestamps.csv",
                        help="Chemin vers le fichier de timestamps profondeur")
    parser.add_argument("--output_folder", type=str,
                        default="../output", help="Dossier de sortie")
    parser.add_argument("--source", type=str, default="image_with_depth",
                        help="Source à utiliser (image: Image RGB sans profondeur, image_with_depth: image RGB avec profondeur, video: partir d'une source vidéo)")
    parser.add_argument("--frequence", type=int, default=20,
                        help="Fréquence d'images à traiter par seconde pour un flux vidéo valeur par défaut 20 hz")
    args = parser.parse_args()
    print(args.image_folder)
    print(args.depth_folder)
    print(args.calibration_file)
    print(args.rgb_timestamps)
    print(args.depth_timestamps)
    print(args.output_folder)
    print(args.source)
    print(args.frequence)
    build_3d_map(
        args.image_folder,
        args.depth_folder,
        args.calibration_file,
        args.rgb_timestamps,
        args.depth_timestamps,
        args.output_folder,
        args.source,
        args.frequence
    )


if __name__ == "__main__":
    main()
    
