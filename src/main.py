from depth.generate_depth import generate_depth_maps
import sys
import argparse
parser = argparse.ArgumentParser(description="Mon application en ligne de commande")


def main():
    # Define paths
    image_folder = sys.argv[2]  # Path to the folder containing images
    output_folder = "path/to/save/depth/maps"
    
    # Generate depth maps
    results = generate_depth_maps(image_folder, output_folder)
    
    # Print results
    print("Depth maps generated successfully.")
    print(f"Results: {results}")
    
    
    



# Ajout des arguments
parser.add_argument("--images_folder", type=str, default="./images_folder", help="Dossier contenant les images")
parser.add_argument("--depth_folder", type=str, default="./depth_folder", help="Dossier contenant les images avec profondeur")
parser.add_argument("--calibration_file", type=str, default="./rtabmap_calib.yaml", help="Chemin vers le fichier de calibration")
parser.add_argument("--rgb_timestamps", type=str, default="./img_timestamps.csv", help="Chemin vers le fichier de timestamps RGB")
parser.add_argument("--depth_timestamps", type=str, default="./depth_timestamps.csv", help="Chemin vers le fichier de timestamps profondeur")
parser.add_argument("--output_folder", type=str, default="./output_folder", help="Dossier de sortie")
parser.add_argument("--source", type =str, default="image_with_depth", help="Source à utiliser (image: Image RGB sans profondeur, image_with_depth: image RGB avec profondeur, video: partir d'une source vidéo)")
parser.add_argument("--frequence", type=int, default=20, help="Fréquence d'images à traiter par seconde pour un flux vidéo valeur par défaut 20 hz")


args = parser.parse_args()


print(args.images_folder)
print(args.depth_folder)
print(args.calibration_file)
print(args.rgb_timestamps)
print(args.depth_timestamps)
print(args.output_folder)
print(args.source)
print(args.frequence)
if __name__ == "__main__":
    main()
    
