import cv2
import numpy as np
import glob
import argparse
import os
from tqdm import tqdm

class CustomFormatter(argparse.ArgumentDefaultsHelpFormatter, argparse.RawTextHelpFormatter):
    pass

def verify_extention(files):
    extensions = {os.path.splitext(f)[1].lower() for f in files}
    if extensions == {'.jpg'} or extensions == {'.png'}:
        return True
    else:
        print(f"Extensions non valides : {extensions}, format .jpg ou .png nécessaire (l'un ou l'autre)")
        return False


def save_calibration_yaml(output_name, output_path, camera_matrix, dist_coeffs, image_size):
    fs = cv2.FileStorage(os.path.join(output_path, output_name), cv2.FILE_STORAGE_WRITE)

    if image_size[1] > image_size[0]: # height > width mode portrait
        local_transform = np.array([
        [0., 0., 1., 0.],
        [-1., 0., 0., 0.],
        [0., -1., 0., 0.]
        ], dtype=np.float32)
    else:  # mode paysage
        local_transform = np.eye(3, 4, dtype=np.float32)

    fs.write("camera_name", output_name)
    fs.write("image_width", image_size[0])
    fs.write("image_height", image_size[1])
    fs.write("camera_matrix", camera_matrix)
    fs.write("distortion_coefficients", dist_coeffs)
    fs.write("local_transform", local_transform.astype(np.float32))    

    fs.release()
    print("==============================")
    print(f"Calibration enregistrée dans : {os.path.join(output_path, output_name)}")


def calibrate(images_path, square_size, nb_cols, nb_rows, output_name, output_path):
    print("==============================")
    print("Initialisation ...")
    # Taille des coins intérieurs de l’échiquier
    chessboard_size = (nb_cols, nb_rows)

    # Préparer les coordonnées 3D du modèle (Z = 0)
    objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2) * square_size

    # Listes pour les points
    objpoints = []  # Points 3D réels
    imgpoints = []  # Points 2D détectés dans l’image

    # Charger les images
    print("==============================")
    images = [f for f in tqdm(os.listdir(images_path), desc="chargement des images") if os.path.isfile(os.path.join(images_path, f))]

    if len(images) < 10:
        print("==============================")
        print(f"Pour une calibration fiable, donner au moins 10 images. Nombre d'images trouvé : {len(images)}")
        print("==============================")
        exit(1)
    
    if not verify_extention(images):
        exit(1)

    undetected_count = 0

    for fname in tqdm(images, desc="Lecture des images"):
        img = cv2.imread(os.path.join(images_path, fname))

        if img is None:
            print("==============================")
            print(f" Image illisible : {fname}")
            continue
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Trouver les coins de l’échiquier
        ret, corners = cv2.findChessboardCorners(gray, chessboard_size, None)

        if ret:
            objpoints.append(objp)
            imgpoints.append(corners)
        
        else:
            print("==============================")
            print(f"Damier non détecté : {fname}")
            undetected_count +=1
            continue

    # Calibration
    print("==============================")
    print("Calibration ...")
    ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

    print("==============================")
    print("Matrice de la caméra :\n", camera_matrix)
    print("Coefficients de distorsion :\n", dist_coeffs)
    print(f"Nombre de damier non detecté : {undetected_count}/{len(images)}")

    print("==============================")
    print("Saving ...")
    save_calibration_yaml(output_name, output_path, camera_matrix, dist_coeffs, gray.shape[::-1])
    print("==============================")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Génération du fichier de calibration d'une caméra via un damier : https://github.com/opencv/opencv/blob/master/doc/pattern.png \n"
            "Pour les images du damier, suivre le tutoriel suivant : https://github.com/mbathe/projet_fil_rouge/blob/main/docs/guide_calibration.md \n"
            ),
        epilog=
            ("Exemples : \n"
             "python calibrate.py --images path/to/images --square-size 24 \n"
             "python calibrate.py --images path/to/images --square-size 24 --cols 10 --rows 7 \n"
             ),
        formatter_class=CustomFormatter
        )

    parser.add_argument("--images",type=str,required=True,help="Chemin vers les images d'entrée (.jpg ou .png).")
    parser.add_argument("--square_size", type=int, required=True, help="Taille des carrés du pattern imprimé en mm.")

    parser.add_argument("--cols",type=int,default=9,help="Nombre de colonnes de coins intérieurs.")
    parser.add_argument("--rows",type=int,default=6,help="Nombre de lignes de coins intérieurs (exclut les bords).")

    parser.add_argument("--output_name",type=str,default="calib.yaml",help="Nom du fichier de configuration enregistré.")
    parser.add_argument("--output_path",type=str,default=os.getcwd(),help="Nombre de lignes de coins intérieurs (exclut les bords).")

    args = parser.parse_args()

    calibrate(args.images, args.square_size, args.cols, args.rows, args.output_name, args.output_path)


    
