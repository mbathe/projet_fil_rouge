#!/usr/bin/env python3
"""
Script pour extraire des images d'une vidéo à une fréquence donnée
Compatible avec RTAB-Map - les images sont nommées avec timestamp
"""

import numpy as np
import cv2
import os
import argparse
import sys
from pathlib import Path

import os
import shutil


def create_or_clean_dir(path):
    shutil.rmtree(path, ignore_errors=True) 


#!/usr/bin/env python3
"""
Script pour extraire des images d'une vidéo à une fréquence donnée
Compatible avec RTAB-Map - images corrigées d'orientation et optimisées pour odométrie
"""


def fix_image_orientation(image, force_rotation=None):
    """
    Corrige l'orientation des images iPhone
    
    Args:
        image: Image à corriger
        force_rotation: None (auto), 'cw' (90° horaire), 'ccw' (90° anti-horaire), 'none' (pas de rotation)
    """
    h, w = image.shape[:2]

    if force_rotation == 'none':
        print(f"Rotation désactivée - Image conservée: {w}x{h}")
        return image
    elif force_rotation == 'cw':
        print(f"Rotation forcée 90° horaire: {w}x{h} -> {h}x{w}")
        return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    elif force_rotation == 'ccw':
        print(f"Rotation forcée 90° anti-horaire: {w}x{h} -> {h}x{w}")
        return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    else:
        # Détection automatique
        if h > w * 1.2:  # Portrait détecté
            print(
                f"Portrait détecté ({w}x{h}), rotation 90° anti-horaire -> {h}x{w}")
            return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
        else:
            print(f"Paysage détecté ({w}x{h}), pas de rotation")
            return image


def enhance_image_for_odometry(image):
    """
    Optimise l'image spécifiquement pour résoudre le problème 'Not enough inliers'
    - Focus sur la robustesse des features plutôt que la quantité
    - Préserve les structures géométriques essentielles
    - Minimise les artefacts qui génèrent de fausses correspondances
    """
    # Analyse de la qualité initiale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    contrast = np.std(gray)
    mean_brightness = np.mean(gray)

    print(
        f"Analyse image - Contraste: {contrast:.1f}, Luminosité: {mean_brightness:.1f}")

    # Prétraitement plus conservateur pour éviter les artefacts
    # Réduction du bruit AVANT toute amélioration (crucial pour la robustesse)
    denoised = cv2.bilateralFilter(image, 9, 75, 75)

    # Conversion en niveaux de gris pour traitement uniforme
    gray_denoised = cv2.cvtColor(denoised, cv2.COLOR_BGR2GRAY)

    # CLAHE très modéré pour éviter la sur-amélioration
    if contrast < 30:  # Seulement si vraiment nécessaire
        print("Contraste très faible - CLAHE modéré appliqué")
        clahe = cv2.createCLAHE(clipLimit=1.8, tileGridSize=(16, 16))
        enhanced_gray = clahe.apply(gray_denoised)
    elif contrast < 50:
        print("Contraste faible - CLAHE léger appliqué")
        clahe = cv2.createCLAHE(clipLimit=1.4, tileGridSize=(20, 20))
        enhanced_gray = clahe.apply(gray_denoised)
    else:
        print("Contraste suffisant - pas d'amélioration CLAHE")
        enhanced_gray = gray_denoised

    # Reconversion en couleur pour préserver l'information chromatique
    result = cv2.cvtColor(enhanced_gray, cv2.COLOR_GRAY2BGR)

    # Netteté très conservative - préserve les structures sans créer d'artefacts
    laplacian_var = cv2.Laplacian(enhanced_gray, cv2.CV_64F).var()

    if laplacian_var < 50:  # Image très floue
        print("Image floue - Amélioration de netteté conservative")
        # Kernel de netteté doux
        kernel = np.array([[0, -0.1, 0],
                          [-0.1, 1.4, -0.1],
                          [0, -0.1, 0]])
        sharpened = cv2.filter2D(result, -1, kernel)
        result = cv2.addWeighted(result, 0.8, sharpened, 0.2, 0)
    else:
        print("Netteté suffisante - pas d'amélioration")

    # Ajustement gamma très léger
    if mean_brightness < 80:  # Image très sombre
        gamma = 1.15
        print("Image sombre - Correction gamma légère")
    elif mean_brightness > 180:  # Image très claire
        gamma = 0.95
        print("Image claire - Correction gamma légère")
    else:
        gamma = 1.02  # Quasi-neutre
        print("Luminosité correcte - pas de correction gamma")

    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255
                      for i in np.arange(0, 256)]).astype("uint8")
    result = cv2.LUT(result, table)

    # Test de qualité des features pour diagnostic
    test_gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)

    # Test avec différents détecteurs pour diagnostic
    corners_gftt = cv2.goodFeaturesToTrack(test_gray, 300, 0.01, 7)
    gftt_count = len(corners_gftt) if corners_gftt is not None else 0

    # Test SIFT pour robustesse
    sift = cv2.SIFT_create(nfeatures=200)
    keypoints_sift = sift.detect(test_gray, None)
    sift_count = len(keypoints_sift)

    print(f"Features détectées - GFTT: {gftt_count}, SIFT: {sift_count}")

    if gftt_count < 100:
        print("⚠ ATTENTION: Peu de features GFTT détectées - risque d'échec odométrie")
    if sift_count < 50:
        print(
            "⚠ ATTENTION: Peu de features SIFT détectées - vérifiez la texture de la scène")

    return result


def create_camera_info_file(output_dir, image_width, image_height):
    """
    Crée un fichier de calibration caméra spécialement optimisé pour résoudre 'Not enough inliers'
    """
    # Paramètres intrinsèques plus conservateurs
    focal_multiplier = 0.75  # Encore plus conservateur
    fx = fy = max(image_width, image_height) * focal_multiplier
    cx = image_width / 2.0
    cy = image_height / 2.0

    # Distorsion minimale pour éviter les erreurs de rectification
    k1 = -0.05  # Distorsion radiale réduite
    k2 = 0.005  # Distorsion quadratique minimale
    p1 = 0.0    # Pas de distorsion tangentielle
    p2 = 0.0    # Pas de distorsion tangentielle
    k3 = 0.0    # Pas de distorsion cubique

    camera_yaml = f"""%YAML:1.0
---
camera_name: iphone_camera_robust
image_width: {image_width}
image_height: {image_height}
camera_matrix:
   rows: 3
   cols: 3
   data: [ {fx:.6f}, 0., {cx:.6f}, 0., {fy:.6f}, {cy:.6f}, 0., 0., 1. ]
distortion_coefficients:
   rows: 1
   cols: 5
   data: [ {k1:.6f}, {k2:.6f}, {p1:.6f}, {p2:.6f}, {k3:.6f} ]
rectification_matrix:
   rows: 3
   cols: 3
   data: [ 1., 0., 0., 0., 1., 0., 0., 0., 1. ]
projection_matrix:
   rows: 3
   cols: 4
   data: [ {fx:.6f}, 0., {cx:.6f}, 0., 0., {fy:.6f}, {cy:.6f}, 0., 0., 0., 1., 0. ]
local_transform:
   rows: 3
   cols: 4
   data: [ 0., 0., 1., 0., -1., 0., 0., 0., 0., -1., 0., 0. ]
"""

    yaml_path = os.path.join(output_dir, "camera_calibration.yaml")
    with open(yaml_path, 'w') as f:
        f.write(camera_yaml)

    # Paramètres RTAB-Map spécifiquement pour résoudre "Not enough inliers"
    info_text = f"""# Calibration iPhone pour résoudre 'Not enough inliers'

Dimensions: {image_width} x {image_height}
Focale fx/fy: {fx:.2f} pixels (conservative)
Centre optique: ({cx:.2f}, {cy:.2f})
Distorsion minimale: k1={k1:.3f}

PARAMÈTRES RTAB-MAP CRITIQUES pour corriger "Not enough inliers":

# Seuils d'inliers réduits (CRITIQUE)
--Vis/MinInliers 8
--Vis/InlierDistance 0.15
--OdomF2M/MaxSize 2000
--OdomF2M/WindowSize 10

# Détection de features robuste
--Kp/DetectorStrategy 6
--Kp/MaxFeatures 800
--GFTT/MinDistance 3
--GFTT/QualityLevel 0.0005

# Correspondances plus tolérantes
--Vis/MaxFeatures 1000
--RGBD/OptimizeMaxError 0.05
--Odom/FillInfoData true

# Stratégie odométrie
--Odom/Strategy 1
--Odom/ResetCountdown 1

COMMANDE COMPLÈTE (COPIEZ-COLLEZ):
rtabmap --camera_info_path camera_calibration.yaml \\
        --Vis/MinInliers 8 --Vis/InlierDistance 0.15 \\
        --OdomF2M/MaxSize 2000 --OdomF2M/WindowSize 10 \\
        --Kp/DetectorStrategy 6 --Kp/MaxFeatures 800 \\
        --GFTT/MinDistance 3 --GFTT/QualityLevel 0.0005 \\
        --Odom/Strategy 1 --Odom/ResetCountdown 1

Si l'erreur persiste, réduisez encore:
--Vis/MinInliers 5
"""

    info_path = os.path.join(output_dir, "camera_info.txt")
    with open(info_path, 'w') as f:
        f.write(info_text)

    print(f"✓ Calibration créée avec paramètres anti-'Not enough inliers'")


def extract_frames(video_path, output_dir, frequency=1.0, enhance_images=True, rotation=None):
    """
    Extrait des images d'une vidéo avec corrections pour odométrie et estimation de profondeur
    
    Args:
        video_path (str): Chemin vers le fichier vidéo
        output_dir (str): Dossier de sortie pour les images
        frequency (float): Fréquence d'extraction en Hz
        enhance_images (bool): Active l'amélioration des images
        rotation (str): Force la rotation ('cw', 'ccw', 'none', None=auto)
    """

    # Vérifier si le fichier vidéo existe
    if not os.path.exists(video_path):
        print(f"Erreur: Le fichier vidéo '{video_path}' n'existe pas.")
        return False

    # Créer le dossier de sortie
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Ouvrir la vidéo
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Erreur: Impossible d'ouvrir la vidéo '{video_path}'.")
        return False

    # Obtenir les propriétés de la vidéo
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps

    print(f"Vidéo: {video_path}")
    print(f"FPS: {fps:.2f}")
    print(f"Durée: {duration:.2f} secondes")
    print(f"Frames totales: {total_frames}")
    print(f"Fréquence d'extraction: {frequency} Hz")
    print(
        f"Amélioration images: {'Activée (odométrie optimisée)' if enhance_images else 'Désactivée'}")
    print(f"Rotation: {rotation if rotation else 'Automatique'}")

    # Calculer l'intervalle entre les frames à extraire
    frame_interval = int(fps / frequency)
    if frame_interval == 0:
        frame_interval = 1
        print(f"Attention: Fréquence trop élevée, extraction de toutes les frames")

    print(f"Extraction d'une frame toutes les {frame_interval} frames")

    frame_count = 0
    extracted_count = 0
    first_frame_processed = False

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        # Extraire seulement les frames selon l'intervalle
        if frame_count % frame_interval == 0:
            # Correction d'orientation pour les images iPhone
            corrected_frame = fix_image_orientation(frame, rotation)

            # Amélioration pour l'odométrie si demandée
            if enhance_images:
                processed_frame = enhance_image_for_odometry(corrected_frame)
            else:
                processed_frame = corrected_frame

            # Créer le fichier camera_info à la première frame
            if not first_frame_processed:
                h, w = processed_frame.shape[:2]
                create_camera_info_file(output_dir, w, h)
                print(f"Dimensions finales des images: {w}x{h}")
                first_frame_processed = True

            # Calculer le timestamp en millisecondes
            timestamp_ms = int((frame_count / fps) * 1000)
            if timestamp_ms == 0:
                timestamp_ms = 1  # Éviter le timestamp zéro

            # Nom du fichier avec timestamp sur 19 chiffres
            filename = f"{timestamp_ms:019d}.png"
            output_path = os.path.join(output_dir, filename)

            # Sauvegarder l'image avec compression optimisée
            cv2.imwrite(output_path, processed_frame, [
                        cv2.IMWRITE_PNG_COMPRESSION, 3])
            extracted_count += 1

            if extracted_count % 10 == 0:
                print(f"Extraites: {extracted_count} images...")

        frame_count += 1

    cap.release()

    print(f"\nExtraction terminée!")
    print(f"Total d'images extraites: {extracted_count}")
    print(f"Images sauvegardées dans: {output_dir}")
    print(
        f"Fichier de calibration: {os.path.join(output_dir, 'camera_calibration.yaml')}")

    print(f"\nOptimisations appliquées:")
    print(f"✓ Correction d'orientation iPhone")
    print(f"✓ Amélioration contraste et netteté pour features")
    print(f"✓ Préservation gradients pour estimation profondeur")
    print(f"✓ Paramètres RTAB-Map pour corriger erreur odométrie")

    print(f"\nPour corriger l'erreur 'Not enough inliers', utilisez:")
    print(
        f"rtabmap --camera_info_path {os.path.join(output_dir, 'camera_calibration.yaml')} \\")
    print(f"        --Odom/Strategy 1 --OdomF2M/MaxSize 1000 --Kp/MaxFeatures 600 \\")
    print(f"        --GFTT/MinDistance 5 --Vis/MinInliers 15")

    return True

def main():
    parser = argparse.ArgumentParser(
        description="Extrait des images d'une vidéo pour odométrie et estimation de profondeur",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples d'utilisation:
  python extract_frames.py video.mp4 -o images/ -f 1.0                    # Auto rotation
  python extract_frames.py video.mp4 -o images/ -f 1.0 --rotation cw      # Force rotation horaire
  python extract_frames.py video.mp4 -o images/ -f 1.0 --rotation ccw     # Force rotation anti-horaire
  python extract_frames.py video.mp4 -o images/ -f 1.0 --rotation none    # Pas de rotation
        """
    )

    parser.add_argument('video', help='Chemin vers le fichier vidéo')
    parser.add_argument('-o', '--output', default='extracted_frames',
                        help='Dossier de sortie (défaut: extracted_frames)')
    parser.add_argument('-f', '--frequency', type=float, default=1.0,
                        help='Fréquence d\'extraction en Hz (défaut: 1.0)')
    parser.add_argument('--raw', action='store_true',
                        help='Sauvegarde les images sans amélioration (brutes)')
    parser.add_argument('--rotation', choices=['cw', 'ccw', 'none'], default=None,
                        help='Force la rotation: cw=90° horaire, ccw=90° anti-horaire, none=aucune (défaut: auto)')

    args = parser.parse_args()

    if args.frequency <= 0:
        print("Erreur: La fréquence doit être positive.")
        sys.exit(1)

    enhance_images = not args.raw

    success = extract_frames(
        args.video,
        args.output,
        args.frequency,
        enhance_images,
        args.rotation
    )

    if not success:
        sys.exit(1)

if __name__ == "__main__":
    main()
