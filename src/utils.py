#!/usr/bin/env python3
"""
Script pour extraire des images d'une vidéo à une fréquence donnée
Compatible avec RTAB-Map - les images sont nommées avec timestamp
"""

import cv2
import os
import argparse
import sys
from pathlib import Path


import os
import shutil


def create_or_clean_dir(path):
    shutil.rmtree(path, ignore_errors=True) 



def extract_frames(video_path, output_dir, frequency=1.0):
    """
    Extrait des images d'une vidéo à la fréquence spécifiée
    
    Args:
        video_path (str): Chemin vers le fichier vidéo
        output_dir (str): Dossier de sortie pour les images
        frequency (float): Fréquence d'extraction en Hz (images par seconde)
    """

    # Vérifier si le fichier vidéo existe
    if not os.path.exists(video_path):
        print(f"Erreur: Le fichier vidéo '{video_path}' n'existe pas.")
        return False

    # Créer le dossier de sortie s'il n'existe pas
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

    # Calculer l'intervalle entre les frames à extraire
    frame_interval = int(fps / frequency)

    if frame_interval == 0:
        frame_interval = 1
        print(f"Attention: Fréquence trop élevée, extraction de toutes les frames")

    print(f"Extraction d'une frame toutes les {frame_interval} frames")

    frame_count = 0
    extracted_count = 0

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        # Extraire seulement les frames selon l'intervalle
        if frame_count % frame_interval == 0:
            # Calculer le timestamp en millisecondes (format RTAB-Map)
            timestamp_ms = int((frame_count / fps) * 1000)

            # S'assurer que le timestamp n'est jamais 0 (problème avec certains logiciels de cartographie)
            if timestamp_ms == 0:
                timestamp_ms = 1

            # Nom du fichier avec timestamp sur 19 chiffres (compatible RTAB-Map)
            filename = f"{timestamp_ms:019d}.png"
            output_path = os.path.join(output_dir, filename)

            # Sauvegarder l'image
            cv2.imwrite(output_path, frame)
            extracted_count += 1

            if extracted_count % 10 == 0:
                print(f"Extraites: {extracted_count} images...")

        frame_count += 1

    cap.release()

    print(f"\nExtraction terminée!")
    print(f"Total d'images extraites: {extracted_count}")
    print(f"Images sauvegardées dans: {output_dir}")

    return True


def main():
    parser = argparse.ArgumentParser(
        description="Extrait des images d'une vidéo pour RTAB-Map",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples d'utilisation:
  python extract_frames.py video.mp4 -o images/ -f 1.0    # 1 image par seconde
  python extract_frames.py video.mp4 -o images/ -f 0.5    # 1 image toutes les 2 secondes
  python extract_frames.py video.mp4 -o images/ -f 2.0    # 2 images par seconde
        """
    )

    parser.add_argument('video', help='Chemin vers le fichier vidéo')
    parser.add_argument('-o', '--output', default='extracted_frames',
                        help='Dossier de sortie (défaut: extracted_frames)')
    parser.add_argument('-f', '--frequency', type=float, default=1.0,
                        help='Fréquence d\'extraction en Hz (défaut: 1.0)')

    args = parser.parse_args()

    if args.frequency <= 0:
        print("Erreur: La fréquence doit être positive.")
        sys.exit(1)

    success = extract_frames(args.video, args.output, args.frequency)

    if not success:
        sys.exit(1)


if __name__ == "__main__":
    main()
