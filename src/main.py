#!/usr/bin/env python3
"""
Cartographie 3D RTAB-Map - Module principal

Ce module contient le point d'entrée principal de l'application de cartographie 3D
utilisant RTAB-Map via Docker, avec support pour différentes sources d'entrée (RGB, RGB-D, vidéo).
"""

import os
import sys
import subprocess
import logging
import argparse
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path

# Import des modules du projet
try:
    from depth.generate_depth import generate_depth_maps
except ImportError as e:
    print(f"Erreur d'importation des modules du projet: {e}")
    print("Assurez-vous que le dossier 'src' est dans votre PYTHONPATH")
    sys.exit(1)

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("cartographie3d.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("cartographie3d")

# Déterminer le répertoire racine du projet
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)


class CartographieError(Exception):
    """Exception personnalisée pour les erreurs liées à la cartographie 3D."""
    pass


class RTAB3DMapper:
    """Classe principale pour la génération de cartographie 3D utilisant RTAB-Map."""

    def __init__(self, args: argparse.Namespace):
        """
        Initialise l'instance de cartographie 3D.
        
        Args:
            args: Arguments de ligne de commande parsés
        """
        self.args = args
        self.validate_args()
        logger.info(f"Initialisation avec source: {args.source}")
        logger.debug(f"Arguments complets: {args}")

    def validate_args(self) -> None:
        """Valide les arguments fournis et les chemins associés."""
        # Vérification des chemins d'entrée
        paths_to_check = {
            "Dossier d'images": self.args.image_folder,
            "Fichier de calibration": self.args.calibration_file,
            "Fichier de timestamps RGB": self.args.rgb_timestamps,
            "Fichier de timestamps profondeur": self.args.depth_timestamps,
        }

        if self.args.source == "image_with_depth":
            paths_to_check["Dossier de profondeur"] = self.args.depth_folder

        for name, path in paths_to_check.items():
            if not os.path.exists(path):
                if name.startswith("Dossier"):
                    # Créer le dossier s'il n'existe pas
                    os.makedirs(path, exist_ok=True)
                    logger.warning(
                        f"{name} '{path}' n'existait pas et a été créé")
                else:
                    logger.warning(f"{name} '{path}' n'existe pas")

        # Création du dossier de sortie s'il n'existe pas
        os.makedirs(self.args.output_folder, exist_ok=True)
        logger.info(f"Dossier de sortie: {self.args.output_folder}")

    def process_source(self) -> None:
        """Traite la source d'entrée selon le type spécifié."""
        if self.args.source == "video":
            self._process_video()
        elif self.args.source == "image":
            self._process_rgb_images()
        elif self.args.source == "image_with_depth":
            self._process_rgbd_images()
        else:
            raise CartographieError(
                f"Type de source inconnu: {self.args.source}")

    def _process_video(self) -> None:
        """Traite une source vidéo pour générer une cartographie 3D."""
        logger.info(
            f"Traitement de la vidéo: {self.args.image_folder} à {self.args.frequence}Hz")
        # À implémenter: extraction des frames et estimation de profondeur
        # TODO: Ajouter l'appel à extract_frames et depth_estimation

        # Une fois les images RGB et profondeur générées, on poursuit avec la cartographie
        self._build_3d_map()

    def _process_rgb_images(self) -> None:
        """Traite des images RGB pour générer une cartographie 3D."""
        logger.info(f"Traitement des images RGB: {self.args.image_folder}")
        # À implémenter: estimation de profondeur pour les images RGB
        # TODO: Ajouter l'appel à depth_estimation

        # Une fois les images de profondeur générées, on poursuit avec la cartographie
        self._build_3d_map()

    def _process_rgbd_images(self) -> None:
        """Traite des images RGB-D pour générer une cartographie 3D."""
        logger.info(
            f"Traitement des images RGB-D: {self.args.image_folder} et {self.args.depth_folder}")
        # On a déjà les images RGB et profondeur, on peut directement passer à la cartographie
        self._build_3d_map()

    def _build_3d_map(self) -> None:
        """
        Exécute RTAB-Map via Docker pour générer la cartographie 3D.
        
        Cette fonction monte les volumes nécessaires dans le conteneur Docker
        et lance le script de cartographie.
        """
        logger.info(
            "Démarrage de la génération de la cartographie 3D avec RTAB-Map")

        try:
            # Construction de la commande Docker
            command = self._build_docker_command()

            # Journalisation de la commande (en masquant les chemins complets pour la sécurité)
            log_command = ' '.join(
                [c.split('/')[-1] if '/' in c else c for c in command])
            logger.debug(f"Exécution de la commande Docker: {log_command}")

            # Exécution de la commande avec affichage en temps réel
            logger.info(
                "Exécution de la commande Docker (affichage en temps réel) :")
            print("\n" + "="*80)
            print("DÉBUT DE L'EXÉCUTION DOCKER - OUTPUT EN TEMPS RÉEL")
            print("="*80 + "\n")

            # Exécution sans capture des flux pour affichage en temps réel dans la console
            result = subprocess.run(command, check=False)

            print("\n" + "="*80)
            print("FIN DE L'EXÉCUTION DOCKER")
            print("="*80 + "\n")

            # Vérification du résultat
            if result.returncode != 0:
                logger.error(
                    f"Erreur lors de l'exécution de Docker (code de retour: {result.returncode})")
                raise CartographieError(
                    f"Échec de la cartographie 3D avec code de retour {result.returncode}")
            else:
                logger.info("Cartographie 3D terminée avec succès")

                # Vérification de la génération des fichiers de sortie
                self._verify_outputs()

        except subprocess.SubprocessError as e:
            logger.error(
                f"Erreur lors de l'exécution du processus Docker: {e}")
            raise CartographieError(f"Échec de l'exécution Docker: {e}")

    def _build_docker_command(self) -> List[str]:
        """
        Construit la commande Docker avec les volumes montés appropriés.
        
        Returns:
            Liste des éléments de la commande Docker
        """
        # On n'utilise pas sudo car Docker est configuré pour s'exécuter sans sudo
        command = [
            "docker", "run", "--rm", "-it",
            "-v", f"{self.args.image_folder}:/rtabmap_ws/rgb_sync_docker",
            "-v", f"{self.args.depth_folder}:/rtabmap_ws/depth_sync_docker",
            "-v", f"{self.args.rgb_timestamps}:/rtabmap_ws/img_timestamps.csv",
            "-v", f"{self.args.depth_timestamps}:/rtabmap_ws/depth_timestamps.csv",
            "-v", f"{self.args.output_folder}:/rtabmap_ws/output",
            "-v", f"{self.args.calibration_file}:/rtabmap_ws/rtabmap_calib.yaml",
            "rtabmap_ubuntu20"
        ]
        return command

    def _verify_outputs(self) -> None:
        """Vérifie que les fichiers de sortie attendus ont été générés."""
        expected_files = ["pointcloud.ply", "rtabmap.db"]
        for file in expected_files:
            file_path = os.path.join(self.args.output_folder, file)
            if os.path.exists(file_path):
                file_size = os.path.getsize(
                    file_path) / (1024 * 1024)  # taille en MB
                logger.info(f"Fichier généré: {file} ({file_size:.2f} MB)")
            else:
                logger.warning(f"Fichier attendu non trouvé: {file}")


def parse_arguments() -> argparse.Namespace:
    """
    Parse les arguments de ligne de commande.
    
    Returns:
        Namespace contenant les arguments parsés
    """
    parser = argparse.ArgumentParser(
        description="Cartographie 3D avec RTAB-Map",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Paramètres d'entrée
    parser.add_argument("--image_folder", type=str,
                        default=os.path.join(
                            PROJECT_ROOT, "data/dataset/deer_walk/images"),
                        help="Dossier contenant les images RGB ou chemin vers le fichier vidéo")
    parser.add_argument("--depth_folder", type=str,
                        default=os.path.join(
                            PROJECT_ROOT, "data/dataset/deer_walk/depth"),
                        help="Dossier contenant les images avec profondeur")
    parser.add_argument("--calibration_file", type=str,
                        default=os.path.join(
                            PROJECT_ROOT, "data/dataset/deer_walk/rtabmap_calib.yaml"),
                        help="Chemin vers le fichier de calibration")
    parser.add_argument("--rgb_timestamps", type=str,
                        default=os.path.join(
                            PROJECT_ROOT, "data/dataset/deer_walk/img_timestamps.csv"),
                        help="Chemin vers le fichier de timestamps RGB")
    parser.add_argument("--depth_timestamps", type=str,
                        default=os.path.join(
                            PROJECT_ROOT, "data/dataset/deer_walk/depth_timestamps.csv"),
                        help="Chemin vers le fichier de timestamps profondeur")
    parser.add_argument("--output_folder", type=str,
                        default=os.path.join(PROJECT_ROOT, "output"),
                        help="Dossier de sortie")

    # Paramètres de traitement
    parser.add_argument("--source", type=str,
                        choices=["image", "image_with_depth", "video"],
                        default="image_with_depth",
                        help="Source à utiliser (image: RGB sans profondeur, image_with_depth: RGB-D, video: vidéo)")
    parser.add_argument("--frequence", type=int, default=20,
                        help="Fréquence d'images à extraire (Hz) pour un flux vidéo")

    # Paramètres avancés
    parser.add_argument("--debug", action="store_true",
                        help="Active les messages de débogage détaillés")
    parser.add_argument("--export_format", type=str,
                        choices=["ply", "obj", "both"],
                        default="ply",
                        help="Format d'exportation du nuage de points")

    return parser.parse_args()


def main():
    """Point d'entrée principal du programme."""
    try:
        # Parsing des arguments
        args = parse_arguments()

        # Configuration du niveau de log
        if args.debug:
            logger.setLevel(logging.DEBUG)
            logger.debug("Mode debug activé")

        # Affichage des paramètres utilisés
        logger.info(f"Source: {args.source}")
        logger.info(f"Dossier d'images: {args.image_folder}")

        # Initialisation et exécution du processus de cartographie
        mapper = RTAB3DMapper(args)
        mapper.process_source()

        logger.info("Traitement terminé avec succès")
        return 0

    except CartographieError as e:
        logger.error(f"Erreur de cartographie: {e}")
        return 1
    except KeyboardInterrupt:
        logger.info("Interruption par l'utilisateur")
        return 130
    except Exception as e:
        logger.exception(f"Erreur inattendue: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
