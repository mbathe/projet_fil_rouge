#!/usr/bin/env python3
"""
Cartographie 3D RTAB-Map - Module principal

Ce module contient le point d'entrée principal de l'application de cartographie 3D
utilisant RTAB-Map via Docker, avec support pour différentes sources d'entrée (RGB, RGB-D, vidéo).

Author: Paul
Date: 2025-05-29
Version: 2.0.0
"""

from utils import create_or_clean_dir
from utils import extract_frames
import json
import logging
import os
import subprocess
import sys
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser as ArgParser, Namespace
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Union
from datetime import datetime

# Configuration des constantes
SCRIPT_DIR = Path(__file__).parent.absolute()
PROJECT_ROOT = SCRIPT_DIR.parent
LOG_MAIN_DIR = PROJECT_ROOT / "logs" / "main"
LOG_RTABMAP_DIR = PROJECT_ROOT / "logs" / "rtabmap"
RTABMAP_CONFIG_FILE = PROJECT_ROOT / "src" / "rtabmap_config.json"
RTABMAP_PARAMS_DIR = PROJECT_ROOT / "src" / "rtabmap" / "params"
OUTPUT_DEPTH_IMAGES = PROJECT_ROOT / "output" / "depth"
OUTPUT_IMAGES = PROJECT_ROOT / "output" / "images"
OUTPUT_RTABMAP = PROJECT_ROOT / "output" / "rtabmap"

# Création du dossier de logs et génération du nom de fichier avec timestamp
LOG_MAIN_DIR.mkdir(parents=True, exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
LOG_FILE = LOG_MAIN_DIR / f"cartographie3d_{timestamp}.log"


# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE, encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("cartographie3d")


class SourceType(Enum):
    """Types de sources d'entrée supportées."""
    IMAGE = "image"
    IMAGE_WITH_DEPTH = "image_with_depth"
    VIDEO = "video"


class ExportFormat(Enum):
    """Formats d'exportation supportés."""
    CLOUD = "--cloud"
    MESH = "--mesh"


@dataclass
class RTABMapConfig:
    """Configuration pour RTAB-Map."""
    reprocess: bool = True
    export_format: ExportFormat = ExportFormat.CLOUD

    def to_dict(self) -> Dict[str, Union[bool, str]]:
        """Convertit la configuration en dictionnaire."""
        return {
            "reprocess": self.reprocess,
            "export_format": self.export_format.value
        }


@dataclass
class CartographyPaths:
    """Conteneur pour tous les chemins utilisés dans la cartographie."""
    image_folder: Path
    depth_folder: Path
    calibration_file: Path
    rgb_timestamps: Path
    output_folder: Path
    depth_timestamps: Path
    video_file: Path

    def __post_init__(self) -> None:
        """Convertit les chemins en objets Path si nécessaire."""
        for field_name, field_value in self.__dict__.items():
            if isinstance(field_value, str):
                setattr(self, field_name, Path(field_value))


class CartographieError(Exception):
    """Exception personnalisée pour les erreurs liées à la cartographie 3D."""
    pass


class PathValidator:
    """Utilitaire pour la validation des chemins."""

    @staticmethod
    def validate_and_create_paths(paths: CartographyPaths, source_type: SourceType) -> None:
        """
        Valide et crée les chemins nécessaires.
        
        Args:
            paths: Conteneur des chemins à valider
            source_type: Type de source pour déterminer les validations nécessaires
            
        Raises:
            CartographieError: Si un chemin requis est invalide
        """
        if source_type == SourceType.IMAGE:
            PathValidator._validate_image_source(paths)
        elif source_type == SourceType.IMAGE_WITH_DEPTH:
            PathValidator._validate_image_with_depth_source(paths)
        elif source_type == SourceType.VIDEO:
            PathValidator._validate_video_source(paths)

        try:
            paths.output_folder.mkdir(parents=True, exist_ok=True)
            logger.debug(
                f"Dossier de sortie créé/vérifié: {paths.output_folder}")
        except OSError as e:
            raise CartographieError(
                f"Impossible de créer le dossier de sortie {paths.output_folder}: {e}")

    @staticmethod
    def _validate_image_source(paths: CartographyPaths) -> None:
        """Valide les chemins pour une source de type IMAGE."""
        if not paths.calibration_file.exists():
            raise CartographieError(
                f"Fichier de calibration manquant: {paths.calibration_file}")

        if not paths.image_folder.exists():
            raise CartographieError(
                f"Dossier d'images inexistant: {paths.image_folder}")

        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
        image_files = [f for f in paths.image_folder.iterdir()
                       if f.is_file() and f.suffix.lower() in image_extensions]

        if not image_files:
            raise CartographieError(
                f"Aucune image trouvée dans le dossier: {paths.image_folder}")

        logger.info(
            f"Validation réussie: {len(image_files)} images trouvées dans {paths.image_folder}")

        if not paths.rgb_timestamps.exists():
            logger.warning(
                f"Fichier de timestamps RGB manquant: {paths.rgb_timestamps}")

    @staticmethod
    def _validate_image_with_depth_source(paths: CartographyPaths) -> None:
        """Valide les chemins pour une source de type IMAGE_WITH_DEPTH."""
        if not paths.calibration_file.exists():
            raise CartographieError(
                f"Fichier de calibration manquant: {paths.calibration_file}")

        if not paths.image_folder.exists():
            raise CartographieError(
                f"Dossier d'images RGB inexistant: {paths.image_folder}")

        if not paths.depth_folder.exists():
            raise CartographieError(
                f"Dossier de profondeur inexistant: {paths.depth_folder}")

        # Vérification que les dossiers contiennent des fichiers
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
        depth_extensions = {'.png', '.tiff', '.tif', '.exr', '.pfm'}

        rgb_files = [f for f in paths.image_folder.iterdir()
                     if f.is_file() and f.suffix.lower() in image_extensions]
        depth_files = [f for f in paths.depth_folder.iterdir()
                       if f.is_file() and f.suffix.lower() in depth_extensions]

        if not rgb_files:
            raise CartographieError(
                f"Aucune image RGB trouvée dans le dossier: {paths.image_folder}")

        if not depth_files:
            raise CartographieError(
                f"Aucune image de profondeur trouvée dans le dossier: {paths.depth_folder}")

        logger.info(
            f"Validation réussie: {len(rgb_files)} images RGB et {len(depth_files)} images de profondeur trouvées")

        # Validation des fichiers de timestamps
        if not paths.rgb_timestamps.exists():
            logger.warning(
                f"Fichier de timestamps RGB manquant: {paths.rgb_timestamps}")

        if not paths.depth_timestamps.exists():
            logger.warning(
                f"Fichier de timestamps de profondeur manquant: {paths.depth_timestamps}")

    @staticmethod
    def _validate_video_source(paths: CartographyPaths) -> None:
        """Valide les chemins pour une source de type VIDEO."""
        if not paths.calibration_file.exists():
            raise CartographieError(
                f"Fichier de calibration manquant: {paths.calibration_file}")

        if not paths.video_file.exists():
            raise CartographieError(
                f"Fichier vidéo inexistant: {paths.video_file}")

        video_extensions = {'.mp4', '.avi', '.mov',
                            '.mkv', '.wmv', '.flv', '.webm'}
        if paths.video_file.suffix.lower() not in video_extensions:
            logger.warning(
                f"Extension de fichier vidéo non standard: {paths.video_file.suffix}")

        logger.info(
            f"Validation réussie: fichier vidéo trouvé - {paths.video_file}")

        try:
            paths.image_folder.mkdir(parents=True, exist_ok=True)
            logger.debug(
                f"Dossier d'images créé pour l'extraction: {paths.image_folder}")
        except OSError as e:
            raise CartographieError(
                f"Impossible de créer le dossier d'images {paths.image_folder}: {e}")


class DockerCommandBuilder:
    """Constructeur de commandes Docker pour RTAB-Map."""

    def __init__(self, paths: CartographyPaths, config: RTABMapConfig):
        """
        Initialise le constructeur de commandes Docker.
        
        Args:
            paths: Chemins pour les volumes Docker
            config: Configuration RTAB-Map
        """
        self.paths = paths
        self.config = config

    def build_command(self) -> List[str]:
        """
        Construit la commande Docker complète.
        
        Returns:
            Liste des éléments de la commande Docker
            
        Raises:
            CartographieError: Si les fichiers de paramètres sont manquants
        """
        try:
            volumes = self._build_volume_mounts()
            command = ["docker", "run", "--rm", "-it"] + \
                volumes + ["rtabmap_ubuntu20"]

            logger.debug(
                f"Commande Docker construite avec {len(volumes)//2} volumes")
            return command

        except Exception as e:
            raise CartographieError(
                f"Erreur lors de la construction de la commande Docker: {e}")

    def _build_volume_mounts(self) -> List[str]:
        """
        Construit la liste des montages de volumes Docker.
        
        Returns:
            Liste des arguments de montage de volumes
        """
        volume_mappings = {
            self.paths.image_folder: "/rtabmap_ws/rgb_sync_docker",
            self.paths.depth_folder: "/rtabmap_ws/depth_sync_docker",
            self.paths.rgb_timestamps: "/rtabmap_ws/img_timestamps.csv",
            self.paths.depth_timestamps: "/rtabmap_ws/depth_timestamps.csv",
            self.paths.output_folder: "/rtabmap_ws/output/rtabmap",
            self.paths.calibration_file: "/rtabmap_ws/rtabmap_calib.yaml",
            RTABMAP_CONFIG_FILE: "/rtabmap_ws/config.json",
            LOG_RTABMAP_DIR: "/rtabmap_ws/logs",
        }

        # Ajout des fichiers de paramètres
        param_files = {
            "export_params.json": "/rtabmap_ws/export_params.json",
            "db_params.json": "/rtabmap_ws/db_params.json",
            "reprocess_params.json": "/rtabmap_ws/reprocess_params.json",
        }

        for param_file, container_path in param_files.items():
            local_path = RTABMAP_PARAMS_DIR / param_file
            volume_mappings[local_path] = container_path

        volumes = []
        for local_path, container_path in volume_mappings.items():
            volumes.extend(["-v", f"{local_path}:{container_path}"])

        return volumes


class OutputVerifier:
    """Vérificateur de fichiers de sortie."""

    EXPECTED_FILES = ["pointcloud.ply", "rtabmap.db"]

    @classmethod
    def verify_outputs(cls, output_folder: Path) -> Dict[str, bool]:
        """
        Vérifie que les fichiers de sortie attendus ont été générés.
        
        Args:
            output_folder: Dossier de sortie à vérifier
            
        Returns:
            Dictionnaire indiquant la présence de chaque fichier attendu
        """
        results = {}

        for file_name in cls.EXPECTED_FILES:
            file_path = output_folder / file_name
            if file_path.exists():
                file_size_mb = file_path.stat().st_size / (1024 * 1024)
                logger.info(
                    f"Fichier généré: {file_name} ({file_size_mb:.2f} MB)")
                results[file_name] = True
            else:
                logger.warning(f"Fichier attendu non trouvé: {file_name}")
                results[file_name] = False

        return results


class ConfigurationManager:
    """Gestionnaire de configuration."""

    @staticmethod
    def write_config(config: RTABMapConfig) -> None:
        """
        Écrit la configuration RTAB-Map dans un fichier JSON.
        
        Args:
            config: Configuration à écrire
            
        Raises:
            CartographieError: Si l'écriture échoue
        """
        try:
            RTABMAP_CONFIG_FILE.parent.mkdir(parents=True, exist_ok=True)

            with open(RTABMAP_CONFIG_FILE, 'w', encoding='utf-8') as f:
                json.dump(config.to_dict(), f, indent=2)

            logger.debug(f"Configuration écrite dans: {RTABMAP_CONFIG_FILE}")

        except (OSError, json.JSONEncodeError) as e:
            raise CartographieError(
                f"Impossible d'écrire la configuration: {e}")


class SourceProcessor:
    """Processeur pour différents types de sources d'entrée."""

    def __init__(self, paths: CartographyPaths, frequency: int = 20):
        """
        Initialise le processeur de sources.
        
        Args:
            paths: Chemins pour le traitement
            frequency: Fréquence pour l'extraction vidéo (Hz)
        """
        self.paths = paths
        self.frequency = frequency

    def process_video(self) -> None:
        """
        Traite une source vidéo pour générer une cartographie 3D.
        
        Note:
            Cette méthode nécessite l'implémentation de l'extraction de frames
            et de l'estimation de profondeur.
        """
        logger.info(
            f"Traitement de la vidéo: {self.paths.image_folder} à {self.frequency}Hz")
        # TODO: Implémenter l'extraction des frames et estimation de profondeur
        os.makedirs(OUTPUT_IMAGES, exist_ok=True)
        try:
            extract_frames(
                video_path=self.paths.video_file,
                output_dir=OUTPUT_IMAGES,
                frequency=self.frequency
            )
            logger.info(
                f"Frames extraites dans: {self.paths.image_folder}")
            self.paths.image_folder = OUTPUT_IMAGES
            self.process_rgb_images()
        except Exception as e:
            logger.error(f"Erreur lors de l'extraction des frames: {e}")
            raise CartographieError(
                f"Échec de l'extraction des frames de la vidéo: {e}")


    def process_rgb_images(self) -> None:
        """
        Traite des images RGB pour générer une cartographie 3D.
        
        Note:
            Cette méthode nécessite l'implémentation de l'estimation de profondeur.
        """
        logger.info(f"Traitement des images RGB: {self.paths.image_folder}")

        try:
            # Import conditionnel pour éviter les erreurs si le module n'est pas disponible
            from depth.generate_depth import generate_depth_maps
            os.makedirs(OUTPUT_DEPTH_IMAGES, exist_ok=True)
            generate_depth_maps(
                image_folder=self.paths.image_folder,
                output_folder=OUTPUT_DEPTH_IMAGES,
                image_extensions=(".png", ".jpg", ".jpeg"),
                output_extension=".tiff"
            )

            # TODO: Appeler generate_depth_maps avec les paramètres appropriés
            logger.warning("Estimation de profondeur non encore implémentée")

        except ImportError as e:
            logger.error(
                f"Module d'estimation de profondeur non disponible: {e}")
            raise CartographieError(
                "Impossible de traiter les images RGB sans module de profondeur")

    def process_rgbd_images(self) -> None:
        """
        Traite des images RGB-D pour générer une cartographie 3D.
        
        Cette méthode peut directement procéder à la cartographie car
        les images de profondeur sont déjà disponibles.
        """
        logger.info(
            f"Traitement des images RGB-D: {self.paths.image_folder} et {self.paths.depth_folder}")

        # Validation de la présence des images de profondeur
        if not self.paths.depth_folder.exists() or not any(self.paths.depth_folder.iterdir()):
            raise CartographieError("Dossier de profondeur vide ou inexistant")


class RTAB3DMapper:
    """Classe principale pour la génération de cartographie 3D utilisant RTAB-Map."""

    def __init__(self, args: Namespace):
        """
        Initialise l'instance de cartographie 3D.
        
        Args:
            args: Arguments de ligne de commande parsés
            
        Raises:
            CartographieError: Si les arguments sont invalides
        """
        try:
            self.source_type = SourceType(args.source)
            self.paths = CartographyPaths(
                image_folder=args.image_folder,
                depth_folder=args.depth_folder,
                calibration_file=args.calibration_file,
                rgb_timestamps=args.rgb_timestamps,
                depth_timestamps=args.depth_timestamps,
                output_folder=OUTPUT_RTABMAP,
                video_file=args.video_file
            )
            self.config = RTABMapConfig(
                reprocess=args.reprocess,
                export_format=ExportFormat(args.export_format)
            )
            self.frequency = args.frequence

            # Validation des chemins selon le type de source
            PathValidator.validate_and_create_paths(
                self.paths, self.source_type)

            # Initialisation des composants
            self.source_processor = SourceProcessor(self.paths, self.frequency)
            self.docker_builder = DockerCommandBuilder(self.paths, self.config)

            logger.info(
                f"Initialisation réussie avec source: {self.source_type.value}")

        except (ValueError, TypeError) as e:
            raise CartographieError(f"Arguments invalides: {e}")

    def process_source(self) -> None:
        """
        Traite la source d'entrée selon le type spécifié.
        
        Raises:
            CartographieError: Si le traitement échoue
        """
        try:
            if self.source_type == SourceType.VIDEO:
                self.source_processor.process_video()
            elif self.source_type == SourceType.IMAGE:
                self.source_processor.process_rgb_images()
            elif self.source_type == SourceType.IMAGE_WITH_DEPTH:
                self.source_processor.process_rgbd_images()

            # Génération de la cartographie 3D
            self._build_3d_map()

        except Exception as e:
            logger.error(
                f"Erreur lors du traitement de la source {self.source_type.value}: {e}")
            raise CartographieError(f"Échec du traitement: {e}")

    def _build_3d_map(self) -> None:
        """
        Exécute RTAB-Map via Docker pour générer la cartographie 3D.
        
        Raises:
            CartographieError: Si l'exécution Docker échoue
        """
        logger.info(
            "Démarrage de la génération de la cartographie 3D avec RTAB-Map")

        try:
            # Écriture de la configuration
            ConfigurationManager.write_config(self.config)

            command = self.docker_builder.build_command()
            self._execute_docker_command(command)

            verification_results = OutputVerifier.verify_outputs(
                self.paths.output_folder)

            if not any(verification_results.values()):
                raise CartographieError("Aucun fichier de sortie généré")

            logger.info("Cartographie 3D terminée avec succès")

        except subprocess.SubprocessError as e:
            raise CartographieError(f"Erreur d'exécution Docker: {e}")
        except Exception as e:
            raise CartographieError(
                f"Erreur lors de la génération de la carte 3D: {e}")

    def _execute_docker_command(self, command: List[str]) -> None:
        """
        Exécute la commande Docker avec gestion d'erreurs.
        
        Args:
            command: Commande Docker à exécuter
            
        Raises:
            subprocess.SubprocessError: Si l'exécution échoue
        """
        # Masquage des chemins sensibles pour les logs
        log_command = ' '.join(
            [c.split('/')[-1] if '/' in c else c for c in command])
        logger.debug(f"Exécution de la commande Docker: {log_command}")

        print("\n" + "="*80)
        print("DÉBUT DE L'EXÉCUTION DOCKER - OUTPUT EN TEMPS RÉEL")
        print("="*80 + "\n")

        try:
            result = subprocess.run(
                command, check=False, timeout=3600)  # Timeout de 1 heure

            print("\n" + "="*80)
            print("FIN DE L'EXÉCUTION DOCKER")
            print("="*80 + "\n")

            if result.returncode != 0:
                raise subprocess.CalledProcessError(result.returncode, command)

        except subprocess.TimeoutExpired:
            logger.error("Timeout de l'exécution Docker (1 heure)")
            raise
        except subprocess.CalledProcessError as e:
            logger.error(f"Erreur d'exécution Docker (code: {e.returncode})")
            raise


class CommandLineParser:
    """Parser d'arguments en ligne de commande."""

    @staticmethod
    def create_parser() -> ArgParser:
        """
        Crée et configure le parser d'arguments.
        
        Returns:
            Parser d'arguments configuré
        """
        parser = ArgParser(
            description="Cartographie 3D avec RTAB-Map",
            formatter_class=ArgumentDefaultsHelpFormatter
        )

        # Groupe des paramètres d'entrée
        input_group = parser.add_argument_group("Paramètres d'entrée")
        input_group.add_argument(
            "--image_folder", type=str,
            default=str(PROJECT_ROOT / "data/datasets/deer_walk/cam0/data"),
            help="Dossier contenant les images RGB"
        )
        input_group.add_argument(
            "--depth_folder", type=str,
            default=str(PROJECT_ROOT / "data/datasets/deer_walk/depth0/data"),
            help="Dossier contenant les images avec profondeur"
        )

        input_group.add_argument(
            "--video_file", type=str,
            default=str(PROJECT_ROOT / "data/datasets/Dear_Walking.mp4"),
            help="Chemin vers le fichier vidéo"
        )

        input_group.add_argument(
            "--calibration_file", type=str,
            default=str(PROJECT_ROOT /
                        "data/datasets/deer_walk/cam0/rtabmap_calib.yaml"),
            help="Chemin vers le fichier de calibration"
        )

        input_group.add_argument(
            "--rgb_timestamps", type=str,
            default=str(PROJECT_ROOT /
                        "data/datasets/deer_walk/cam0/data.csv"),
            help="Chemin vers le fichier de timestamps RGB"
        )
        input_group.add_argument(
            "--depth_timestamps", type=str,
            default=str(PROJECT_ROOT /
                        "data/datasets/deer_walk/depth0/data.csv"),
            help="Chemin vers le fichier de timestamps profondeur"
        )

        # Groupe des paramètres de traitement
        processing_group = parser.add_argument_group(
            "Paramètres de traitement")
        processing_group.add_argument(
            "--source", type=str,
            choices=[e.value for e in SourceType],
            default=SourceType.IMAGE_WITH_DEPTH.value,
            help="Source à utiliser"
        )
        processing_group.add_argument(
            "--frequence", type=int, default=20,
            help="Fréquence d'images à extraire (Hz) pour un flux vidéo"
        )
        processing_group.add_argument(
            "--reprocess", type=bool, default=True,
            help="Retraiter les données existantes"
        )
        processing_group.add_argument(
            "--export_format", type=str,
            choices=[e.value for e in ExportFormat],
            default=ExportFormat.CLOUD.value,
            help="Format d'exportation du nuage de points"
        )

        # Groupe des paramètres de sortie
        output_group = parser.add_argument_group("Paramètres de sortie")
        output_group.add_argument(
            "--debug", action="store_true",
            help="Active les messages de débogage détaillés"
        )

        return parser

    @classmethod
    def parse_arguments(cls) -> Namespace:
        """
        Parse les arguments de ligne de commande.
        
        Returns:
            Namespace contenant les arguments parsés
        """
        parser = cls.create_parser()
        return parser.parse_args()


def setup_logging(debug: bool = False) -> None:
    """
    Configure le système de logging.
    
    Args:
        debug: Active le mode debug si True
    """
    if debug:
        logger.setLevel(logging.DEBUG)
        logger.debug("Mode debug activé")

    logger.info("Système de logging initialisé")
    logger.info(f"Fichier de log: {LOG_FILE}")


def main() -> int:
    """
    Point d'entrée principal du programme.
    
    Returns:
        Code de retour (0 pour succès, autre pour erreur)
    """
    try:
        # Parsing des arguments
        args = CommandLineParser.parse_arguments()
        
        args.depth_image_folder = args.depth_folder if args.source == SourceType.IMAGE_WITH_DEPTH else OUTPUT_DEPTH_IMAGES
        if args.source == SourceType.VIDEO:
            args.image_folder = OUTPUT_IMAGES
            args.depth_folder = OUTPUT_DEPTH_IMAGES
            create_or_clean_dir(OUTPUT_RTABMAP)
            create_or_clean_dir(OUTPUT_DEPTH_IMAGES)


        # Configuration du logging
        setup_logging(args.debug)

        # Log de démarrage avec informations sur l'exécution
        logger.info("="*60)
        logger.info("DÉMARRAGE DE LA CARTOGRAPHIE 3D RTAB-MAP")
        logger.info(f"Timestamp d'exécution: {timestamp}")
        logger.info(f"Fichier de log: {LOG_FILE}")
        logger.info("="*60)

        # Création du dossier de sortie principal
        output_dir = Path(OUTPUT_RTABMAP)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Affichage des paramètres principaux
        logger.info(f"Source: {args.source}")
        logger.info(f"Dossier d'images: {OUTPUT_RTABMAP}")
        logger.info(f"Dossier de sortie: {OUTPUT_RTABMAP}")

        # Initialisation et exécution du processus de cartographie
        mapper = RTAB3DMapper(args)
        mapper.process_source()

        logger.info("Traitement terminé avec succès")
        logger.info("="*60)
        logger.info("FIN DE L'EXÉCUTION")
        logger.info("="*60)
        return 0

    except CartographieError as e:
        logger.error(f"Erreur de cartographie: {e}")
        logger.error("="*60)
        logger.error("ÉCHEC DE L'EXÉCUTION")
        logger.error("="*60)
        return 1
    except KeyboardInterrupt:
        logger.info("Interruption par l'utilisateur")
        logger.info("="*60)
        logger.info("ARRÊT PAR L'UTILISATEUR")
        logger.info("="*60)
        return 130
    except Exception as e:
        logger.exception(f"Erreur inattendue: {e}")
        logger.error("="*60)
        logger.error("ERREUR INATTENDUE")
        logger.error("="*60)
        return 1


if __name__ == "__main__":
    sys.exit(main())
