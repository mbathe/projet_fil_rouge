#!/usr/bin/env python3
"""
Script pour extraire des images d'une vidéo à une fréquence donnée
Compatible avec RTAB-Map - les images sont nommées avec timestamp
"""

import logging
import json
from typing import Dict, Optional, Any
from dataclasses import dataclass
import platform
import numpy as np
import cv2
import argparse
import sys
from pathlib import Path
import shutil
import glob
from tqdm import tqdm
import shutil
import json
import argparse

import os

# Répertoire du script
script_dir = os.path.dirname(os.path.abspath(__file__))



SHOW_PROGRESS = True  # Set to False to disable progress bars

WORKSPACE_ROOT = os.getcwd()
DEFAULT_FPS = 20.0
DEFAULT_START_TIME = 1400000000.0








def get_os_version():
    """Retourne le système d'exploitation et sa version"""
    os_name = platform.system()
    
    if os_name == "Linux":
        # Vérifier si c'est Ubuntu
        if os.path.exists("/etc/os-release"):
            with open("/etc/os-release", "r") as f:
                for line in f:
                    if line.startswith("NAME="):
                        distro = line.split("=")[1].strip().strip('"')
                    elif line.startswith("VERSION_ID="):
                        version = line.split("=")[1].strip().strip('"')
                        return f"{distro}", f"{version}"
        return "Linux", f"(version inconnue)"
    
    elif os_name == "Windows":
        return "Windows", f"{platform.release()}"
    
    elif os_name == "Darwin":
        return "macOS", f"{platform.mac_ver()[0]}"
    
    else:
        return "{os_name}", f"(version inconnue)"








class MultiPlatformPathManager:
    def __init__(self, host_source_path, base_local_dir: str =Path(script_dir)/"rtabmap"/"script/rtabmap_ws"):
        self.is_linux = platform.system() == "Linux"
        self.base_local_dir = Path(base_local_dir)

        # Configuration des chemins Docker (pour Linux)
        self.docker_paths = {
            "image_folder": "/rtabmap_ws/rgb_sync_docker",
            "depth_folder": "/rtabmap_ws/depth_sync_docker",
            "output_folder": "/rtabmap_ws/output/rtabmap",
            "calibration_file": "/rtabmap_ws/rtabmap_calib.yaml",
            "config_file": "/rtabmap_ws/config.json",
            "log_dir": "/rtabmap_ws/logs"
        }

        # Configuration des chemins locaux (pour Windows/macOS)
        self.local_paths = {
            "image_folder": self.base_local_dir / "rgb_sync_docker",
            "depth_folder": self.base_local_dir / "depth_sync_docker",
            "output_folder": self.base_local_dir / "output" / "rtabmap",
            "calibration_file": self.base_local_dir / "rtabmap_calib.yaml",
            "config_file": self.base_local_dir / "config.json",
            "log_dir": self.base_local_dir / "logs",
            "db_params": self.base_local_dir / "db_params.json",
            "export_params": self.base_local_dir / "export_params.json",
            "reprocess_params": self.base_local_dir / "reprocess_params.json"
        }

        # Chemins sources sur la machine hôte (à adapter selon votre structure)
        self.host_source_paths = host_source_path

    def get_paths(self) -> Dict[str, Any]:
        """Retourne les chemins appropriés selon la plateforme"""
        if self.is_linux:
            return self.docker_paths
        else:
            return {key: str(path) for key, path in self.local_paths.items()}

    def setup_local_environment(self):
        """Configure l'environnement local pour Windows/macOS"""
        if self.is_linux:
            print("Environnement Linux détecté - utilisation de Docker")
            return

        print("Configuration de l'environnement local pour Windows/macOS...")

        # Créer la structure de dossiers
        self.create_local_directories()

        # Copier les fichiers nécessaires
        self.copy_files_to_local()

        print(f"Environnement local configuré dans : {self.base_local_dir}")

    def create_local_directories(self):
        """Crée tous les dossiers nécessaires en local après suppression des existants"""
        directories_to_create = [
            self.local_paths["image_folder"],
            self.local_paths["depth_folder"],
            self.local_paths["output_folder"],
            self.local_paths["log_dir"]
        ]

        for directory in directories_to_create:
            if directory.exists():
                shutil.rmtree(directory)  # Supprime le dossier existant
                print(f"Dossier supprimé : {directory}")
            directory.mkdir(parents=True, exist_ok=True)
            print(f"Dossier créé : {directory}")

    def copy_files_to_local(self):
        """Copie les fichiers depuis les sources vers l'environnement local après suppression des existants"""
        files_to_copy = [
            ("calibration_file", "file"),
            ("config_file", "file"),
            ("db_params", "file"),
            ("export_params", "file"),
            ("reprocess_params", "file"),
            ("image_folder", "directory"),
            ("depth_folder", "directory")
        ]

        for key, file_type in files_to_copy:
            source = Path(self.host_source_paths.get(key, ""))
            destination = self.local_paths[key]

            if destination.exists():
                if file_type == "file":
                    destination.unlink()  # Supprime le fichier existant
                    print(f"Fichier supprimé : {destination}")
                elif file_type == "directory":
                    shutil.rmtree(destination)  # Supprime le dossier existant
                    print(f"Dossier supprimé : {destination}")

            if not source.exists():
                print(f" Source non trouvée : {source}")
                continue

            try:
                if file_type == "file":
                    shutil.copy2(source, destination)
                    print(f"Fichier copié : {source} -> {destination}")
                elif file_type == "directory":
                    shutil.copytree(source, destination)
                    print(f"Dossier copié : {source} -> {destination}")
            except Exception as e:
                print(f"Erreur lors de la copie {source} -> {destination}: {e}")

    def get_docker_volume_mounts(self) -> list:
        """Génère les arguments de montage de volumes pour Docker"""
        if not self.is_linux:
            return []

        mounts = []
        for key, docker_path in self.docker_paths.items():
            host_path = self.host_source_paths.get(key)
            if host_path and Path(host_path).exists():
                # Pour les dossiers de sortie et logs, créer s'ils n'existent pas
                if key in ["output_folder", "log_dir"]:
                    Path(host_path).mkdir(parents=True, exist_ok=True)
                mounts.append(f"-v {os.path.abspath(host_path)}:{docker_path}")

        return mounts
    
    def delete_base_dir(self):
        """Supprime le répertoire de base local"""
        if self.base_local_dir.exists():
            shutil.rmtree(self.base_local_dir)
        else:
            print(f"Aucun répertoire à supprimer : {self.base_local_dir}")


def create_or_clean_dir(path):
    shutil.rmtree(path, ignore_errors=True)


#!/usr/bin/env python3
"""
Script pour extraire des images d'une vidéo à une fréquence donnée
Compatible avec RTAB-Map - images corrigées d'orientation et optimisées pour odométrie
"""


@dataclass
class RTABMAPPaths:
    """Classe pour gérer les chemins RTABMAP selon le système d'exploitation"""

    def __init__(self):
        self.os_type = self._detect_os()
        self.base_path = self._get_base_path()

        # Configuration des chemins selon l'OS
        self._setup_paths()

    def _detect_os(self) -> str:
        """Détecte le système d'exploitation"""
        system = platform.system().lower()
        if system == "windows":
            return "windows"
        elif system == "linux":
            return "linux"
        elif system == "darwin":
            return "macos"
        else:
            return "unknown"

    def _get_base_path(self) -> str:
        """Retourne le chemin de base selon l'OS"""
        if self.os_type == "linux":
            return "/rtabmap_ws"
        else:  # Windows/macOS - utilise le répertoire courant
            current_dir = os.getcwd()
            return os.path.join(current_dir, "rtabmap_ws")

    def _setup_paths(self):
        """Configure tous les chemins"""
        if self.os_type == "linux":
            # Linux - chemins absolus originaux
            self.image_folder = "/rtabmap_ws/rgb_sync_docker"
            self.depth_folder = "/rtabmap_ws/depth_sync_docker"
            self.output_folder = "/rtabmap_ws/output/rtabmap"
            self.calibration_file = "/rtabmap_ws/rtabmap_calib.yaml"
            self.config_file = "/rtabmap_ws/config.json"
            self.log_dir = "/rtabmap_ws/logs"
        else:  # Windows/macOS - chemins relatifs au répertoire courant
            self.image_folder = os.path.join(self.base_path, "rgb_sync_docker")
            self.depth_folder = os.path.join(
                self.base_path, "depth_sync_docker")
            self.output_folder = os.path.join(
                self.base_path, "output", "rtabmap")
            self.calibration_file = os.path.join(
                self.base_path, "rtabmap_calib.yaml")
            self.config_file = os.path.join(self.base_path, "config.json")
            self.log_dir = os.path.join(self.base_path, "logs")

    def create_directories(self) -> Dict[str, bool]:
        """Crée tous les répertoires nécessaires"""
        directories = [
            self.image_folder,
            self.depth_folder,
            self.output_folder,
            self.log_dir,
            os.path.dirname(self.calibration_file),
            os.path.dirname(self.config_file)
        ]

        results = {}
        for directory in set(directories):  # Supprime les doublons
            try:
                Path(directory).mkdir(parents=True, exist_ok=True)
                results[directory] = True
                print(f"✓ Répertoire créé/vérifié: {directory}")
            except Exception as e:
                results[directory] = False
                print(f"✗ Erreur création répertoire {directory}: {e}")

        return results

    def check_paths_exist(self) -> Dict[str, bool]:
        """Vérifie l'existence de tous les chemins"""
        paths_to_check = {
            "image_folder": self.image_folder,
            "depth_folder": self.depth_folder,
            "output_folder": self.output_folder,
            "log_dir": self.log_dir,
            "calibration_file": self.calibration_file,
            "config_file": self.config_file
        }

        results = {}
        for name, path in paths_to_check.items():
            exists = os.path.exists(path)
            results[name] = exists
            status = "✓" if exists else "✗"
            print(f"{status} {name}: {path}")

        return results

    def get_all_paths(self) -> Dict[str, str]:
        """Retourne tous les chemins sous forme de dictionnaire"""
        return {
            "os_type": self.os_type,
            "base_path": self.base_path,
            "image_folder": self.image_folder,
            "depth_folder": self.depth_folder,
            "output_folder": self.output_folder,
            "calibration_file": self.calibration_file,
            "config_file": self.config_file,
            "log_dir": self.log_dir
        }

    def save_config(self, config_path: Optional[str] = None) -> bool:
        """Sauvegarde la configuration des chemins en JSON"""
        if config_path is None:
            config_path = self.config_file

        try:
            config_data = self.get_all_paths()
            with open(config_path, 'w', encoding='utf-8') as f:
                json.dump(config_data, f, indent=4, ensure_ascii=False)
            print(f"✓ Configuration sauvegardée: {config_path}")
            return True
        except Exception as e:
            print(f"✗ Erreur sauvegarde configuration: {e}")
            return False

    def load_config(self, config_path: Optional[str] = None) -> bool:
        """Charge la configuration depuis un fichier JSON"""
        if config_path is None:
            config_path = self.config_file

        try:
            if os.path.exists(config_path):
                with open(config_path, 'r', encoding='utf-8') as f:
                    config_data = json.load(f)

                # Mise à jour des attributs
                for key, value in config_data.items():
                    if hasattr(self, key):
                        setattr(self, key, value)

                print(f"✓ Configuration chargée: {config_path}")
                return True
            else:
                print(f"✗ Fichier de configuration non trouvé: {config_path}")
                return False
        except Exception as e:
            print(f"✗ Erreur chargement configuration: {e}")
            return False

    def setup_logging(self, log_level: str = "INFO") -> logging.Logger:
        """Configure le logging pour RTABMAP"""
        log_file = os.path.join(self.log_dir, "rtabmap.log")

        # Créer le répertoire de logs s'il n'existe pas
        Path(self.log_dir).mkdir(parents=True, exist_ok=True)

        # Configuration du logger
        logger = logging.getLogger("rtabmap")
        logger.setLevel(getattr(logging, log_level.upper()))

        # Handler pour fichier
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)

        # Handler pour console
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)

        # Format des logs
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)

        # Ajouter les handlers
        if not logger.handlers:  # Éviter les doublons
            logger.addHandler(file_handler)
            logger.addHandler(console_handler)

        logger.info(f"Logger configuré - OS: {self.os_type}")
        return logger

    def get_relative_paths(self, reference_path: str) -> Dict[str, str]:
        """Calcule les chemins relatifs par rapport à un répertoire de référence"""
        ref_path = Path(reference_path)
        paths = self.get_all_paths()

        relative_paths = {}
        for name, path in paths.items():
            if isinstance(path, str) and os.path.isabs(path):
                try:
                    rel_path = os.path.relpath(path, reference_path)
                    relative_paths[name] = rel_path
                except ValueError:
                    # Chemins sur des lecteurs différents (Windows)
                    relative_paths[name] = path
            else:
                relative_paths[name] = path

        return relative_paths


# Classe principale d'utilisation
class RTABMAPManager:
    """Gestionnaire principal pour RTABMAP avec gestion multi-OS"""

    def __init__(self):
        self.paths = RTABMAPPaths()
        self.logger = None

    def initialize(self) -> bool:
        """Initialise complètement l'environnement RTABMAP"""
        print(f"🔧 Initialisation RTABMAP sur {self.paths.os_type.upper()}")
        if self.paths.os_type == "linux":
            print(f"Chemin de base: {self.paths.base_path} (système)")
        else:
            print(
                f"Chemin de base: {self.paths.base_path} (répertoire courant)")
            print(f" Répertoire d'exécution: {os.getcwd()}")

        # Créer les répertoires
        print("\n Création des répertoires...")
        dir_results = self.paths.create_directories()

        # Vérifier les chemins
        print("\n Vérification des chemins...")
        path_results = self.paths.check_paths_exist()

        # Configurer le logging
        print("\n Configuration du logging...")
        self.logger = self.paths.setup_logging()

        # Sauvegarder la configuration
        print("\n Sauvegarde de la configuration...")
        config_saved = self.paths.save_config()

        # Résumé
        success = all(dir_results.values()) and config_saved
        status = "SUCCÈS" if success else "ÉCHEC"
        print(f"\n{status} - Initialisation terminée")

        return success

    def get_summary(self) -> str:
        """Retourne un résumé de la configuration"""
        paths = self.paths.get_all_paths()

        if self.paths.os_type == "linux":
            location_info = "Chemins système absolus"
        else:
            location_info = f"Répertoire courant: {os.getcwd()}"

        summary = f"""
            🔧 Configuration RTABMAP
            ========================
            Système d'exploitation: {paths['os_type'].upper()}
            {location_info}
            Chemin de base: {paths['base_path']}

            Répertoires:
            - Images RGB: {paths['image_folder']}
            - Images Depth: {paths['depth_folder']}
            - Sortie: {paths['output_folder']}
            - Logs: {paths['log_dir']}

            Fichiers:
            - Calibration: {paths['calibration_file']}
            - Configuration: {paths['config_file']}
        """
        return summary


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
