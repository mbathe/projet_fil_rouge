#!/usr/bin/env python3
"""
Script pour télécharger Depth Anything V2 depuis le dépôt officiel GitHub
"""

import os
import subprocess
import sys
import urllib.request
import urllib.error
import logging
import logging.handlers
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv
load_dotenv()

# Peut être 'small', 'base' ou 'large'
DEPTH_ANYTHING_TYPE = os.getenv("DEPTH_ANYTHING_TYPE", "small")

SCRIPT_DIR = Path(__file__).parent.absolute()
PROJECT_ROOT = SCRIPT_DIR.parent
LOG_DIR = PROJECT_ROOT / "logs" / "scripts"


def setup_logging():
    """Configure le système de logging avec fichier et console"""
    # Créer le répertoire de logs s'il n'existe pas
    log_dir = Path(PROJECT_ROOT)
    log_dir.mkdir(exist_ok=True)

    # Nom du fichier de log avec timestamp
    log_filename = log_dir / \
        f"download_depth_anything_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

    # Configuration du logger principal
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    # Formatter pour les logs
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # Handler pour fichier avec rotation
    file_handler = logging.handlers.RotatingFileHandler(
        log_filename, maxBytes=10*1024*1024, backupCount=5
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)

    # Handler pour console
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter('%(levelname)s - %(message)s')
    console_handler.setFormatter(console_formatter)

    # Ajout des handlers
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    logging.info(f"Logging configuré - Fichier: {log_filename}")
    return logger

def run_command(cmd, cwd=None):
    """Exécute une commande shell et gère les erreurs"""
    try:
        logging.debug(f"Exécution de la commande: {cmd} (cwd: {cwd})")
        result = subprocess.run(
            cmd if isinstance(cmd, str) else ' '.join(cmd),
            shell=True,
            check=True,
            capture_output=True,
            text=True,
            cwd=cwd
        )
        logging.debug(
            f"Commande exécutée avec succès: {result.stdout[:200]}...")
        return result.stdout
    except subprocess.CalledProcessError as e:
        logging.error(f"Erreur lors de l'exécution de: {cmd}")
        logging.error(f"Code d'erreur: {e.returncode}")
        logging.error(f"Sortie d'erreur: {e.stderr}")
        return None


def download_file_with_progress(url, destination):
    """
    Télécharge un fichier avec barre de progression

    Args:
        url (str): URL du fichier à télécharger
        destination (Path): Chemin de destination
    """
    try:
        logging.info(f"Début du téléchargement: {url}")
        logging.debug(f"Destination: {destination}")

        def progress_hook(block_num, block_size, total_size):
            if total_size > 0:
                percent = min(
                    100, (block_num * block_size * 100) // total_size)
                bar_length = 30
                filled = int(bar_length * percent // 100)
                bar = '█' * filled + '░' * (bar_length - filled)
                progress_msg = f'Téléchargement: {bar} {percent}% ({block_num * block_size // (1024*1024)}/{total_size // (1024*1024)} MB)'
                print(f'\r{progress_msg}', end='', flush=True)

                # Log de progression tous les 10%
                if percent % 10 == 0 and block_num > 0:
                    logging.debug(f"Progression téléchargement: {percent}%")

        urllib.request.urlretrieve(url, destination, progress_hook)
        print()  # Nouvelle ligne après la barre de progression
        logging.info("Téléchargement terminé avec succès")
        return True
    except urllib.error.URLError as e:
        logging.error(f"Erreur lors du téléchargement: {e}")
        print(f"\nErreur lors du téléchargement: {e}")
        return False
    except Exception as e:
        logging.error(f"Erreur inattendue lors du téléchargement: {e}")
        print(f"\nErreur inattendue: {e}")
        return False


def check_git_installed():
    """Vérifie si Git est installé"""
    try:
        subprocess.run(["git", "--version"], check=True, capture_output=True)
        logging.info("Git est installé et disponible")
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        logging.error("Git n'est pas installé ou non disponible")
        return False


def download_model_weights(project_path):
    """
    Télécharge les poids du modèle Depth Anything V2

    Args:
        project_path (Path): Chemin du projet DepthAnythingV2
    """
    logging.info("Début du téléchargement des poids du modèle")
    print("\nTéléchargement des poids du modèle...")

    # Création du répertoire checkpoints
    checkpoints_dir = project_path / "checkpoints"
    checkpoints_dir.mkdir(exist_ok=True)
    logging.debug(f"Répertoire checkpoints créé: {checkpoints_dir}")

    model_small_url = "https://huggingface.co/depth-anything/Depth-Anything-V2-Metric-Hypersim-Small/resolve/main/depth_anything_v2_metric_hypersim_vits.pth?download=true"
    model_base_url = "https://huggingface.co/depth-anything/Depth-Anything-V2-Metric-Hypersim-Base/resolve/main/depth_anything_v2_metric_hypersim_vitb.pth?download=true"
    model_large_url = "https://huggingface.co/depth-anything/Depth-Anything-V2-Metric-Hypersim-Large/resolve/main/depth_anything_v2_metric_hypersim_vitl.pth?download=true"

    # URL et destination du fichier de poids
    model_url = model_small_url if DEPTH_ANYTHING_TYPE == "small" else (
        model_base_url if DEPTH_ANYTHING_TYPE == "base" else model_large_url
    )
    model_file = checkpoints_dir / "depth_anything_v2_metric_hypersim_vits.pth" if DEPTH_ANYTHING_TYPE == "small" else (
        checkpoints_dir / "depth_anything_v2_metric_hypersim_vitb.pth" if DEPTH_ANYTHING_TYPE == "base" else checkpoints_dir /
        "depth_anything_v2_metric_hypersim_vitl.pth"
    )

    logging.info(f"Type de modèle sélectionné: {DEPTH_ANYTHING_TYPE}")
    logging.debug(f"URL du modèle: {model_url}")
    logging.debug(f"Fichier destination: {model_file}")

    # Vérifier si le fichier existe déjà
    if model_file.exists():
        file_size = model_file.stat().st_size / (1024 * 1024)  # Taille en MB
        logging.warning(
            f"Le fichier de poids existe déjà ({file_size:.1f} MB)")
        response = input(
            f"Le fichier de poids existe déjà ({file_size:.1f} MB). Le re-télécharger ? (Y/N): ")
        if response.lower() not in ['y', 'yes', 'o', 'oui']:
            logging.info("Utilisation du fichier de poids existant")
            print("Utilisation du fichier de poids existant.")
            return True
        else:
            model_file.unlink()  # Supprimer le fichier existant
            logging.info("Ancien fichier de poids supprimé")

    logging.info(f"Téléchargement vers: {model_file}")
    print(f"Téléchargement vers: {model_file}")
    print("Cela peut prendre plusieurs minutes selon votre connexion...")

    success = download_file_with_progress(model_url, model_file)

    if success and model_file.exists():
        file_size = model_file.stat().st_size / (1024 * 1024)  # Taille en MB
        logging.info(
            f"Poids du modèle téléchargé avec succès ! (Taille: {file_size:.1f} MB)")
        print(
            f"Poids du modèle téléchargé avec succès ! (Taille: {file_size:.1f} MB)")
        return True
    else:
        logging.error("Échec du téléchargement des poids du modèle")
        print("Échec du téléchargement des poids du modèle.")
        return False


def download_depth_anything_v2(target_directory="./src/depth/DepthAnythingV2"):
    """
    Télécharge Depth Anything V2 dans le répertoire spécifié

    Args:
        target_directory (str): Chemin du répertoire de destination
    """
    logging.info(
        f"Début du téléchargement de Depth Anything V2 vers: {target_directory}")

    # Vérification de Git
    if not check_git_installed():
        logging.error("Git n'est pas installé sur le système")
        print("Git n'est pas installé sur votre système.")
        print("Veuillez installer Git pour continuer.")
        return False

    # Création du chemin absolu
    target_path = Path(target_directory).resolve()
    parent_dir = target_path.parent

    # Création du répertoire parent si nécessaire
    parent_dir.mkdir(parents=True, exist_ok=True)
    logging.debug(f"Répertoire parent créé/vérifié: {parent_dir}")

    logging.info(f"Répertoire de destination: {target_path}")
    print(f"Répertoire de destination: {target_path}")

    # Clone du dépôt
    repo_url = "https://github.com/DepthAnything/Depth-Anything-V2.git"
    logging.info(f"Début du clonage depuis: {repo_url}")
    print(f"Téléchargement de Depth Anything V2 depuis {repo_url}...")

    clone_cmd = ["git", "clone", repo_url, str(target_path)]
    logging.debug(f"Commande de clonage: {clone_cmd}")
    print("Téléchargement du modèle", clone_cmd)
    result = run_command(clone_cmd)

    if result is None:
        logging.error("Échec du téléchargement du dépôt")
        print("Échec du téléchargement du dépôt.")
        return False

    logging.info("Dépôt téléchargé avec succès")
    print("Dépôt téléchargé avec succès !")

    # Vérification du contenu
    if target_path.exists():
        files = list(target_path.iterdir())
        logging.info(f"Contenu téléchargé: {len(files)} éléments")
        print(f"Contenu téléchargé ({len(files)} éléments):")
        for file in files[:10]:  # Affiche les 10 premiers éléments
            print(f"   - {file.name}")
            logging.debug(f"Fichier téléchargé: {file.name}")
        if len(files) > 10:
            print(f"   ... et {len(files) - 10} autres éléments")
            logging.debug(f"... et {len(files) - 10} autres éléments")

    # Installation des dépendances (optionnel)
    requirements_file = target_path / "requirements.txt"
    if requirements_file.exists():
        logging.info("Fichier requirements.txt trouvé")
        response = input(
            "\nUn fichier requirements.txt a été trouvé. Voulez-vous installer les dépendances ? (Y/N): ")
        if response.lower() in ['y', 'yes', 'o', 'oui']:
            logging.info("Installation des dépendances demandée")
            print("Installation des dépendances...")
            install_cmd = f'pip install -r "{requirements_file}"'
            install_result = run_command(install_cmd)
            if install_result is not None:
                logging.info("Dépendances installées avec succès")
                print("Dépendances installées avec succès !")
            else:
                logging.error("Erreur lors de l'installation des dépendances")
                print("Erreur lors de l'installation des dépendances.")
        else:
            logging.info(
                "Installation des dépendances ignorée par l'utilisateur")

    # Téléchargement des poids du modèle
    response = input(
        "\nVoulez-vous télécharger les poids du modèle Depth Anything V2 Small ? (Y/N): ")
    if response.lower() not in ['n', 'no', 'non']:
        weights_success = download_model_weights(target_path)
        if not weights_success:
            logging.warning(
                "Le projet a été téléchargé mais les poids du modèle ont échoué")
            print("Le projet a été téléchargé mais les poids du modèle ont échoué.")
            print("Vous pouvez les télécharger manuellement plus tard.")
    else:
        logging.info("Téléchargement des poids ignoré par l'utilisateur")

    logging.info(f"Depth Anything V2 téléchargé dans: {target_path}")
    print(f"\nDepth Anything V2 a été téléchargé dans: {target_path}")
    return True


def main():
    """Fonction principale"""
    # Configuration du logging en premier
    logger = setup_logging()

    logging.info("=" * 50)
    logging.info("Début du script de téléchargement Depth Anything V2")
    logging.info("=" * 50)

    print("Script de téléchargement Depth Anything V2")
    print("=" * 50)

    # Demande du répertoire de destination
    default_dir = "./src/depth/DepthAnythingV2"
    user_dir = input(
            f"Répertoire de destination (défaut: {default_dir}): ").strip() or default_dir

    logging.info(f"Répertoire de destination choisi: {user_dir}")

    # Téléchargement
    success = download_depth_anything_v2(user_dir)

    if success:
        logging.info("Téléchargement terminé avec succès")
        print("\nTéléchargement terminé avec succès !")
        print(f"Le projet se trouve dans: {Path(user_dir).resolve()}")
        print("\nProchaines étapes suggérées:")
        print("   1. Lire le README.md pour les instructions d'utilisation")
        print("   2. Vérifier les requirements système")
        print("   3. Télécharger les modèles pré-entraînés si nécessaire")
    else:
        logging.error("Échec du téléchargement")
        print("\nÉchec du téléchargement.")
        sys.exit(1)

    logging.info("Fin du script")


if __name__ == "__main__":
    main()
