#!/usr/bin/env python3
"""
Script pour télécharger Depth Anything V2 depuis le dépôt officiel GitHub
"""

import os
import subprocess
import sys
import urllib.request
import urllib.error
from pathlib import Path


def run_command(cmd, cwd=None):
    """Exécute une commande shell et gère les erreurs"""
    try:
        result = subprocess.run(
            cmd, 
            shell=True, 
            check=True, 
            capture_output=True, 
            text=True,
            cwd=cwd
        )
        return result.stdout
    except subprocess.CalledProcessError as e:
        print(f"Erreur lors de l'exécution de: {cmd}")
        print(f"Code d'erreur: {e.returncode}")
        print(f"Sortie d'erreur: {e.stderr}")
        return None


def download_file_with_progress(url, destination):
    """
    Télécharge un fichier avec barre de progression
    
    Args:
        url (str): URL du fichier à télécharger
        destination (Path): Chemin de destination
    """
    try:
        def progress_hook(block_num, block_size, total_size):
            if total_size > 0:
                percent = min(100, (block_num * block_size * 100) // total_size)
                bar_length = 30
                filled = int(bar_length * percent // 100)
                bar = '█' * filled + '░' * (bar_length - filled)
                print(f'\r📥 Téléchargement: {bar} {percent}% ({block_num * block_size // (1024*1024)}/{total_size // (1024*1024)} MB)', end='', flush=True)
        
        urllib.request.urlretrieve(url, destination, progress_hook)
        print()  # Nouvelle ligne après la barre de progression
        return True
    except urllib.error.URLError as e:
        print(f"\n❌ Erreur lors du téléchargement: {e}")
        return False
    except Exception as e:
        print(f"\n❌ Erreur inattendue: {e}")
        return False


def check_git_installed():
    """Vérifie si Git est installé"""
    try:
        subprocess.run(["git", "--version"], check=True, capture_output=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def download_model_weights(project_path):
    """
    Télécharge les poids du modèle Depth Anything V2 Base
    
    Args:
        project_path (Path): Chemin du projet DepthAnythingV2
    """
    print("\n📦 Téléchargement des poids du modèle...")
    
    # Création du répertoire checkpoints
    checkpoints_dir = project_path / "checkpoints"
    checkpoints_dir.mkdir(exist_ok=True)
    
    # URL et destination du fichier de poids
    model_url = "https://huggingface.co/depth-anything/Depth-Anything-V2-Base/resolve/main/depth_anything_v2_vitb.pth?download=true"
    model_file = checkpoints_dir / "depth_anything_v2_vitb.pth"
    
    # Vérifier si le fichier existe déjà
    if model_file.exists():
        file_size = model_file.stat().st_size / (1024 * 1024)  # Taille en MB
        response = input(f"Le fichier de poids existe déjà ({file_size:.1f} MB). Le re-télécharger ? (y/N): ")
        if response.lower() not in ['y', 'yes', 'o', 'oui']:
            print("✅ Utilisation du fichier de poids existant.")
            return True
        else:
            model_file.unlink()  # Supprimer le fichier existant
    
    print(f"📂 Téléchargement vers: {model_file}")
    print("⏳ Cela peut prendre plusieurs minutes selon votre connexion...")
    
    success = download_file_with_progress(model_url, model_file)
    
    if success and model_file.exists():
        file_size = model_file.stat().st_size / (1024 * 1024)  # Taille en MB
        print(f"✅ Poids du modèle téléchargé avec succès ! (Taille: {file_size:.1f} MB)")
        return True
    else:
        print("❌ Échec du téléchargement des poids du modèle.")
        return False
    """Vérifie si Git est installé"""
    try:
        subprocess.run(["git", "--version"], check=True, capture_output=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def download_depth_anything_v2(target_directory="./src/depth/DepthAnythingV2"):
    """
    Télécharge Depth Anything V2 dans le répertoire spécifié
    
    Args:
        target_directory (str): Chemin du répertoire de destination
    """
    
    # Vérification de Git
    if not check_git_installed():
        print("❌ Git n'est pas installé sur votre système.")
        print("Veuillez installer Git pour continuer.")
        return False
    
    # Création du chemin absolu
    target_path = Path(target_directory).resolve()
    parent_dir = target_path.parent
    
    # Création du répertoire parent si nécessaire
    parent_dir.mkdir(parents=True, exist_ok=True)
    
    # Création du répertoire de destination si nécessaire
    target_path.mkdir(parents=True, exist_ok=True)
    
    print(f"📂 Répertoire de destination: {target_path}")
    
    # Vérification si le répertoire existe déjà
    if target_path.exists():
        response = input(f"Le répertoire {target_path} existe déjà. Voulez-vous le supprimer et recommencer ? (y/N): ")
        if response.lower() in ['y', 'yes', 'o', 'oui']:
            import shutil
            shutil.rmtree(target_path)
            print("✅ Répertoire existant supprimé.")
        else:
            print("❌ Téléchargement annulé.")
            return False
    
    # Clone du dépôt
    repo_url = "https://github.com/DepthAnything/Depth-Anything-V2.git"
    print(f"📥 Téléchargement de Depth Anything V2 depuis {repo_url}...")
    
    clone_cmd = f"git clone {repo_url} {target_path}"
    result = run_command(clone_cmd)
    
    if result is None:
        print("❌ Échec du téléchargement du dépôt.")
        return False
    
    print("✅ Dépôt téléchargé avec succès !")
    
    # Vérification du contenu
    if target_path.exists():
        files = list(target_path.iterdir())
        print(f"📁 Contenu téléchargé ({len(files)} éléments):")
        for file in files[:10]:  # Affiche les 10 premiers éléments
            print(f"   - {file.name}")
        if len(files) > 10:
            print(f"   ... et {len(files) - 10} autres éléments")
    
    # Installation des dépendances (optionnel)
    requirements_file = target_path / "requirements.txt"
    if requirements_file.exists():
        response = input("\n📦 Un fichier requirements.txt a été trouvé. Voulez-vous installer les dépendances ? (y/N): ")
        if response.lower() in ['y', 'yes', 'o', 'oui']:
            print("📦 Installation des dépendances...")
            install_cmd = f"pip install -r {requirements_file}"
            install_result = run_command(install_cmd)
            if install_result is not None:
                print("✅ Dépendances installées avec succès !")
            else:
                print("⚠️  Erreur lors de l'installation des dépendances.")
    
    # Téléchargement des poids du modèle
    response = input("\n🤖 Voulez-vous télécharger les poids du modèle Depth Anything V2 Base ? (Y/n): ")
    if response.lower() not in ['n', 'no', 'non']:
        weights_success = download_model_weights(target_path)
        if not weights_success:
            print("⚠️  Le projet a été téléchargé mais les poids du modèle ont échoué.")
            print("💡 Vous pouvez les télécharger manuellement plus tard.")
    
    print(f"\n🎉 Depth Anything V2 a été téléchargé dans: {target_path}")
    return True


def main():
    """Fonction principale"""
    print("🚀 Script de téléchargement Depth Anything V2")
    print("=" * 50)
    
    # Demande du répertoire de destination
    default_dir = "./src/depth/DepthAnythingV2"
    user_dir = input(f"Répertoire de destination (défaut: {default_dir}): ").strip()
    
    if not user_dir:
        user_dir = default_dir
    
    # Téléchargement
    success = download_depth_anything_v2(user_dir)
    
    if success:
        print("\n✅ Téléchargement terminé avec succès !")
        print(f"📂 Le projet se trouve dans: {Path(user_dir).resolve()}")
        print("\n💡 Prochaines étapes suggérées:")
        print("   1. Lire le README.md pour les instructions d'utilisation")
        print("   2. Vérifier les requirements système")
        print("   3. Télécharger les modèles pré-entraînés si nécessaire")
    else:
        print("\n❌ Échec du téléchargement.")
        sys.exit(1)


if __name__ == "__main__":
    main()