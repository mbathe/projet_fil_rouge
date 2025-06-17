import requests
import zipfile
import logging
from pathlib import Path
from datetime import datetime


def setup_logging():
    """Configure le logging basique."""
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_dir /"scripts" / f"download_{timestamp}.log"),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


def download_dataset(url, download_dir="data/datasets", extract_name="deer_walk"):
    """
    Télécharge, extrait et organise le dataset avec gestion d'erreurs et logs.
    
    Args:
        url: URL du fichier zip
        download_dir: Dossier de téléchargement
        extract_name: Nom du dossier d'extraction
    """
    logger = setup_logging()

    try:
        # Configuration des chemins
        download_path = Path(download_dir)
        zip_path = download_path / "temp.zip"
        extract_path = download_path
        extract_path_check = download_path / "deer_walk"

        # Créer les dossiers
        download_path.mkdir(exist_ok=True)
        logger.info(f"Dossier créé : {download_path}")

        # Vérifier si déjà présent
        if extract_path_check.exists():
            logger.info("Dataset déjà présent")
            files_count = sum(
                1 for _ in extract_path.rglob('*') if _.is_file())
            total_size = sum(
                f.stat().st_size for f in extract_path.rglob('*') if f.is_file())
            print(
                f"Dataset existant : {files_count} fichiers, {total_size/(1024*1024):.1f} MB")
            return str(extract_path.absolute())

        # Téléchargement
        logger.info(f"Téléchargement depuis {url}")
        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status()

        total_size = int(response.headers.get('content-length', 0))
        logger.info(f"Taille : {total_size/(1024*1024):.1f} MB")

        with open(zip_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)

        logger.info("Téléchargement terminé")

        # Extraction
        logger.info("Extraction du fichier zip...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            # Test d'intégrité
            if zip_ref.testzip():
                raise zipfile.BadZipFile("Archive corrompue")

            zip_ref.extractall(extract_path)

        # Analyse du contenu
        files_count = sum(1 for _ in extract_path.rglob('*') if _.is_file())
        total_extracted = sum(
            f.stat().st_size for f in extract_path.rglob('*') if f.is_file())

        logger.info(
            f"Extraction terminée : {files_count} fichiers, {total_extracted/(1024*1024):.1f} MB")

        # Nettoyer le zip
        zip_path.unlink()
        logger.info("Fichier zip temporaire supprimé")

        print(f"\n✅ Succès !")
        print(f"📁 Dataset disponible : {extract_path.absolute()}")
        print(f"📊 {files_count} fichiers ({total_extracted/(1024*1024):.1f} MB)")

        return str(extract_path.absolute())

    except requests.exceptions.RequestException as e:
        logger.error(f"Erreur de téléchargement : {e}")
        raise Exception(f"Échec du téléchargement : {e}")

    except zipfile.BadZipFile as e:
        logger.error(f"Archive corrompue : {e}")
        raise Exception(f"Fichier zip invalide : {e}")

    except Exception as e:
        logger.error(f"Erreur : {e}")
        # Nettoyer en cas d'erreur
        if zip_path.exists():
            zip_path.unlink()
        raise

    finally:
        # Assurer le nettoyage
        if 'zip_path' in locals() and zip_path.exists():
            try:
                zip_path.unlink()
            except:
                pass


def main():
    """Fonction principale."""
    url = "https://www.doc.ic.ac.uk/~wl208/lmdata/deer_walk.zip"

    try:
        dataset_path = download_dataset(url)
        print(f"\n🎯 Dataset prêt à l'usage dans : {dataset_path}")

    except Exception as e:
        print(f"\n Erreur : {e}")
        print("Consultez les logs pour plus de détails.")
        return False

    return True


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
