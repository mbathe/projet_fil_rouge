from dotenv import load_dotenv
import os
import gdown
import zipfile
import logging
load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(os.path.join(
            os.path.dirname(__file__), '..'), "script.log")),
        logging.StreamHandler()
    ]
)

error_handler = logging.FileHandler(os.path.join(os.path.join(
    os.path.dirname(__file__), '..'), "error.log"))
error_handler.setLevel(logging.ERROR)
error_handler.setFormatter(logging.Formatter(
    '%(asctime)s - %(levelname)s - %(message)s'))


logging.getLogger().addHandler(error_handler)


def download_dataset_from_drive(file_id, output_directory):
    """
    Télécharge un fichier ZIP depuis Google Drive et le décompresse.
    
    Args:
        file_id (str): L'ID du fichier Google Drive
        output_directory (str): Répertoire de destination pour le téléchargement et la décompression
    
    Returns:
        str: Chemin du fichier ZIP téléchargé
    """
    os.makedirs(output_directory, exist_ok=True)
    url = f'https://drive.google.com/uc?id={file_id}'

    output_zip = os.path.join(output_directory, 'deer_walk.zip')

    try:
        logging.info(f"Téléchargement du fichier depuis : {url}")
        gdown.download(url, output_zip, quiet=False)
        logging.info(f"Décompression du fichier dans : {output_directory}")
        with zipfile.ZipFile(output_zip, 'r') as zip_ref:
            zip_ref.extractall(output_directory)

        os.remove(output_zip)

        logging.info("Téléchargement et décompression terminés avec succès.")
        return output_zip

    except Exception as e:
        logging.error(
            f"Erreur lors du téléchargement ou de la décompression : {e}")
        return None


if __name__ == "__main__":
    file_id = os.getenv("DATASET_DRIVE_ID")
    output_dir = os.getenv("DIR_DATASET")
    downloaded_file = download_dataset_from_drive(file_id, output_dir)
