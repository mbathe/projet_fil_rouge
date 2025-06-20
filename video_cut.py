import subprocess
import os

# TODO: Implémenter pour faciliter l'utilisation des autres commandes
def cmd(command):

    cmd = command.split()

    try:
        output = subprocess.check_output(cmd, stderr=subprocess.STDOUT)
        return output
    except Exception as e:
        print("Erreur d'execution de commande :", e)
        return None

# TODO: Implémenter get framerate pour automatiser le choix des fps
def get_framerate(video_path):

    output = cmd(f'ffprobe -v 0 -select_streams v:0 -show_entries stream=avg_frame_rate -of default=noprint_wrappers=1:nokey=1 {video_path}')
    framerate_str = output.decode().strip()
    num, denom = map(int, framerate_str.split('/'))

    return round(num / denom)

# TODO: Implémenter pour découper les vidéos en images
def video_cut(video_path, output_folder, output_format="image_%10d.png"):

    os.makedirs(output_folder, exist_ok=True)
    output = cmd(f"ffmpeg -i {video_path} {os.path.join(output_folder, output_format)}")





path = "/Users/loux/Desktop/CODE/PROJET_FILROUGE/github/projet_fil_rouge/data/dataset/vid_salle.mp4"
framerate = get_framerate(path)
print(framerate)

output_folder = "/Users/loux/Desktop/CODE/PROJET_FILROUGE/github/projet_fil_rouge/data/dataset/vid_salle/"
os.makedirs(output_folder, exist_ok=True)
output_images_folder = "/Users/loux/Desktop/CODE/PROJET_FILROUGE/github/projet_fil_rouge/data/dataset/vid_salle/images"
video_cut(path, output_images_folder)