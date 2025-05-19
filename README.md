
# RTAB-Map - Pipeline de Génération de Nuage de Points 3D

Ce script Python automatise la préparation d’un dataset RGB-D pour RTAB-Map, la conversion des timestamps, l’exécution de RTAB-Map pour le traitement SLAM, et l’exportation du nuage de points en `.ply`.

---

## 📌 Fonctionnalités

1. **Vérifie et valide les fichiers CSV** (`img_timestamps.csv`, `depth_timestamps.csv`).
2. **Nettoie et copie** les dossiers `rgb_sync/` et `depth_sync/`.
3. **Supprime et remplace** le fichier de calibration `rtabmap_calib.yaml`.
4. **Renomme** les fichiers RGB et profondeur avec leur `timestamp`.
5. **Exécute RTAB-Map** sur les données préparées.
6. **Exporte** un fichier `.ply` contenant le nuage de points 3D.

---

## 🧾 Arborescence attendue

Avant exécution :

```
dataset/
├── rgb/                   # Images RGB originales (.png)
├── depth/                 # Images de profondeur originales (.png)
├── img_timestamps.csv     # Timestamp des images RGB
├── depth_timestamps.csv   # Timestamp des images profondeur
└── calib.yaml             # Fichier de calibration caméra
```

Après exécution :

```
output_dir/
├── rgb_sync/              # Images RGB renommées avec timestamp
├── depth_sync/            # Images Depth renommées
├── rtabmap_calib.yaml     # Fichier de calibration copié
├── img_timestamps.csv     # Copie validée
├── depth_timestamps.csv   # Copie validée
└── pointcloud.ply         # Nuage de points 3D exporté
```

---

## 🚀 Exécution

### Commande

```bash
python script.py <chemin_rgb> <chemin_depth> <fichier_calibration> <csv_rgb> <csv_depth>
```

### Exemple

```bash
python script.py ./dataset/rgb ./dataset/depth ./dataset/calib.yaml ./dataset/img_timestamps.csv ./dataset/depth_timestamps.csv
```

---

## ✅ Pré-requis

- Python 3.x
- Bibliothèques : `pandas`, `numpy`
- RTAB-Map installé avec :
  - `rtabmap-rgbd_dataset`
  - `rtabmap-export`

### Installation des bibliothèques Python

```bash
pip install pandas numpy
```

---

## 📂 Explication des modules

### 1. `prepare_dataset(...)`

- Vérifie que les CSV contiennent `timestamp` et `filename`.
- Supprime les anciens dossiers `rgb_sync`, `depth_sync`, les fichiers `rtabmap_calib.yaml`, et les anciens CSV.
- Copie les fichiers dans le dossier de travail.

### 2. `convert_to_timestamps(...)`

- Renomme chaque image en fonction de la colonne `timestamp` du CSV.
- Gère les doublons ou les timestamps à 0 en ajoutant un léger décalage aléatoire (0.001 à 0.005).

### 3. `main()`

- Lance `rtabmap-rgbd_dataset` avec des paramètres de configuration.
- Puis exécute `rtabmap-export` pour générer un fichier `.ply`.

---

## ⚠️ Validation CSV

Les fichiers CSV doivent avoir **exactement deux colonnes** :  
- `timestamp` : nombre (float ou int)
- `filename` : nom exact de l’image (avec extension)

Exemple :

```csv
timestamp,filename
1713456011.123456,rgb_001.png
1713456011.323456,rgb_002.png
```

Si ces colonnes ne sont pas présentes, le script s’arrête avec un message d’erreur.

---

## ❗ Attention aux erreurs

- **Colonnes manquantes dans le CSV** :
  ```
  [ERREUR] Le fichier 'img_timestamps.csv' doit contenir les colonnes 'timestamp' et 'filename'
  ```
- **Fichier de calibration absent** :
  ```
  [DELETE] Ancien fichier de calibration supprimé
  [OK] Nouveau fichier de calibration copié
  ```
- **Timestamps dupliqués** :
  ```
  [INFO] Adjusted duplicate/zero timestamp for rgb_001.png: 1713456011.128456
  ```

---

## 🧠 Personnalisation possible

- Ajouter des logs dans un fichier.
- Générer plusieurs `.ply` par séquence.
- Ajouter un filtre spatial sur le nuage de points généré.
- Gérer les fichiers `.jpg` ou `.tiff`.

---