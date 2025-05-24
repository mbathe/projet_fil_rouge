# Projet de Cartographie 3D avec RTAB-Map



![](https://github.com/mbathe/projet_fil_rouge/blob/main/deek_walk_3dmap.png)

## 📋 Aperçu du projet

Ce projet permet de générer une cartographie 3D à partir de différentes sources d'entrée :
- Vidéos (segmentées en images)
- Images RGB (avec estimation de profondeur)
- Images RGB-D existantes

Le workflow principal consiste à :
1. **Acquisition des données** : vidéo ou séquence d'images
2. **Estimation de profondeur** : utilisation du modèle DepthAnythingV2 pour créer des images de profondeur
3. **Cartographie 3D** : utilisation de RTAB-Map via Docker pour générer un modèle 3D
4. **Exportation** : nuage de points au format .ply ou mesh pour visualisation et analyse

## 🔍 Technologies clés

- **RTAB-Map** (Real-Time Appearance-Based Mapping) : Framework de SLAM pour la cartographie 3D
- **DepthAnythingV2** : Modèle de deep learning pour l'estimation de profondeur à partir d'images RGB
- **Docker** : Conteneurisation des dépendances complexes
- **Python** : Orchestration du pipeline complet

## 🏗️ Architecture du projet

```
projet/
├── data/                  # Données d'exemple
├── notebook/              # Notebooks d'expérimentation
├── src/
│   ├── depth/             # Code pour l'estimation de profondeur
│   ├── rtabmap/           # Code pour la génération de cartographie 3D
│   └── main.py            # Point d'entrée de l'application
├── output/                # Base de données RTAB-Map, fichiers mesh et cloud
├── weight/                # Poids des modèles de deep learning
└── scripts/               # Scripts utilitaires
```

## 📦 Modules principaux

### 1. Estimation de profondeur
- Utilise le modèle **DepthAnythingV2** pour générer des cartes de profondeur à partir d'images RGB
- Traite soit des images individuelles, soit extrait des frames d'une vidéo
- Calibre et normalise les données de profondeur pour RTAB-Map

### 2. Cartographie RTAB-Map
- Utilise les paires RGB-D pour construire une représentation 3D
- Génère une base de données de l'environnement avec informations de localisation
- Exécute les algorithmes de SLAM pour aligner les images dans l'espace 3D

### 3. Exportation et visualisation
- Génère des nuages de points 3D (.ply)
- Crée des maillages 3D (mesh)
- Offre des options de projection 2D du modèle 3D

## 🛠️ Installation et configuration

### Prérequis
- Python 3.8+
- Docker
- GPU recommandé pour l'inférence du modèle de profondeur

### Installation de Docker

Pour installer Docker, veuillez suivre la documentation officielle de Docker correspondant à votre système d'exploitation :
- **Site d'installation officiel** : [https://docs.docker.com/engine/install/](https://docs.docker.com/engine/install/)
- Choisissez votre distribution Linux, ou Windows/macOS selon votre système

### Configuration de Docker sans sudo (important)

⚠️ **IMPORTANT** : Comme Docker est invoqué directement depuis le code Python de ce projet, il est **crucial** de configurer Docker pour qu'il fonctionne sans sudo sur les systèmes Linux. Sans cette configuration, les scripts Python ne pourront pas exécuter les commandes Docker correctement.

Suivez les instructions de post-installation pour votre plateforme :
- **Documentation post-installation** : [https://docs.docker.com/engine/install/linux-postinstall/](https://docs.docker.com/engine/install/linux-postinstall/)

Les étapes principales sont :
1. Ajouter votre utilisateur au groupe Docker
2. Appliquer les changements de groupe
3. Vérifier l'installation sans sudo
4. Configurer Docker pour démarrer au boot

### Configuration de l'environnement

1. **Installation des dépendances Python** :
```bash
pip install -r requirements.txt
```

2. **Téléchargement des poids du modèle** :
```bash
# Le script téléchargera automatiquement les poids lors de la première exécution
# ou vous pouvez les télécharger manuellement dans le dossier weights/
```

### Construction de l'image Docker personnalisée

Le projet utilise une image Docker personnalisée qui contient RTAB-Map et les scripts nécessaires pour la génération de cartographie 3D.

⚠️ **IMPORTANT** : Avant d'exécuter le programme principal, vous devez construire l'image Docker :

```bash
sudo docker build -t rtabmap_ubuntu20 .
```

Le `Dockerfile` à la racine du projet contient les instructions pour :
1. Construire l'image Docker avec RTAB-Map et toutes les dépendances nécessaires
2. Injecter le script `./src/rtabmap/rtabmap_script.py` dans l'image
3. Configurer l'environnement d'exécution pour la cartographie 3D

Ce script est automatiquement appelé lorsque le conteneur Docker est exécuté depuis le code Python, et il prend en charge la génération de la cartographie 3D.

**Note** : Il n'est pas nécessaire d'installer RTAB-Map séparément ou de télécharger une autre image Docker, car le Dockerfile configure tout ce qui est nécessaire.

**Note** : Chaque fois que vous modifiez le contenu du répertoire `./src/rtabmap/`, vous devez reconstruire l'image Docker pour que les changements soient pris en compte.

## 🚀 Utilisation

### ⚠️ Chemins absolus obligatoires

**Important** : Comme le programme utilise Docker avec des montages de volumes, tous les chemins doivent être **absolus** et non relatifs. Les chemins relatifs ne fonctionneront pas car Docker nécessite des chemins complets pour monter les volumes correctement.

Dans les exemples ci-dessous, remplacez `<PROJECT_ROOT>` par le chemin absolu vers la racine de votre projet.

### Exemple complet avec chemins absolus

```bash
python3 <PROJECT_ROOT>/src/main.py \
  --image_folder "<PROJECT_ROOT>/data/images" \
  --depth_folder "<PROJECT_ROOT>/data/depth" \
  --calibration_file "<PROJECT_ROOT>/data/rtabmap_calib.yaml" \
  --rgb_timestamps "<PROJECT_ROOT>/data/img_timestamps.csv" \
  --depth_timestamps "<PROJECT_ROOT>/data/depth_timestamps.csv" \
  --output_folder "<PROJECT_ROOT>/output"
```

Par exemple, si votre projet est situé dans `/home/utilisateur/cartographie3d`, tous les chemins doivent commencer par cette racine.

### Mode vidéo (à partir d'une source vidéo)

```bash
python src/main.py --source video --images_folder ./chemin/vers/la/video.mp4 --output_folder ./output_folder --frequence 5
```

### Mode images (RGB sans profondeur)

```bash
python src/main.py --source image --images_folder ./chemin/vers/images --output_folder ./output_folder
```

### Mode RGB-D (images avec profondeur)

```bash
python src/main.py --source image_with_depth --images_folder ./chemin/vers/images/rgb --depth_folder ./chemin/vers/images/depth --output_folder ./output_folder
```

### Arguments disponibles

Voici la liste complète des arguments acceptés par le script :

```
--images_folder        Dossier contenant les images RGB ou chemin vers le fichier vidéo (défaut: "./images_folder")
--depth_folder         Dossier contenant les images de profondeur (défaut: "./depth_folder")
--calibration_file     Chemin vers le fichier de calibration de caméra (défaut: "./rtabmap_calib.yaml")
--rgb_timestamps       Chemin vers le fichier CSV de timestamps RGB (défaut: "./img_timestamps.csv")
--depth_timestamps     Chemin vers le fichier CSV de timestamps profondeur (défaut: "./depth_timestamps.csv")
--output_folder        Dossier de sortie pour tous les résultats (défaut: "./output_folder")
--source               Source à utiliser: "image" (RGB sans profondeur), "image_with_depth" (RGB-D), "video" (vidéo)
                       (défaut: "image_with_depth")
--frequence            Fréquence d'extraction d'images depuis la vidéo en Hz (défaut: 20)
```

### Exemples d'utilisation

#### Traitement vidéo avec une fréquence de 10 Hz
```bash
python src/main.py --source video --images_folder ./data/video.mp4 --output_folder ./results --frequence 10
```

#### Traitement d'images RGB avec génération de profondeur
```bash
python src/main.py --source image --images_folder ./data/rgb_images --output_folder ./results
```

#### Traitement d'images RGB-D existantes avec fichiers de timestamps
```bash
python src/main.py --source image_with_depth --images_folder ./data/rgb --depth_folder ./data/depth --rgb_timestamps ./data/rgb_timestamps.csv --depth_timestamps ./data/depth_timestamps.csv --output_folder ./results
```

## 📊 Format des données

### Structure pour les séquences d'images
Les images doivent être nommées de manière séquentielle ou avec des timestamps.

### Format CSV pour les timestamps
Si vous utilisez des timestamps personnalisés, le CSV doit contenir :
- `timestamp` : nombre (float ou int)
- `filename` : nom exact de l'image (avec extension)

Exemple :
```csv
timestamp,filename
1713456011.123456,rgb_001.png
1713456011.323456,rgb_002.png
```

## 🔧 Paramètres avancés de RTAB-Map

Le projet expose plusieurs paramètres RTAB-Map pour les utilisateurs avancés :
- Paramètres d'odométrie visuelle
- Options de loop closure
- Filtrage de nuage de points
- Paramètres d'optimisation du maillage

### Fichiers de configuration paramètres

Le répertoire `<PROJECT_ROOT>/src/rtabmap/rtabmap_params/` contient trois fichiers JSON qui permettent de configurer finement le comportement de RTAB-Map :

1. **`export_params.json`** : Paramètres pour l'exportation des nuages de points et meshes
   - Format d'exportation (PLY, OBJ, etc.)
   - Densité des nuages de points
   - Options de texture et coloration
   - Filtres d'export (distance, bruit, etc.)

2. **`generate_db_params.json`** : Paramètres pour la génération initiale de la base de données
   - Paramètres de détection de feature points
   - Options de calibration de caméra
   - Paramètres d'optimisation de la carte
   - Configuration des correspondances de feature

3. **`reprocess_params.json`** : Paramètres pour le retraitement d'une base de données existante
   - Options de filtrage
   - Paramètres de re-optimisation
   - Techniques de loop closure
   - Configuration des ajustements globaux

Ces fichiers peuvent être modifiés selon vos besoins pour affiner les résultats de la cartographie 3D.

Consultez la documentation RTAB-Map complète pour plus de détails sur les paramètres disponibles : [Documentation RTAB-Map](http://wiki.ros.org/rtabmap_ros/Tutorials/Advanced%20Parameter%20Tuning)

## 🧠 Extensions et personnalisations

- Intégration d'autres modèles d'estimation de profondeur
- Filtrage spatial sur le nuage de points généré
- Support pour différents formats d'image (.jpg, .tiff, etc.)
- Ajout de logs détaillés
- Parallélisation des traitements pour améliorer les performances

## 📜 Licence

Ce projet est sous licence MIT.

---