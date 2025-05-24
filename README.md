# Projet de Cartographie 3D avec RTAB-Map

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

2. **Téléchargement de l'image Docker RTAB-Map** :
```bash
docker pull introlab3it/rtabmap:latest
```

3. **Téléchargement des poids du modèle** :
```bash
# Le script téléchargera automatiquement les poids lors de la première exécution
# ou vous pouvez les télécharger manuellement dans le dossier weights/
```

## 🚀 Utilisation

### Mode vidéo

```bash
python src/main.py --input video.mp4 --output-dir ./output --mode video --fps 5
```

### Mode images

```bash
python src/main.py --input ./images_folder --output-dir ./output --mode images
```

### Mode RGB-D

```bash
python src/main.py --input ./rgbd_folder --depth ./depth_folder --output-dir ./output --mode rgbd
```

### Options importantes

- `--depth-model` : Modèle d'estimation de profondeur (default: "depthanything")
- `--resolution` : Résolution de sortie (default: "1280x720")
- `--export-format` : Format d'exportation ("ply", "mesh", ou "both")
- `--with-texture` : Ajouter la texture au maillage 3D

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

Consultez la documentation RTAB-Map complète pour plus de détails.

## 🧠 Extensions et personnalisations

- Intégration d'autres modèles d'estimation de profondeur
- Filtrage spatial sur le nuage de points généré
- Support pour différents formats d'image (.jpg, .tiff, etc.)
- Ajout de logs détaillés
- Parallélisation des traitements pour améliorer les performances

## 📜 Licence

Ce projet est sous licence MIT.

---