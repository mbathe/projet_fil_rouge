# Utilise Ubuntu 20.04
FROM ubuntu:20.04

#  Empêche les prompts interactifs et fixe le fuseau horaire
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Europe/Paris

# 1. Mise à jour système + outils de base
RUN apt-get update && apt-get install -y \
    tzdata \
    curl \
    gnupg2 \
    lsb-release \
    software-properties-common \
    git \
    wget \
    cmake \
    build-essential \
    python3 \
    python3-pip \
    python3-dev \
    && apt-get clean

# 2. Installation de catkin-tools et dépendances ROS Python via pip
RUN pip3 install --no-cache-dir \
    catkin-tools \
    rospkg \
    empy

# 3. Ajout des dépôts ROS avec la nouvelle méthode GPG
RUN curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg && \
    echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros/ubuntu $(lsb_release -cs) main" | tee /etc/apt/sources.list.d/ros-latest.list > /dev/null

# 4. Installation de ROS Noetic complet
RUN apt-get update && apt-get install -y \
    ros-noetic-desktop-full \
    python3-rosdep \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# 5. Initialisation de rosdep
RUN rosdep init && rosdep update

# 6. Prépare le sourcing automatique de ROS dans le conteneur
SHELL ["/bin/bash", "-c"]
RUN echo "source /opt/ros/noetic/setup.bash" >> ~/.bashrc

# 7. Installation des dépendances pour compiler RTAB-Map
#    (sans libgtsam-dev qui n'existe pas)
RUN apt-get update && apt-get install -y \
    libsqlite3-dev \
    libpcl-dev \
    libopencv-dev \
    libqt5svg5-dev \
    qtbase5-dev \
    libvtk7-dev \
    libboost-all-dev \
    libproj-dev \
    libopenni2-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# 8. Compilation de RTAB-Map depuis GitHub (repo officiel)
WORKDIR /opt
RUN git clone https://github.com/introlab/rtabmap.git && \
    mkdir -p rtabmap/build && \
    cd rtabmap/build && \
    cmake .. -DCMAKE_BUILD_TYPE=Release && \
    make -j"$(nproc)" && \
    make install

# 9. Installation des dépendances Python
RUN python3 -m pip install --no-cache-dir \
    pandas \
    python-dotenv \
    tqdm

# 10. Création des répertoires nécessaires
RUN mkdir -p /rtabmap_ws/rgb_sync \
    /rtabmap_ws/depth_sync

# 11. Configuration des bibliothèques
RUN echo "/usr/local/lib" >> /etc/ld.so.conf.d/rtabmap.conf && ldconfig
ENV LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH

# 12. Copie de ton script Python dans le conteneur
WORKDIR /rtabmap_ws
COPY ./src/rtabmap/script/ /rtabmap_ws/

# 13. Commande par défaut : exécute ton script Python
CMD ["python3", "/rtabmap_ws/rtabmap.py"]