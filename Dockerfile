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

# 3. Ajout des dépôts ROS
RUN curl -sSL http://packages.ros.org/ros.key | apt-key add - && \
    echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" \
      > /etc/apt/sources.list.d/ros-latest.list

# 4. Installation de ROS Noetic complet
RUN apt-get update && apt-get install -y \
    ros-noetic-desktop-full \
    python3-rosdep \
    && apt-get clean

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
    && apt-get clean

# 8. Compilation de RTAB-Map depuis GitHub (repo officiel)
WORKDIR /opt
RUN git clone https://github.com/introlab/rtabmap.git && \
    mkdir -p rtabmap/build && \
    cd rtabmap/build && \
    cmake .. -DCMAKE_BUILD_TYPE=Release && \
    make -j"$(nproc)" && \
    make install

# 9. Copie de ton script Python dans le conteneur
WORKDIR /rtabmap_ws
COPY ./src/rtabmap/script/ /rtabmap_ws/

# 10. Commande par défaut : exécute ton script Python

RUN python3 -m pip install pandas
RUN pip install python-dotenv
RUN pip install tqdm

RUN mkdir -p /rtabmap_ws/rgb_sync
RUN mkdir -p /rtabmap_ws/depth_sync


RUN echo "/usr/local/lib" >> /etc/ld.so.conf.d/rtabmap.conf && ldconfig
ENV LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH



CMD ["python3", "/rtabmap_ws/rtabmap.py"]
