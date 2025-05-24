#sudo docker run --rm -it   -v /home/paul/Cours/projet_fil_rouge/deer_walk/cam0/data:/rtabmap_ws/rgb_sync_docker   -v /home/paul/Cours/projet_fil_rouge/deer_walk/depth0/data:/rtabmap_ws/depth_sync_docker   -v /home/paul/Cours/projet_fil_rouge/img_timestamps.csv:/rtabmap_ws/img_timestamps.csv   -v /home/paul/Cours/projet_fil_rouge/output:/rtabmap_ws/output   -v /home/paul/Cours/projet_fil_rouge/depth_timestamps.csv:/rtabmap_ws/depth_timestamps.csv   -v /home/paul/Cours/projet_fil_rouge/rtabmap_calib.yaml:/rtabmap_ws/rtabmap_calib.yaml   rtabmap_ubuntu20


sudo docker run --rm -it   -v /home/paul/Cours/projet_fil_rouge/deer_walk/cam0/data:/rtabmap_ws/rgb_sync_docker   -v /home/paul/Cours/projet_fil_rouge/deer_walk/depth0/data:/rtabmap_ws/depth_sync_docker   -v /home/paul/Cours/projet_fil_rouge/img_timestamps.csv:/rtabmap_ws/img_timestamps.csv   -v /home/paul/Cours/projet_fil_rouge/output:/rtabmap_ws/output   -v /home/paul/Cours/projet_fil_rouge/depth_timestamps.csv:/rtabmap_ws/depth_timestamps.csv   -v /home/paul/Cours/projet_fil_rouge/rtabmap_calib.yaml:/rtabmap_ws/rtabmap_calib.yaml   rtabmap_ubuntu20

sudo docker rmi rtabmap_ubuntu20

sudo docker build -t rtabmap_ubuntu20 .