# 3D Mapping Project with RTAB-Map

![](https://github.com/mbathe/projet_fil_rouge/blob/main/data/deer_walk_3dmap.png)

## Project Overview

This project enables the generation of a 3D map from various input sources:
- Monocular Videos (split into images, with depth estimation)
- RGB images (with depth estimation)
- Existing RGB-D images

The main workflow consists of:
1. **Data acquisition**: video or image sequence
2. **Depth estimation**: using the DepthAnythingV2 metric model to reconstruct metric depth images
3. **3D mapping**: using RTAB-Map via Docker to generate a 3D cartography
4. **Export**: point cloud or mesh in .ply format for visualization and analysis

## Key Technologies

- **RTAB-Map** (Real-Time Appearance-Based Mapping): SLAM framework for 3D mapping
- **DepthAnythingV2**: Deep learning model for metric depth estimation from RGB images
- **Docker**: Containerization of complex dependencies
- **Python**: Orchestration of the complete pipeline

## Project Structure

```
project/
├── data/                  # Example data
├── notebook/              # Experimentation notebooks
├── src/
│   ├── depth/             # Depth estimation code
│   ├── rtabmap/           # 3D mapping code
│   └── main.py            # Application entry point
├── output/                # RTAB-Map database, mesh and cloud files
├── weight/                # Deep learning model weights
└── scripts/               # Utility scripts
```

## Main Modules

### 1. Depth Estimation
- Uses the **DepthAnythingV2** model to generate depth maps from RGB images
- Processes either individual images or extracts frames from a video
- Calibrates and normalizes depth data for RTAB-Map

### 2. RTAB-Map 3D Mapping
- Uses RGB-D pairs to build a 3D representation
- Generates a database of the environment with localization information
- Runs SLAM algorithms to align images in 3D space

### 3. Export and Visualization
- Generates 3D point clouds (.ply)
- Creates 3D meshes
- Offers 2D projection options of the 3D model

## Installation and Setup

### Prerequisites
- Python 3.8+
- Docker
- GPU recommended for depth model inference

### Docker Installation

To install Docker, follow the official Docker documentation for your operating system:
- **Official installation site**: [https://docs.docker.com/engine/install/](https://docs.docker.com/engine/install/)
- Choose your Linux distribution, or Windows/macOS as appropriate

### Docker Without Sudo (important)

**IMPORTANT**: Since Docker is invoked directly from the Python code, it is **crucial** to configure Docker to run without sudo on Linux systems. Without this, the Python scripts will not be able to execute Docker commands properly.

Follow the post-installation instructions for your platform:
- **Post-install documentation**: [https://docs.docker.com/engine/install/linux-postinstall/](https://docs.docker.com/engine/install/linux-postinstall/)

Main steps:
1. Add your user to the Docker group
2. Apply group changes
3. Verify installation without sudo
4. Set Docker to start on boot

### Environment Setup

#### 1. **Install Python dependencies**:
```bash
pip install -r requirements.txt
```

#### 2. **Automatic Download of Depth Anything V2**

The Python script (`download_depth_anything.py`) allows you to automatically download the official **Depth Anything V2 metric** repository as well as the associated model weights.

##### Usage

**2.1 Set the model type**  
   Set the `DEPTH_ANYTHING_TYPE` environment variable in the .env file according to the desired model type:  
   - `small` (default)
   - `base` 
   - `large`

**2.2 Run the script**  
   Execute the script from the terminal:
   ```bash
   python scripts/download_depth_anything.py
   ```

**2.3 Follow the instructions**  
   The script will guide you to choose the destination folder, install dependencies, and download the model weights corresponding to the selected type.



#### 3. **Download the test dataset**:

Run the following command at the project root to download the deer_walk test image dataset to ./data/deer_walk/ (set by the DIR_DATASET environment variable):
```bash
python scripts/download_dataset.py
```

Caution : this dataset is intended for example purposes only, as it is under non-commercial license. 
### Build the Custom Docker Image

The project uses a custom Docker image that contains RTAB-Map and the necessary scripts for 3D mapping.

**IMPORTANT**: Before running the main program, you must build the Docker image:

```bash
sudo docker build -t rtabmap_ubuntu20 .
```
on Linux or 

```bash
docker build -t rtabmap_ubuntu20 .
```

The `Dockerfile` at the project root:
1. Builds the Docker image with RTAB-Map and all required dependencies
2. Injects the script `./src/rtabmap/rtabmap_script.py` into the image
3. Sets up the runtime environment for 3D mapping

This script is automatically called when the Docker container is run from the Python code and handles the 3D mapping process.

**Note**: You do not need to install RTAB-Map separately or download another Docker image, as the Dockerfile sets up everything required.

**Note**: Every time you modify the contents of the `./src/rtabmap/` directory, you must rebuild the Docker image for changes to take effect.

## Usage

### Absolute Paths Required

**Important**: Since the program uses Docker with volume mounts, all paths must be **absolute** and not relative. Relative paths will not work because Docker requires full paths to mount volumes correctly.

In the examples below, replace `<PROJECT_ROOT>` with the absolute path to your project root.

### Full Example with Absolute Paths

```bash
python3 <PROJECT_ROOT>/src/main.py \
  --image_folder "<PROJECT_ROOT>/data/images" \
  --depth_folder "<PROJECT_ROOT>/data/depth" \
  --calibration_file "<PROJECT_ROOT>/data/rtabmap_calib.yaml"
```

For example, if your project is located at `/home/user/cartographie3d`, all paths should start with that root.

### Using the deer_walk Test Data

If you have run the above command to download the test dataset, you can then run the command below to generate the map from this dataset. Note that the database file and generated mesh or cloud files will be in the ./output directory at the project root.

```bash
python3 <PROJECT_ROOT>/projet_fil_rouge/src/main.py
```

### Video Mode (from a video source)

```bash
python src/main.py --source video --image_folder ./path/to/video.mp4 --frequence 5
```

### Image Mode (RGB without depth)

```bash
python src/main.py --source image --image_folder ./path/to/images
```

### RGB-D Mode (images with depth)

```bash
python src/main.py --source image_with_depth --image_folder ./path/to/images/rgb --depth_folder ./path/to/images/depth
```

### Available Arguments

Here is the full list of arguments accepted by the script:

```
--image_folder        Folder containing RGB images or path to the video file (default: "./image_folder")
--depth_folder         Folder containing depth images (default: "./depth_folder")
--calibration_file     Path to the camera calibration file (default: "./rtabmap_calib.yaml")
--source               Source to use: "image" (RGB without depth), "image_with_depth" (RGB-D), "video" (video)
                       (default: "image_with_depth")
--frequence            Frame extraction frequency from video in Hz (default: 20)
```

### Usage Examples

#### Video processing at 10 Hz
```bash
python src/main.py --source video --image_folder ./data/video.mp4 --frequence 10
```

#### RGB image processing with depth estimation
```bash
python src/main.py --source image --image_folder ./data/rgb_images
```

#### Existing RGB-D image processing with timestamp files
```bash
python src/main.py --source image_with_depth --image_folder ./data/rgb --depth_folder ./data/depth
```

## Data Format

### Structure for Image Sequences
Images should be named sequentially or with timestamps.

### CSV Format for Timestamps
If you use custom timestamps, the CSV must contain:
- `timestamp`: number (float or int)
- `filename`: exact image name (with extension)

Example:
```csv
timestamp,filename
1713456011.123456,rgb_001.png
1713456011.323456,rgb_002.png
```

## Advanced RTAB-Map Parameters

The project exposes several RTAB-Map parameters for advanced users:
- Visual odometry parameters
- Loop closure options
- Point cloud filtering
- Mesh optimization parameters

### Parameter Configuration Files

The `<PROJECT_ROOT>/src/rtabmap/rtabmap_params/` directory contains three JSON files to finely configure RTAB-Map behavior:

1. **`export_params.json`**: Parameters for exporting point clouds and meshes
   - Export format (PLY, OBJ, etc.)
   - Point cloud density
   - Texture and coloring options
   - Export filters (distance, noise, etc.)

2. **`generate_db_params.json`**: Parameters for initial database generation
   - Feature point detection parameters
   - Camera calibration options
   - Map optimization parameters
   - Feature matching configuration

3. **`reprocess_params.json`**: Parameters for reprocessing an existing database
   - Filtering options
   - Re-optimization parameters
   - Loop closure techniques
   - Global adjustment configuration

You can modify these files as needed to fine-tune your 3D mapping results.

See the full RTAB-Map documentation for more details: [RTAB-Map Documentation](http://wiki.ros.org/rtabmap_ros/Tutorials/Advanced%20Parameter%20Tuning)

## Extensions and Customization

- Integration of other depth estimation models
- Spatial filtering on the generated point cloud
- Support for different image formats (.jpg, .tiff, etc.)
- Detailed logging
- Parallelization for improved performance


## Notebooks

The requirements needed to run the notebooks are listed separately, as they are not required to run the rest of the code. You can install those by running `pip install -r notebooks/requirements.txt`

The notebooks are intended to :
* Show how to visualize the cloudpoints generated using Python and Open3D,
* Explore and present various techniques for rendering the structure of a scene or room in 2D, with the aim of capturing its spatial layout (for cartography purpose for example).

