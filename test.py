import yaml
import configparser
import sys
from pathlib import Path


import yaml
import sys


def yaml_to_ini(yaml_path, ini_path, image_path, depth_path):
    with open(yaml_path, 'r') as f:
        lines = f.readlines()

    # Ignore the first line if it starts with "%YAML"
    if lines[0].startswith("%YAML"):
        lines = lines[1:]

    calib = yaml.safe_load("".join(lines))

    with open(ini_path, 'w') as f:
        f.write("[camera]\n")
        f.write("model=PINHOLE\n")
        f.write("width={}\n".format(calib["image_width"]))
        f.write("height={}\n".format(calib["image_height"]))
        f.write("fx={}\n".format(calib["camera_matrix"]["data"][0]))
        f.write("fy={}\n".format(calib["camera_matrix"]["data"][4]))
        f.write("cx={}\n".format(calib["camera_matrix"]["data"][2]))
        f.write("cy={}\n".format(calib["camera_matrix"]["data"][5]))
        f.write("distortion_model=plumb_bob\n")
        f.write("d0={}\n".format(calib["distortion_coefficients"]["data"][0]))
        f.write("d1={}\n".format(calib["distortion_coefficients"]["data"][1]))
        f.write("d2={}\n".format(calib["distortion_coefficients"]["data"][2]))
        f.write("d3={}\n".format(calib["distortion_coefficients"]["data"][3]))
        f.write("d4={}\n".format(calib["distortion_coefficients"]["data"][4]))
        f.write("rgb={}\n".format(image_path))
        f.write("depth={}\n".format(depth_path))


if __name__ == '__main__':
    if len(sys.argv) != 5:
        print("Usage: python3 yaml2ini.py <calib.yaml> <out.ini> <rgb_dir> <depth_dir>")
        sys.exit(1)
    yaml_to_ini(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])
