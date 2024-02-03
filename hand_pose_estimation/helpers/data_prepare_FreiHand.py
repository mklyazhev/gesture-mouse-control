import os
import glob
import shutil
import argparse
import json
from PIL import Image
import numpy as np


def projectPoints(xyz, K):
    xyz = np.array(xyz)
    K = np.array(K)
    uv = np.matmul(K, xyz.T).T
    return uv[:, :2] / uv[:, -1:]

# <class-index> <x> <y> <width> <height> <px1> <py1> <px2> <py2> ... <pxn> <pyn>
# parser = argparse.ArgumentParser(description="Prepare data for YOLOv8")
# parser.add_argument("srcdir", type=str, help="Source directory with data")
# parser.add_argument("destdir", type=str, help="Destination directory for converted data")
# parser.add_argument("-m", "--mode", dest="mode", default="train", type=str, help="Which mode data will be converted")
# args = parser.parse_args()

SOURCE_DIR = "../../../../datasets/FreiHAND_pub_v2"  #args.srcdir
DESTINATION_DIR = "../../../../datasets/FreiHAND_yolo"  #args.destdir
MODE = "test"#args.mode
BOX_ADD_SIZE = 10
if MODE == "train":
    START = 0
    END = 26048
elif MODE == "test":
    START = 26048
    END = 32560
else:
    print(f"Unsupported mode - {MODE}!")

path_K_matrix = glob.glob(f"{SOURCE_DIR}/training_K.json")[0]
# path_K_matrix = r"C:\Users\mikhail.klyazhev\Desktop\study\PycharmProjects\datasets\FreiHAND_pub_v2\training_K.json"
with open(path_K_matrix, "r") as f:
    K_matrix = np.array(json.load(f))

path_anno = glob.glob(f"{SOURCE_DIR}/training_xyz.json")[0]
# path_anno = r"C:\Users\mikhail.klyazhev\Desktop\study\PycharmProjects\datasets\FreiHAND_pub_v2\training_xyz.json"
with open(path_anno, "r") as f:
    anno = np.array(json.load(f))

image_names = glob.glob(f"{SOURCE_DIR}/training/rgb/*.jpg")[START:END]

count_samples = len(image_names)
count_success = 0

for i in range(len(image_names)):
    image_path = image_names[i]
    sample_name = os.path.splitext(os.path.basename(image_path))[0]
    keypoints = projectPoints(anno[i], K_matrix[i])

    image = Image.open(image_path)
    image.close()

    # Hand = 0
    class_idx = 0

    # Find box borders
    x_min = min([kp[0] for kp in keypoints])
    x_max = max([kp[0] for kp in keypoints])
    y_min = min([kp[1] for kp in keypoints])
    y_max = max([kp[1] for kp in keypoints])

    # Find box center coordinates
    x_center = (x_max - ((x_max - x_min) / 2)) / image.width
    y_center = (y_max - ((y_max - y_min) / 2)) / image.height

    # Calculate free space before image edges
    free_space_x = image.width - x_max
    free_space_y = image.height - y_max

    # Calculate box width/height
    width = (x_max - x_min + min((free_space_x * 2, x_min * 2, BOX_ADD_SIZE))) / image.width
    height = (y_max - y_min + min((free_space_y * 2, y_min * 2, BOX_ADD_SIZE))) / image.height

    # Find x/y points coordinates
    x_list = [kp[0] / image.width for kp in keypoints]
    y_list = [kp[1] / image.height for kp in keypoints]

    # Prepare string for writing in txt
    points_string = ""
    for j in range(len(keypoints)):
        if j != len(keypoints) - 1:
            add_space = " "
        else:
            add_space = ""

        point = f"{x_list[j]} {y_list[j]}{add_space}"
        points_string += point

    completed_string = f"{class_idx} {x_center} {y_center} {width} {height} {points_string}"

    # Save sample
    new_image_path = f"{DESTINATION_DIR}/{MODE}/{sample_name}.jpg"
    new_label_path = f"{DESTINATION_DIR}/{MODE}/{sample_name}.txt"
    shutil.copy(image_path, new_image_path)
    with open(new_label_path, "w") as f:
        f.write(completed_string)

    count_success += 1
    print(f"\rProgress:{i + 1}/{count_samples} Successfully converts:{count_success}", end="", flush=True)
