import os
import glob
import shutil
import argparse
import json
from PIL import Image


# <class-index> <x> <y> <width> <height> <px1> <py1> <px2> <py2> ... <pxn> <pyn>
# parser = argparse.ArgumentParser(description="Prepare data for YOLOv8")
# parser.add_argument("srcdir", type=str, help="Source directory with data")
# parser.add_argument("destdir", type=str, help="Destination directory for converted data")
# parser.add_argument("-m", "--mode", dest="mode", default="train", type=str, help="Which mode data will be converted")
# args = parser.parse_args()

SOURCE_DIR = "../../../data/HandDB"#args.srcdir
DESTINATION_DIR = "../../../../datasets/HandDB_yolo"  #args.destdir
MODE = "train"#args.mode
INCORRECT_DATA_LABEL = 0.
BOX_ADD_SIZE = 10

target_dir = [d for d in glob.glob(f"{SOURCE_DIR}/*") if MODE in d][0]
labels_names = glob.glob(f"{target_dir}/*.json")
image_names = glob.glob(f"{target_dir}/*.jpg")

count_samples = len(image_names)
count_success = 0

for i in range(len(image_names)):
    image_path = image_names[i]
    label_path = labels_names[i]
    sample_name = os.path.splitext(os.path.basename(image_path))[0]

    image = Image.open(image_path)
    image.close()

    with open(label_path) as f:
        sample_json = json.loads(f.read())

    points = sample_json["hand_pts"]

    # Check if sample has incorrect keypoints (element with position index 2 in each point)
    if INCORRECT_DATA_LABEL in [p[2] for p in points]:
        continue

    # Hand = 0
    class_idx = 0

    # Find box center coordinates
    x_center = sample_json["hand_box_center"][0] / image.width
    y_center = sample_json["hand_box_center"][1] / image.height

    # Find box borders
    x_min = min([l[0] for l in sample_json["hand_pts"]])
    x_max = max([l[0] for l in sample_json["hand_pts"]])
    y_min = min([l[1] for l in sample_json["hand_pts"]])
    y_max = max([l[1] for l in sample_json["hand_pts"]])

    # Calculate free space before image edges
    free_space_x = image.width - x_max
    free_space_y = image.height - y_max

    # Calculate box width/height
    width = (x_max - x_min + min((free_space_x * 2, x_min * 2, BOX_ADD_SIZE))) / image.width
    height = (y_max - y_min + min((free_space_y * 2, y_min * 2, BOX_ADD_SIZE))) / image.height

    # Find x/y points coordinates
    x_list = [p[0] / image.width for p in points]
    y_list = [p[1] / image.height for p in points]

    # Prepare string for writing in txt
    points_string = ""
    for j in range(len(points)):
        if j != len(points) - 1:
            add_space = " "
        else:
            add_space = ""

        point = f"{x_list[j]} {y_list[j]}{add_space}"
        points_string += point

    completed_string = f"{class_idx} {x_center} {y_center} {width} {height} {points_string}"

    new_image_path = f"{DESTINATION_DIR}/{MODE}/{sample_name}.jpg"
    new_label_path = f"{DESTINATION_DIR}/{MODE}/{sample_name}.txt"
    shutil.copy(image_path, new_image_path)
    with open(new_label_path, "w") as f:
        f.write(completed_string)

    count_success += 1
    print(f"\rProgress:{i + 1}/{count_samples} Successfully converts:{count_success}", end="", flush=True)
