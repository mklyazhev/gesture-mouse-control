import os
import glob
import shutil
import argparse
from pathlib import Path
import mediapipe as mp
import cv2

# SOTA hand detection util
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()

# Visualization util
mp_drawing = mp.solutions.drawing_utils

# <class-index> <x> <y> <width> <height> <px1> <py1> <px2> <py2> ... <pxn> <pyn>
# parser = argparse.ArgumentParser(description="Prepare data for YOLOv8")
# parser.add_argument("srcdir", type=str, help="Source directory with data")
# parser.add_argument("destdir", type=str, help="Destination directory for converted data")
# parser.add_argument("-m", "--mode", dest="mode", default="train", type=str, help="Which mode data will be converted")
# args = parser.parse_args()


SOURCE_DIR = "../../../datasets/HaGRID_512"  # args.srcdir
DESTINATION_DIR = "../../../datasets/HaGRID_YOLO_keypoints"  # args.destdir
MODE = "train"  # args.mode
PART_SIZE = -1
BOX_ADD_SIZE = 10


def get_filename(path: str, include_extension: bool = False):
    name, extension = os.path.splitext(os.path.basename(path))
    if include_extension:
        return "".join((name, extension))
    else:
        return name


def create_dir(path: str, exist: bool = True):
    Path(path).mkdir(parents=True, exist_ok=exist)


if MODE not in ("train", "test"):
    print(f"Unsupported mode: {MODE}")

print("Starting...")
images_dir = [d for d in glob.glob(f"{SOURCE_DIR}/images/*") if MODE in d][0]
classes = [get_filename(cl) for cl in glob.glob(f"{images_dir}/*") if get_filename(cl)]

print("Calculate samples count...")
count_samples = sum([len(glob.glob(f"{images_dir}/{cl}/*.jpg")[:PART_SIZE]) for cl in classes])
count_success = 0

print("Preparing needed dirs...")
create_dir(f"{DESTINATION_DIR}/{MODE}")

print("Start converting...")
for cl in classes:

    images = glob.glob(f"{images_dir}/{cl}/*.jpg")[:PART_SIZE]

    for i in range(len(images)):
        image_path = images[i]
        sample_name = os.path.splitext(os.path.basename(image_path))[0]

        image = cv2.imread(image_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Hand = 0
        class_idx = 0

        # Detect keypoints using mediapipe
        results = hands.process(image_rgb)

        image_height, image_width, _ = image.shape
        completed_string = ""

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                keypoints = [
                    (hand_landmarks.landmark[i].x, hand_landmarks.landmark[i].y)
                    for i in range(len(hand_landmarks.landmark))
                ]

                # Find box borders
                x_min = min([kp[0] for kp in keypoints])
                x_max = max([kp[0] for kp in keypoints])
                y_min = min([kp[1] for kp in keypoints])
                y_max = max([kp[1] for kp in keypoints])

                # Find box center coordinates
                x_center = (x_max - ((x_max - x_min) / 2))
                y_center = (y_max - ((y_max - y_min) / 2))

                # Calculate free space before image edges
                free_space_x = 1. - x_max
                free_space_y = 1. - y_max

                # Calculate box width/height
                width = (x_max - x_min + min((free_space_x * 2, x_min * 2, BOX_ADD_SIZE / image_width)))
                height = (y_max - y_min + min((free_space_y * 2, y_min * 2, BOX_ADD_SIZE / image_height)))

                # Find x/y points coordinates
                x_list = [kp[0] for kp in keypoints]
                y_list = [kp[1] for kp in keypoints]

                # Prepare string for writing in txt
                keypoints_string = ""
                for j in range(len(keypoints)):
                    if j != len(keypoints) - 1:
                        add_space = " "
                    else:
                        add_space = "\n"

                    keypoint = f"{x_list[j]} {y_list[j]}{add_space}"
                    keypoints_string += keypoint

                completed_string += f"{class_idx} {x_center} {y_center} {width} {height} {keypoints_string}"

        # Save sample
        new_image_path = f"{DESTINATION_DIR}/{MODE}/{sample_name}.jpg"
        new_label_path = f"{DESTINATION_DIR}/{MODE}/{sample_name}.txt"
        shutil.copy(image_path, new_image_path)
        with open(new_label_path, "w") as f:
            f.write(completed_string)

        count_success += 1
        print(f"\rProgress: {count_success}/{count_samples}", end="", flush=True)

print("\nCompleted!")
