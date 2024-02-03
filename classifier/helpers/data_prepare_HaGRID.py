import argparse
import glob
import json
import os
from pathlib import Path
import shutil

# TODO: сделать рабочий аргпарс
# TODO: сделать автоматическое создание директорий и вынести длинные функции os.path.basename в отдельные функции utils,
#  которые будут использоваться и в детекторе, и в классификаторе

# <class-index> <x_center> <y_center> <width> <height>
# parser = argparse.ArgumentParser(description="Prepare data for YOLOv8")
# parser.add_argument("srcdir", type=str, help="Source directory with data")
# parser.add_argument("destdir", type=str, help="Destination directory for converted data")
# parser.add_argument("-m", "--mode", dest="mode", default="train", type=str, help="Which mode data will be converted")
# parser.add_argument("-ps", "--part-size", dest="part_size", default="-1", type=int, help="Subsample size of each class. All samples will be used by default")
# args = parser.parse_args()

SOURCE_DIR = "../../../datasets/hagrid_dataset_512"  # args.srcdir
DESTINATION_DIR = "../../../datasets/HaGRID_ResNext_classification"  # args.destdir
MODE = "train"  # args.mode
PART_SIZE = 10000  # -1
TARGET_CLASSES = (
    "like",
    "dislike",
    "ok",
    "one",
    "two_up",
    "three",
    "fist",
    "palm",
    "stop",
    "stop_inverted",
)


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
labels_dir = [d for d in glob.glob(f"{SOURCE_DIR}/annotations/*") if MODE in d][0]
classes = [get_filename(cl) for cl in glob.glob(f"{images_dir}/*") if get_filename(cl) in TARGET_CLASSES]

print("Calculate samples count...")
count_samples = sum([len(glob.glob(f"{images_dir}/{cl}/*.jpg")[:PART_SIZE]) for cl in classes])
count_success = 0

print("Preparing needed dirs...")
create_dir(f"{DESTINATION_DIR}/images/{MODE}")
create_dir(f"{DESTINATION_DIR}/annotations/{MODE}")

print("Start converting...")
for cl in classes:
    labels_path = glob.glob(f"{labels_dir}/{cl}.json")[0]
    with open(labels_path, "r") as f:
        labels = json.load(f)

    images = glob.glob(f"{images_dir}/{cl}/*.jpg")[:PART_SIZE]
    new_labels = {}

    # Creating class dir
    create_dir(f"{DESTINATION_DIR}/images/{MODE}/{cl}")

    for i in range(len(images)):
        image_path = images[i]
        sample_name = os.path.splitext(os.path.basename(image_path))[0]
        new_labels[sample_name] = labels[sample_name]

        # Save sample
        new_image_path = f"{DESTINATION_DIR}/images/{MODE}/{cl}/{sample_name}.jpg"
        shutil.copy(image_path, new_image_path)

        count_success += 1
        print(f"\rProgress: {count_success}/{count_samples}", end="", flush=True)

    with open(f"{DESTINATION_DIR}/annotations/{MODE}/{cl}.json", "w") as f:
        json.dump(new_labels, f, indent=4)

print("\nCompleted!")
