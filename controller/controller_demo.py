import argparse
import logging
import time
from enum import Enum
from typing import Optional, Tuple

import cv2
import numpy as np
import torch
from omegaconf import OmegaConf
from PIL import Image
from torch import Tensor
from torchvision.transforms import functional as f
from ultralytics import YOLO

from classifier.utils import build_model
from classifier.constants import gestures, hands
from controller.gesture_handler import GestureHandler

import warnings
warnings.filterwarnings("ignore")

logging.basicConfig(format="[LINE:%(lineno)d] %(levelname)-8s [%(asctime)s]  %(message)s", level=logging.INFO)

COLOR_GREEN = (0, 255, 0)
COLOR_RED = (0, 0, 255)
FONT = cv2.FONT_HERSHEY_SIMPLEX


class Mode(Enum):
    DETECT = 1
    CLASSIFICATION = 2
    CONTROL = 3

# Добавить вывод режима запуска, добавить вывод активности контроллера, добавить рамку рабочей области
# Вынести соотношение сторон в конфигурационный файл (16 : 10)

class Demo:
    @staticmethod
    def preprocess(img: np.ndarray) -> Tuple[Tensor, Tuple[int, int], Tuple[int, int]]:
        """
        Preproc image for model input
        Parameters
        ----------
        img: np.ndarray
            input image
        """
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(img)
        width, height = image.size

        image = image.resize((224, 224))

        img_tensor = f.pil_to_tensor(image)
        img_tensor = f.convert_image_dtype(img_tensor)
        img_tensor = img_tensor[None, :, :, :]
        return img_tensor, (width, height)

    @staticmethod
    def draw_workspace_rect(frame, screen_aspect_ratio):
        h, w, _ = frame.shape

        scaled_w = w - (0.3 * w)
        scaled_h = scaled_w * screen_aspect_ratio

        center_x = w / 2
        center_y = h / 2

        cv2.rectangle(
            frame,
            (int(center_x - (scaled_w / 2)), int(center_y + (scaled_h / 2))),
            (int(center_x + (scaled_w / 2)), int(center_y - (scaled_h / 2))),
            color=COLOR_GREEN,
            thickness=2
        )

        return center_x, center_y, scaled_w, scaled_h

    @staticmethod
    def is_in_workspace(coords, xywh):
        x, y, w, h = xywh

        if not ((x - w / 2) <= coords[0] <= (x + w / 2)):
            return False

        if not ((y - h / 2) <= coords[1] <= (y + h / 2)):
            return False

        return True

    @staticmethod
    def scale_coords(coords, workspace_xywh, screen_wh):
        ws_x, ws_y, ws_w, ws_h = workspace_xywh
        s_w, s_h = screen_wh
        x, y = coords

        new_x = (x - (ws_x - ws_w / 2)) * (s_w / ws_w)
        new_y = (y - (ws_y - ws_h / 2)) * (s_h / ws_h)

        return new_x, new_y

    @staticmethod
    def run(detector=None, classifier=None, handler=None, mode=3) -> None:
        """
        Run demonstration in selected mode
        Parameters
        ----------
        detector : TorchVisionModel
            Detector model
        classifier : TorchVisionModel
            Classifier model
        mode : int
            Running mode
        """

        # Check that mode is supported
        try:
            Mode(mode)
        except ValueError:
            raise ValueError(
                f"Invalid mode: {mode}!\n\
                Supported modes:\n\
                {Mode.DETECT.name} = {Mode.DETECT.value}\n\
                {Mode.CLASSIFY.name} = {Mode.CLASSIFY.value}\n\
                {Mode.CONTROL.name} = {Mode.CONTROL.value}\n"
            )

        cap = cv2.VideoCapture(0)
        t1 = cnt = 0

        screen_wh = handler.controller.get_screen_size()
        screen_ar = handler.controller.get_screen_aspect_ratio()

        while cap.isOpened():
            delta = time.time() - t1
            t1 = time.time()

            ret, frame = cap.read()
            frame = cv2.flip(frame, 1)  # Mirroring cam frame for usability (fps 5)
            # Draw selected mode
            cv2.putText(
                frame,
                f"Mode: {Mode(mode).name}, controller: {'on' if handler.controller.active else 'off'}",
                org=(10, 60),
                fontFace=FONT,
                fontScale=1,
                color=COLOR_GREEN,
                thickness=2
            )

            if ret:
                if mode == 3:
                    # Calculate and drawing workspace for gesture control
                    workspace_xywh = Demo.draw_workspace_rect(frame, screen_ar)

                if mode >= 1:  # If mode support detection
                    bboxes = detector(frame, imgsz=320, verbose=False)[0].boxes.xywh

                    if bboxes.numel():  # Check that tensor is not empty
                        bbox = bboxes[0]
                        bb_x, bb_y, bb_w, bb_h = bbox
                        cv2.rectangle(
                            frame,
                            (int(bb_x - (bb_w / 2)), int(bb_y + (bb_h / 2))),
                            (int(bb_x + (bb_w / 2)), int(bb_y - (bb_h / 2))),
                            color=COLOR_RED,
                            thickness=2
                        )

                        if mode >= 2:  # If mode support classification
                            cropped_frame = frame[
                                int(bb_y - (bb_h / 2)):int((bb_y - (bb_h / 2)) + bb_h),
                                int(bb_x - (bb_w / 2)):int((bb_x - (bb_w / 2)) + bb_w)
                            ]
                            processed_frame, size = Demo.preprocess(cropped_frame)
                            with torch.no_grad():
                                output = classifier(processed_frame)
                            gesture = gestures[int(output["gesture"].argmax(dim=1)) + 1]
                            hand = hands[int(output["leading_hand"].argmax(dim=1)) + 1]
                            cv2.putText(
                                frame,
                                f"{gesture}, {hand}",
                                org=(10, 90),
                                fontFace=FONT,
                                fontScale=1,
                                color=COLOR_RED,
                                thickness=2
                            )

                            if mode == 3:  # If mode support controller
                                handler.queue.append(gesture)  # Add gestures in queue for dynamic gestures
                                coords = (bb_x, bb_y)
                                if Demo.is_in_workspace(coords, workspace_xywh):
                                    coords = Demo.scale_coords(coords, workspace_xywh, screen_wh)
                                    args = {"coords": coords}
                                    handler.process(gesture, args)

                fps = 1 / delta
                # frame = cv2.flip(frame, 1)  # Mirroring cam frame for usability
                cv2.putText(
                    frame,
                    f"FPS: {fps :02.1f}",
                    org=(10, 30),
                    fontFace=FONT,
                    fontScale=1,
                    color=COLOR_GREEN,
                    thickness=2
                )
                cnt += 1

                cv2.imshow("Frame", frame)

                key = cv2.waitKey(1)
                if key == ord("q"):
                    return
            else:
                cap.release()
                cv2.destroyAllWindows()


def parse_arguments(params: Optional[Tuple] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Demo detection...")

    parser.add_argument("-p", "--path_to_config", required=True, type=str, help="Path to config")

    parser.add_argument("-lm", "--landmarks", required=False, action="store_true", help="Use landmarks")

    known_args, _ = parser.parse_known_args(params)
    return known_args


classifier = build_model(
    model_name="ResNet18",
    num_classes=11,
    checkpoint=r"C:\Users\mikhail.klyazhev\Desktop\study\PycharmProjects\MouseCursorHandControl\classifier\weights\ResNet18-HaGRID_full-5ep.pth",
    device="cpu",
    pretrained=True,
    freezed=False,
    ff=False,
)
classifier.eval()

detector = YOLO("..\detector\weights\YOLOv8n-HaGRID_ss-3ep.pt")

handler = GestureHandler()

Demo.run(detector, classifier, handler, 3)
