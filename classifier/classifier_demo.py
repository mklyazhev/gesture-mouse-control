import argparse
import logging
import time
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
from classifier.constants import gestures

logging.basicConfig(format="[LINE:%(lineno)d] %(levelname)-8s [%(asctime)s]  %(message)s", level=logging.INFO)

COLOR = (0, 255, 0)
FONT = cv2.FONT_HERSHEY_SIMPLEX


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
    def run(classifier) -> None:
        """
        Run detection model and draw bounding boxes on frame
        Parameters
        ----------
        classifier : TorchVisionModel
            Classifier model
        """

        cap = cv2.VideoCapture(0)
        t1 = cnt = 0

        ###
        detector = YOLO("..\detector\weights\YOLOv8n-HaGRID_ss-3ep.pt")
        ###

        while cap.isOpened():
            delta = time.time() - t1
            t1 = time.time()

            ret, frame = cap.read()
            if ret:

                ###
                # print(detector(frame))
                boxes = detector(frame, imgsz=320)[0].boxes.xywh
                if boxes.numel():  # Check is tensor not empty
                    bbox = boxes[0]
                    cv2.rectangle(
                        frame,
                        (int(bbox[0] - (bbox[2] / 2)), int(bbox[1] + (bbox[3] / 2))),
                        (int(bbox[0] + (bbox[2] / 2)), int(bbox[1] - (bbox[3] / 2))),
                        color=(0, 0, 255),
                        thickness=2
                    )
                    cropped = frame[
                              int(bbox[1] - (bbox[3] / 2)):int((bbox[1] - (bbox[3] / 2)) + bbox[3]),
                              int(bbox[0] - (bbox[2] / 2)):int((bbox[0] - (bbox[2] / 2)) + bbox[2])
                              ]
                    processed_frame, size = Demo.preprocess(cropped)
                    with torch.no_grad():
                        output = classifier(processed_frame)
                    label = output["gesture"].argmax(dim=1)
                else:
                    # processed_frame, size = Demo.preprocess(frame)
                    label = 10
                ###

                # processed_frame, size = Demo.preprocess(frame)
                # with torch.no_grad():
                #     output = classifier(processed_frame)
                # label = output["gesture"].argmax(dim=1)

                # print(targets[int(label) + 1])
                cv2.putText(
                    frame, gestures[int(label) + 1], (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), thickness=3
                )
                fps = 1 / delta
                cv2.putText(frame, f"FPS: {fps :02.1f}, Frame: {cnt}", (30, 30), FONT, 1, (255, 0, 255), 2)
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


# resnet18-hagrid_full-5ep-last
model = build_model(
    model_name="ResNet18",
    num_classes=10 + 1,
    checkpoint=r"C:\Users\mikhail.klyazhev\Desktop\study\PycharmProjects\MouseCursorHandControl\src\classifier\weights\ResNet18-HaGRID_full-5ep.pth",
    device="cpu",
    pretrained=True,
    freezed=False,
    ff=False,
)
model.eval()
Demo.run(model)
