import os
from typing import Dict, List
import cv2
import numpy as np
from db import get_config

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)
profile_face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_profileface.xml"
)
plate_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_russian_plate_number.xml"
)

for root, dirs, files in os.walk("./client/restmp", topdown=False):
    for name in files:
        os.remove(os.path.join(root, name))
    for name in dirs:
        os.rmdir(os.path.join(root, name))
    break

def privacy_protect(image: np.ndarray, protect_type: str) -> np.ndarray:
    list_type = protect_type.split(",")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if "face" in list_type:
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
        profile_faces = profile_face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
        for x, y, w, h in faces:
            image[y: y + h, x: x + w] = cv2.blur(image[y: y + h, x: x + w], (30, 30))
        for x, y, w, h in profile_faces:
            image[y: y + h, x: x + w] = cv2.blur(image[y: y + h, x: x + w], (30, 30))
    if "plate" in list_type:
        plates = plate_cascade.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=5)
        for x, y, w, h in plates:
            image[y: y + h, x: x + w] = cv2.blur(image[y: y + h, x: x + w], (30, 30))
    return image

def adjust_resolution(frame: np.ndarray) -> np.ndarray:
    # 检查有shape属性的对象是否为图像
    if not hasattr(frame, "shape"):
        return frame
    height, width, _ = frame.shape
    if height >= 1080 or width >= 1080:
        # 将分辨率调整为720p
        frame = cv2.resize(frame, (1280, 720), interpolation=cv2.INTER_LINEAR)
    return frame

def draw_detections(image: np.ndarray, detections: List[Dict[str, any]]) -> np.ndarray:
    for detection in detections:
        name = detection["name"]
        confidence = detection["confidence"]
        box = detection["box"]
        x1, y1 = int(box["x1"] * image.shape[1]), int(box["y1"] * image.shape[0])
        x2, y2 = int(box["x2"] * image.shape[1]), int(box["y2"] * image.shape[0])

        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = f"{name}: {confidence:.2f}"
        cv2.putText(
            image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2
        )

    return image

def compress_jpg(image: np.ndarray, rate: int = 90) -> np.ndarray:
    if rate == 100:
        return image
    _, buffer = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), rate])
    return cv2.imdecode(buffer, cv2.IMREAD_COLOR)
