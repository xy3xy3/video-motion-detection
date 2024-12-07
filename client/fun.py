import io
import os
from typing import Dict, List
import cv2
import numpy as np
import onnxruntime
from runtime import detection
from db import get_config
from PIL import Image


# 初始化 ONNX 模型
model_path = "./model/FastestDet.onnx"  # 模型路径
session = onnxruntime.InferenceSession(model_path, providers=["CPUExecutionProvider"])
# `['CUDAExecutionProvider', 'CPUExecutionProvider']`
for root, dirs, files in os.walk("./client/restmp", topdown=False):
    for name in files:
        os.remove(os.path.join(root, name))
    for name in dirs:
        os.rmdir(os.path.join(root, name))
    break

def privacy_protect(image: np.ndarray, protect_type: str) -> np.ndarray:
    """
    隐私保护函数，利用 ONNX 模型检测并模糊人脸、车牌区域。
    """
    list_type = protect_type.split(",")
    input_width, input_height = 352, 352  # 模型输入大小
    thresh = 0.5  # 检测阈值

    # 使用 ONNX 模型检测
    detections = detection(session, image, input_width, input_height, thresh)

    # 检查检测结果是否有效
    if not detections or detections == [None]:
        print("无识别结果")
        return image  # 如果检测为空或无效，直接返回原图

    for det in detections:
        if det is None or len(det) != 6:  # 跳过无效的检测结果
            continue

        # 强制将坐标值转换为整数
        x1, y1, x2, y2 = map(int, det[:4])
        score, cls_index = det[4], det[5]
        print(f"识别结果: {score}, {cls_index}")
        # 检查类别是否需要保护
        if cls_index == 0 and "face" in list_type:  # 0 是人脸
            if y1 < y2 and x1 < x2 and y1 >= 0 and x1 >= 0 and y2 <= image.shape[0] and x2 <= image.shape[1]:
                image[y1:y2, x1:x2] = cv2.blur(image[y1:y2, x1:x2], (30, 30))
        elif cls_index == 1 and "plate" in list_type:  # 1 是车牌
            if y1 < y2 and x1 < x2 and y1 >= 0 and x1 >= 0 and y2 <= image.shape[0] and x2 <= image.shape[1]:
                image[y1:y2, x1:x2] = cv2.blur(image[y1:y2, x1:x2], (30, 30))

    return image

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

def privacy_protect_old(image: np.ndarray, protect_type: str) -> np.ndarray:
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


def compress_jpg_pil(image: np.ndarray, rate: int = 90) -> np.ndarray:
    if rate == 100:
        return image
    # 将 numpy 数组转换为 PIL 图像
    pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    # 压缩图像
    buffer = io.BytesIO()
    pil_image.save(buffer, format="JPEG", quality=rate)
    buffer.seek(0)
    # 将压缩后的图像转换回 numpy 数组
    compressed_image = np.array(Image.open(buffer))
    return cv2.cvtColor(compressed_image, cv2.COLOR_RGB2BGR)
if __name__ == "__main__":
    # 读取图片
    img = cv2.imread("./static/testimg/image3.png")
    # 模型输入的宽高
    input_width, input_height = 352, 352
    # 测试隐私保护
    img = privacy_protect(img,"face,plate")
    # 显示图片
    cv2.imwrite("./result.jpg", img)