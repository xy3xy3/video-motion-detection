import json
import os
from typing import Dict, List
import requests
import cv2
import numpy as np
import time

from db import get_config

# 加载Haar级联分类器
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)
plate_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_russian_plate_number.xml"
)

# 清除restmp内的文件
for root, dirs, files in os.walk("./client/restmp", topdown=False):
    for name in files:
        os.remove(os.path.join(root, name))
    for name in dirs:
        os.rmdir(os.path.join(root, name))
    break


# 隐私保护函数
def privacy_protect(image: np.ndarray,protect_type: str) -> np.ndarray:
    list_type = protect_type.split(",")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if "face" in list_type:
        # 检测人脸
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
        for x, y, w, h in faces:
            image[y : y + h, x : x + w] = cv2.blur(image[y : y + h, x : x + w], (30, 30))
    if "plate" in list_type:
        # 车牌打码函数
        plates = plate_cascade.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=5)
        for x, y, w, h in plates:
            image[y : y + h, x : x + w] = cv2.blur(image[y : y + h, x : x + w], (30, 30))
    return image


# 解析检测结果并绘制到图像上
def draw_detections(image: np.ndarray, detections: List[Dict[str, any]]) -> np.ndarray:
    for detection in detections:
        name = detection["name"]
        confidence = detection["confidence"]
        box = detection["box"]
        x1, y1 = int(box["x1"] * image.shape[1]), int(box["y1"] * image.shape[0])
        x2, y2 = int(box["x2"] * image.shape[1]), int(box["y2"] * image.shape[0])

        # 绘制边界框
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # 绘制标签和置信度
        label = f"{name}: {confidence:.2f}"
        cv2.putText(
            image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2
        )

    return image


# 发送图像数据到服务器并接收检测结果
def send_file(file_path: str, original_image: np.ndarray, output_path: str) -> dict:
    url = get_config("server_url") + "/predict/"
    with open(file_path, "rb") as f:
        start_time = time.time()  # 记录开始时间
        response = requests.post(url, files={"file": f})
        end_time = time.time()  # 记录结束时间
        receive_time = end_time - start_time  # 计算接收用时
        print(f"接收用时: {receive_time:.2f} 秒")
        if response.status_code == 200:
            detections = json.loads(response.json())
            print("检测结果:", detections)
            image_with_detections = draw_detections(original_image, detections)
            cv2.imwrite(output_path, image_with_detections)
            return detections
        else:
            print("请求失败:", response.json())
            return {}


# 压缩jpg
def compress_jpg(image: np.ndarray, rate:int = 90) -> np.ndarray:
    if rate == 100:
        return image
    _, buffer = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), rate])
    return cv2.imdecode(buffer, cv2.IMREAD_COLOR)


# 捕获图像并发送到服务器
def capture_and_send() -> None:
    cap = cv2.VideoCapture(0)
    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1
        if frame_count % 30 == 0:  # 每隔30帧发送一次
            frame = compress_jpg(frame)
            original_frame = frame.copy()
            frame = privacy_protect(frame)
            file_path = f"./client/restmp/frame_{frame_count}.jpg"
            output_path = f"./client/restmp/result_{frame_count}.jpg"
            # 本地临时文件用于上传
            cv2.imwrite(file_path, frame)
            send_file(file_path, original_frame, output_path)
            # 删除本地临时文件
            os.remove(file_path)
    cap.release()


# 读取视频文件并发送到服务器
def capture_and_send_from_video(video_path: str) -> None:
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1
        if frame_count % 30 == 0:  # 每隔30帧发送一次
            original_frame = frame.copy()
            frame = privacy_protect(frame)
            file_path = f"./client/restmp/frame_{frame_count}.jpg"
            output_path = f"./client/restmp/result_{frame_count}.jpg"
            # 本地临时文件用于上传
            cv2.imwrite(file_path, frame)
            send_file(file_path, original_frame, output_path)
            # 删除本地临时文件
            os.remove(file_path)
    cap.release()


if __name__ == "__main__":
    video_path = "./client/test.mp4"  # 视频文件路径
    # 调用摄像头版
    # capture_and_send()
    # 调用本地当前目录test.mp4模拟版
    # capture_and_send_from_video(video_path)
    print(cv2.data.haarcascades )
