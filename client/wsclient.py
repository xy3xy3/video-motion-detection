import asyncio
import json
import os
import websockets
import base64
import cv2
import numpy as np
import time
from typing import List, Dict

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

# 人脸模糊函数
def blur_faces(image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5
    )
    for x, y, w, h in faces:
        roi = image[y : y + h, x : x + w]
        roi = cv2.GaussianBlur(roi, (23, 23), 30)
        image[y : y + h, x : x + w] = roi
    return image

# 车牌打码函数
def blur_license_plates(image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    plates = plate_cascade.detectMultiScale(
        gray, scaleFactor=1.05, minNeighbors=5
    )
    for x, y, w, h in plates:
        plate_region = image[y : y + h, x : x + w]
        plate_region = cv2.GaussianBlur(plate_region, (15, 15), 0)
        image[y : y + h, x : x + w] = plate_region
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
# 带错误处理的WebSocket发送函数
async def ws_send(websocket, message):
    remote_ip = websocket.remote_address[0]
    try:
        await websocket.send(message)
    except websockets.exceptions.ConnectionClosedError:
        print(f"ws_send无法发送消息，远程链接已关闭: {remote_ip}")
    except Exception as e:
        print(f"ws_send发生错误: {str(e)}")
# 发送图像数据到服务器
async def send_file(websocket, frame: np.ndarray, image_name: str) -> None:
    try:
        start_time = time.time()  # 记录开始时间
        original_image = frame.copy()
        # 应用人脸和车牌模糊
        encode_data = blur_faces(frame)
        encode_data = blur_license_plates(encode_data)
        # 转为JPEG格式并发送
        _, buffer = cv2.imencode('.jpg', encode_data, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        image_data = buffer.tobytes()
        await ws_send(websocket, base64.b64encode(image_data).decode())
        await ws_send(websocket, f"END {image_name}")
        response = await websocket.recv()
        end_time = time.time()  # 记录结束时间
        receive_time = end_time - start_time  # 计算接收用时
        print(f"接收到的检测结果: type={type(response)} \n {response}")
        print(f"接收用时: {receive_time:.2f} 秒")
        if response != "[]":
            # 解析响应并绘制检测结果到图像
            detections = json.loads(response)
            image_with_detections = draw_detections(original_image, detections)
            # 保存结果图像
            result_path = os.path.join("./client/restmp", image_name)
            success = cv2.imwrite(result_path, image_with_detections)
            if success:
                print(f"结果图像已保存到: {result_path}")
            else:
                print(f"保存结果图像失败: {result_path} 报错：{cv2.error()}")
    except websockets.exceptions.ConnectionClosedError as e:
        print(f"连接已关闭: {e}")
    except Exception as e:
        print(f"发生错误: {str(e)}")

# 捕获图像并发送到服务器
async def capture_and_send(url: str, token: str) -> None:
    cap = cv2.VideoCapture(0)
    frame_count = 0
    async with websockets.connect(
        url, extra_headers={"Authorization": "Bearer " + token}
    ) as websocket:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame_count += 1
            image_name = f"frame_{frame_count}.jpg"
            await send_file(websocket, frame, image_name)
        cap.release()

# 读取视频文件并发送到服务器
async def capture_and_send_from_video(url: str, token: str, video_path: str) -> None:
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    async with websockets.connect(
        url, extra_headers={"Authorization": "Bearer " + token}
    ) as websocket:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame_count += 1
            if frame_count % 30 == 0:  # 每隔30帧发送一次
                image_name = f"frame_{frame_count}.jpg"
                await send_file(websocket, frame, image_name)
        cap.release()

if __name__ == "__main__":
    url = "ws://127.0.0.1:6789"
    # 调用摄像头版
    # asyncio.run(capture_and_send(
    #     "ws://127.0.0.1:6789",
    #     "1234"  # 确保这里的token与服务器端一致
    # ))
    # 调用本地当前目录test.mp4模拟版
    asyncio.run(
        capture_and_send_from_video(
            url, "1234", os.path.dirname(os.path.abspath(__file__)).replace("\\", "/")+"/test.mp4"  # 确保这里的token与服务器端一致  # 视频文件路径
        )
    )
