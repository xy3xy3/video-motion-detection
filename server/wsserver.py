import asyncio
import torch
import websockets
import base64
import os
from datetime import datetime
from ultralytics import YOLO
import time
import traceback 
from PIL import Image
import numpy as np
from io import BytesIO

secret_token = "1234"
log_record = False
# 确保 tmp 文件夹存在
if not os.path.exists("tmp"):
    os.makedirs("tmp")
if not os.path.exists("restmp"):
    os.makedirs("restmp")

# 带错误处理的WebSocket发送函数
async def ws_send(websocket, message):
    remote_ip = websocket.remote_address[0]
    try:
        await websocket.send(message)
    except websockets.exceptions.ConnectionClosedError:
        print(f"无法发送消息，远程链接已关闭: {remote_ip}")

# 处理图像并使用YOLOv8进行检测
async def process_image(image_data, image_name):
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print("推理设备", device)
    model = YOLO("./models/yolov8n.pt")
    start_time = time.time()  # 记录开始时间
    if log_record:
        with open(image_name, "wb") as image_file:
            image_file.write(image_data)
    try:
        # 将图像数据转换为NumPy数组
        image = Image.open(BytesIO(image_data))
        image_np = np.array(image)
        res = model.predict(source=image_np, device=device,
                            classes=[0, 2, 3, 5, 6, 7, 9, 11, 12],
                            show_boxes=True, show_conf=True, show_labels=True,
                            conf=0.3)[0]
    except Exception as e:
        print(f"Error during YOLO prediction: {e}")
        return {}

    if res is None:
        print("Error: YOLO prediction result is None.")
        return {}
    
    end_time = time.time()  # 记录结束时间
    inference_time = end_time - start_time  # 计算推理用时
    print(f"推理用时: {inference_time:.2f} 秒")
    # 防止下面报错
    data = res.boxes or res.obb
    if data is None:
        return "[]"
    if log_record:
        clean_tmp_folder("tmp", 10)
        clean_tmp_folder("restmp", 10)
        res.save(filename=f"restmp/{image_name.split('/')[-1]}")  # 保存结果图片    
    return res.tojson(True)

# 清理临时文件夹，只保留最近n个文件
def clean_tmp_folder(folder_path, n):
    files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
    files.sort(key=os.path.getmtime)
    if len(files) > n:
        for file in files[:-n]:
            os.remove(file)

# 接收并处理图像数据
async def save_and_process_image(websocket, temp_path):
    data = ""
    async for message in websocket:
        if message.startswith("END"):
            file_name = message.split(" ")[1]
            image_data = base64.b64decode(data)
            detection_results = await process_image(image_data, temp_path)
            await ws_send(websocket, detection_results)
            remote_ip = websocket.remote_address[0]
            print(f"处理了来自 {remote_ip} 的 {file_name}")
        else:
            data = message

# 处理WebSocket连接
async def handler(websocket, path):
    global secret_token
    remote_ip = websocket.remote_address[0]
    try:
        token = websocket.request_headers['Authorization']
        if token != "Bearer " + secret_token:
            await ws_send(websocket, "授权失败")
            print(f"来自 {remote_ip} 的未授权访问尝试")
        else:
            print(f"授权来自 {remote_ip} 的连接")
            while True:
                temp_path = f"tmp/temp_image_{datetime.now().strftime('%Y%m%d%H%M%S%f')}.jpg"
                await save_and_process_image(websocket, temp_path)
    except websockets.exceptions.ConnectionClosedError as e:
        print(f"远程链接关闭: {remote_ip}, error: {e}")
    except Exception as e:
        error_message = f"错误: {str(e)}"
        detailed_traceback = traceback.format_exc()
        print(f"来自 {remote_ip} 的错误: {detailed_traceback}")
        await ws_send(websocket, f"{error_message}\n详细错误信息: {detailed_traceback}")

# 主函数，启动WebSocket服务器
async def main():
    async with websockets.serve(handler, "0.0.0.0", 6789):
        await asyncio.Future()

try:
    asyncio.run(main())
except KeyboardInterrupt:
    print("服务器手动停止")
