import asyncio
import torch
import websockets
import base64
import os
import json
from datetime import datetime
from ultralytics import YOLO
import time

secret_token = "1234"

# 确保 tmp 文件夹存在
if not os.path.exists("tmp"):
    os.makedirs("tmp")
if not os.path.exists("restmp"):
    os.makedirs("restmp")

# 处理图像并使用YOLOv8进行检测
async def process_image(image_data, image_name):
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print("推理设备", device)
    model = YOLO("./models/yolov8n.pt")
    start_time = time.time()  # 记录开始时间

    with open(image_name, "wb") as image_file:
        image_file.write(image_data)
    
    try:
        res = model.predict(source=image_name, device=device,
                            classes=[0, 2, 3, 5, 6, 7, 9, 11, 12],
                            show_boxes=True, show_conf=True, show_labels=True,
                            conf=0.3)[0]
    except Exception as e:
        print(f"Error during YOLO prediction: {e}")
        return {}

    if res is None:
        print("Error: YOLO prediction result is None.")
        return {}
    
    res.show()
    end_time = time.time()  # 记录结束时间
    inference_time = end_time - start_time  # 计算推理用时
    print(f"推理用时: {inference_time:.2f} 秒")
    res.save(filename=f"restmp/{image_name.split('/')[-1]}")  # 保存结果图片

    # 清理临时文件夹，只保留最近10个图片
    clean_tmp_folder("tmp", 10)
    clean_tmp_folder("restmp", 10)

    return json.loads(res.tojson())

# 清理临时文件夹，只保留最近n个文件
def clean_tmp_folder(folder_path, n):
    files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
    files.sort(key=os.path.getmtime)
    if len(files) > n:
        for file in files[:-n]:
            os.remove(file)

# 接收并处理图像数据
async def save_and_process_image(websocket, temp_path):
    file_chunks = []
    async for message in websocket:
        if message.startswith("END"):
            file_name = message.split(" ")[1]
            image_data = b''.join(base64.b64decode(chunk) for chunk in file_chunks)
            detection_results = await process_image(image_data, temp_path)
            await websocket.send(json.dumps(detection_results))
            break
        else:
            file_chunks.append(message)
    remote_ip = websocket.remote_address[0]
    print(f"处理了来自 {remote_ip} 的 {file_name}")

# 处理WebSocket连接
async def handler(websocket, path):
    global secret_token
    remote_ip = websocket.remote_address[0]
    try:
        token = websocket.request_headers['Authorization']
        if token == "Bearer " + secret_token:
            print(f"授权来自 {remote_ip} 的连接")
            temp_path = f"tmp/temp_image_{datetime.now().strftime('%Y%m%d%H%M%S%f')}.jpg"
            await save_and_process_image(websocket, temp_path)
        else:
            await websocket.send("授权失败")
            print(f"来自 {remote_ip} 的未授权访问尝试")
    except websockets.exceptions.ConnectionClosedError as e:
        print(f"远程链接关闭: {remote_ip}, error: {e}")
    except Exception as e:
        try:
            await websocket.send(f"错误: {str(e)}")
        except websockets.exceptions.ConnectionClosedError:
            print(f"无法发送消息，远程链接已关闭: {remote_ip}")
        print(f"来自 {remote_ip} 的错误: {str(e)}")

# 主函数，启动WebSocket服务器
async def main():
    async with websockets.serve(handler, "0.0.0.0", 6789):
        await asyncio.Future()

try:
    asyncio.run(main())
except KeyboardInterrupt:
    print("服务器手动停止")
