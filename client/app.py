import base64
import os
import threading
import asyncio
from datetime import datetime
import numpy as np
import cv2
from fastapi import FastAPI, Request, Form, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import websockets
from io import BytesIO
from db import *
import httpclient

app = FastAPI()
templates = Jinja2Templates(directory="templates")

app.config = {
    "UPLOAD_FOLDER": "static/restmp"
}

# 配置静态文件目录
app.mount("/static", StaticFiles(directory="static"), name="static")

os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

processing_thread = None
is_processing = False

websocket_clients = []

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/start_processing")
async def start_processing(input_type: str = Form(...), video_path: str = Form(None)):
    global processing_thread, is_processing

    if is_processing:
        return JSONResponse(content={"status": "processing"}, status_code=400)

    start_time = datetime.now()
    log = create_log(start_time=start_time, end_time=None)
    log_id = log.id

    if input_type == "camera":
        processing_thread = threading.Thread(target=process_camera, args=(log_id,))
    elif input_type == "video":
        processing_thread = threading.Thread(target=process_video, args=(log_id, video_path))
    else:
        return JSONResponse(content={"status": "invalid input type"}, status_code=400)

    is_processing = True
    processing_thread.start()

    return JSONResponse(content={"status": "started", "log_id": log_id})

@app.post("/stop_processing")
async def stop_processing():
    global is_processing
    is_processing = False
    return JSONResponse(content={"status": "stopped"})

@app.get("/set", response_class=HTMLResponse)
async def get_settings(request: Request):
    config = {
        "server_url": get_config("server_url"),
        "protect_type": get_config("protect_type"),
        "compress": get_config("compress")
    }
    return templates.TemplateResponse("set.html", {"request": request, "config": config})

@app.post("/set")
async def update_settings(request: Request):
    data = await request.json()
    server_url = data.get("server_url")
    protect_type = data.get("protect_type")
    compress = data.get("compress")
    save_config("server_url", server_url)
    save_config("protect_type", protect_type)
    save_config("compress", str(compress))

    return JSONResponse(content={"status": "success"})

def process_camera(log_id):
    global is_processing
    rate = int(get_config("compress"))
    protect_type = get_config("protect_type")
    cap = cv2.VideoCapture(0)
    frame_count = 0
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    while is_processing and cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1
        original_frame = frame.copy()
        frame = httpclient.privacy_protect(frame, protect_type)
        frame = httpclient.compress_jpg(frame, rate)
        loop.run_until_complete(send_frame_ws(frame, log_id, original_frame))

    cap.release()
    update_log(log_id, end_time=datetime.now())
    is_processing = False

def process_video(log_id, video_path):
    global is_processing
    rate = int(get_config("compress"))
    protect_type = get_config("protect_type")
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    while is_processing and cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1
        original_frame = frame.copy()
        frame = httpclient.privacy_protect(frame, protect_type)
        frame = httpclient.compress_jpg(frame, rate)
        loop.run_until_complete(send_frame_ws(frame, log_id, original_frame))

    cap.release()
    update_log(log_id, end_time=datetime.now())
    is_processing = False

# 发送给服务端
async def send_frame_ws(frame, log_id, original_frame):
    server_url = get_config("server_url").replace("http", "ws") + "/ws/predict"
    async with websockets.connect(server_url) as websocket:
        _, buffer = cv2.imencode('.jpg', frame)
        frame_base64 = base64.b64encode(buffer).decode('utf-8')
        await websocket.send(frame_base64)  # 发送base64编码的图像数据
        data = await websocket.recv()
        process_detection_result(data, log_id, original_frame)

# 处理服务端的结果
def process_detection_result(data, log_id, original_frame):
    detection_result = json.loads(data)
    create_frame(log_id, time=datetime.now(), data=detection_result)
    # 本地画框
    image_with_detections = httpclient.draw_detections(original_frame, detection_result)
    # base64编码 opencv
    _, buffer = cv2.imencode('.jpg', image_with_detections)
    img_str = base64.b64encode(buffer).decode()
    # 发送给前端
    for client in websocket_clients:
        asyncio.create_task(send_image_to_client(client, img_str))

async def send_image_to_client(client, img_str):
    try:
        await client.send_text(img_str)
    except Exception as e:
        print(f"Failed to send image to client: {e}")


@app.websocket("/ws/stream")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    websocket_clients.append(websocket)
    try:
        while True:
            data = await websocket.receive_text()
    except WebSocketDisconnect:
        websocket_clients.remove(websocket)

@app.get("/restmp/{filename}")
async def send_file(filename: str):
    file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    if os.path.exists(file_path):
        return FileResponse(file_path)
    else:
        raise HTTPException(status_code=404, detail="File not found")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5000)
