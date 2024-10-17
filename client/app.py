import base64
import os
import asyncio
from datetime import datetime
import numpy as np
import cv2
from fastapi import (
    FastAPI,
    Request,
    Form,
    WebSocket,
    WebSocketDisconnect,
    HTTPException,
)
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import websockets
from db import *
import fun
from videos import router as videos_router
from frames import router as frames_router

app = FastAPI()
templates = Jinja2Templates(directory="templates")

# 配置静态文件目录
app.mount("/static", StaticFiles(directory="static"), name="static")


is_processing = False

websocket_clients = []

# 导入并包括视频管理的路由
app.include_router(videos_router, prefix="/videos")
app.include_router(frames_router, prefix="/frames")


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/console", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("console.html", {"request": request})


@app.post("/start_processing")
async def start_processing(input_type: str = Form(...), video_path: str = Form(None)):
    global is_processing

    if is_processing:
        return JSONResponse(content={"status": "processing"}, status_code=400)

    start_time = datetime.now()
    log = create_log(start_time=start_time, end_time=None)
    log_id = log.id

    if input_type == "camera":
        asyncio.create_task(process_camera(log_id))
    elif input_type == "video":
        asyncio.create_task(process_video(log_id, video_path))
    else:
        return JSONResponse(content={"status": "invalid input type"}, status_code=400)

    is_processing = True

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
        "compress": get_config("compress"),
        "grayscale": get_config("grayscale"),
    }
    return templates.TemplateResponse(
        "set.html", {"request": request, "config": config}
    )


@app.post("/set")
async def update_settings(request: Request):
    data = await request.json()
    server_url = data.get("server_url")
    protect_type = data.get("protect_type")
    compress = data.get("compress")
    grayscale = data.get("grayscale")
    save_config("server_url", server_url)
    save_config("protect_type", protect_type)
    save_config("compress", str(compress))
    save_config("grayscale", str(grayscale))

    return JSONResponse(content={"status": "success"})


@app.get("/log", response_class=HTMLResponse)
async def log(request: Request):
    return templates.TemplateResponse("log.html", {"request": request})


@app.get("/log/list")
async def log_list(request: Request):
    # 从请求中获取分页参数
    page = int(request.query_params.get("page", 1))
    limit = int(request.query_params.get("limit", 10))

    # 计算分页的起始索引
    start_index = (page - 1) * limit
    end_index = page * limit
    logs = get_all_logs()
    log_list = [
        {
            "id": log.id,
            "start_time": log.start_time.strftime("%Y-%m-%d %H:%M:%S"),
            "end_time": (
                log.end_time.strftime("%Y-%m-%d %H:%M:%S") if log.end_time else None
            ),
        }
        for log in logs
    ]
    log_list = log_list[start_index:end_index]
    return JSONResponse(
        content={"code": 0, "msg": "ok", "data": log_list, "count": len(log_list)}
    )


@app.delete("/log/{log_id}")
async def del_log(log_id: int):
    delete_log(log_id)
    return JSONResponse(content={"status": "success"})


# 摄像头处理
async def process_camera(log_id: int):
    global is_processing
    rate = int(get_config("compress"))
    protect_type = get_config("protect_type")
    grayscale = get_config("grayscale")
    cap = cv2.VideoCapture(0)
    # 设置摄像头分辨率为720p
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    frame_count = 0
    while is_processing and cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1
        original_frame = frame.copy()
        frame = fun.privacy_protect(frame, protect_type)
        frame = fun.compress_jpg(frame, rate)
        if grayscale:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        await send_frame_ws(frame, log_id, original_frame)
        # await fake_process(frame, log_id, original_frame)
    cap.release()
    update_log(log_id, end_time=datetime.now())
    is_processing = False


# 视频处理
async def process_video(log_id: int, video_path: str):
    global is_processing
    rate = int(get_config("compress"))
    protect_type = get_config("protect_type")
    grayscale = get_config("grayscale")
    video_path = "./static/videos/" + video_path
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    while is_processing and cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1
        frame = fun.adjust_resolution(frame)
        original_frame = frame.copy()
        frame = fun.privacy_protect(frame, protect_type)
        frame = fun.compress_jpg(frame, rate)
        if grayscale:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        await send_frame_ws(frame, log_id, original_frame)
        # await fake_process(frame, log_id, original_frame)
    cap.release()
    update_log(log_id, end_time=datetime.now())
    is_processing = False


# 测试隐私保护
async def fake_process(frame: np.ndarray, log_id: int, original_frame: np.ndarray):
    # 使用 asyncio.to_thread 在后台线程中编码图像
    _, buffer = await asyncio.to_thread(cv2.imencode, ".jpg", frame)
    img_str = base64.b64encode(buffer).decode()
    await asyncio.to_thread(
        create_frame, log_id, time=datetime.now(), data=None, base64=img_str
    )
    await send_image_to_client(img_str)


# 发送帧到远程websocket
async def send_frame_ws(frame: np.ndarray, log_id: int, original_frame: np.ndarray):
    server_url = get_config("server_url").replace("http", "ws") + "/ws/predict"
    async with websockets.connect(server_url) as websocket:
        _, buffer = cv2.imencode(".jpg", frame)
        frame_base64 = base64.b64encode(buffer).decode("utf-8")
        await websocket.send(frame_base64)
        data = await websocket.recv()
        await process_detection_result(data, log_id, original_frame)


# 处理服务器的检测结果
async def process_detection_result(data: str, log_id: int, original_frame: np.ndarray):
    detection_result = json.loads(data)
    image_with_detections = fun.draw_detections(original_frame, detection_result)
    _, buffer = cv2.imencode(".jpg", image_with_detections)
    img_str = base64.b64encode(buffer).decode()
    create_frame(log_id, time=datetime.now(), data=detection_result, base64=img_str)
    await send_image_to_client(img_str)


# 发送图片到客户端
async def send_image_to_client(img_str: str):
    if len(websocket_clients) == 0:
        print("No clients connected")
        return
    for client in websocket_clients:
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
            await websocket.receive_text()
    except WebSocketDisconnect:
        websocket_clients.remove(websocket)


@app.websocket("/ws/stream/{log_id}")
async def websocket_endpoint(websocket: WebSocket, log_id: int):
    await websocket.accept()
    frames = session.query(Frame).filter_by(log_id=log_id).all()
    try:
        for frame in frames:
            await websocket.send_text(frame.base64)
            await asyncio.sleep(0.05)
    except WebSocketDisconnect:
        print("log推送 disconnect")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=5000)
