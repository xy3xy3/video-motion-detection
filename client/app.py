import asyncio
import time
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from contextlib import asynccontextmanager
import cv2
import websockets
import base64
import numpy as np
from datetime import datetime
from db import *
import fun
from videos import router as videos_router
from frames import router as frames_router
import multiprocessing
from multiprocessing import Queue, Process
global_websocket = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global global_websocket
    await establish_websocket()
    # 创建并启动一个新进程来处理队列
    process = Process(target=process_frame_queue, args=(frame_queue,))
    process.start()
    # 应用启动
    yield
    # 应用关闭，终止进程
    process.terminate()
    process.join()
    if global_websocket and not global_websocket.closed:
        await global_websocket.close()


app = FastAPI(lifespan=lifespan)
templates = Jinja2Templates(directory="templates")

app.mount("/static", StaticFiles(directory="static"), name="static")

is_processing = False
websocket_clients = []

# 导入并包括视频管理的路由
app.include_router(videos_router, prefix="/videos")
app.include_router(frames_router, prefix="/frames")

# 创建全局帧队列
frame_queue = Queue()  # 使用多进程队列


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
        "record_db": get_config("record_db"),
        "show_protect": get_config("show_protect"),
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
    record_db = data.get("record_db")
    show_protect = data.get("show_protect")
    save_config("server_url", server_url)
    save_config("protect_type", protect_type)
    save_config("compress", str(compress))
    save_config("grayscale", str(grayscale))
    save_config("record_db", str(record_db))
    save_config("show_protect", str(show_protect))

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
                log.end_time.strftime(
                    "%Y-%m-%d %H:%M:%S") if log.end_time else None
            ),
        }
        for log in logs
    ]
    log_list = log_list[start_index:end_index]
    return JSONResponse(
        content={"code": 0, "msg": "ok",
                 "data": log_list, "count": len(log_list)}
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
    grayscale = bool(get_config("grayscale")=="1")
    show_protect = bool(get_config("show_protect")=="1")
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    frame_count = 0
    target_fps = 30  # 设置目标帧率
    while is_processing and cap.isOpened():
        start_time = time.time()
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1
        original_frame = frame.copy()
        frame = fun.privacy_protect(frame, protect_type)
        if show_protect:
            original_frame = frame
        frame = fun.compress_jpg(frame, rate)
        if grayscale:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        await send_frame_ws(frame, log_id, original_frame, 1)
        # 控制帧率
        elapsed_time = time.time() - start_time
        if elapsed_time < 1 / target_fps:
            await asyncio.sleep(1 / target_fps - elapsed_time)
    cap.release()
    update_log(log_id, end_time=datetime.now())
    is_processing = False

# 视频处理
async def process_video(log_id: int, video_path: str):
    global is_processing,websocket_clients
    rate = int(get_config("compress"))
    protect_type = get_config("protect_type")
    grayscale = bool(get_config("grayscale")=="1")
    show_protect = bool(get_config("show_protect")=="1")
    video_path = "./static/videos/" + video_path
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    while is_processing and cap.isOpened():
        print("="*50)
        #死循环,等待客户端链接建立
        while is_processing and len(websocket_clients) == 0:
            print("waiting for client")
            await asyncio.sleep(1)
            if not is_processing:
                break
        if not is_processing:
            break
        start_time = time.time()
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1
        original_frame = frame.copy()
        # frame = fun.adjust_resolution(frame)
        frame = fun.privacy_protect(frame, protect_type)
        if show_protect:
            original_frame = frame
        protect_time = time.time()
        print(f"protect_time {(protect_time - start_time) * 1000:.4f}ms")
        if rate < 100:
            print("压缩图片")
            frame = fun.compress_jpg(frame, rate)
        compress_time = time.time()
        print(f"compress_time rate{rate} {(compress_time - protect_time) * 1000:.4f}ms")
        if grayscale:
            print("转为灰度")
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        process_time = time.time()
        print(f"frame process time {(process_time - start_time) * 1000:.4f}ms")
        res = await send_frame_ws(frame, log_id, original_frame,1)
        end_time = time.time()
        print(f"signgle frame time {(end_time - start_time) * 1000:.4f}ms res:{res}")
        if not res:
            break
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


async def establish_websocket():
    global global_websocket
    server_url = get_config("server_url").replace("http", "ws") + "/ws/predict"
    global_websocket = await websockets.connect(server_url)



# 在send_frame_ws里使用global_websocket
async def send_frame_ws(frame: np.ndarray, log_id: int, original_frame: np.ndarray,send=True):
    global global_websocket
    if global_websocket is None or global_websocket.closed:
        await establish_websocket()

    start_time = time.time()  # 记录开始时间
    _, buffer = cv2.imencode(".jpg", frame)
    if send:
        encode_time = time.time()  # 记录编码时间
        await global_websocket.send(buffer.tobytes())
        send_time = time.time()  # 记录发送时间
        data = await global_websocket.recv()
        recv_time = time.time()  # 记录接收时间
        # 处理返回的检测结果
        detection_result = json.loads(data)
        image_with_detections = fun.draw_detections(
            original_frame, detection_result)
        _, buffer = cv2.imencode(".jpg", image_with_detections)
        draw_time = time.time()  # 记录绘制时间
        # 将帧数据放入队列
        frame_queue.put((log_id, detection_result, buffer.tobytes()))
        queue_time = time.time()  # 记录放入队列时间
    else:
        await asyncio.sleep(0.01)
    # 同时发送给前端用户
    res = await send_image_to_client(buffer.tobytes())
    client_send_time = time.time()  # 记录发送给前端时间

    # print(f"Encode time: {(encode_time - start_time) * 1000:.4f}ms")
    # print(f"Send time: {(send_time - encode_time) * 1000:.4f}ms")
    # print(f"Receive time: {(recv_time - send_time) * 1000:.4f}ms")
    # print(f"Draw time: {(draw_time - recv_time) * 1000:.4f}ms")
    # print(f"Queue time: {(queue_time - draw_time) * 1000:.4f}ms")
    # print(f"Client send time: {(client_send_time - queue_time) * 1000:.4f}ms")
    print(f"send_frame_ws time {(client_send_time - start_time) * 1000:.4f}ms")
    return res

# 发送图片到客户端
async def send_image_to_client(buffer: bytes):
    if len(websocket_clients) == 0:
        print("No clients connected")
        return False
    for client in websocket_clients:
        try:
            # 直接发送二进制数据
            await client.send_bytes(buffer)
        except Exception as e:
            print(f"Failed to send image to client: {e}")
            return False
    return True
# 后台任务：从队列中读取帧并写入数据库（在进程中运行）


def process_frame_queue(frame_queue: Queue):
    record_db = get_config("record_db")
    while True:
        try:
            log_id, detection_result, frame_data = frame_queue.get()
            if record_db == "1":
                create_frame(log_id, time=datetime.now(
                ), data=detection_result, base64=base64.b64encode(frame_data).decode())
            else:
                continue  # 跳过不必要的操作
        except Exception as e:
            print(f"Error processing frame: {e}")


@app.websocket("/ws/stream")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("加入新链接")
    websocket_clients.append(websocket)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        print("客户端断开连接")
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

    uvicorn.run("app:app", host="0.0.0.0", port=5000, reload=True)
