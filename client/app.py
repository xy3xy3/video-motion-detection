import base64
import os
import queue
import threading
from flask import Flask, render_template, request, jsonify, Response, send_from_directory
from datetime import datetime
import cv2
import json
import httpclient
from db import *

app = Flask(__name__)
app.config["TEMPLATES_AUTO_RELOAD"] = True
app.config["UPLOAD_FOLDER"] = "static/restmp"

# 创建目录，如果不存在
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

send_frame_count = 3
# 创建全局变量以便线程控制
processing_thread = None
is_processing = False

sse_clients = []

# 首页
@app.route("/")
def index():
    return render_template("index.html")


# 启动视频处理
@app.route("/start_processing", methods=["POST"])
def start_processing():
    global processing_thread, is_processing

    if is_processing:
        return jsonify({"status": "processing"}), 400

    input_type = request.form.get("input_type")
    video_path = request.form.get("video_path", "")

    # 创建日志记录
    start_time = datetime.now()
    log = create_log(start_time=start_time, end_time=None)
    log_id = log.id

    if input_type == "camera":
        processing_thread = threading.Thread(target=process_camera, args=(log_id,))
    elif input_type == "video":
        processing_thread = threading.Thread(
            target=process_video, args=(log_id, video_path)
        )
    else:
        return jsonify({"status": "invalid input type"}), 400

    is_processing = True
    processing_thread.start()

    return jsonify({"status": "started", "log_id": log_id})


# 停止视频处理
@app.route("/stop_processing", methods=["POST"])
def stop_processing():
    global is_processing
    is_processing = False
    return jsonify({"status": "stopped"})

# 处理摄像头视频
def process_camera(log_id):
    global is_processing, send_frame_count

    cap = cv2.VideoCapture(0)
    frame_count = 0
    while is_processing and cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1
        if frame_count % send_frame_count == 0:
            original_frame = frame.copy()
            frame = httpclient.privacy_protect(frame)
            file_path = os.path.join(
                app.config["UPLOAD_FOLDER"], f"{log_id}_frame_{frame_count}.jpg"
            ).replace("\\", "/").replace("//","/")
            output_path = os.path.join(
                app.config["UPLOAD_FOLDER"], f"{log_id}_result_{frame_count}.jpg"
            ).replace("\\", "/").replace("//","/")
            cv2.imwrite(file_path, frame)
            data = httpclient.send_file(file_path, original_frame, output_path)
            create_frame(log_id, path=output_path, time=datetime.now(), data=data)
            os.remove(file_path)
            send_frame_to_clients(output_path)

    cap.release()
    update_log(log_id, end_time=datetime.now())
    is_processing = False

def send_frame_to_clients(frame_path):
    with open(frame_path, "rb") as f:
        img = f.read()
        img_base64 = base64.b64encode(img).decode("utf-8")
        sse_data = f"data: {json.dumps({'img_base64': img_base64})}\n\n"
        for client in sse_clients:
            client.put(sse_data)


# 处理视频文件
def process_video(log_id, video_path):
    global is_processing, send_frame_count
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    while is_processing and cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1
        if frame_count % send_frame_count == 0:
            frame = httpclient.compress_jpg(frame)
            original_frame = frame.copy()
            frame = httpclient.privacy_protect(frame)
            file_path = os.path.join(
                app.config["UPLOAD_FOLDER"], f"{log_id}_frame_{frame_count}.jpg"
            ).replace("\\", "/").replace("//","/")
            output_path = os.path.join(
                app.config["UPLOAD_FOLDER"], f"{log_id}_result_{frame_count}.jpg"
            ).replace("\\", "/").replace("//","/")
            cv2.imwrite(file_path, frame)
            data = httpclient.send_file(file_path, original_frame, output_path)
            create_frame(log_id, path=output_path, time=datetime.now(), data=data)
            os.remove(file_path)
            send_frame_to_clients(output_path)

    cap.release()
    update_log(log_id, end_time=datetime.now())
    is_processing = False


@app.route('/stream')
def stream():
    def generate():
        client_queue = queue.Queue()
        sse_clients.append(client_queue)
        try:
            while True:
                data = client_queue.get()
                yield data
        except GeneratorExit:
            sse_clients.remove(client_queue)
    return Response(generate(), content_type='text/event-stream')


# 提供图像文件服务
@app.route("/restmp/<filename>")
def send_file(filename):
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename)


if __name__ == "__main__":
    app.run(debug=True)
