import base64
import json
import cv2
import torch
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from ultralytics import YOLO
import numpy as np

app = FastAPI()

# 加载YOLO模型
device = "cuda:0" if torch.cuda.is_available() else "cpu"
model = YOLO("./models/yolov8m.pt")

@app.websocket("/ws/predict")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_text()  # 改为接收文本消息
            frame_data = base64.b64decode(data)
            nparr = np.frombuffer(frame_data, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            # 进行YOLO模型预测
            res = model.predict(source=frame, device=device,
                                classes=[0, 2, 3, 5, 6, 7, 9, 11, 12],
                                show_boxes=True, show_conf=True, show_labels=True,
                                conf=0.3)[0]
            if res is None:
                await websocket.send_text(json.dumps({"error": "YOLO prediction result is None."}))
                continue

            boxes_data = res.boxes or res.obb
            if boxes_data is None:
                await websocket.send_text(json.dumps({}))
                continue
            
            await websocket.send_text(res.tojson(True))
    except WebSocketDisconnect:
        print("WebSocket connection closed")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
