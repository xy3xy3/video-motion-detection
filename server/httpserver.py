from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import torch
from ultralytics import YOLOv10
from PIL import Image
import numpy as np
from io import BytesIO
import time
import base64
import json
import cv2
import torch
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
import numpy as np
from ultralytics.nn.autobackend import default_class_names

app = FastAPI()

# 加载YOLO模型
device = "cuda:0" if torch.cuda.is_available() else "cpu"
# m 20.2ms n 15-25ms s 14-21ms
model = YOLOv10("./models/yolov10s.pt")
model.model.names=default_class_names("coco8.yaml")

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
@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    start_time = time.time()  # 记录开始时间
    try:
        image = Image.open(BytesIO(await file.read()))
        image_np = np.array(image)
        res = model.predict(source=image_np, device=device,
                            classes=[0, 2, 3, 5, 6, 7, 9, 11, 12],# 选择需要检测的类别
                            show_boxes=True, show_conf=True, show_labels=True,# 显示边界框、置信度和标签
                            conf=0.3,  # 调整置信度阈值
                            )[0]
        if res is None:
            return JSONResponse(content={"error": "YOLO prediction result is None."}, status_code=400)

        end_time = time.time()  # 记录结束时间
        inference_time = end_time - start_time  # 计算推理用时
        print(f"推理用时: {inference_time:.2f} 秒")
        
        data = res.boxes or res.obb
        if data is None:
            return JSONResponse(content={})
        return JSONResponse(content=res.tojson(True))
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
