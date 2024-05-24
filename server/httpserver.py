from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import torch
from ultralytics import YOLO
from PIL import Image
import numpy as np
from io import BytesIO
import time

app = FastAPI()

# 加载YOLO模型
device = "cuda:0" if torch.cuda.is_available() else "cpu"
model = YOLO("./models/yolov8n.pt")

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    start_time = time.time()  # 记录开始时间
    try:
        image = Image.open(BytesIO(await file.read()))
        image_np = np.array(image)
        res = model.predict(source=image_np, device=device,
                            classes=[0, 2, 3, 5, 6, 7, 9, 11, 12],
                            show_boxes=True, show_conf=True, show_labels=True,
                            conf=0.3)[0]
        if res is None:
            return JSONResponse(content={"error": "YOLO prediction result is None."}, status_code=400)

        end_time = time.time()  # 记录结束时间
        inference_time = end_time - start_time  # 计算推理用时
        print(f"推理用时: {inference_time:.2f} 秒")

        return JSONResponse(content=res.tojson(True))
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
