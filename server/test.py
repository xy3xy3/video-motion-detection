from ultralytics import YOLO

import torch
device = "cuda:0" if torch.cuda.is_available() else "cpu"

print(f"使用设备: {device}")
# 加载预训练的YOLOv11n模型
model = YOLO(r"./models/yolo11n.pt")

# 对'bus.jpg'图像进行推理，并获取结果
results = model.predict(r"test_image/image.png", save=True, imgsz=640, conf=0.5)

# 处理返回的结果
for result in results:
    boxes = result.boxes       # 获取边界框信息
    result.show()              # 显示结果