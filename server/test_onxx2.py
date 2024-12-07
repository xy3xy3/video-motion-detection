import cv2
import numpy as np
import onnxruntime as ort
from PIL import Image, ImageDraw

# 类别名称映射
names = []
with open("./models/coco.names", "r") as f:
    for line in f.readlines():
        names.append(line.strip())
#填写CLASS_NAMES
for i,name in enumerate(names):
    CLASS_NAMES = {i: name for i, name in enumerate(names)}
# 加载ONNX模型
onnx_model_path = './models/yolo11n.onnx'
session = ort.InferenceSession(onnx_model_path, providers=['CPUExecutionProvider'])

# 获取模型输入和输出信息
input_name = session.get_inputs()[0].name
output_names = [output.name for output in session.get_outputs()]

# 预处理函数
def preprocess_image(image_path, input_size=(640, 640)):
    # 读取图像并调整颜色空间
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # 保持纵横比缩放并填充
    def letterbox(img, new_shape=input_size, color=(114, 114, 114)):
        shape = img.shape[:2]
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        new_unpad = (int(shape[1] * r), int(shape[0] * r))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
        dw /= 2
        dh /= 2
        if shape != new_unpad:
            img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
        return img, r, (dw, dh)

    image_letterboxed, ratio, pad = letterbox(image, new_shape=input_size)

    # 转换为模型输入格式
    image_np = image_letterboxed.astype(np.float32)
    image_np = np.transpose(image_np, (2, 0, 1)) / 255.0
    image_np = np.expand_dims(image_np, axis=0)

    return image_np, ratio, pad, image

# 后处理函数
def postprocess_output(outputs, ratio, pad, input_size, conf_thres=0.5, iou_thres=0.45):
    predictions = np.squeeze(outputs[0])
    boxes = predictions[:, :4]
    scores = predictions[:, 4]
    classes = predictions[:, 5]

    # 过滤低置信度
    valid_indices = scores > conf_thres
    boxes = boxes[valid_indices]
    scores = scores[valid_indices]
    classes = classes[valid_indices]

    # 缩放回原始图像尺寸
    boxes[:, [0, 2]] -= pad[0]  # x坐标减去填充
    boxes[:, [1, 3]] -= pad[1]  # y坐标减去填充
    boxes[:, :4] /= ratio
    boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, input_size[0])
    boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, input_size[1])

    # 转换为xyxy格式
    boxes[:, 2] = boxes[:, 0] + boxes[:, 2]
    boxes[:, 3] = boxes[:, 1] + boxes[:, 3]

    # 非极大值抑制
    indices = cv2.dnn.NMSBoxes(boxes.tolist(), scores.tolist(), conf_thres, iou_thres)
    indices = indices.flatten() if len(indices) > 0 else []

    # 提取最终检测结果
    final_boxes = boxes[indices].astype(int)
    final_scores = scores[indices]
    final_classes = classes[indices].astype(int)

    return final_boxes, final_scores, final_classes

# 主函数
def main():
    image_path = 'test_image/image.png'
    input_size = (640, 640)

    # 预处理图像
    image_np, ratio, pad, original_image = preprocess_image(image_path, input_size)

    # 模型推理
    outputs = session.run(output_names, {input_name: image_np})

    # 后处理
    boxes, scores, classes = postprocess_output(outputs, ratio, pad, input_size)
    print(boxes)
    print(scores)
    print(classes)
    # 可视化
    image_draw = Image.fromarray(original_image)
    draw = ImageDraw.Draw(image_draw)
    for box, score, cls in zip(boxes, scores, classes):
        x0, y0, x1, y1 = box
        label = f"{CLASS_NAMES[cls]}: {score:.2f}"
        draw.rectangle([x0, y0, x1, y1], outline='red', width=3)
        draw.text((x0, y0 - 15), label, fill='red')

    # 保存结果图像
    image_draw.save('det_result_picture.jpg')
    print("图像已保存为 det_result_picture.jpg")

if __name__ == '__main__':
    main()