"""
功能：
1. 支持JPEG/PNG格式图片批量处理
2. 采用SSD算法进行人脸检测（置信度阈值0.5）
3. 使用ONNX格式的AffectNet-7模型进行表情分类
4. 自动保存带标注框的结果图像

使用流程：
1. 准备输入图像文件夹
2. 运行脚本自动处理
3. 查看输出文件夹中的结果

参数说明：
--folder : 必需参数，指定待处理图片所在的文件夹路径

文件结构：
├── 模型文件（与脚本同级目录）
│   ├── affectnet7_kdef_lr5_ep12.onnx
│   ├── deploy.prototxt.txt
│   └── res10_300x300_ssd_iter_140000.caffemodel
├── 输入结构
│   └── /civi/data/FER/DAN/demo_images/hard_image/ (--folder)
│       ├── *.jpg
│       └── *.png
└── 输出结构
    └── /civi/data/FER/DAN/demo_images/hard_output/ (output_dir)
        └── *_result.png
"""

import os
import cv2
import numpy as np
import onnxruntime
from PIL import Image
import argparse
from torchvision import transforms

# 配置参数
onnx_model_path = "affectnet7_kdef_lr5_ep12.onnx"
labels = ['neutral', 'happy', 'sad', 'surprise', 'fear', 'disgust', 'anger']
local_prototxt_path = "deploy.prototxt.txt"
local_ssd_checkpoint_path = "res10_300x300_ssd_iter_140000.caffemodel"

# 初始化ONNX推理会话
ort_session = onnxruntime.InferenceSession(onnx_model_path)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--folder', default="/civi/data/FER/DAN/demo_images/hard_image/",  # 文件夹
                        type=str, help='Folder containing images for evaluation.')
    return parser.parse_args()


def preprocess(image):
    """与训练时一致的预处理流程"""
    # 确保输入为PIL.Image对象
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image.astype(np.uint8))
    # 调整尺寸
    image = image.resize((224, 224))
    # 转换为numpy数组（必须先转换）
    img = np.array(image).astype(np.float32)
    # 使用float32类型的归一化参数
    mean = np.array([123.675, 116.28, 103.53], dtype=np.float32)
    std = np.array([58.395, 57.12, 57.375], dtype=np.float32)
    img = (img - mean) / std
    # 转换通道顺序为CHW
    img = img.transpose(2, 0, 1)
    # 添加batch维度并保持float32
    return np.expand_dims(img, axis=0).astype(np.float32)


def predict_emotion(input_data):
    """执行ONNX推理"""
    # ONNX推理
    ort_inputs = {ort_session.get_inputs()[0].name: input_data}
    ort_outs = ort_session.run(None, ort_inputs)
    # 获取预测结果
    pred = np.argmax(ort_outs[0])
    return labels[pred]


def get_ssd_face_detector():
    ssd_face_detector = cv2.dnn.readNetFromCaffe(
        prototxt=local_prototxt_path,
        caffeModel=local_ssd_checkpoint_path,
    )
    return ssd_face_detector


transform = transforms.Compose(
    transforms=[transforms.ToPILImage(), transforms.ToTensor()]
)


def convert_to_square(xmin, ymin, xmax, ymax):
    # convert to square location
    center_x = (xmin + xmax) // 2
    center_y = (ymin + ymax) // 2

    square_length = ((xmax - xmin) + (ymax - ymin)) // 2 // 2
    square_length *= 1.1

    xmin = int(center_x - square_length)
    ymin = int(center_y - square_length)
    xmax = int(center_x + square_length)
    ymax = int(center_y + square_length)
    return xmin, ymin, xmax, ymax


def fer(path):
    # 创建保存目录
    output_dir = "/civi/data/FER/DAN/demo_images/hard_output/"
    os.makedirs(output_dir, exist_ok=True)

    # cv2方法读取图片
    img = cv2.imread(path)
    img0 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    cv_image = cv2.cvtColor(np.array(img0), cv2.COLOR_RGB2BGR)  # 转换为OpenCV格式
    h, w = img.shape[:2]
    # 使用SSD模型要求的前处理（300x300分辨率，特定归一化参数）
    blob = cv2.dnn.blobFromImage(
        cv2.resize(img, (300, 300)),  # 调整输入尺寸
        # img,
        1.0,
        (300, 300),
        (104.0, 177.0, 123.0),  # ImageNet数据集均值（BGR顺序）
        False,  # 不交换RB通道
        False,  # 不裁剪图像
    )
    # 通过SSD网络进行人脸检测
    face_detector = get_ssd_face_detector()
    face_detector.setInput(blob)
    faces = face_detector.forward()

    face_results = []
    # 遍历所有检测到的人脸候选框(faces.shape[2]为检测结果数量)
    for i in range(0, faces.shape[2]):
        confidence = faces[0, 0, i, 2]  # 获取检测置信度
        if confidence < 0.5:  # 过滤低置信度结果
            continue
        # 得到检测框的相对坐标(faces.shape[3][3~6]为坐标信息(0~1))
        raw_box = faces[0, 0, i, 3:7]

        # ====允许部分边界溢出====
        valid_threshold = 0.8  # 有效区域阈值（可调整）
        box_area = (raw_box[2] - raw_box[0]) * (raw_box[3] - raw_box[1])  # 原始区域面积
        # 裁剪到合法范围 [0,1]
        clipped_box = np.clip(raw_box, 0, 1)
        clipped_area = (clipped_box[2] - clipped_box[0]) * (clipped_box[3] - clipped_box[1])  # 有效区域面积
        # 当有效区域面积占比小于阈值时过滤
        if clipped_area / box_area < valid_threshold:
            continue

        # 相对坐标转换为绝对坐标
        xmin, ymin, xmax, ymax = (
                raw_box * np.array([w, h, w, h])
        ).astype("int")
        # 将矩形框转换为正方形（扩大1.1倍避免面部截断）
        xmin, ymin, xmax, ymax = convert_to_square(xmin, ymin, xmax, ymax)
        # 过滤无效坐标（防止宽高为负值）
        if xmax <= xmin or ymax <= ymin:
            continue
        # 存储有效人脸区域坐标
        face_results.append(
            {
                "xmin": xmin,
                "ymin": ymin,
                "xmax": xmax,
                "ymax": ymax,
            }
        )
    if len(face_results) == 0:
        return 'noface'
    for face in face_results:
        xmin = face["xmin"]
        ymin = face["ymin"]
        xmax = face["xmax"]
        ymax = face["ymax"]
        if xmin < 0: xmin = 0
        if ymin < 0: ymin = 0
        if xmax > w: xmax = w
        if ymax > h: ymax = h
        # 提取人脸区域（使用灰度图像）
        face_image = img0[ymin:ymax, xmin:xmax]
        # 剔除过小区域
        if face_image.shape[0] < 10 or face_image.shape[1] < 10:
            continue

        face_pil = Image.fromarray(face_image)
        face_img = preprocess(face_pil)
        out = predict_emotion(face_img)

        # 新增可视化部分
        cv2.rectangle(cv_image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)  # 画人脸框
        cv2.putText(cv_image, out, (xmin + 5, ymin + 20), cv2.FONT_HERSHEY_SIMPLEX,
                    0.9, (0, 255, 0), 2)  # 写标签

    # 生成保存路径（例：输入demo.jpg → 输出demo_images/demo_result.jpg）
    filename = os.path.basename(path)
    output_path = os.path.join(output_dir, f"{os.path.splitext(filename)[0]}_result.png")
    cv2.imwrite(output_path, cv_image)

    return out


if __name__ == "__main__":
    args = parse_args()

    for filename in os.listdir(args.folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):  # 过滤图片格式
            image_path = os.path.join(args.folder, filename)
            label = fer(image_path)
            if label == 'noface':
                print(f'No face detected in {filename}')
                continue
            print(f'emotion label: {label}')
