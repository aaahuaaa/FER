"""
功能：
1. 使用SSD算法检测视频中的人脸
2. 采用ONNX格式的AffectNet模型进行表情分类
3. 在视频中实时标注人脸框和表情标签

参数说明：
--video : 输入视频文件路径（支持MP4/AVI等格式）
--output : 输出视频文件路径（建议使用.avi格式保证编码兼容性）

文件要求：
1. 模型文件：
   - affectnet7_original.onnx : 表情识别模型
   - deploy.prototxt.txt : SSD人脸检测模型配置
   - res10_300x300_ssd_iter_140000.caffemodel : SSD人脸检测模型
2. 需与脚本放在同一目录
"""

import os
import cv2
import numpy as np
import onnxruntime
from PIL import Image
import argparse
from torchvision import transforms

# 配置参数
onnx_model_path = "affectnet7_original.onnx"
labels = ['neutral', 'happy', 'sad', 'surprise', 'fear', 'disgust', 'anger']
local_prototxt_path = "deploy.prototxt.txt"
local_ssd_checkpoint_path = "res10_300x300_ssd_iter_140000.caffemodel"

# 初始化ONNX推理会话
ort_session = onnxruntime.InferenceSession(onnx_model_path)


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


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--video', default='demo_videos/input/Video_05131556.mp4',
                        type=str, help='Input video file path')
    parser.add_argument('--output', default='demo_videos/output/output_05131556.avi',
                        type=str, help='Output video file path')
    return parser.parse_args()


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


def fer(frame):

    cv_image = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)  # 转换为OpenCV格式
    h, w = frame.shape[:2]
    # 使用SSD模型要求的前处理（300x300分辨率，特定归一化参数）
    blob = cv2.dnn.blobFromImage(
        cv2.resize(frame, (300, 300)),  # 调整输入尺寸
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
        if confidence < 0.3:  # 过滤低置信度结果
            continue
        # 得到检测框的相对坐标(faces.shape[3][3~6]为坐标信息(0~1))
        raw_box = faces[0, 0, i, 3:7]

        # 设置部分边界溢出，防止人脸框过多
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
        face_image = frame[ymin:ymax, xmin:xmax]
        # 剔除过小区域
        if face_image.shape[0] < 10 or face_image.shape[1] < 10:
            continue

        face_pil = Image.fromarray(face_image)
        face_img = preprocess(face_pil)
        out = predict_emotion(face_img)
        print(f'emotion label: {out}')

        # 新增可视化部分
        cv2.rectangle(cv_image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)  # 画人脸框
        cv2.putText(cv_image, out, (xmin + 5, ymin + 20), cv2.FONT_HERSHEY_SIMPLEX,
                    0.9, (0, 255, 0), 2)  # 写标签

    return cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)


def main():
    args = parse_args()

    # 初始化视频捕捉
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print("Error opening video file")
        return

    # 获取视频参数
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # 初始化视频写入
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(args.output, fourcc, fps, (width, height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # 人脸检测和表情识别
        frame = fer(frame)

        # 显示和保存
        out.write(frame)

    # 释放资源
    cap.release()
    out.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
