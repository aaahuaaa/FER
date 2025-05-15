import os
import argparse
from PIL import Image
import numpy as np
import cv2
import torch
from torchvision import transforms
from networks.dan import DAN

local_prototxt_path = "deploy.prototxt.txt"
local_ssd_checkpoint_path = "res10_300x300_ssd_iter_140000.caffemodel"


def parse_args():
    parser = argparse.ArgumentParser()
    # parser.add_argument('--image', default="demo_images/Online_image/sadness2.png",  # 文件
    #                     type=str, help='Image file for evaluation.')

    parser.add_argument('--folder', default="E:/Faces/asian_faces",  # 文件夹
                        type=str, help='Folder containing images for evaluation.')

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


class Model():
    def __init__(self):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.data_transforms = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        self.labels = ['neutral', 'happy', 'sad', 'surprise', 'fear', 'disgust', 'anger', 'contempt']

        self.model = DAN(num_head=4, num_class=8, pretrained=False)
        checkpoint = torch.load('./checkpoints/affecnet8_epoch5_acc0.6209.pth',
                                map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'], strict=True)
        self.model.to(self.device)
        self.model.eval()
        self.face_detector = get_ssd_face_detector()

    def fer(self, path):
        # 创建保存目录
        output_dir = "E:/Faces/asian_faces_DAN-outputs-4"
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
        self.face_detector.setInput(blob)
        faces = self.face_detector.forward()

        face_results = []
        # 遍历所有检测到的人脸候选框(faces.shape[2]为检测结果数量)
        for i in range(0, faces.shape[2]):
            confidence = faces[0, 0, i, 2]  # 获取检测置信度
            if confidence < 0.5:  # 过滤低置信度结果
                continue
            # 得到检测框的相对坐标(faces.shape[3][3~6]为坐标信息(0~1))
            raw_box = faces[0, 0, i, 3:7]

            # # ====严格过滤====
            # if np.any(raw_box < 0) or np.any(raw_box > 1):
            #     continue

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
            # 转换为PIL图像
            face_image = Image.fromarray(face_image)
            # 添加数据预处理流程
            face_tensor = self.data_transforms(face_image)  # 应用标准化处理
            face_tensor = face_tensor.unsqueeze(0).to(self.device)  # 增加batch维度并送设备
            with torch.set_grad_enabled(False):
                out, _, _ = self.model(face_tensor)  # 前向传播（DAN模型输出num_class=8）
                _, pred = torch.max(out, 1)  # 找到最大概率的类别索引
                index = int(pred)  # 转换为整数
                label = self.labels[index]  # 获取对应的情感标签

            # 新增可视化部分
            cv2.rectangle(cv_image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)  # 画人脸框
            cv2.putText(cv_image, label, (xmin + 5, ymin + 20), cv2.FONT_HERSHEY_SIMPLEX,
                        0.9, (0, 255, 0), 2)  # 写标签

        # 生成保存路径（例：输入demo.jpg → 输出demo_images/demo_result.jpg）
        filename = os.path.basename(path)
        output_path = os.path.join(output_dir, f"{os.path.splitext(filename)[0]}_result.png")
        cv2.imwrite(output_path, cv_image)

        return label


if __name__ == "__main__":
    args = parse_args()

    model = Model()

    # image = args.image
    # assert os.path.exists(image), "Failed to load image file."

    for filename in os.listdir(args.folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):  # 过滤图片格式
            image_path = os.path.join(args.folder, filename)
            label = model.fer(image_path)
            if label == 'noface':
                print(f'No face detected in {filename}')
                continue
            print(f'emotion label: {label}')

    # print(f'emotion label: {label}')
