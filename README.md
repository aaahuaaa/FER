# FER —— DNN + DAN
checkpoints和models下载连接：[baidunetdisk](https://pan.baidu.com/s/1VcVeDTwHY8QAu2_1l_Hzlg?pwd=1234) 提取码: 1234 <br>
datasets下载链接：[baidunetdisk](https://pan.baidu.com/s/1EF566pKNxnXp8vMNvDftZw?pwd=1234) 提取码: 1234 <br>
`项目根目录：（192.168.3.113服务器）"/civi/data/FER/DAN/" `<br>
## 项目结构
```
project/
├── datasets/             # 数据集文件夹
├── checkpoints/          # 训练所得参数包文件夹
├── models/               # resnet在msceleb预训练包文件夹
├── networks/             # 模型架构文件夹
├── utils/                # 工具文件夹（内含pth转onnx脚本）
├── affectnet.py          # 训练脚本
├── xxx_demo.py           # 分images输入和video输入，pth的predict脚本
├── onnx_xxx.py           # 分images输入和video输入，onnx的predict脚本
├── deploy.prototxt.txt   # 【人脸识别】cv2.dnn的模型文件
└── res10_300x300_ssd_iter_140000.caffemodel #【人脸识别】cv2.dnn的参数文件
```
## 数据集
### 【AffectNet】
- 路径："~/datasets/AffectNet/"
- 简介：原版图源为网络爬虫（40W+），该版是经人清洗后的删减版（4W+），train和val中各有8个类别，其中第八个类别可训(AffectNet8)可不训(AffectNet7)。类别号0~6分别对应'neutral', 'happy', 'sad', 'surprise', 'fear', 'disgust', 'anger'，类别号7对应'contempt'。
### 【KDEF】
  - 路径："~/datasets/KDEF/"
  - 简介：45个外国男&45个外国女在不同角度/光照条件下做7种表情。命名规则详情见KDEF的官方说明—— "~/datasets/KDEF/ReadThis/KDEF.doc"
## 训练及测试流程
### 模型训练：（以AffectNet7为准）
```
cd /civi/data/FER/DAN/  # 进入DAN项目目录
CUDA_VISIBLE_DEVICES=0 python affectnet.py --aff_path 'datasets/AffectNet/' --epochs 15 --num_class 7
```
`训练之后进入"~/checkpoints"中，找到val时acc>0.63的.pth（该acc阈值可以手动修改）`
```
(309)   if args.num_class == 7 and acc > 0.63:
```
### 获取静态包
`pth包转onnx`
```
python utils/pth2onnx.py --pth 'checkpoints/affecnet7_epoch6_acc0.6569.pth' --out 'affectnet7.onnx' --num_class 7
```
### 模型推理
`手动修改配置参数中的onnx包路径`
```
(28) onnx_model_path = "affectnet7.onnx"
```
### 启动脚本对视频进行推理
```
python onnx_video_predict.py --video 'demo_videos/input/Video_05131556.mp4' --output 'demo_videos/output/output_05131556.avi'
```

