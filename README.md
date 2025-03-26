# YOLO11 公路裂缝检测和分割项目

基于YOLO11的公路裂缝检测与分割模型，用于高速公路养护和安全监测。本项目在原始YOLO架构基础上进行了多项改进，包括EMA注意力机制、分割头和组合损失函数。

## 项目亮点

1. **基于YOLO11的改进架构**：使用最新的YOLO11模型作为基础，针对裂缝检测场景进行优化。
2. **EMA注意力机制**：引入EMA (有效多尺度注意力) 机制，更好地处理不同尺度的裂缝特征。
3. **分割头设计**：添加分割头，除提供裂缝的位置和类别外，还能给出准确的裂缝形状轮廓。
4. **组合损失函数**：结合边界框损失、分类损失和分割损失，更好地优化模型。
5. **完整的训练、评估和预测流程**：从数据准备到模型部署的全流程支持。

## 模型架构

![Model Architecture](https://mermaid.ink/img/pako:eNqFkMEKwjAMhl-l5KwHe4AWPKl48OBpi-C1NC6usi5jrQ6R8d5tHaoHLzkkf_J_aZqKJmeIIRV65S_iXaPFZ7p-cBLQIRXg-iSKpnNOwI0OmmQ9AcG2vkFpXWv7vGvwAR2ZHt1Z0wvbRKdz9dpdI9dNEATnZXaajOCRvbZnrCDjt3mj8khzrAR6ZL_kK_pPfljX7hfIIV4vwbOiJ0YrpkdWCYTaNL0kF4hNnLXMhWgHsWEvMTHE0LURxEbFbCbLxFvDJmkh1vZNEP8Cnk1q4A)

## 安装

### 环境要求

- Python 3.8+
- PyTorch 1.12+
- CUDA (如需GPU加速)

### 安装步骤

1. 克隆仓库：
   ```bash
   git clone https://github.com/yourusername/yolo11-highway-cracks.git
   cd yolo11-highway-cracks
   ```

2. 安装依赖：
   ```bash
   pip install -r requirements.txt
   ```

3. 安装开发模式：
   ```bash
   python setup.py develop
   ```

## 数据准备

1. 准备数据集，并按以下目录结构组织：
   ```
   datasets/
   ├── train/
   │   ├── images/
   │   └── labels/
   ├── val/
   │   ├── images/
   │   └── labels/
   └── test/
       ├── images/
       └── labels/
   ```

2. 标签格式应为YOLO格式：每行`class_id x_center y_center width height`。如果有分割标签，还需要添加分割点坐标。

3. 确保`datasets/data.yaml`配置正确。

## 使用方法

### 训练模型

```bash
# 基本训练命令
python main.py --data datasets/data.yaml --weights yolov8n.pt

# 高级训练选项
python main.py --data datasets/data.yaml --weights yolov8n.pt --batch 16 --epochs 300 --imgsz 640 --device 0
```

### 评估模型

```bash
# 评估训练好的模型
python eval.py --weights runs/train/exp/weights/best.pt --data datasets/data.yaml --save-plots

# 高级评估选项
python eval.py --weights runs/train/exp/weights/best.pt --data datasets/data.yaml --batch 16 --imgsz 640 --device 0 --save-json --save-plots
```

### 预测

```bash
# 对单张图像进行预测
python predict.py --weights runs/train/exp/weights/best.pt --source path/to/image.jpg

# 对视频或目录进行预测
python predict.py --weights runs/train/exp/weights/best.pt --source path/to/video.mp4
python predict.py --weights runs/train/exp/weights/best.pt --source path/to/images/

# 更多预测选项
python predict.py --weights runs/train/exp/weights/best.pt --source path/to/images/ --conf-thres 0.4 --save-txt --save-masks
```

## 模型参数说明

| 参数              | 说明                    | 默认值              |
|------------------|-----------------------|-------------------|
| `--data`         | 数据集配置文件             | datasets/data.yaml |
| `--weights`      | 初始权重或预训练模型路径      | yolov8n.pt         |
| `--model`        | 模型配置文件路径            | ultralytics/cfg/models/11/yolo11-SegHead.yaml |
| `--batch`        | 训练批次大小              | 16                |
| `--imgsz`        | 图像尺寸                 | 640               |
| `--epochs`       | 训练轮数                 | 300               |
| `--enable_augment` | 启用高级图像增强           | False             |
| `--attention`    | 注意力机制类型             | ema               |
| `--segment`      | 启用分割头                | True              |
| `--combined_loss` | 使用组合损失函数            | True              |

## 项目文件结构

```
yolo11-highway-cracks/
├── datasets/              # 数据集目录
│   └── data.yaml         # 数据集配置
├── ultralytics/          # YOLO核心代码
│   ├── cfg/             # 配置文件
│   │   ├── models/     # 模型配置
│   │   │   └── 11/    # YOLO11模型配置
│   │   │       └── yolo11-SegHead.yaml  # 我们的分割模型配置
│   │   ├── nn/              # 神经网络模块
│   │   │   └── modules/    # 网络组件
│   │   │       └── seg_head.py  # 分割头实现
│   │   └── utils/           # 工具函数
│   │       └── combined_loss.py  # 组合损失函数
│   ├── main.py               # 主训练脚本
│   ├── eval.py               # 评估脚本
│   └── predict.py            # 预测脚本
└── README.md             # 本文档
```

## 引用

如果您在研究中使用了本项目，请引用：

```
@software{yolo11_highway_cracks,
  author = {Your Name},
  title = {YOLO11 Highway Crack Detection and Segmentation},
  year = {2023},
  url = {https://github.com/yourusername/yolo11-highway-cracks}
}
```

## 许可证

本项目遵循 AGPL-3.0 许可证。
