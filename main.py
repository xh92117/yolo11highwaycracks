import argparse
import os
from ultralytics import YOLO
from ultralytics.data.custom_augment import CustomAugment
from ultralytics.nn.modules.attention import CBAM, EMAAttention
from ultralytics.nn.modules.seg_head import DetectionSegmentationHead, SegmentationHead
from ultralytics.utils.combined_loss import CombinedLoss
from ultralytics.utils import LOGGER

# 安装命令
# python setup.py develop

# 数据集示例百度云链接
# 链接：https://pan.baidu.com/s/19FM7XnKEFC83vpiRdtNA8A?pwd=n93i 
# 提取码：n93i 

def parse_args():
    parser = argparse.ArgumentParser(description='YOLO11公路裂缝检测训练脚本')

    # 基本参数
    parser.add_argument('--data', type=str, default='datasets/data.yaml', help='数据集配置文件')
    parser.add_argument('--weights', type=str, default='yolov8n.pt', help='初始权重文件')
    parser.add_argument('--model', type=str, default='ultralytics/cfg/models/11/yolo11-SegHead.yaml', help='模型配置文件')
    parser.add_argument('--epochs', type=int, default=300, help='训练轮数')
    parser.add_argument('--batch', type=int, default=16, help='批次大小')
    parser.add_argument('--imgsz', type=int, default=640, help='图像尺寸')
    parser.add_argument('--device', default='', help='训练设备 (如 0,1,2,3 或 cpu)')
    parser.add_argument('--project', default='runs/train', help='项目名称')
    parser.add_argument('--name', default='exp', help='实验名称')

    # 改进开关
    parser.add_argument('--enable_augment', action='store_true', help='启用高级图像增强')
    parser.add_argument('--attention', type=str, choices=['none', 'cbam', 'ema'], default='ema',
                        help='使用注意力机制类型 (none, cbam, ema)')
    parser.add_argument('--segment', action='store_true', default=True, help='启用分割头')
    parser.add_argument('--combined_loss', action='store_true', default=True, help='使用改进的组合损失函数')

    # 损失权重
    parser.add_argument('--box_weight', type=float, default=7.5, help='边界框损失权重')
    parser.add_argument('--cls_weight', type=float, default=0.5, help='分类损失权重')
    parser.add_argument('--dfl_weight', type=float, default=1.5, help='DFL损失权重')
    parser.add_argument('--mask_weight', type=float, default=2.0, help='掩码损失权重')

    return parser.parse_args()


def main(args=None):
    # 如果没有提供参数，解析命令行参数
    if args is None:
        args = parse_args()

    # 创建输出目录
    os.makedirs(os.path.join(args.project, args.name), exist_ok=True)

    # 配置信息
    config = {
        'data': args.data,
        'epochs': args.epochs,
        'batch': args.batch,
        'imgsz': args.imgsz,
        'device': args.device,
        'project': args.project,
        'name': args.name,
    }

    # 打印改进配置
    LOGGER.info('正在使用以下改进:')
    LOGGER.info(f'- 基础模型: YOLO11')
    LOGGER.info(f'- 高级图像增强: {"✓" if args.enable_augment else "✗"}')
    LOGGER.info(f'- 注意力机制: {args.attention}')
    LOGGER.info(f'- 分割头: {"✓" if args.segment else "✗"}')
    LOGGER.info(f'- 组合损失函数: {"✓" if args.combined_loss else "✗"}')

    # 选择模型配置文件
    model_path = args.model

    # 创建模型
    LOGGER.info(f'加载模型配置: {model_path}')
    model = YOLO(model_path)
    
    # 加载预训练权重
    if args.weights:
        LOGGER.info(f'加载预训练权重: {args.weights}')
        model.load(args.weights)

    # 设置损失函数
    if args.combined_loss:
        # 设置损失函数权重
        config['box_weight'] = args.box_weight
        config['cls_weight'] = args.cls_weight
        config['dfl_weight'] = args.dfl_weight
        config['mask_weight'] = args.mask_weight

    # 启动训练
    LOGGER.info('开始训练...')
    model.train(**config)

    # 模型验证
    LOGGER.info('模型验证...')
    model.val(data=args.data)

    # 保存模型
    LOGGER.info('导出模型...')
    model.export()

    return model


if __name__ == '__main__':
    # 从命令行运行
    main()
    
    # 或直接使用预设参数运行 (无需命令行参数)
    # config = {
    #     'data': 'datasets/data.yaml',
    #     'weights': 'yolov8n.pt',
    #     'model': 'ultralytics/cfg/models/11/yolo11-SegHead.yaml',
    #     'epochs': 300,
    #     'batch': 16,
    #     'enable_augment': True,
    #     'attention': 'ema',
    #     'segment': True,
    #     'combined_loss': True,
    # }
    # main(argparse.Namespace(**config))
