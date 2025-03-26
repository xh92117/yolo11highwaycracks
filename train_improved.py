import argparse
import os
from ultralytics import YOLO
from ultralytics.data.custom_augment import CustomAugment
from ultralytics.nn.modules.attention import CBAM, EMAAttention
from ultralytics.nn.modules.seg_head import DetectionSegmentationHead, SegmentationHead
from ultralytics.utils.combined_loss import CombinedLoss
from ultralytics.utils import LOGGER


def parse_args():
    parser = argparse.ArgumentParser(description='YOLOv8改进版训练脚本')

    # 基本参数
    parser.add_argument('--data', type=str, default='data.yaml', help='数据集配置文件')
    parser.add_argument('--weights', type=str, default='yolov8n.pt', help='初始权重文件')
    parser.add_argument('--epochs', type=int, default=100, help='训练轮数')
    parser.add_argument('--batch', type=int, default=16, help='批次大小')
    parser.add_argument('--imgsz', type=int, default=640, help='图像尺寸')
    parser.add_argument('--device', default='', help='训练设备 (如 0,1,2,3 或 cpu)')
    parser.add_argument('--project', default='runs/train', help='项目名称')
    parser.add_argument('--name', default='exp', help='实验名称')

    # 改进开关
    parser.add_argument('--CustomAugment', action='store_true', help='启用高级图像增强')
    parser.add_argument('--attention', type=str, choices=['none', 'cbam', 'ema'], default='none',
                        help='使用注意力机制类型 (none, cbam, ema)')
    parser.add_argument('--segment', action='store_true', help='启用分割头')
    parser.add_argument('--combined-loss', action='store_true', help='使用改进的组合损失函数')

    # 损失权重
    parser.add_argument('--box-weight', type=float, default=7.5, help='边界框损失权重')
    parser.add_argument('--cls-weight', type=float, default=0.5, help='分类损失权重')
    parser.add_argument('--mask-weight', type=float, default=2.0, help='掩码损失权重')

    return parser.parse_args()


def main(args):
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
    LOGGER.info(f'- 高级图像增强: {"✓" if args.enable_augment else "✗"}')
    LOGGER.info(f'- 注意力机制: {args.attention}')
    LOGGER.info(f'- 分割头: {"✓" if args.segment else "✗"}')
    LOGGER.info(f'- 组合损失函数: {"✓" if args.combined_loss else "✗"}')

    # 创建模型
    model = YOLO(args.weights)

    # 根据命令行参数选择配置文件
    if args.segment:
        # 使用带分割头的模型配置
        config['cfg'] = 'ultralytics/cfg/models/v8/yolov8_improved.yaml'
    elif args.attention != 'none':
        # 选择注意力机制配置
        if args.attention == 'cbam':
            config['cfg'] = 'ultralytics/cfg/models/v8/yolov8_cbam.yaml'
        elif args.attention == 'ema':
            config['cfg'] = 'ultralytics/cfg/models/v8/yolov8_ema.yaml'

    # 添加图像增强
    if args.enable_augment:
        # 注册自定义增强器
        from ultralytics.data.augment import v8_transforms
        # 替换或修改v8_transforms函数添加自定义增强

    # 添加组合损失函数
    if args.combined_loss:
        # 设置损失函数权重
        config['box_weight'] = args.box_weight
        config['cls_weight'] = args.cls_weight
        config['mask_weight'] = args.mask_weight

    # 启动训练
    model.train(**config)

    # 保存模型
    model.export()


if __name__ == '__main__':
    args = parse_args()
    main(args)