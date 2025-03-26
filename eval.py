import argparse
import os
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from ultralytics import YOLO
from ultralytics.utils import LOGGER


def parse_args():
    parser = argparse.ArgumentParser(description='YOLO11公路裂缝检测评估脚本')

    # 基本参数
    parser.add_argument('--data', type=str, default='datasets/data.yaml', help='数据集配置文件')
    parser.add_argument('--weights', type=str, required=True, help='模型权重文件')
    parser.add_argument('--imgsz', type=int, default=640, help='图像尺寸')
    parser.add_argument('--batch', type=int, default=16, help='批次大小')
    parser.add_argument('--device', default='', help='评估设备 (如 0,1,2,3 或 cpu)')
    parser.add_argument('--save-json', action='store_true', help='保存结果为JSON文件')
    parser.add_argument('--save-plots', action='store_true', help='保存结果图表')
    parser.add_argument('--iou-thres', type=float, default=0.65, help='IOU阈值')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='置信度阈值')

    return parser.parse_args()


def evaluate_model(args):
    LOGGER.info(f'加载模型: {args.weights}')
    model = YOLO(args.weights)

    # 运行验证
    results = model.val(
        data=args.data,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        plots=args.save_plots,
        iou=args.iou_thres,
        conf=args.conf_thres,
        save_json=args.save_json
    )

    # 保存结果
    output_dir = Path(args.weights).parent / 'eval_results'
    os.makedirs(output_dir, exist_ok=True)

    # 提取关键指标
    metrics = {
        'mAP50': results.box.map50,
        'mAP50-95': results.box.map,
        'precision': results.box.mp,
        'recall': results.box.mr,
        'f1-score': 2 * results.box.mp * results.box.mr / (results.box.mp + results.box.mr + 1e-10),
    }

    # 如果有分割结果
    if hasattr(results, 'seg'):
        metrics.update({
            'mask_mAP50': results.seg.map50,
            'mask_mAP50-95': results.seg.map,
            'mask_precision': results.seg.mp,
            'mask_recall': results.seg.mr,
            'mask_f1-score': 2 * results.seg.mp * results.seg.mr / (results.seg.mp + results.seg.mr + 1e-10),
        })

    # 打印结果
    LOGGER.info(f'评估结果:')
    for k, v in metrics.items():
        LOGGER.info(f'- {k:15s}: {v:.5f}')

    # 保存结果为JSON
    if args.save_json:
        with open(output_dir / 'metrics.json', 'w') as f:
            json.dump(metrics, f, indent=4)

    return metrics, output_dir


def plot_metrics(metrics, output_dir):
    """绘制评估指标图表"""
    os.makedirs(output_dir, exist_ok=True)

    # 绘制检测指标
    plt.figure(figsize=(10, 6))
    plt.bar(['mAP50', 'mAP50-95', 'Precision', 'Recall', 'F1-Score'],
            [metrics['mAP50'], metrics['mAP50-95'], metrics['precision'],
             metrics['recall'], metrics['f1-score']])
    plt.title('公路裂缝检测评估指标')
    plt.ylim(0, 1.0)
    plt.ylabel('分数')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.savefig(output_dir / 'detection_metrics.png', dpi=300, bbox_inches='tight')

    # 如果有分割指标，绘制分割结果
    if 'mask_mAP50' in metrics:
        plt.figure(figsize=(10, 6))
        plt.bar(['Mask mAP50', 'Mask mAP50-95', 'Mask Precision', 'Mask Recall', 'Mask F1-Score'],
                [metrics['mask_mAP50'], metrics['mask_mAP50-95'], metrics['mask_precision'],
                 metrics['mask_recall'], metrics['mask_f1-score']])
        plt.title('公路裂缝分割评估指标')
        plt.ylim(0, 1.0)
        plt.ylabel('分数')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.savefig(output_dir / 'segmentation_metrics.png', dpi=300, bbox_inches='tight')


def visualize_predictions(model, data_yaml, output_dir, num_samples=5):
    """可视化模型预测结果"""
    from ultralytics.data.dataset import YOLODataset
    from ultralytics.utils.plotting import plot_results
    import yaml
    
    # 加载数据集配置
    with open(data_yaml, 'r') as f:
        data_dict = yaml.safe_load(f)
    
    # 准备测试图像目录
    test_dir = Path(data_dict.get('path', '.')) / data_dict.get('test', 'test/images')
    
    # 获取测试图像
    test_images = list(Path(test_dir).glob('*.jpg')) + list(Path(test_dir).glob('*.png'))
    if len(test_images) == 0:
        LOGGER.warning(f"未找到测试图像: {test_dir}")
        return
    
    # 随机选择样本
    import random
    sample_images = random.sample(test_images, min(num_samples, len(test_images)))
    
    # 保存目录
    vis_dir = output_dir / 'visualizations'
    os.makedirs(vis_dir, exist_ok=True)
    
    # 预测并保存结果
    for img_path in sample_images:
        results = model.predict(source=str(img_path), save=True, save_dir=vis_dir)
        LOGGER.info(f"已保存预测结果: {img_path.name}")


def main():
    args = parse_args()
    metrics, output_dir = evaluate_model(args)

    # 可视化结果
    if args.save_plots:
        plot_metrics(metrics, output_dir)
        # 可视化一些预测结果
        model = YOLO(args.weights)
        visualize_predictions(model, args.data, output_dir)


if __name__ == '__main__':
    main() 