import argparse
import os
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from ultralytics import YOLO
from ultralytics.utils import LOGGER


def parse_args():
    parser = argparse.ArgumentParser(description='YOLOv8改进版评估脚本')

    # 基本参数
    parser.add_argument('--data', type=str, default='data.yaml', help='数据集配置文件')
    parser.add_argument('--weights', type=str, required=True, help='模型权重文件')
    parser.add_argument('--imgsz', type=int, default=640, help='图像尺寸')
    parser.add_argument('--batch', type=int, default=16, help='批次大小')
    parser.add_argument('--device', default='', help='评估设备 (如 0,1,2,3 或 cpu)')
    parser.add_argument('--save-json', action='store_true', help='保存结果为JSON文件')
    parser.add_argument('--save-plots', action='store_true', help='保存结果图表')

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
        plots=args.save_plots
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

    return metrics


def main():
    args = parse_args()
    metrics = evaluate_model(args)

    # 可视化结果
    if args.save_plots:
        plot_metrics(metrics, Path(args.weights).parent / 'eval_results')


def plot_metrics(metrics, output_dir):
    """绘制评估指标图表"""
    os.makedirs(output_dir, exist_ok=True)

    # 绘制检测指标
    plt.figure(figsize=(10, 6))
    plt.bar(['mAP50', 'mAP50-95', 'Precision', 'Recall', 'F1-Score'],
            [metrics['mAP50'], metrics['mAP50-95'], metrics['precision'],
             metrics['recall'], metrics['f1-score']])
    plt.title('Object Detection Metrics')
    plt.ylim(0, 1.0)
    plt.ylabel('Score')
    plt.savefig(output_dir / 'detection_metrics.png')

    # 如果有分割指标，绘制分割结果
    if 'mask_mAP50' in metrics:
        plt.figure(figsize=(10, 6))
        plt.bar(['Mask mAP50', 'Mask mAP50-95', 'Mask Precision', 'Mask Recall', 'Mask F1-Score'],
                [metrics['mask_mAP50'], metrics['mask_mAP50-95'], metrics['mask_precision'],
                 metrics['mask_recall'], metrics['mask_f1-score']])
        plt.title('Segmentation Metrics')
        plt.ylim(0, 1.0)
        plt.ylabel('Score')
        plt.savefig(output_dir / 'segmentation_metrics.png')


if __name__ == '__main__':
    main()