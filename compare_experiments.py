import argparse
import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description='对比实验结果分析')
    parser.add_argument('--results-dir', type=str, required=True, help='包含实验结果的目录')
    parser.add_argument('--output-dir', type=str, default='analysis', help='分析结果保存目录')
    return parser.parse_args()


def load_experiment_results(results_dir):
    """加载所有实验结果"""
    results = []

    # 遍历实验目录
    for exp_dir in Path(results_dir).glob('exp*'):
        if not exp_dir.is_dir():
            continue

        # 尝试加载配置和结果
        config_path = exp_dir / 'args.json'
        metrics_path = exp_dir / 'eval_results/metrics.json'

        if not config_path.exists() or not metrics_path.exists():
            continue

        # 读取配置和指标
        with open(config_path, 'r') as f:
            config = json.load(f)

        with open(metrics_path, 'r') as f:
            metrics = json.load(f)

        # 组合数据
        exp_data = {
            'experiment': exp_dir.name,
            'augment': config.get('enable_augment', False),
            'attention': config.get('attention', 'none'),
            'segment': config.get('segment', False),
            'combined_loss': config.get('combined_loss', False),
        }
        exp_data.update(metrics)

        results.append(exp_data)

    return pd.DataFrame(results)


def analyze_results(df, output_dir):
    """分析实验结果"""
    os.makedirs(output_dir, exist_ok=True)

    # 1. 保存表格数据
    df.to_csv(f'{output_dir}/results_comparison.csv', index=False)

    # 2. 绘制mAP比较图
    plt.figure(figsize=(12, 8))
    sns.barplot(x='experiment', y='mAP50', data=df)
    plt.title('mAP@0.5 Comparison')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/map50_comparison.png')

    # 3. 绘制改进影响对比图
    plt.figure(figsize=(14, 8))

    # 创建改进组合标签
    df['improvements'] = df.apply(lambda row:
                                  f"Aug{'✓' if row['augment'] else '✗'}_"
                                  f"Att{row['attention']}_"
                                  f"Seg{'✓' if row['segment'] else '✗'}_"
                                  f"Loss{'✓' if row['combined_loss'] else '✗'}", axis=1)

    # 画mAP和分割精度对比
    sns.barplot(x='improvements', y='mAP50', data=df, color='blue', label='mAP50')

    if 'mask_mAP50' in df.columns:
        plt.figure(figsize=(14, 8))
        sns.barplot(x='improvements', y='mask_mAP50', data=df, color='orange', label='Mask mAP50')

    plt.title('Effect of Different Improvements')
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'{output_dir}/improvements_comparison.png')

    # 4. 注意力机制对比
    if len(df['attention'].unique()) > 1:
        plt.figure(figsize=(10, 6))
        sns.barplot(x='attention', y='mAP50', data=df)
        plt.title('Effect of Attention Mechanisms')
        plt.savefig(f'{output_dir}/attention_comparison.png')

    # 5. 损失函数权重分析
    if 'combined_loss' in df.columns and df['combined_loss'].any():
        loss_df = df[df['combined_loss']]
        if len(loss_df) > 1:
            plt.figure(figsize=(12, 6))
            plt.scatter(loss_df['box_weight'], loss_df['mAP50'], label='Box Weight')
            plt.scatter(loss_df['cls_weight'], loss_df['mAP50'], label='Cls Weight')
            plt.scatter(loss_df['mask_weight'], loss_df['mAP50'], label='Mask Weight')
            plt.title('Effect of Loss Weights on Performance')
            plt.xlabel('Weight Value')
            plt.ylabel('mAP50')
            plt.legend()
            plt.savefig(f'{output_dir}/loss_weights_analysis.png')


def main():
    args = parse_args()

    # 加载实验结果
    results_df = load_experiment_results(args.results_dir)

    if len(results_df) == 0:
        print('未找到实验结果!')
        return

    print(f'加载了 {len(results_df)} 个实验结果')

    # 分析结果
    analyze_results(results_df, args.output_dir)
    print(f'分析结果已保存到 {args.output_dir}')

    # 打印最佳结果
    best_map = results_df.loc[results_df['mAP50'].idxmax()]
    print('\n最佳mAP50结果:')
    print(f'- 实验: {best_map["experiment"]}')
    print(f'- mAP50: {best_map["mAP50"]:.5f}')
    print(f'- 配置: 增强={best_map["augment"]}, 注意力={best_map["attention"]}, '
          f'分割={best_map["segment"]}, 组合损失={best_map["combined_loss"]}')

    if 'mask_mAP50' in results_df.columns:
        best_mask = results_df.loc[results_df['mask_mAP50'].idxmax()]
        print('\n最佳掩码mAP50结果:')
        print(f'- 实验: {best_mask["experiment"]}')
        print(f'- 掩码mAP50: {best_mask["mask_mAP50"]:.5f}')


if __name__ == '__main__':
    main()