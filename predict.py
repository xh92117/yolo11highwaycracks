import argparse
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from ultralytics import YOLO
from ultralytics.utils import LOGGER, colorstr
from PIL import Image


def parse_args():
    parser = argparse.ArgumentParser(description='YOLO11公路裂缝检测预测脚本')

    parser.add_argument('--weights', type=str, required=True, help='模型权重文件')
    parser.add_argument('--source', type=str, required=True, help='要预测的图像、视频或目录路径')
    parser.add_argument('--imgsz', type=int, default=640, help='图像尺寸')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='置信度阈值')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IOU阈值')
    parser.add_argument('--max-det', type=int, default=300, help='最大检测数量')
    parser.add_argument('--device', default='', help='推理设备 (如 cuda:0 或 cpu)')
    parser.add_argument('--view-img', action='store_true', help='显示结果')
    parser.add_argument('--save-txt', action='store_true', help='保存为txt结果')
    parser.add_argument('--save-conf', action='store_true', help='保存置信度分数')
    parser.add_argument('--save-crop', action='store_true', help='保存裁剪的预测对象')
    parser.add_argument('--hide-labels', action='store_true', help='隐藏标签')
    parser.add_argument('--hide-conf', action='store_true', help='隐藏置信度')
    parser.add_argument('--save-masks', action='store_true', default=True, help='保存分割掩码')
    parser.add_argument('--line-thickness', type=int, default=3, help='边界框线条厚度(像素)')
    parser.add_argument('--dnn', action='store_true', help='使用OpenCV DNN进行ONNX推理')
    parser.add_argument('--retina-masks', action='store_true', help='使用高分辨率分割掩码')
    parser.add_argument('--half', action='store_true', help='使用FP16半精度推理')
    parser.add_argument('--project', default='runs/predict', help='保存结果的项目路径')
    parser.add_argument('--name', default='exp', help='保存结果的实验名称')
    parser.add_argument('--exist-ok', action='store_true', help='现有项目/名称是否可以覆盖')

    return parser.parse_args()


def predict(args):
    """
    执行预测并保存结果
    """
    LOGGER.info(f"{colorstr('predict:')} 权重={args.weights}, 源数据={args.source}")
    save_dir = Path(os.path.join(args.project, args.name))
    if not args.exist_ok:
        os.makedirs(save_dir, exist_ok=args.exist_ok)
    LOGGER.info(f"结果将保存到 {save_dir}")

    # 加载模型
    model = YOLO(args.weights)
    model.info()  # 输出模型信息

    # 预测配置
    pred_args = {
        'conf': args.conf_thres,
        'iou': args.iou_thres,
        'max_det': args.max_det,
        'device': args.device,
        'save': True,
        'save_txt': args.save_txt,
        'save_conf': args.save_conf,
        'save_crop': args.save_crop,
        'hide_labels': args.hide_labels,
        'hide_conf': args.hide_conf,
        'half': args.half,
        'dnn': args.dnn,
        'line_thickness': args.line_thickness,
        'visualize': False,
        'retina_masks': args.retina_masks,
        'project': args.project,
        'name': args.name,
    }

    # 执行预测
    results = model.predict(source=args.source, imgsz=args.imgsz, **pred_args)

    # 分析结果
    for result in results:
        img_path = Path(result.path)
        save_path = save_dir / img_path.name

        # 如果有分割掩码，单独保存掩码结果
        if hasattr(result, 'masks') and result.masks is not None and args.save_masks:
            save_mask_path = save_dir / f"{img_path.stem}_mask{img_path.suffix}"
            save_mask(result, save_mask_path)
    
    LOGGER.info(f"预测完成。结果保存在: {save_dir}")
    return results


def save_mask(result, save_path):
    """将预测的分割掩码保存为彩色图像"""
    if result.masks is None:
        return
    
    # 原始图像
    img = Image.fromarray(result.orig_img)
    
    # 合并所有掩码
    masks = result.masks.data
    if len(masks) == 0:
        return
    
    # 创建彩色掩码
    mask_overlay = np.zeros_like(result.orig_img)
    for i, mask in enumerate(masks):
        color = np.array([0, 0, 255])  # 红色用于公路裂缝
        mask_np = mask.cpu().numpy()
        mask_overlay[mask_np > 0.5] = color
    
    # 创建半透明覆盖
    alpha = 0.5
    overlay = Image.fromarray(mask_overlay)
    img_with_masks = Image.blend(img.convert("RGB"), overlay.convert("RGB"), alpha)
    
    # 保存结果
    img_with_masks.save(save_path)
    LOGGER.info(f"掩码保存到: {save_path}")


def main():
    args = parse_args()
    predict(args)


if __name__ == '__main__':
    main() 