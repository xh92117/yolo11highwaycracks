import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics.utils.metrics import bbox_iou
from ultralytics.utils.tal import TaskAlignedAssigner


class CombinedLoss:
    """组合目标检测和分割的损失函数"""

    def __init__(self, model, box_weight=7.5, cls_weight=0.5, dfl_weight=1.5, mask_weight=2.0):
        device = next(model.parameters()).device

        # 损失函数权重
        self.hyp = {
            'box': box_weight,  # 边界框损失权重
            'cls': cls_weight,  # 分类损失权重
            'dfl': dfl_weight,  # DFL损失权重
            'mask': mask_weight,  # 掩码损失权重
        }

        # 任务对齐分配器
        self.assigner = TaskAlignedAssigner(
            topk=10,
            num_classes=model.nc,
            alpha=0.5,
            beta=6.0
        )

        # 定义损失函数
        self.bce_cls = nn.BCEWithLogitsLoss(reduction='none')
        self.bce_mask = nn.BCEWithLogitsLoss(reduction='mean')
        self.dice_loss = DiceLoss()

    def __call__(self, preds, batch):
        """
        计算综合损失
        Args:
            preds: 模型预测 [检测预测, 分割预测]
            batch: 数据批次
        Returns:
            loss: 总损失和各部分损失
        """
        det_preds, seg_preds = preds
        device = det_preds.device

        # 初始化损失
        loss = torch.zeros(5, device=device)  # box, cls, dfl, mask, total

        # 1. 检测损失计算
        # 分配正样本
        target_bboxes = batch['bboxes']
        target_labels = batch['cls']

        assigned_targets = self.assigner(
            det_preds[..., 5:],  # cls preds
            det_preds[..., :4],  # bbox preds
            target_bboxes,
            target_labels
        )

        # 计算边界框损失
        iou = bbox_iou(
            det_preds[..., :4][assigned_targets.indices],
            assigned_targets.bboxes,
            xywh=True
        )
        loss[0] = (1.0 - iou).mean() * self.hyp['box']

        # 计算分类损失
        cls_loss = self.bce_cls(
            det_preds[..., 5:],
            assigned_targets.cls_targets
        ).mean()
        loss[1] = cls_loss * self.hyp['cls']

        # 2. 分割损失计算
        # 掩码真值
        target_masks = batch['masks'].float()

        # BCE 损失
        bce_mask_loss = self.bce_mask(seg_preds, target_masks)
        # Dice 损失 - 更关注形状
        dice_mask_loss = self.dice_loss(
            torch.sigmoid(seg_preds),
            target_masks
        )
        # 组合掩码损失
        mask_loss = 0.5 * bce_mask_loss + 0.5 * dice_mask_loss
        loss[3] = mask_loss * self.hyp['mask']

        # 总损失
        loss[4] = loss[0] + loss[1] + loss[2] + loss[3]

        return loss


class DiceLoss(nn.Module):
    """Dice损失函数 - 更关注分割形状"""

    def __init__(self, smooth=1.0):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, pred, target):
        pred = pred.contiguous().view(-1)
        target = target.contiguous().view(-1)

        intersection = (pred * target).sum()
        dice = (2. * intersection + self.smooth) / (
                pred.sum() + target.sum() + self.smooth
        )

        return 1. - dice