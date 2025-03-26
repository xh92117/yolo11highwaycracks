import torch.nn as nn
from ultralytics.nn.modules.conv import Conv


class SegmentationHead(nn.Module):
    """分割头模块"""

    def __init__(self, c1, nc=1):
        super().__init__()
        self.nc = nc

        # 上采样和特征处理
        self.conv1 = Conv(c1, c1 // 2)
        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.conv2 = Conv(c1 // 2, c1 // 4)
        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.conv3 = Conv(c1 // 4, c1 // 8)
        self.up3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)

        # 输出层 - 二分类掩码
        self.final = nn.Conv2d(c1 // 8, 1, kernel_size=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.up1(x)
        x = self.conv2(x)
        x = self.up2(x)
        x = self.conv3(x)
        x = self.up3(x)
        x = self.final(x)
        return x


class DetectionSegmentationHead(nn.Module):
    """组合检测和分割的多任务头"""

    def __init__(self, c1, nc=1):
        super().__init__()
        self.nc = nc

        # 检测头 - 复用原始检测头
        self.detect = nn.Conv2d(c1, (nc + 5) * 3, 1)

        # 分割头
        self.segment = SegmentationHead(c1, nc)

    def forward(self, x):
        det_out = self.detect(x)  # 检测输出
        seg_out = self.segment(x)  # 分割输出

        return det_out, seg_out