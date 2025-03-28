# Ultralytics YOLO 🚀, AGPL-3.0 license
"""
Ultralytics modules.

Example:
    Visualize a module with Netron.
    ```python
    from ultralytics.nn.modules import *
    import torch
    import os

    x = torch.ones(1, 128, 40, 40)
    m = Conv(128, 128)
    f = f'{m._get_name()}.onnx'
    torch.onnx.export(m, x, f)
    os.system(f'onnxsim {f} {f} && open {f}')
    ```
"""

from .block import (C1, C2, C3, C3TR, DFL, SPP, SPPF, Bottleneck, BottleneckCSP, C2f, C3Ghost, C3x, GhostBottleneck,
                    HGBlock, HGStem, Proto, RepC3, ResNetLayer,C2f_Att, C2f_DCN, C2f_DCN2, BiFPN_Concat2, BiFPN_Concat3, SPDConv, CSPOmniKernel, SBA, FeaturePyramidSharedConv, C2f_DeepDBB, C2f_SHSA_CGLU, C2f_SHSA)
from .conv import (CBAM, ChannelAttention, Concat, Conv, Conv2, ConvTranspose, DWConv, DWConvTranspose2d, Focus,
                   GhostConv, LightConv, RepConv, SpatialAttention,EMA_attention, DSConv, ResBlock_CBAM)
from .head import Classify, Detect, Pose, RTDETRDecoder, Segment
from .seg_head import SegmentationHead, DetectionSegmentationHead
from .transformer import (AIFI, MLP, DeformableTransformerDecoder, DeformableTransformerDecoderLayer, LayerNorm2d,
                          MLPBlock, MSDeformAttn, TransformerBlock, TransformerEncoderLayer, TransformerLayer)



__all__ = ('Conv', 'Conv2', 'LightConv', 'RepConv', 'DWConv', 'DWConvTranspose2d', 'ConvTranspose', 'Focus',
           'GhostConv', 'ChannelAttention', 'SpatialAttention', 'CBAM', 'EMA_attention','Concat', 'TransformerLayer',
           'TransformerBlock', 'MLPBlock', 'LayerNorm2d', 'DFL', 'HGBlock', 'HGStem', 'SPP', 'SPPF', 'C1', 'C2', 'C3',
           'C2f', 'C3x', 'C3TR', 'C3Ghost', 'GhostBottleneck', 'Bottleneck', 'BottleneckCSP', 'Proto', 'Detect','SegmentationHead', 'DetectionSegmentationHead',
           'Segment', 'Pose', 'Classify', 'TransformerEncoderLayer', 'RepC3', 'RTDETRDecoder', 'AIFI',
           'DeformableTransformerDecoder', 'DeformableTransformerDecoderLayer', 'MSDeformAttn', 'MLP', 'ResNetLayer', 'C2f_Att', 'C2f_DCN', 'C2f_DCN2','BiFPN_Concat2',  'BiFPN_Concat3', 'SPDConv', 'CSPOmniKernel', 'DSConv', 'SBA', 'FeaturePyramidSharedConv', 'C2f_DeepDBB','ResBlock_CBAM','C2f_SHSA', 'C2f_SHSA_CGLU')
