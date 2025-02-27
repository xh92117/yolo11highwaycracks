######################################## APConv AAAI2025   by AI Little monster start  ########################################
 
import torch
import torch.nn as nn
import torch.nn.functional as F
 
from ultralytics.nn.modules.conv import Conv,autopad
from ultralytics.nn.modules.block import Bottleneck, C2f,C3k,C3k2
 
from mmcv.cnn import ConvModule
from mmengine.model import caffe2_xavier_init, constant_init
 
from ultralytics.nn.modules.conv import Conv
 
 
class ContextAggregation(nn.Module):
    """
    Context Aggregation Block.
    Args:
        in_channels (int): Number of input channels.
        reduction (int, optional): Channel reduction ratio. Default: 1.
        conv_cfg (dict or None, optional): Config dict for the convolution
            layer. Default: None.
    """
 
    def __init__(self, in_channels, reduction=1):
        super(ContextAggregation, self).__init__()
        self.in_channels = in_channels
        self.reduction = reduction
        self.inter_channels = max(in_channels // reduction, 1)
 
        conv_params = dict(kernel_size=1, act_cfg=None)
 
        self.a = ConvModule(in_channels, 1, **conv_params)
        self.k = ConvModule(in_channels, 1, **conv_params)
        self.v = ConvModule(in_channels, self.inter_channels, **conv_params)
        self.m = ConvModule(self.inter_channels, in_channels, **conv_params)
 
        self.init_weights()
 
    def init_weights(self):
        for m in (self.a, self.k, self.v):
            caffe2_xavier_init(m.conv)
        constant_init(self.m.conv, 0)
 
    def forward(self, x):
        # n, c = x.size(0)
        n = x.size(0)
        c = self.inter_channels
        # n, nH, nW, c = x.shape
 
        # a: [N, 1, H, W]
        a = self.a(x).sigmoid()
 
        # k: [N, 1, HW, 1]
        k = self.k(x).view(n, 1, -1, 1).softmax(2)
 
        # v: [N, 1, C, HW]
        v = self.v(x).view(n, 1, c, -1)
 
        # y: [N, C, 1, 1]
        y = torch.matmul(v, k).view(n, c, 1, 1)
        y = self.m(y) * a
 
        return x + y
 
 
 
class PConv_att(nn.Module):  
    ''' Pinwheel-shaped Convolution using the Asymmetric Padding method. '''
    
    def __init__(self, c1, c2, k, s):
        super().__init__()
 
        # self.k = k
        p = [(k, 0, 1, 0), (0, k, 0, 1), (0, 1, k, 0), (1, 0, 0, k)]
        self.pad = [nn.ZeroPad2d(padding=(p[g])) for g in range(4)]
        self.cw = Conv(c1, c2 // 4, (1, k), s=s, p=0)
        self.ch = Conv(c1, c2 // 4, (k, 1), s=s, p=0)
        self.cat = Conv(c2, c2, 2, s=1, p=0)
        self.att = ContextAggregation(c1)
 
    def forward(self, x):
        x =   self.att(x)
        yw0 = self.cw(self.pad[0](x))
        yw1 = self.cw(self.pad[1](x))
        yh0 = self.ch(self.pad[2](x))
        yh1 = self.ch(self.pad[3](x))
        return self.cat(torch.cat([yw0, yw1, yh0, yh1], dim=1))
 
######################################## APConv AAAI2025   by AI Little monster END  ########################################
