import torch.nn as nn
import torch.nn.functional as F
import torch
import time

class ResnetBlock(nn.Module):
    def __init__(self, dim, dilation, norm_layer=nn.Identity, use_dropout=False, use_bias=True):
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, dilation, norm_layer, use_dropout, use_bias)

    def build_conv_block(self, dim, dilation, norm_layer, use_dropout, use_bias):
        conv_block = []

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=dilation, dilation=dilation, bias=use_bias),
                       PALayer(dim),CALayer(dim),
                       norm_layer(dim),
                       nn.PReLU(),
                       ]

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=dilation, dilation=dilation, bias=use_bias),
                       PALayer(dim),CALayer(dim),
                       norm_layer(dim),
                       nn.PReLU()]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)
        return out

class BasicBlock(nn.Module):
    def __init__(self, type, inplane, outplane, stride):
        super(BasicBlock, self).__init__()
        conv_block = []
        if type == "Conv":
            conv_block += [nn.Conv2d(inplane, outplane, kernel_size=3, stride=stride, padding=1)]
        elif type == "Deconv":
            conv_block += [nn.ConvTranspose2d(inplane, outplane, kernel_size=4, stride=stride, padding=1)]

        conv_block +=[nn.Identity(outplane),
                      PALayer(outplane),CALayer(outplane),
                      nn.PReLU(),]

        self.conv_block = nn.Sequential(*conv_block)

    def forward(self, x):
        out = self.conv_block(x)
        return out


class PALayer(nn.Module):
    """
        This module implements Pixel-wise Attention (PA),
        See https://arxiv.org/abs/1911.07559v2 for more information
    """

    def __init__(self, channel: int):
        super(PALayer, self).__init__()
        self.pa0 = nn.Conv2d(channel, channel // 8, 1, padding=0, bias=True)
        self.relu = nn.ReLU(inplace=True)
        self.pa1 = nn.Sequential(
            nn.Conv2d(channel // 8, 1, 1, padding=0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.pa1(self.relu(self.pa0(x)))
        return x * y


class CALayer(nn.Module):
    """
       This module implements Channel-wise Attention (CA),
       See https://arxiv.org/abs/1911.07559v2 for more information
    """

    def __init__(self, channel: int):
        super(CALayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.ca0 = nn.Conv2d(channel, channel // 8, 1, padding=0, bias=True)
        self.relu0 = nn.ReLU(inplace=True)
        self.ca1 = nn.Conv2d(channel // 8, channel, 1, padding=0, bias=True)
        self.sig = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.avg_pool(x)
        y = self.sig(self.ca1(self.relu0(self.ca0(y))))
        return x * y

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('norm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

class AttConvBlock(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, kernel_size=3, padding=None, stride=1, bias=True, dilation=1):
        super(AttConvBlock, self).__init__()
        padding = padding or dilation * (kernel_size - 1) // 2
        self.model = nn.Sequential(
            PALayer(in_channels),
            CALayer(in_channels),
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                      padding=padding, stride=stride, bias=bias, dilation=dilation),
        )

    def forward(self, x):
        return self.model(x)

    
    
class DownBlock(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, kernel_size=3, padding=None, stride=1,
                 norm=False, bias=True, dilation=1):
        super(DownBlock, self).__init__()
        padding = padding or dilation * (kernel_size - 1) // 2

        self.model = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                      padding=padding, stride=stride, bias=bias, dilation=dilation),
            PALayer(out_channels),
            CALayer(out_channels)
        )
        self.norm = nn.InstanceNorm(out_channels) if norm else None
        self.relu = nn.PReLU()
    
    def forward(self, input_features):
        input_features = self.model(input_features)
        if self.norm:
            input_features = self.norm(input_features)
        out_features = self.relu(input_features)
        return out_features

    
class UpBlock(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, kernel_size=3, padding=None, stride=1,
                 norm=False, bias=True, dilation=1):
        super(UpBlock, self).__init__()
        padding = padding or dilation * (kernel_size - 1) // 2


        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.model = nn.Sequential(
            nn.Conv2d(in_channels * 2, out_channels, kernel_size=kernel_size,
                      padding=padding, stride=stride, bias=bias, dilation=dilation),
            PALayer(out_channels),
            CALayer(out_channels)
        )
        self.norm = nn.InstanceNorm(out_channels) if norm else None
        self.relu = nn.PReLU()

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        x = self.model(x)
        if self.norm:
            x = self.norm(x)
        out_features = self.relu(x)
        return out_features

class UNet(nn.Module):
    def __init__(self, in_channels=64):
        super(UNet, self).__init__()
        self.down1 = DownBlock(in_channels, in_channels)
        self.down2 = DownBlock(in_channels, in_channels)
        self.down3 = DownBlock(in_channels, in_channels)
        self.down4 = DownBlock(in_channels, in_channels)
        self.up1 = UpBlock(in_channels, in_channels)
        self.up2 = UpBlock(in_channels, in_channels)
        self.up3 = UpBlock(in_channels, in_channels)
        self.up4 = UpBlock(in_channels, in_channels)

    def forward(self, x):
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        u1 = self.up1(d4, d3)
        u2 = self.up2(u1, d2)
        u3 = self.up3(u2, d1)
        u4 = self.up4(u3, x)
        return u4
        
class Smoother(nn.Module):
    def __init__(self, inference_only=True):
        r'''
            inference_only : If inference_only is True, ignore side-outputs
        '''
        super(Smoother, self).__init__()
        self.is_cuda = torch.cuda.is_available()
        self.inference_only = inference_only

        self.backbone0 = nn.Sequential(*[
                        BasicBlock("Conv", 4, 32, 1),
                        BasicBlock("Conv", 32,32, 1),
                        ResnetBlock(32, 1)
        ])
        
        self.backbone1 = nn.Sequential(*[
                        ResnetBlock(32, 1),
                        ResnetBlock(32, 1),
                        nn.MaxPool2d(2)
        ])


        self.backbone2 = nn.Sequential(*[
                        ResnetBlock(32, 1),
                        ResnetBlock(32, 1),
                        nn.MaxPool2d(2)
        ])

        if not inference_only:
            self.up1 = UpBlock(32, 32)   
        self.up21 = UpBlock(32, 32)     
        self.up22 = UpBlock(32, 32) 
        body0, body1, body2 = [UNet(32)], [UNet(32)], [UNet(32)]
        self.body0, self.body1, self.body2 = nn.Sequential(*body0), nn.Sequential(*body1), nn.Sequential(*body2)
        
        outy = [BasicBlock("Conv", 32, 32, 1)]
        outy += [nn.Conv2d(32, 3, kernel_size=3, stride=1, padding=1)]

        self.outy = nn.Sequential(*outy)

        
    def forward(self, x, edges, inference=True):

        in_stage0 = self.backbone0(torch.cat([x, edges], dim=1))
        body_stage0 = self.body0(in_stage0)
        in_stage1 = self.backbone1(body_stage0)
        body_stage1 = self.body1(in_stage1)
        in_stage2 = self.backbone2(body_stage1)
        body_stage2 = self.body2(in_stage2)
        out_stage2 = self.outy(self.up21(self.up22(body_stage2, body_stage1), body_stage0))

        if not inference:
            out_stage0 = self.outy(body_stage0) #+ x
            out_stage1 = self.outy(self.up1(body_stage1, body_stage0)) #+ x
            return out_stage0, out_stage1, out_stage2
        
        return out_stage2

