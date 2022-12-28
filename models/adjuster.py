import torch.nn as nn
import torch.nn.functional as F
import torch
from cc_torch import connected_components_labeling

def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


def edge_calc(image: torch.Tensor) -> torch.Tensor:
    r"""
    Calculate image gradients horizontally and vertically, then add them up.
    Args :
        images : torch.Tensor, input image in [N, C, H, W] shape
    Return :
        Edge maps, torch.Tensor
    """

    edg_x, edg_y = F.pad(torch.abs(image[:, :, :, :-1] - image[:, :, :, 1:]), (1, 0, 0, 0)), \
                   F.pad(torch.abs(image[:, :, :-1, :] - image[:, :, 1:, :]), (0, 0, 1, 0))
    return (edg_x + edg_y)

class GuiderResidualBlock(nn.Module):
    def __init__(self, in_features, dilation):
        super(GuiderResidualBlock, self).__init__()

        conv_block = [#PALayer(in_features),
                      #CALayer(in_features),
                      nn.Conv2d(in_features, in_features, 3, padding=dilation, dilation=dilation),
                      nn.InstanceNorm2d(in_features),
                      nn.PReLU(),
                      nn.Conv2d(in_features, in_features, 1, padding=0),
                      nn.InstanceNorm2d(in_features),
                      nn.PReLU(),
                      ]
        self.conv_block = nn.Sequential(*conv_block)

    def forward(self, x):
        return x + self.conv_block(x)


class BasicBlock(nn.Module):
    def __init__(self, type, inplane, outplane, stride, leaky=True):
        super(BasicBlock, self).__init__()
        conv_block = []
        if type == "Conv":
            conv_block += [nn.Conv2d(inplane, outplane, kernel_size=3, stride=stride, padding=1)]
        elif type == "Deconv":
            conv_block += [nn.ConvTranspose2d(inplane, outplane, kernel_size=4, stride=stride, padding=1)]

        if leaky:
            conv_block += [nn.InstanceNorm2d(outplane), nn.PReLU()]
        else :
            conv_block += [nn.InstanceNorm2d(outplane),
                           nn.ReLU(True)]

        self.conv_block = nn.Sequential(*conv_block)

    def forward(self, x):
        out = self.conv_block(x)
        return out


class ConvBlock(nn.Sequential):
    def __init__(self, in_channel, out_channel, ker_size, padd, stride):
        super(ConvBlock,self).__init__()
        self.add_module('conv',nn.Conv2d(in_channel ,out_channel,kernel_size=ker_size,stride=stride,padding=padd)),
        self.add_module('norm', nn.InstanceNorm2d(out_channel)),
        self.add_module('leakyReLU', nn.PReLU())

class Adjuster(nn.Module):
    def __init__(self, in_channels=3, nfc=32, fuse_option="concat", out_channels=3):
        super(Adjuster, self).__init__()
        self.nfc = nfc
        self.num_layer = 6
        N = self.nfc

        self.to_feature = nn.Sequential(*([ConvBlock(in_channels, N, 3, 1, 1)] + [
            GuiderResidualBlock(32, 1) for _ in range(self.num_layer - 2)
        ]))

        self.body = nn.Sequential(*[GuiderResidualBlock(32, 2 ** (i // 2)) for i in range(8)
                                    ] + [GuiderResidualBlock(32, 1)])

        self.siamese_body = nn.Sequential(*[GuiderResidualBlock(32, 1)]) #[GuiderResidualBlock(32, 2 ** (i // 2)) for i in range(8)] +
        self.fuser = nn.Sequential(
            nn.Conv2d(32 + 32, 32, 1, 1, 0, bias=True),
            nn.InstanceNorm2d(16),
            # nn.LeakyReLU(0.2),
            nn.PReLU()
        )

        self.to_edge = nn.Sequential(
            nn.Conv2d(32, 16, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm2d(16),
            #nn.LeakyReLU(0.2),
            nn.PReLU(),
            nn.Conv2d(16, 1, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()
            )

        for k, v in self.named_parameters():
            print("k: {},  requires_grad: {}".format(k, v.requires_grad))

    def forward(self, x, lam=0, inference=False):
        input_curr = x
        n, c, h, w = x.shape

        grad = torch.clip(torch.max(edge_calc(x), dim=-3, keepdim=True)[0], 0, 1)

        x2 = self.to_feature(x)
        state, siamese_state = self.body(x2), self.siamese_body(x2)
        actual_lam = max(0.2, lam)
        tmp = self.fuser(torch.cat([state, actual_lam * siamese_state], dim=-3)) 
        mask =  self.to_edge(tmp)
        if not inference : return mask * grad 
        mask_org = mask.clone()
        thresh = 0.05
        mask_org[mask < thresh] = 0
        mask_org[mask >= thresh] = torch.clamp(torch.exp(mask_org[mask >= thresh]) - 1, 0, 1)
        thresh = 0.1
        
        if (lam < 0.2):
            mask[mask < thresh] = 0
            mask[mask >= thresh] = 1
            mask = (mask).type(torch.uint8)
            mask = connected_components_labeling(mask.squeeze(1), torch.tensor(lam))#
            edge = mask.unsqueeze(1) #* mask_org
            return edge * mask_org * grad
        
        else : return mask_org * grad if lam < 0.9 else torch.sqrt(mask_org * grad)
