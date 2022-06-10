''' network architecture for Group-based Bi-Directional Recurrent Wavelet Neural Network (GBR-WNN) '''

import functools
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import models.archs.arch_util as arch_util

class GBR_WNN(nn.Module):
    def __init__(self, nf=128, nframes=7, RBs=10, scale=4):
        super(GBR_WNN, self).__init__()
        self.nf = nf
        self.center = nframes // 2
        self.nframes = nframes
        self.scale = scale

        ### Discrete Wavelet Transform
        self.DWT = DWT()

        ### Temporal Attention in Temporal Wavelet Attention (TWA) Module
        self.temporal_attention = TemporalAttention(nframes=3) # L0, Current, L1 frames

        ### Reconstruction
        self.recon_feature = Reconstruction(nf=nf, RBs=RBs, scale=scale)

        ### Space-to-Depth
        self.space_to_depth = PixelUnShuffle(scale)

    def forward(self, c_x, r0_x, r0_h, r0_o, r1_x, r1_h, r1_o, first, last):
        # c_x : current input [B, 3, H, W]
        # r0_x : l0 neighbor input [B, 3, H, W] / r0_h : l0 neighbor hidden [B, nf, H, W] / r0_o : l0 neighbor prediction [B, 3, scale * scale * H, scale * scale * W]
        # r1_x : l1 neighbor input / r1_h : l1 neighbor hidden / r1_o : l1 neighbor prediction
        # first : bool. true if current frame is first frame
        # last : bool. true if current frame is last frame

        B, C, H, W = c_x.shape # [B, 3, H, W]

        ### [Temporal Attention in Temporal Wavelet Attention (TWA) Module]
        x = torch.cat([r0_x.unsqueeze(1), c_x.unsqueeze(1), r1_x.unsqueeze(1)], dim=1) # [B, 3, 3, H, W]
        x_att = self.temporal_attention(x) # [B, nf, H, W]

        ### [Spatial Attention using DWT in Temporal Wavelet Attention (TWA) Module]
        x_att = x_att.unsqueeze(1)  # [B, 1, nf, H, W]
        dwt_x_att = self.DWT(x_att)  # [B, 1, 4, nf, H//2, W//2]
        dwt_x_att = dwt_x_att.squeeze(1)  # [B, 4, nf, H//2, W//2]
        x_att = x_att.squeeze(1)  # [B, nf, H, W]

        dwt_x_att_mul = []
        for i in range(4):
            up_dwt_x_att = F.interpolate(dwt_x_att[:, i, :, :, :], scale_factor=2, mode='bilinear', align_corners=False)
            up_dwt_x_att_sig = torch.sigmoid(up_dwt_x_att)
            up_dwt_x_att_mul = x_att * up_dwt_x_att_sig  # [B, nf, H, W]
            dwt_x_att_mul.append(up_dwt_x_att_mul)
        dwt_x_att_mul = torch.stack(dwt_x_att_mul, dim=1).view(B, -1, H, W)  # [B, 4, nf, H, W] -> [B, 4*nf, H, W]

        ### [Reconstruction]
        if first:
            x_h, x_o = self.recon_feature(dwt_x_att_mul, r0_h, r0_o, r1_h, r1_o)
        elif last:
            r0_o = self.space_to_depth(r0_o)
            x_h, x_o = self.recon_feature(dwt_x_att_mul, r0_h, r0_o, r1_h, r1_o)
        else:
            r0_o = self.space_to_depth(r0_o)
            r1_o = self.space_to_depth(r1_o)
            x_h, x_o = self.recon_feature(dwt_x_att_mul, r0_h, r0_o, r1_h, r1_o)

        ### [Upsampling]
        x_o = F.pixel_shuffle(x_o, self.scale) + F.interpolate(c_x, scale_factor=self.scale, mode='bilinear', align_corners=False)

        return x_h, x_o

class TemporalAttention(nn.Module):
    ''' Temporal Attention in Temporal Wavelet Attention Module
    '''

    def __init__(self, nf=128, nframes=7, center=1, input_nf=3):
        super(TemporalAttention, self).__init__()
        self.center = center
        self.nframes = nframes

        # temporal attention (before fusion conv)
        self.tAtt_1 = nn.Conv2d(input_nf, nf, 3, 1, 1, bias=True)
        self.tAtt_2 = nn.Conv2d(input_nf, nf, 3, 1, 1, bias=True)

        # final spatial attention
        self.sAtt_1 = nn.Conv2d(input_nf * nframes, nf, 3, 1, 1, bias=True)
        self.sAtt_2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.sAtt_3 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.sAtt_4 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.sAtt_5 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)

        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, aligned_fea):
        B, N, C, H, W = aligned_fea.size()  # N video frames
        #### temporal attention
        emb_ref = self.tAtt_2(aligned_fea[:, self.center, :, :, :].clone())
        emb = self.tAtt_1(aligned_fea.view(-1, C, H, W)).view(B, N, -1, H, W)  # [B, N, C(nf), H, W]

        cor_l = []
        for i in range(N):
            emb_nbr = emb[:, i, :, :, :]
            cor_tmp = torch.sum(emb_nbr * emb_ref, 1).unsqueeze(1)  # B, 1, H, W
            cor_l.append(cor_tmp)
        cor_prob = torch.sigmoid(torch.cat(cor_l, dim=1))  # B, N, H, W
        cor_prob = cor_prob.unsqueeze(2).repeat(1, 1, C, 1, 1)  # B, N, C, H, W
        aligned_fea = aligned_fea * cor_prob  # B, N, C, H, W

        #### fusion
        att = self.lrelu(self.sAtt_1(aligned_fea.view(B, -1, H, W)))  # B, nf (128), H, W
        att_add = att
        att = self.lrelu(self.sAtt_2(att))
        att = self.lrelu(self.sAtt_3(att))
        att = att + att_add  # B, nf (128), H, W
        att = self.lrelu(self.sAtt_4(att))
        att = self.lrelu(self.sAtt_5(att))

        return att

class Reconstruction(nn.Module):
    ''' Reconstruction
    '''

    def __init__(self, nf=128, RBs=10, scale=4):
        super(Reconstruction, self).__init__()

        self.conv_1 = nn.Conv2d(scale ** 2 * 3 * 2 + nf * 2 + nf * 4, nf, (3, 3), stride=(1, 1), padding=(1, 1))
        self.conv_2 = nn.Conv2d(nf, nf, (3, 3), stride=(1, 1), padding=(1, 1))
        self.conv_3 = nn.Conv2d(nf, nf, (3, 3), stride=(1, 1), padding=(1, 1))
        self.conv_4 = nn.Conv2d(nf, nf, (3, 3), stride=(1, 1), padding=(1, 1))
        self.conv_5 = nn.Conv2d(nf, nf, (3, 3), stride=(1, 1), padding=(1, 1))

        ### Residual Block
        ResidualBlock_noBN_f = functools.partial(arch_util.ResidualBlock_noBN, nf=nf)
        self.recon_trunk = make_layer(ResidualBlock_noBN_f, RBs)

        self.conv_6 = nn.Conv2d(nf, nf, (3, 3), stride=(1, 1), padding=(1, 1))
        self.conv_7 = nn.Conv2d(nf, nf, (3, 3), stride=(1, 1), padding=(1, 1))
        self.conv_8 = nn.Conv2d(nf, nf, (3, 3), stride=(1, 1), padding=(1, 1))
        self.conv_9 = nn.Conv2d(nf, nf, (3, 3), stride=(1, 1), padding=(1, 1))
        self.conv_h = nn.Conv2d(nf, nf, (3, 3), stride=(1, 1), padding=(1, 1))
        self.conv_o = nn.Conv2d(nf, scale ** 2 * 3, (3, 3), stride=(1, 1), padding=(1, 1))

        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

        initialize_weights([self.conv_1, self.conv_h, self.conv_o], 0.1)

    def forward(self, x, r0_h, r0_o, r1_h, r1_o):
        x = torch.cat((x, r0_h, r0_o, r1_h, r1_o), dim=1) # [B, 96+256+512, H, W]
        x = self.lrelu(self.conv_1(x))
        x = self.lrelu(self.conv_2(x))
        x = self.lrelu(self.conv_3(x))
        x = self.lrelu(self.conv_4(x))
        x = self.lrelu(self.conv_5(x))
        x = self.recon_trunk(x)
        x = self.lrelu(self.conv_6(x))
        x = self.lrelu(self.conv_7(x))
        x = self.lrelu(self.conv_8(x))
        x = self.lrelu(self.conv_9(x))
        x_h = self.lrelu(self.conv_h(x))
        x_o = self.conv_o(x)

        return x_h, x_o

def pixel_unshuffle(input, upscale_factor):
    batch_size, channels, in_height, in_width = input.size()
    out_height = in_height // upscale_factor
    out_width = in_width // upscale_factor

    input_view = input.contiguous().view(
        batch_size, channels, out_height, upscale_factor,
        out_width, upscale_factor)

    channels *= upscale_factor ** 2
    unshuffle_out = input_view.permute(0, 1, 3, 5, 2, 4).contiguous()
    return unshuffle_out.view(batch_size, channels, out_height, out_width)

class PixelUnShuffle(nn.Module):
    def __init__(self, upscale_factor):
        super(PixelUnShuffle, self).__init__()
        self.upscale_factor = upscale_factor

    def forward(self, input):
        return pixel_unshuffle(input, self.upscale_factor)

    def extra_repr(self):
        return 'upscale_factor={}'.format(self.upscale_factor)

class DWT(nn.Module):
    def __init__(self):
        super(DWT, self).__init__()
        self.requires_grad = False

    def forward(self, x):
        return self.dwt(x)

    def dwt(self, x):

        x01 = x[:, :, :, 0::2, :] / 2
        x02 = x[:, :, :, 1::2, :] / 2
        x1 = x01[:, :, :, :, 0::2]
        x2 = x02[:, :, :, :, 0::2]
        x3 = x01[:, :, :, :, 1::2]
        x4 = x02[:, :, :, :, 1::2]
        x_LL = x1 + x2 + x3 + x4
        x_HL = -x1 - x2 + x3 + x4
        x_LH = -x1 + x2 - x3 + x4
        x_HH = x1 - x2 - x3 + x4

        return torch.cat((x_LL.unsqueeze(2), x_HL.unsqueeze(2), x_LH.unsqueeze(2), x_HH.unsqueeze(2)), 2)

def initialize_weights(net_l, scale=0.1):
    if not isinstance(net_l, list):
        net_l = [net_l]
    for net in net_l:
        for m in net.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                m.weight.data *= scale  # for residual block
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias.data, 0.0)

def make_layer(block, n_layers):
    layers = []
    for _ in range(n_layers):
        layers.append(block())
    return nn.Sequential(*layers)