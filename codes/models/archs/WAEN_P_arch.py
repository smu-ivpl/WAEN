''' network architecture for WAEN-P '''
import functools
import torch
import torch.nn as nn
import torch.nn.functional as F
import models.archs.arch_util as arch_util

class WAEN_P(nn.Module):
    def __init__(self, nframes=7, nf=128, RBs=10):
        super(WAEN_P, self).__init__()
        self.nf = nf
        self.center = nframes // 2

        ### Residual Block
        ResidualBlock_noBN_f = functools.partial(arch_util.ResidualBlock_noBN, nf=nf)

        ### Discrete Wavelet Transform
        self.DWT = DWT()

        ### Inverse Wavelet Transform
        self.IWT = IWT()

        ### Wavelet Embedding
        self.wavelet_embedding = WaveletEmbedding(nframes=nframes)

        ### TSA
        self.tsa_embedding = TSA_Fusion(nf=nf, nframes=nframes, center=self.center)

        ### ResReconstruction for each sub-band and att
        self.recon_ll = ResReconstruction(input_nf=nframes*3*2, nf=nf, rec_m_RBs=5, filter_size=5, padd_size=2)
        self.recon_h = ResReconstruction(input_nf=nframes*3*2, nf=nf, rec_m_RBs=5, filter_size=3, padd_size=1)

        ### Final Res Reconstruction
        self.misr_conv1 = nn.Conv2d(nf * 2, nf, 3, 1, 1, bias=True) #3+128
        self.misr_feature_extraction = arch_util.make_layer(ResidualBlock_noBN_f, RBs)
        self.misr_conv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.misr_upconv1 = nn.Conv2d(nf, nf * 4, 3, 1, 1, bias=True)
        self.misr_upconv2 = nn.Conv2d(nf, 64 * 4, 3, 1, 1, bias=True)
        self.misr_pixel_shuffle = nn.PixelShuffle(2)
        self.misr_HRconv = nn.Conv2d(64, 64, 3, 1, 1, bias=True)
        self.misr_conv_last = nn.Conv2d(64, 3, 3, 1, 1, bias=True)

        #### activation function
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, x):
        B, N, C, H, W = x.size()  # N video frames
        x_center = x[:, self.center, :, :, :].contiguous()

        ### [Embedding]
        ### Wavelet Embedding
        w_emb_ll, w_emb_hl, w_emb_lh, w_emb_hh = self.wavelet_embedding(x)

        ### TSA Embedding
        tsa_emb = self.tsa_embedding(x)

        ### Concat (Final Embedding)
        w_emb_ll = torch.cat((x.view(B, -1, H, W), w_emb_ll), 1)  # B, 5*3*2, H, W
        w_emb_hl = torch.cat((x.view(B, -1, H, W), w_emb_hl), 1)  # B, 5*3*2, H, W
        w_emb_lh = torch.cat((x.view(B, -1, H, W), w_emb_lh), 1)  # B, 5*3*2, H, W
        w_emb_hh = torch.cat((x.view(B, -1, H, W), w_emb_hh), 1)  # B, 5*3*2, H, W

        ### [Reconstruction for each sub-band]
        fea_ll = self.recon_ll(w_emb_ll)  # B, 128, H, W
        fea_hl = self.recon_h(w_emb_hl)  # B, 128, H, W
        fea_lh = self.recon_h(w_emb_lh)  # B, 128, H, W
        fea_hh = self.recon_h(w_emb_hh)  # B, 128, H, W
        fea_wavelet = torch.cat((fea_ll.view(B, self.nf, 1, H, W), fea_hl.view(B, self.nf, 1, H, W), fea_lh.view(B, self.nf, 1, H, W), fea_hh.view(B, self.nf, 1, H, W)), 2)  # B, self.nf, 4, H, W
        fea_wavelet = self.IWT(fea_wavelet)  # B, self.nf, 1, H*2, W*2
        fea_wavelet = F.upsample(fea_wavelet.view(B, -1, H * 2, W * 2), size=(H, W), mode='bilinear', align_corners=False)  # B, self.nf, H, W

        ### [Fusion and Up-sampling, Reconstruction final]
        fused_fea = torch.cat((fea_wavelet, tsa_emb), 1) # in_ch = 256
        feature = self.lrelu(self.misr_conv1(fused_fea))
        feature = self.misr_feature_extraction(feature)
        feature = self.lrelu(self.misr_conv2(feature))
        feature = self.lrelu(self.misr_pixel_shuffle(self.misr_upconv1(feature)))
        feature = self.lrelu(self.misr_pixel_shuffle(self.misr_upconv2(feature)))
        feature = self.lrelu(self.misr_HRconv(feature))
        feature = self.misr_conv_last(feature)
        base = F.interpolate(x_center, scale_factor=4, mode='bilinear', align_corners=False)
        feature += base

        return feature

class WaveletEmbedding(nn.Module):
    ''' ResNet based Reconstruction module
    '''

    def __init__(self, nframes=5):
        super(WaveletEmbedding, self).__init__()

        ### Discrete Wavelet Transform
        self.DWT = DWT()

        self.wavelet_conv1_ll = nn.Conv2d(3 * nframes, 3 * nframes * 4, 5, 1, 2, bias=True)
        self.wavelet_conv1_hl = nn.Conv2d(3 * nframes, 3 * nframes * 4, 3, 1, 1, bias=True)
        self.wavelet_conv1_lh = nn.Conv2d(3 * nframes, 3 * nframes * 4, 3, 1, 1, bias=True)
        self.wavelet_conv1_hh = nn.Conv2d(3 * nframes, 3 * nframes * 4, 3, 1, 1, bias=True)
        self.wavelet_pixel_shuffle = nn.PixelShuffle(2)
        self.wavelet_conv2_ll = nn.Conv2d(3 * nframes, 3 * nframes, 5, 1, 2, bias=True)
        self.wavelet_conv2_hl = nn.Conv2d(3 * nframes, 3 * nframes, 3, 1, 1, bias=True)
        self.wavelet_conv2_lh = nn.Conv2d(3 * nframes, 3 * nframes, 3, 1, 1, bias=True)
        self.wavelet_conv2_hh = nn.Conv2d(3 * nframes, 3 * nframes, 3, 1, 1, bias=True)

        #### activation function
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, feature):
        B, N, C, H, W = feature.size()

        w_emb = self.DWT(feature)  # B, 5, 3*4, H // 2, W // 2

        w_emb_ll = w_emb[:, :, 0:3, :, :].contiguous().view(B, -1, H // 2, W // 2) # B, 5*3, H // 2, W // 2
        w_emb_hl = w_emb[:, :, 3:6, :, :].contiguous().view(B, -1, H // 2, W // 2) # B, 5*3, H // 2, W // 2
        w_emb_lh = w_emb[:, :, 6:9, :, :].contiguous().view(B, -1, H // 2, W // 2) # B, 5*3, H // 2, W // 2
        w_emb_hh = w_emb[:, :, 9:12, :, :].contiguous().view(B, -1, H // 2, W // 2) # B, 5*3, H // 2, W // 2

        w_emb_ll = self.lrelu(self.wavelet_conv1_ll(w_emb_ll))  # B, 5*3*4, H // 2, W // 2
        w_emb_hl = self.lrelu(self.wavelet_conv1_hl(w_emb_hl))  # B, 5*3*4, H // 2, W // 2
        w_emb_lh = self.lrelu(self.wavelet_conv1_lh(w_emb_lh))  # B, 5*3*4, H // 2, W // 2
        w_emb_hh = self.lrelu(self.wavelet_conv1_hh(w_emb_hh))  # B, 5*3*4, H // 2, W // 2

        w_emb_ll = self.wavelet_pixel_shuffle(w_emb_ll)  # B, 5*3, H, W
        w_emb_hl = self.wavelet_pixel_shuffle(w_emb_hl)  # B, 5*3, H, W
        w_emb_lh = self.wavelet_pixel_shuffle(w_emb_lh)  # B, 5*3, H, W
        w_emb_hh = self.wavelet_pixel_shuffle(w_emb_hh)  # B, 5*3, H, W

        w_emb_ll = self.lrelu(self.wavelet_conv2_ll(w_emb_ll))  # B, 5*3, H, W
        w_emb_hl = self.lrelu(self.wavelet_conv2_hl(w_emb_hl))  # B, 5*3, H, W
        w_emb_lh = self.lrelu(self.wavelet_conv2_lh(w_emb_lh))  # B, 5*3, H, W
        w_emb_hh = self.lrelu(self.wavelet_conv2_hh(w_emb_hh))  # B, 5*3, H, W

        return w_emb_ll, w_emb_hl, w_emb_lh, w_emb_hh

class ResReconstruction(nn.Module):
    ''' ResNet based ResReconstruction module
    '''

    def __init__(self, input_nf=30, nf=128, rec_m_RBs=10, filter_size=3, padd_size=1):
        super(ResReconstruction, self).__init__()

        ### Residual Block
        ResidualBlock_noBN_f = functools.partial(arch_util.ResidualBlock_noBN, nf=nf)

        self.misr_conv1 = nn.Conv2d(input_nf, nf, filter_size, 1, padd_size, bias=True)
        self.misr_feature_extraction = arch_util.make_layer(ResidualBlock_noBN_f, rec_m_RBs)
        self.misr_conv2 = nn.Conv2d(nf, nf, filter_size, 1, padd_size, bias=True)
        self.misr_HRconv = nn.Conv2d(nf, nf, filter_size, 1, padd_size, bias=True)

        #### activation function
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, tsa_fea):
        MISR_fea = self.lrelu(self.misr_conv1(tsa_fea))
        MISR_fea = self.misr_feature_extraction(MISR_fea)
        MISR_fea = self.lrelu(self.misr_conv2(MISR_fea))
        MISR_fea = self.lrelu(self.misr_HRconv(MISR_fea))
        #MISR_fea = self.misr_conv_last(MISR_fea)

        return MISR_fea

class TSA_Fusion(nn.Module):
    ''' Temporal Spatial Attention fusion module
    Temporal: correlation;
    Spatial: 3 pyramid levels.
    '''

    def __init__(self, nf=128, nframes=5, center=2):
        super(TSA_Fusion, self).__init__()
        self.center = center
        # temporal attention (before fusion conv)
        self.tAtt_1 = nn.Conv2d(3, nf, 3, 1, 1, bias=True)
        self.tAtt_2 = nn.Conv2d(3, nf, 3, 1, 1, bias=True)

        # fusion conv: using 1x1 to save parameters and computation
        self.fea_fusion = nn.Conv2d(nframes * 3, nf, 1, 1, bias=True)

        # spatial attention (after fusion conv)
        self.sAtt_1 = nn.Conv2d(nframes * 3, nf, 1, 1, bias=True)
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)
        self.avgpool = nn.AvgPool2d(3, stride=2, padding=1)
        self.sAtt_2 = nn.Conv2d(nf * 2, nf, 1, 1, bias=True)
        self.sAtt_3 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.sAtt_4 = nn.Conv2d(nf, nf, 1, 1, bias=True)
        self.sAtt_5 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.sAtt_L1 = nn.Conv2d(nf, nf, 1, 1, bias=True)
        self.sAtt_L2 = nn.Conv2d(nf * 2, nf, 3, 1, 1, bias=True)
        self.sAtt_L3 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.sAtt_add_1 = nn.Conv2d(nf, nf, 1, 1, bias=True)
        self.sAtt_add_2 = nn.Conv2d(nf, nf, 1, 1, bias=True)

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
        cor_prob = cor_prob.unsqueeze(2).repeat(1, 1, C, 1, 1).view(B, -1, H, W)
        aligned_fea = aligned_fea.view(B, -1, H, W) * cor_prob

        #### fusion
        fea = self.lrelu(self.fea_fusion(aligned_fea))

        #### spatial attention
        att = self.lrelu(self.sAtt_1(aligned_fea))
        att_max = self.maxpool(att)
        att_avg = self.avgpool(att)
        att = self.lrelu(self.sAtt_2(torch.cat([att_max, att_avg], dim=1)))
        # pyramid levels
        att_L = self.lrelu(self.sAtt_L1(att))
        att_max = self.maxpool(att_L)
        att_avg = self.avgpool(att_L)
        att_L = self.lrelu(self.sAtt_L2(torch.cat([att_max, att_avg], dim=1)))
        att_L = self.lrelu(self.sAtt_L3(att_L))
        att_L = F.interpolate(att_L, scale_factor=2, mode='bilinear', align_corners=False)

        att = self.lrelu(self.sAtt_3(att))
        att = att + att_L
        att = self.lrelu(self.sAtt_4(att))
        att = F.interpolate(att, scale_factor=2, mode='bilinear', align_corners=False)
        att = self.sAtt_5(att)
        att_add = self.sAtt_add_2(self.lrelu(self.sAtt_add_1(att)))
        att = torch.sigmoid(att)

        fea = fea * att * 2 + att_add
        return fea

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

        return torch.cat((x_LL, x_HL, x_LH, x_HH), 2)

class IWT(nn.Module):
    def __init__(self):
        super(IWT, self).__init__()
        self.requires_grad = False

    def forward(self, x):
        return self.iwt(x)

    def iwt(self, x):
        r = 2
        in_batch, in_frame, in_channel, in_height, in_width = x.size()

        out_batch, out_frame, out_channel, out_height, out_width = in_batch, in_frame, int(
            in_channel / (r ** 2)), r * in_height, r * in_width
        x1 = x[:, :, 0:out_channel, :, :] / 2
        x2 = x[:, :, out_channel:out_channel * 2, :, :] / 2
        x3 = x[:, :, out_channel * 2:out_channel * 3, :, :] / 2
        x4 = x[:, :, out_channel * 3:out_channel * 4, :, :] / 2

        h = torch.zeros([out_batch, out_frame, out_channel, out_height, out_width]).float().cuda()

        h[:, :, :, 0::2, 0::2] = x1 - x2 - x3 + x4
        h[:, :, :, 1::2, 0::2] = x1 - x2 + x3 - x4
        h[:, :, :, 0::2, 1::2] = x1 + x2 - x3 - x4
        h[:, :, :, 1::2, 1::2] = x1 + x2 + x3 + x4

        return h