''' network architecture for WAEN-S '''
import functools
import torch
import torch.nn as nn
import torch.nn.functional as F
import models.archs.arch_util as arch_util

class WAEN_S(nn.Module):
    def __init__(self, nframes=7, nf=128, RBs=40):
        super(WAEN_S, self).__init__()
        self.nf = nf
        self.center = nframes // 2

        ResidualBlock_noBN_f = functools.partial(arch_util.ResidualBlock_noBN, nf=nf)

        ### Discrete Wavelet Transform
        self.DWT = DWT()

        ### Wavelet Embedding
        self.wavelet_conv1 = nn.Conv2d(3 * 4, 3 * 4 * 4, 3, 1, 1, bias=True)
        self.wavelet_pixel_shuffle = nn.PixelShuffle(2)
        self.wavelet_conv2 = nn.Conv2d(3 * 4, 3 * 4, 3, 1, 1, bias=True)

        ### TSA Fusion
        self.tsa_fusion = TSA_Fusion(nf=nf, nframes=nframes, center=self.center)

        ### MISR
        self.misr_conv1 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
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

        ### Wavelet Embedding
        x_dwt = self.DWT(x)  # B, N, 3*4, H // 2, W // 2
        x_dwt = self.lrelu(self.wavelet_conv1(x_dwt.view(-1, 3 * 4, H // 2, W // 2)))  # B, N, 3*4*4, H // 2, W // 2
        x_dwt_ch1 = x_dwt[:, 0:12, :, :]  # B, N, 3*4, H // 2, W // 2
        x_dwt_ch2 = x_dwt[:, 12:24, :, :]  # B, N, 3*4, H // 2, W // 2
        x_dwt_ch3 = x_dwt[:, 24:36, :, :]  # B, N, 3*4, H // 2, W // 2
        x_dwt_ch4 = x_dwt[:, 36:48, :, :]  # B, N, 3*4, H // 2, W // 2
        x_dwt_ch1 = self.wavelet_pixel_shuffle(x_dwt_ch1)  # B, N, 3, H, W
        x_dwt_ch2 = self.wavelet_pixel_shuffle(x_dwt_ch2)  # B, N, 3, H, W
        x_dwt_ch3 = self.wavelet_pixel_shuffle(x_dwt_ch3)  # B, N, 3, H, W
        x_dwt_ch4 = self.wavelet_pixel_shuffle(x_dwt_ch4)  # B, N, 3, H, W
        x_dwt = self.lrelu(
            self.wavelet_conv2(torch.cat((x_dwt_ch1, x_dwt_ch2, x_dwt_ch3, x_dwt_ch4), 1)))  # B, N, 3*4, H, W

        ### Concat X and Wavelet (Final Embedding)
        x_ll_emb = torch.cat((x, x_dwt[:, 0:3, :, :].view(B, N, -1, H, W)), 2)  # B, N, 6, H, W
        x_hl_emb = torch.cat((x, x_dwt[:, 3:6, :, :].view(B, N, -1, H, W)), 2)  # B, N, 6, H, W
        x_lh_emb = torch.cat((x, x_dwt[:, 6:9, :, :].view(B, N, -1, H, W)), 2)  # B, N, 6, H, W
        x_hh_emb = torch.cat((x, x_dwt[:, 9:12, :, :].view(B, N, -1, H, W)), 2)  # B, N, 6, H, W

        ### TSA Fusion
        fea = self.tsa_fusion(x_ll_emb, x_hl_emb, x_lh_emb, x_hh_emb)

        ### MISR
        MISR_fea = self.lrelu(self.misr_conv1(fea))
        MISR_fea = self.misr_feature_extraction(MISR_fea)
        MISR_fea = self.lrelu(self.misr_conv2(MISR_fea))
        MISR_fea = self.lrelu(self.misr_pixel_shuffle(self.misr_upconv1(MISR_fea)))
        MISR_fea = self.lrelu(self.misr_pixel_shuffle(self.misr_upconv2(MISR_fea)))
        MISR_fea = self.lrelu(self.misr_HRconv(MISR_fea))
        MISR_fea = self.misr_conv_last(MISR_fea)
        base = F.interpolate(x_center, scale_factor=4, mode='bilinear', align_corners=False)
        MISR_fea += base

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
        self.tAtt_1 = nn.Conv2d(6, nf // 4, 3, 1, 1, bias=True)
        self.tAtt_2 = nn.Conv2d(6, nf // 4, 3, 1, 1, bias=True)

        # fusion conv: using 1x1 to save parameters and computation
        self.fea_fusion = nn.Conv2d(nframes * 6 * 4, nf, 1, 1, bias=True)

        # spatial attention (after fusion conv)
        self.sAtt_1 = nn.Conv2d(nframes * 6 * 4, nf, 1, 1, bias=True)
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

    def forward(self, aligned_fea_ll, aligned_fea_hl, aligned_fea_lh, aligned_fea_hh):
        B, N, C, H, W = aligned_fea_ll.size()  # N video frames | C 6

        #### temporal attention (LL)
        emb_ref_ll = self.tAtt_2(aligned_fea_ll[:, self.center, :, :, :].clone())
        emb_ll = self.tAtt_1(aligned_fea_ll.view(-1, C, H, W)).view(B, N, -1, H, W)  # [B, N, C(nf // 2), H, W]
        cor_l_ll = []
        for i in range(N):
            emb_nbr_ll = emb_ll[:, i, :, :, :]
            cor_tmp_ll = torch.sum(emb_nbr_ll * emb_ref_ll, 1).unsqueeze(1)  # B, 1, H, W
            cor_l_ll.append(cor_tmp_ll)
        cor_prob_ll = torch.sigmoid(torch.cat(cor_l_ll, dim=1))  # B, N, H, W
        cor_prob_ll = cor_prob_ll.unsqueeze(2).repeat(1, 1, C, 1, 1).view(B, -1, H, W)
        aligned_fea_ll = aligned_fea_ll.view(B, -1, H, W) * cor_prob_ll

        #### temporal attention (HL)
        emb_ref_hl = self.tAtt_2(aligned_fea_hl[:, self.center, :, :, :].clone())
        emb_hl = self.tAtt_1(aligned_fea_hl.view(-1, C, H, W)).view(B, N, -1, H, W)  # [B, N, C(nf // 2), H, W]
        cor_l_hl = []
        for i in range(N):
            emb_nbr_hl = emb_hl[:, i, :, :, :]
            cor_tmp_hl = torch.sum(emb_nbr_hl * emb_ref_hl, 1).unsqueeze(1)  # B, 1, H, W
            cor_l_hl.append(cor_tmp_hl)
        cor_prob_hl = torch.sigmoid(torch.cat(cor_l_hl, dim=1))  # B, N, H, W
        cor_prob_hl = cor_prob_hl.unsqueeze(2).repeat(1, 1, C, 1, 1).view(B, -1, H, W)
        aligned_fea_hl = aligned_fea_hl.view(B, -1, H, W) * cor_prob_hl

        #### temporal attention (LH)
        emb_ref_lh = self.tAtt_2(aligned_fea_lh[:, self.center, :, :, :].clone())
        emb_lh = self.tAtt_1(aligned_fea_lh.view(-1, C, H, W)).view(B, N, -1, H, W)  # [B, N, C(nf // 2), H, W]
        cor_l_lh = []
        for i in range(N):
            emb_nbr_lh = emb_lh[:, i, :, :, :]
            cor_tmp_lh = torch.sum(emb_nbr_lh * emb_ref_lh, 1).unsqueeze(1)  # B, 1, H, W
            cor_l_lh.append(cor_tmp_lh)
        cor_prob_lh = torch.sigmoid(torch.cat(cor_l_lh, dim=1))  # B, N, H, W
        cor_prob_lh = cor_prob_lh.unsqueeze(2).repeat(1, 1, C, 1, 1).view(B, -1, H, W)
        aligned_fea_lh = aligned_fea_lh.view(B, -1, H, W) * cor_prob_lh

        #### temporal attention (HH)
        emb_ref_hh = self.tAtt_2(aligned_fea_hh[:, self.center, :, :, :].clone())
        emb_hh = self.tAtt_1(aligned_fea_hh.view(-1, C, H, W)).view(B, N, -1, H, W)  # [B, N, C(nf // 2), H, W]
        cor_l_hh = []
        for i in range(N):
            emb_nbr_hh = emb_hh[:, i, :, :, :]
            cor_tmp_hh = torch.sum(emb_nbr_hh * emb_ref_hh, 1).unsqueeze(1)  # B, 1, H, W
            cor_l_hh.append(cor_tmp_hh)
        cor_prob_hh = torch.sigmoid(torch.cat(cor_l_hh, dim=1))  # B, N, H, W
        cor_prob_hh = cor_prob_hh.unsqueeze(2).repeat(1, 1, C, 1, 1).view(B, -1, H, W)
        aligned_fea_hh = aligned_fea_hh.view(B, -1, H, W) * cor_prob_hh

        #### fusion
        fea = self.lrelu(self.fea_fusion(torch.cat([aligned_fea_ll, aligned_fea_hl, aligned_fea_lh, aligned_fea_hh], dim=1)))

        #### spatial attention
        att = self.lrelu(self.sAtt_1(torch.cat([aligned_fea_ll, aligned_fea_hl, aligned_fea_lh, aligned_fea_hh], dim=1)))
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