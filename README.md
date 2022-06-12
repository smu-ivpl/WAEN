# Wavelet Attention Embedding Networks for Video Super-Resolution
##### Young-Ju Choi, Young-Woon Lee, and Byung-Gyu Kim
##### Intelligent Vision Processing Lab. (IVPL), Sookmyung Women's University, Seoul, Republic of Korea
----------------------------
##### This repository is the official PyTorch implementation of the paper published in _2020 25th International Conference on Pattern Recognition (ICPR)_.
[![paper](https://img.shields.io/badge/paper-PDF-<COLOR>.svg)](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9412623)

----------------------------
## Summary of paper
#### Abstract
> Recently, Video super-resolution (VSR) has become more crucial as the resolution of display has been grown. The majority of deep learning-based VSR methods combine the convolutional neural networks (CNN) with motion compensation or alignment module to estimate a high-resolution (HR) frame from low-resolution (LR) frames. However, most of the previous methods deal with the spatial features equally and may result in the misaligned temporal features by the pixel-based motion compensation and alignment module. It can lead to the damaging effect on the accuracy of the estimated HR feature. In this paper, we propose a wavelet attention embedding network (WAEN), including wavelet embedding network (WENet) and attention embedding network (AENet), to fully exploit the spatio-temporal informative features. The WENet is operated as a spatial feature extractor of individual low and high-frequency information based on 2-D Haar discrete wavelet transform. The meaningful temporal feature is extracted in the AENet through utilizing the weighted attention map between frames. Experimental results verify that the proposed method achieves superior performance compared with state-of-the-art methods.
>

#### Network Architecture
<p align="center">
  <img width="800" src="fig1.png">
</p>

<p align="center">
  <img width="800" src="fig2.png">
</p>

<p align="center">
  <img width="800" src="fig3.png">
</p>

#### Experimental Results
<p align="center">
  <img width="800" src="table.png">
</p>

<p align="center">
  <img width="800" src="fig5.png">
</p>

----------------------------
## Getting Started
#### Dependencies and Installation
#### Dataset Preparation
#### Model Zoo

----------------------------
## Training

----------------------------
## Testing

----------------------------
## Citation
    @inproceedings{choi2021wavelet,
      title={Wavelet attention embedding networks for video super-resolution},
      author={Choi, Young-Ju and Lee, Young-Woon and Kim, Byung-Gyu},
      booktitle={2020 25th International Conference on Pattern Recognition (ICPR)},
      pages={7314--7320},
      year={2021},
      organization={IEEE}
    }
----------------------------
## Acknowledgement
The codes are heavily based on [EDVR](https://github.com/xinntao/EDVR). Thanks for their awesome works.


