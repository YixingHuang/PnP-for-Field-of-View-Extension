# PnP-for-Field-of-View-Extension

This repository is the deep learning implementation for the paper "Data Extrapolation from Learned Prior Images for Truncation Correction in Computed Tomography" (In Updating)

In this paper, a plug-and-play (PnP) method is proposed for truncation correction in computed tomography (CT). Truncation corrections means cupping artifact reduction inside the field-of-view (FOV) and anatomical structure restoration outside the FOV (FOV extension). The PnP method mainly consists of three steps: artifact reduction using deep learning,
data extrapolation from learned prior images, and image reconstruction from extrapolated data. In the first and the last steps, various deep learning methods and conventional image reconstruction methods can be plugged in, respectively.


![pipeline](https://github.com/YixingHuang/PnP-for-Field-of-View-Extension/blob/main/pipeline.png)


The codes in this repository contains the [FBPConvNet](https://ieeexplore.ieee.org/document/7949028) and [Pix2pixGAN](https://arxiv.org/abs/1611.07004) for **artifact reduction using deep learning**.

1. The FBPConvNet is fundamentally the [U-Net](https://arxiv.org/abs/1505.04597) architecture. Our implementation is modified from the [implementation of Jakeret et al.](https://github.com/jakeret/tf_unet), which uses Tensorflow 1. The updated version with Tensorflow 2 can be found [here](https://github.com/jakeret/unet).
