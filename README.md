# PnP-for-Field-of-View-Extension

This repository is the deep learning implementation for the paper "Data Extrapolation from Learned Prior Images for Truncation Correction in Computed Tomography" (**In Updating**)

In this paper, a plug-and-play (PnP) method is proposed for truncation correction in computed tomography (CT). Truncation corrections means cupping artifact reduction inside the field-of-view (FOV) and anatomical structure restoration outside the FOV (FOV extension). The PnP method mainly consists of three steps: artifact reduction using deep learning,
data extrapolation from learned prior images, and image reconstruction from extrapolated data. In the first and the last steps, various deep learning methods and conventional image reconstruction methods can be plugged in, respectively.


![pipeline](https://github.com/YixingHuang/PnP-for-Field-of-View-Extension/blob/main/DescriptionImages/pipeline.png)

At the stage of artifact reduction using deep learning, a deep learning model is trained to reduce truncation artifacts like the following:
![example](https://github.com/YixingHuang/PnP-for-Field-of-View-Extension/blob/main/DescriptionImages/Example.png)

Please make sure that the error inside the FOV is small inside the artifact image; otherwise there might be problems in constructing the training and test datasets.

The codes in this repository contains the [FBPConvNet](https://ieeexplore.ieee.org/document/7949028) and [Pix2pixGAN](https://arxiv.org/abs/1611.07004) for **artifact reduction using deep learning**.

## FBPConvNet
The FBPConvNet is fundamentally the [U-Net](https://arxiv.org/abs/1505.04597) architecture. Our implementation is modified from the [implementation of Jakeret et al.](https://github.com/jakeret/tf_unet), which uses Tensorflow 1. The updated version with Tensorflow 2 can be found [here](https://github.com/jakeret/unet). The main modification includes the following points: a. Change the last layer for a regressional neural network; b. Change the deconvolutional layer to conv + bilinear up-sampling to avoid checkerboard artifacts; c. Use l2 loss function for training; d. Use tif files as input and output files.

## Pix2pixGAN
The pix2pixGAN implementation is modified from this [source](https://github.com/affinelayer/pix2pix-tensorflow), which also uses Tensorflow 1. Our modification includes the following points: a. Using L2 loss instead of L1 loss; b. Read tif files as input and output as raw files to keep precision of intensity values; c. Change the U-Net to a standard one instead of the simplified one. The codes are in the "pix2pix": folder. More details about the installation and running can be found from the [source](https://github.com/affinelayer/pix2pix-tensorflow).

# Installation
Please check the requirements.txt in the FBPConvNet and Pix2pixGAN folders respectively for installation. Assuming Python 3.6 or 3.7, Tensorflow_GPU == 1.14.0.

# Data
The data for the simulation experiments are from the [AAPM Low Dose CT Grand Challenge](https://www.aapm.org/grandchallenge/lowdosect/#). Please contact the organizers to get the access to the data.

# Acknowledgement for reference repos
 - [U-Net](https://github.com/jakeret/tf_unet)
 - [Pix2pixGAN](https://github.com/affinelayer/pix2pix-tensorflow)
