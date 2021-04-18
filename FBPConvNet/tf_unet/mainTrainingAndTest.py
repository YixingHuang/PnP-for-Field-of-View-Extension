# -*- coding: utf-8 -*-
"""
Created on Tue Dec 12 13:21:53 2017

@author: z003hpfz
"""
import unet, util, image_util,SEUNet
import image_gen
import numpy as np
import tensorflow as tf
# preparing data loading
#training data folder, the images should be named in pairs. For example, "data1.tif" and "data1_mask.tif"
data_provider = image_util.ImageDataProvider("YourFolder/tf_unet/TrainingData/*.tif")
data_provider_val = image_util.ImageDataProvider("YourFolder/tf_unet/ValidationData/*.tif")
# setup & training
#SE-U-Net parameters
net = SEUNet.FBPUNet(layers=5, features_root=64, channels=1, n_class=1, cost="l2_loss",cost_kwargs=dict(regularizer
                                                                                                         =0.001))
trainer = SEUNet.Trainer(net, prediction_path="prediction", optimizer="adam")
modelPath = trainer.train(data_provider, data_provider_val, "YourFolder/tf_unet/tf_unet/Results", training_iters=32, epochs=500, restore= False)
#modelPath = "YourFolder/tf_unet/tf_unet/Results/model.cpkt"
pathTesting = "YourFolder/tf_unet/"
pathStore = pathTesting + "evaluation/"
net.predictAllSingleSlice(modelPath, pathTesting, pathStore, 256)
