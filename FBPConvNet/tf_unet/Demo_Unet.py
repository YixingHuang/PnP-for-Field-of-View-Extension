#!/usr/bin/env python
# _*_ coding:utf-8 _*_
import unet, util, image_util
import image_gen
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

# preparing data loading
data_provider = image_util.ImageDataProvider("F:/LowDoseChallenge/experiment16_parallel120_256/*.tif")

# setup & training
#net = unet.Unet(layers=3, features_root=6, channels=1, n_class=2)
net = unet.Unet(layers=5, features_root=64, channels=1, n_class=1, cost="l2_loss",cost_kwargs=dict(regularizer=0.0001))
trainer = unet.Trainer(net, optimizer="adam")
#path = trainer.train(data_provider, "C:/Users/z003hpfz/tf_unet/tf_unet/Results", training_iters=32, epochs=50, restore=False)
path = "F:/LowDoseChallenge/experiment19_fan120_256/prediction150/Results/model.cpkt"
data_provider_evaluate = image_util.ImageDataProvider("D:/Tasks/DeepLearning/experiment19_fan120_256_WithoutTumors/*tif")
data, label = data_provider_evaluate(1)
# verification
prediction = net.predict(path, data)
unet.error_rate(prediction, util.crop_to_shape(label, prediction.shape))
sess = tf.Session()
pmax = tf.reduce_max(prediction)
print(sess.run(pmax))
print(sess.run(tf.reduce_min(prediction)))
print(sess.run(tf.reduce_max(label)))
print(sess.run(tf.reduce_min(label)))
img = util.combine_img_prediction(data, label, prediction)
util.save_image(img, "prediction.jpg")
















