import tensorflow as tf
import numpy as np
import util
from tifffile import imsave
from layers import (weight_variable, weight_variable_devonc, bias_variable,
                            conv2d, deconv2d, max_pool, avg_unpool, crop_and_concat, pixel_wise_softmax_2,
                            cross_entropy)
#import tifffile as tif

sess = tf.Session()

in_channels = 15 # 3 for RGB, 32, 64, 128, ...
out_channels = 15 # 128, 256, ...
ones_3d = np.ones((256,256,in_channels)) # input is 3d, in_channels = 32
for i in range(256):
    for j in range(256):
        for k in range(in_channels):
            ones_3d[ i, j, k] = 1
temp0 = util.fixOutputChannelDiemension(ones_3d)
#imsave("out.tif", output_array.astype(np.float32))
imsave("ones_3d.tif", temp0.astype(np.float32))
#ones_3d = tif.imread('C:/Users/liuling/Desktop/Master Thesis/Data/single_256/19_backup/final/data000.tif')
# filter must have 3d-shpae x number of filters = 4D
#weight_4d = tf.truncated_normal([3, 3, 15, 64], stddev=0.1)
stddev = np.sqrt(2 / (3 ** 2 * out_channels))
weight_4d = tf.truncated_normal([3, 3, in_channels, out_channels], stddev)

#weight_4d = tf.truncated_normal([1, 1, 15, 64], stddev=0.1)
flag = tf.InteractiveSession().run(weight_4d).astype(np.float32)
for i in range(3):
    for j in range(3):
        for k in range(in_channels):
            for l in range(out_channels):
                flag[i,j,k,l]=1
strides_2d = [1, 1, 1, 1]

in_3d = tf.constant(ones_3d, dtype=tf.float32)
filter_4d = tf.constant(flag, dtype=tf.float32)

in_width = int(in_3d.shape[0])
in_height = int(in_3d.shape[1])

filter_width = int(filter_4d.shape[0])
filter_height = int(filter_4d.shape[1])

input_3d   = tf.reshape(in_3d, [1, in_height, in_width, in_channels])
kernel_4d = tf.reshape(filter_4d, [filter_height, filter_width, in_channels, out_channels])

#output stacked shape is 3D = 2D x N matrix
output_3d = tf.nn.conv2d(input_3d, kernel_4d, strides=strides_2d, padding='SAME')

diff = output_3d - ones_3d
loss = tf.nn.l2_loss(diff)
output_array = tf.InteractiveSession().run(output_3d)
print(output_array.shape)
output_array = output_array.reshape(256,256,out_channels)
temp = util.fixOutputChannelDiemension(output_array)
#imsave("out.tif", output_array.astype(np.float32))
imsave("out.tif", temp.astype(np.float32))
#lossValue =  tf.InteractiveSession().run(loss)
#print(lossValue)
t1 = tf.constant([[[[1.0, 1., 1.], [2., 2., 2.]], [[3., 3., 3.], [4., 4., 4.]]]])
tf.shape(t1)  # [2, 2, 3]
t2 = tf.constant([[[[4., 4., 4.], [4., 4., 4.]], [[4., 4., 4.], [4., 4., 4.]]]])
t3 = t1-t2
loss2 = tf.nn.l2_loss(t3)
lossValue =  tf.InteractiveSession().run(loss2)
print(lossValue)
batch_mean, batch_var = tf.nn.moments(t1, [0, 1, 2])
print(tf.InteractiveSession().run(batch_mean))
print(tf.InteractiveSession().run(batch_var))
#tif.imsave("output.tif", output_array.reshape(64, 256, 256).astype(np.float32))
#print (sess.run(output_3d))










