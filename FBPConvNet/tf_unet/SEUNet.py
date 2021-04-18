# tf_unet is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# 
# tf_unet is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License
# along with tf_unet.  If not, see <http://www.gnu.org/licenses/>.


'''
Created on Jul 28, 2016

author: jakeret
'''
from __future__ import print_function, division, absolute_import, unicode_literals
#from vgg_perceptual_loss import compute_perceptual_loss
import os
import shutil
import numpy as np
from collections import OrderedDict
import logging
import tensorflow as tf
import util
import image_util
from tflearn.layers.conv import global_avg_pool
from tifffile import imsave
from layers import (weight_variable, weight_variable_devonc, bias_variable,
                            conv2d, deconv2d, max_pool, avg_unpool, crop_and_concat, pixel_wise_softmax_2,
                            cross_entropy)

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')

# Ling' modify: batch normalization
epsilon = 1e-3


def batch_norm_wrapper(inputs, is_training, decay=0.999):
    scale = tf.Variable(tf.ones([inputs.shape[-1]]))
    beta = tf.Variable(tf.zeros([inputs.shape[-1]]))
    pop_mean = tf.Variable(tf.zeros([inputs.shape[-1]]), trainable=False)
    pop_var = tf.Variable(tf.ones([inputs.shape[-1]]), trainable=False)

    if is_training is not None:
        batch_mean, batch_var = tf.nn.moments(inputs, [0, 1, 2])
        train_mean = tf.assign(pop_mean,
                               pop_mean * decay + batch_mean * (1 - decay))
        train_var = tf.assign(pop_var,
                              pop_var * decay + batch_var * (1 - decay))
        with tf.control_dependencies([train_mean, train_var]):
            return tf.nn.batch_normalization(inputs,
                                             batch_mean, batch_var, beta, scale, epsilon)
    else:
        return tf.nn.batch_normalization(inputs,
                                         pop_mean, pop_var, beta, scale, epsilon)

def createFBPUNet(x, keep_prob, channels, n_class, is_training = None, layers=5, features_root=64, filter_size=3,
                  pool_size=2, summaries=False):
    """
    Creates a new convolutional unet for the given parametrization.

    :param x: input tensor, shape [?,nx,ny,channels]
    :param keep_prob: dropout probability tensor
    :param channels: number of channels in the input image
    :param n_class: number of output labels
    """

    out, variables, offset = create_conv_net(x, keep_prob, channels, n_class, is_training, layers, features_root,
                                             filter_size,pool_size, summaries)
    # out2 = tf.subtract(x, out)
    # without subtraction, when the target image is the artifact image
    out2 = out

    return out2, variables, offset


def squeeze_excitation_layer(input_x, out_dim, ratio, layer_name):
    with tf.name_scope(layer_name) :


        squeeze = Global_Average_Pooling(input_x)

        excitation = Fully_connected(squeeze, units=out_dim / ratio, layer_name=layer_name+'_fully_connected1')
        excitation = Relu(excitation)
        excitation = Fully_connected(excitation, units=out_dim, layer_name=layer_name+'_fully_connected2')
        excitation = Sigmoid(excitation)

        excitation = tf.reshape(excitation, [-1,1,1,out_dim])
        scale = input_x * excitation

        return scale



def get_incoming_shape(incoming):
    """ Returns the incoming data shape """
    if isinstance(incoming, tf.Tensor):
        return incoming.get_shape().as_list()
    elif type(incoming) in [np.array, np.ndarray, list, tuple]:
        return np.shape(incoming)
    else:
        raise Exception("Invalid incoming layer.")

def Global_Average_Pooling(x):
    return global_avg_pool(x, name='Global_avg_pooling')

def Relu(x):
    return tf.nn.relu(x)


def Sigmoid(x) :
    return tf.nn.sigmoid(x)

def Fully_connected(x, units, layer_name='fully_connected') :
    with tf.name_scope(layer_name) :
        return tf.layers.dense(inputs=x, use_bias=False, units=units)

def create_conv_net(x, keep_prob, channels, n_class, is_training = None,layers=5, features_root=64, filter_size=3, pool_size=2, summaries=False):
    """
    Creates a new convolutional unet for the given parametrization.

    :param x: input tensor, shape [?,nx,ny,channels]
    :param keep_prob: dropout probability tensor
    :param channels: number of channels in the input image
    :param n_class: number of output labels
    :param layers: number of layers in the net
    :param features_root: number of features in the first layer
    :param filter_size: size of the convolution filter
    :param pool_size: size of the max pooling operation
    :param summaries: Flag if summaries should be created
    """

    logging.info(
        "Layers {layers}, features {features}, filter size {filter_size}x{filter_size}, pool size: {pool_size}x{pool_size}".format(
            layers=layers,
            features=features_root,
            filter_size=filter_size,
            pool_size=pool_size))
    # Placeholder for the input image
    nx = tf.shape(x)[1]
    ny = tf.shape(x)[2]
    x_image = tf.reshape(x, tf.stack([-1, nx, ny, channels]))
    in_node = x_image
    

    with tf.variable_scope("U_net"):
        weights = []
        biases = []
        convs = []
        pools = OrderedDict()
        deconv = OrderedDict()
    
      
        dw_h_convs = OrderedDict()
        up_h_convs = OrderedDict()

        in_size = 1000
        size = in_size
        reduction_ratio = 4
    
        for layer in range(0, layers):
            features = 2 ** layer * features_root
            stddev = np.sqrt(2 / (filter_size ** 2 * channels))
            if layer == 0:
                w1 = weight_variable([filter_size, filter_size, channels, features], stddev)
            else:
                w1 = weight_variable([filter_size, filter_size, features // 2, features], stddev)

            w2 = weight_variable([filter_size, filter_size, features, features], stddev)
            b1 = bias_variable([features])
            b2 = bias_variable([features])

            conv1 = conv2d(in_node, w1, keep_prob)
            tmp_h_conv_bn = batch_norm_wrapper(conv1, is_training)
            tmp_h_conv = tf.nn.relu(tmp_h_conv_bn + b1)

            tmp_h_conv_se = squeeze_excitation_layer(tmp_h_conv, features, reduction_ratio, 'squeeze_layer1_')
            conv2 = conv2d(tmp_h_conv_se, w2, keep_prob)
            tmp_h_conv_bn2 = batch_norm_wrapper(conv2, is_training)
            tmp_h_conv2 = tf.nn.relu(tmp_h_conv_bn2 + b2)
            dw_h_convs[layer] = squeeze_excitation_layer(tmp_h_conv2, features, reduction_ratio, 'squeeze_layer2_')
            weights.append((w1, w2))
            biases.append((b1, b2))
            convs.append((conv1, conv2))

        #size -= 4
            if layer < layers - 1:
                pools[layer] = max_pool(dw_h_convs[layer], pool_size)
                in_node = pools[layer]
                size /= 2

        in_node = dw_h_convs[layers - 1]

    # up layers
        for layer in range(layers - 2, -1, -1):
            features = 2 ** (layer + 1) * features_root
            stddev = np.sqrt(2 / (filter_size ** 2 * features))
            wd = weight_variable([pool_size, pool_size, features, features//2], stddev)
            bd = bias_variable([features // 2])
            shape_down = tf.shape(in_node)
            shape_up = [shape_down[1]*2, shape_down[2]*2]
            in_node_resize = tf.image.resize_images(in_node, shape_up, tf.image.ResizeMethod.NEAREST_NEIGHBOR)
            h_deconv = tf.nn.relu(conv2d(in_node_resize, wd, keep_prob) + bd)
            h_deconv_bn = batch_norm_wrapper(h_deconv, is_training)

            h_deconv_concat = crop_and_concat(dw_h_convs[layer], h_deconv_bn)
            deconv[layer] = h_deconv_concat

            w1 = weight_variable([filter_size, filter_size, features, features // 2], stddev)
            w2 = weight_variable([filter_size, filter_size, features // 2, features // 2], stddev)
            b1 = bias_variable([features // 2])
            b2 = bias_variable([features // 2])

            conv1 = conv2d(h_deconv_concat, w1, keep_prob)
            h_conv_bn = batch_norm_wrapper(conv1, is_training)
            h_conv_temp3 = tf.nn.relu(h_conv_bn + b1)
            h_conv_se = squeeze_excitation_layer(h_conv_temp3, (features // 2), reduction_ratio, 'right_squeeze_layer1_')
            conv2 = conv2d(h_conv_se, w2, keep_prob)
            h_conv_bn2 = batch_norm_wrapper(conv2, is_training)
            h_conv_temp4 = tf.nn.relu(h_conv_bn2 + b2)
            in_node = squeeze_excitation_layer(h_conv_temp4,  (features // 2),  reduction_ratio, 'right_squeeze_layer2_')


            up_h_convs[layer] = in_node

            weights.append((w1, w2))
            biases.append((b1, b2))
            convs.append((conv1, conv2))

            size *= 2
        #size -= 4

    # Output Map
        weight = weight_variable([1, 1, features_root, n_class], stddev)
        conv = conv2d(in_node, weight, tf.constant(1.0))
        output_map = conv
    #output_map = batch_norm_wrapper(tf.nn.relu(conv + bias), is_training)
        up_h_convs["out"] = output_map

    if summaries:
        for i, (c1, c2) in enumerate(convs):
            tf.summary.image('summary_conv_%02d_01_0' % i, get_image_summary(c1, idx=0))
            tf.summary.image('summary_conv_%02d_02_0' % i, get_image_summary(c2, idx=0))
            tf.summary.image('summary_conv_%02d_01_1' % i, get_image_summary(c1, idx=1))
            tf.summary.image('summary_conv_%02d_02_1' % i, get_image_summary(c2, idx=1))
            tf.summary.image('summary_conv_%02d_01_2' % i, get_image_summary(c1, idx=2))
            tf.summary.image('summary_conv_%02d_02_2' % i, get_image_summary(c2, idx=2))
        for i, (q1, q2) in enumerate(weights):
            tf.summary.image('summary_weight_%02d_01_0' % i, get_image_summary(q1, idx=0))
            tf.summary.image('summary_weight_%02d_02_0' % i, get_image_summary(q2, idx=0))
            tf.summary.image('summary_weight_%02d_01_1' % i, get_image_summary(q1, idx=1))
            tf.summary.image('summary_weight_%02d_02_1' % i, get_image_summary(q2, idx=1))
            tf.summary.image('summary_weight_%02d_01_2' % i, get_image_summary(q1, idx=2))
            tf.summary.image('summary_weight_%02d_02_2' % i, get_image_summary(q2, idx=2))
        for k in pools.keys():
            tf.summary.image('summary_pool_%02d' % k, get_image_summary(pools[k]))

        for k in deconv.keys():
            tf.summary.image('summary_deconv_concat_%02d' % k, get_image_summary(deconv[k]))

        for k in dw_h_convs.keys():
            tf.summary.histogram("dw_convolution_%02d" % k + '/activations', dw_h_convs[k])

        for k in up_h_convs.keys():
            tf.summary.histogram("up_convolution_%s" % k + '/activations', up_h_convs[k])

    variables = []

    for w1, w2 in weights:
        variables.append(w1)
        variables.append(w2)

    for b1, b2 in biases:
        variables.append(b1)
        variables.append(b2)

    return output_map, variables, int(in_size - size)


class FBPUNet(object):
    """
    A unet implementation

    :param channels: (optional) number of channels in the input image
    :param n_class: (optional) number of output labels
    :param cost: (optional) name of the cost function. Default is 'cross_entropy'
    :param cost_kwargs: (optional) kwargs passed to the cost function. See Unet._get_cost for more options
    """

    def __init__(self, channels=3, n_class=1, cost="cross_entropy", cost_kwargs={}, **kwargs):
        tf.reset_default_graph()

        self.n_class = n_class
        self.summaries = kwargs.get("summaries", True)
        self.is_Training = tf.placeholder(tf.bool, shape = ())
        self.x = tf.placeholder("float", shape=[None, None, None, channels])
        self.y = tf.placeholder("float", shape=[None, None, None, n_class])
        self.keep_prob = tf.placeholder(tf.float32)  # dropout (keep probability)

        # logits: output map
        logits, self.variables, self.offset = createFBPUNet(self.x, self.keep_prob, channels, n_class, self.is_Training, **kwargs)

        self.cost = self._get_cost(logits, cost, cost_kwargs)

        self.gradients_node = tf.gradients(self.cost, self.variables)

        self.cross_entropy = tf.reduce_mean(cross_entropy(tf.reshape(self.y, [-1, n_class]),
                                                          tf.reshape(pixel_wise_softmax_2(logits), [-1, n_class])))

        self.predicter = logits
        self.rmse = tf.metrics.root_mean_squared_error(self.y, logits)
        self.correct_pred = tf.equal(tf.argmax(self.predicter, 3), tf.argmax(self.y, 3))
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_pred, tf.float32))

    def _get_cost(self, logits, cost_name, cost_kwargs):
        """
        Constructs the cost function, either cross_entropy, weighted cross_entropy or dice_coefficient.
        Optional arguments are:
        class_weights: weights for the different classes in case of multi-class imbalance
        regularizer: power of the L2 regularizers added to the loss function
        """

        flat_logits = tf.reshape(logits, [-1, self.n_class])
        flat_labels = tf.reshape(self.y, [-1, self.n_class])

        if cost_name == "cross_entropy":
            class_weights = cost_kwargs.pop("class_weights", None)

            if class_weights is not None:
                class_weights = tf.constant(np.array(class_weights, dtype=np.float32))

                weight_map = tf.multiply(flat_labels, class_weights)
                weight_map = tf.reduce_sum(weight_map, axis=1)

                loss_map = tf.nn.softmax_cross_entropy_with_logits(logits=flat_logits,
                                                                   labels=flat_labels)
                weighted_loss = tf.multiply(loss_map, weight_map)

                loss = tf.reduce_mean(weighted_loss)

            else:
                loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=flat_logits,
                                                                              labels=flat_labels))
        elif cost_name == "dice_coefficient":
            eps = 1e-5
            prediction = pixel_wise_softmax_2(logits)
            intersection = tf.reduce_sum(prediction * self.y)
            union = eps + tf.reduce_sum(prediction) + tf.reduce_sum(self.y)
            loss = -(2 * intersection / (union))
        elif cost_name == "l2_loss":
            diff = flat_logits - flat_labels
            loss = tf.nn.l2_loss(diff)
#        elif cost_name == "perceptual_loss":
#            logits_3d = tf.concat([logits, logits, logits], axis = 3)
#            label_3d = tf.concat([self.y, self.y, self.y], axis = 3)
#            loss = compute_perceptual_loss(logits_3d, label_3d)
        else:
            raise ValueError("Unknown cost function: " % cost_name)

        regularizer = cost_kwargs.pop("regularizer", None)

        if regularizer is not None:
            regularizers = sum([tf.nn.l2_loss(variable) for variable in self.variables])
            loss = loss + (regularizer * regularizers)
        return loss

    def predict(self, model_path, x_test):
        """
        Uses the model to create a prediction for the given data

        :param model_path: path to the model checkpoint to restore
        :param x_test: Data to predict on. Shape [n, nx, ny, channels]
        :returns prediction: The unet prediction Shape [n, px, py, labels] (px=nx-self.offset/2)
        """

        init = tf.global_variables_initializer()
        # Ling's modify: using tensorflow-GPU
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        # restrict GPU's space
        config.gpu_options.per_process_gpu_memory_fraction = 1

        with tf.Session(config = config) as sess:
            # Initialize variables
            sess.run(init)

            # Restore model weights from previously saved model
            self.restore(sess, model_path)

            y_dummy = np.empty((x_test.shape[0], x_test.shape[1], x_test.shape[2], self.n_class))
            prediction = sess.run(self.predicter, feed_dict={self.x: x_test, self.y: y_dummy, self.keep_prob: 1.})

        return prediction
       
    def predictAll(self, model_path, data_provider_evaluate, path_store, num):
        init = tf.global_variables_initializer()
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        # restrict GPU's space
        config.gpu_options.per_process_gpu_memory_fraction = 1

        with tf.Session(config = config) as sess:
            # Initialize variables
            sess.run(init)

            # Restore model weights from previously saved model
            self.restore(sess, model_path)
            for idx in range(num):
                x_test, label = data_provider_evaluate(1)
                y_dummy = np.empty((x_test.shape[0], x_test.shape[1], x_test.shape[2], self.n_class))
                prediction = sess.run(self.predicter, feed_dict={self.x: x_test, self.y: y_dummy, self.keep_prob: 1.})
                if len(prediction.shape) == 3:
                    out = prediction.reshape([-1,prediction.shape[2],prediction.shape[3]])
                    temp = util.fixOutputChannelDiemension(out)
                    imsave("final_prediction.tif", temp.astype(np.float32))
                else:
                    out = prediction.reshape(-1, prediction.shape[2])
                    imsave("%s%s_final_prediction.tif" %(path_store, idx), out.astype(np.float32))
        
        return 1
   
       
    def predictAllSingleSlice(self, model_path, pathTesting, path_store, num):
        init = tf.global_variables_initializer()
        # Ling's modify: using tensorflow-GPU
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        # restrict GPU's space
        config.gpu_options.per_process_gpu_memory_fraction = 1

        with tf.Session(config = config) as sess:
            # Initialize variables
            sess.run(init)

            # Restore model weights from previously saved model
            self.restore(sess, model_path)
            for idx in range(num):
                pathTestingSingleSlice= pathTesting +str(idx) + "/*.tif"
                data_provider_evaluate = image_util.ImageDataProvider(pathTestingSingleSlice)
                x_test, label = data_provider_evaluate(1)
                y_dummy = np.empty((x_test.shape[0], x_test.shape[1], x_test.shape[2], self.n_class))
                prediction = sess.run(self.predicter, feed_dict={self.x: x_test, self.y: y_dummy, self.keep_prob: 1.})
                if len(prediction.shape) == 3:
                    out = prediction.reshape([-1,prediction.shape[2],prediction.shape[3]])
                    temp = util.fixOutputChannelDiemension(out)
                    imsave("final_prediction.tif", temp.astype(np.float32))
                else:
                    out = prediction.reshape(-1, prediction.shape[2])
                    imsave("%s%s_final_prediction.tif" %(path_store, idx), out.astype(np.float32))
        
        return 1

 
    def save(self, sess, model_path):
        """
        Saves the current session to a checkpoint

        :param sess: current session
        :param model_path: path to file system location
        """

        saver = tf.train.Saver()
        save_path = saver.save(sess, model_path)
        return save_path

    def restore(self, sess, model_path):
        """
        Restores a session from a checkpoint

        :param sess: current session instance
        :param model_path: path to file system checkpoint location
        """

        saver = tf.train.Saver()
        saver.restore(sess, model_path)
        logging.info("Model restored from file: %s" % model_path)


class Trainer(object):
    """
    Trains a unet instance

    :param net: the unet instance to train
    :param batch_size: size of training batch
    :param optimizer: (optional) name of the optimizer to use (momentum or adam)
    :param opt_kwargs: (optional) kwargs passed to the learning rate (momentum opt) and to the optimizer
    """

#    prediction_path = "prediction"
    verification_batch_size = 25

    def __init__(self, net, prediction_path="prediction", batch_size=4, optimizer="momentum", opt_kwargs={}):
        self.net = net
        self.prediction_path = prediction_path
        self.batch_size = batch_size
        self.optimizer = optimizer
        self.opt_kwargs = opt_kwargs

    def _get_optimizer(self, training_iters, global_step):
        if self.optimizer == "momentum":
            learning_rate = self.opt_kwargs.pop("learning_rate", 0.2)
            decay_rate = self.opt_kwargs.pop("decay_rate", 0.95)
            momentum = self.opt_kwargs.pop("momentum", 0.2)

            self.learning_rate_node = tf.train.exponential_decay(learning_rate=learning_rate,
                                                                 global_step=global_step,
                                                                 decay_steps=training_iters,
                                                                 decay_rate=decay_rate,
                                                                 staircase=True)

            optimizer = tf.train.MomentumOptimizer(learning_rate=self.learning_rate_node, momentum=momentum,
                                                   **self.opt_kwargs).minimize(self.net.cost,
                                                                               global_step=global_step)
        elif self.optimizer == "adam":
            learning_rate = self.opt_kwargs.pop("learning_rate", 0.001)
            decay_rate = self.opt_kwargs.pop("decay_rate", 0.98)
            t_vars = tf.trainable_variables()
            train_vars = [var for var in t_vars if var.name.startswith("U_net")]
            self.learning_rate_node = tf.Variable(learning_rate)
            optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate_node,
                                               **self.opt_kwargs).minimize(self.net.cost,
                                                                           global_step=global_step, var_list = train_vars)

        return optimizer

    def _initialize(self, training_iters, output_path, restore):
        global_step = tf.Variable(0)

        self.norm_gradients_node = tf.Variable(tf.constant(0.0, shape=[len(self.net.gradients_node)]))

        if self.net.summaries:
            tf.summary.histogram('norm_grads', self.norm_gradients_node)

        tf.summary.scalar('loss', self.net.cost)
        tf.summary.scalar('cross_entropy', self.net.cross_entropy)
        tf.summary.scalar('accuracy', self.net.accuracy)
        self.global_step = global_step
        self.optimizer = self._get_optimizer(training_iters, global_step)
        tf.summary.scalar('learning_rate', self.learning_rate_node)
                     
        self.summary_op = tf.summary.merge_all()
        init = tf.global_variables_initializer()

        prediction_path = os.path.abspath(self.prediction_path)
        output_path = os.path.abspath(output_path)

        if not restore:
            logging.info("Removing '{:}'".format(prediction_path))
            shutil.rmtree(prediction_path, ignore_errors=True)
            logging.info("Removing '{:}'".format(output_path))
            shutil.rmtree(output_path, ignore_errors=True)

        if not os.path.exists(prediction_path):
            logging.info("Allocating '{:}'".format(prediction_path))
            os.makedirs(prediction_path)

        if not os.path.exists(output_path):
            logging.info("Allocating '{:}'".format(output_path))
            os.makedirs(output_path)

        return init

    def train(self, data_provider, data_provider_validation, output_path, training_iters=10, epochs=100, dropout=0.75, display_step=1,
              restore=False, write_graph=False):
        """
        Lauches the training process

        :param data_provider: callable returning training and verification data
        :param output_path: path where to store checkpoints
        :param training_iters: number of training mini batch iteration
        :param epochs: number of epochs
        :param dropout: dropout probability
        :param display_step: number of steps till outputting stats
        :param restore: Flag if previous model should be restored
        :param write_graph: Flag if the computation graph should be written as protobuf file to the output path
        """
        save_path = os.path.join(output_path, "model.cpkt")
        if epochs == 0:
            return save_path
        
        #change learning rate

        init = self._initialize(training_iters, output_path, restore)

        with tf.Session() as sess:
            if write_graph:
                tf.train.write_graph(sess.graph_def, output_path, "graph.pb", False)

            sess.run(init)
            myinit = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
            sess.run(myinit)
            # Ling's modify
            #tf.initialize_all_variables().run()

            if restore:
                ckpt = tf.train.get_checkpoint_state(output_path)
                if ckpt and ckpt.model_checkpoint_path:
                    self.net.restore(sess, ckpt.model_checkpoint_path)
 

            test_x, test_y = data_provider(self.verification_batch_size)
            test_x_vali, test_y_vali =data_provider_validation(self.verification_batch_size)
            pred_shape = self.store_prediction(sess, test_x, test_y,test_x_vali, test_y_vali, "_init")

            summary_writer = tf.summary.FileWriter(output_path, graph=sess.graph)
            logging.info("Start optimization")

            avg_gradients = None
            for epoch in range(epochs):
                if epoch == 200:
                    self.opt_kwargs.pop("learning_rate", 0.0001)
                if epoch == 400:
                    self.opt_kwargs.pop("learning_rate", 0.00001)
       
                total_loss = 0
                for step in range((epoch * training_iters), ((epoch + 1) * training_iters)):
                    batch_x, batch_y = data_provider(self.batch_size)

                
                    _, loss, lr, gradients = sess.run(
                                                          (self.optimizer, self.net.cost, self.learning_rate_node, self.net.gradients_node),
                                                          feed_dict={self.net.x: batch_x,
                                                                     self.net.y: util.crop_to_shape(batch_y, pred_shape),
                                       self.net.keep_prob: dropout,
                                       self.net.is_Training: True})
                   
                    if avg_gradients is None:
                        avg_gradients = [np.zeros_like(gradient) for gradient in gradients]
                    for i in range(len(gradients)):
                        avg_gradients[i] = (avg_gradients[i] * (1.0 - (1.0 / (step + 1)))) + (gradients[i] / (step + 1))

                    norm_gradients = [np.linalg.norm(gradient) for gradient in avg_gradients]


                    if step % display_step == 0:
                        self.output_minibatch_stats(sess, summary_writer, step, batch_x,
                                                    util.crop_to_shape(batch_y, pred_shape))

                    total_loss += loss

                self.output_epoch_stats(epoch, total_loss, training_iters, lr)
                self.store_prediction(sess, test_x, test_y, test_x_vali, test_y_vali, "epoch_%s" % epoch)

                save_path = self.net.save(sess, save_path)
            logging.info("Optimization Finished!")

            return save_path

    def store_prediction(self, sess, batch_x, batch_y, batch_x_vali, batch_y_vali, name):
        prediction = sess.run(self.net.predicter, feed_dict={self.net.x: batch_x,
                                                             self.net.y: batch_y,
                                                             self.net.keep_prob: 1.})

        pred_shape = prediction.shape

        loss = sess.run(self.net.cost, feed_dict={self.net.x: batch_x,
                                                  self.net.y: util.crop_to_shape(batch_y, pred_shape),
                                                  self.net.keep_prob: 1.})


        rmse_train = sess.run(self.net.rmse,feed_dict={self.net.x: batch_x,
                                                  self.net.y: util.crop_to_shape(batch_y, pred_shape),
                                                  self.net.keep_prob: 1.})

        rmse_vali = sess.run(self.net.rmse,feed_dict={self.net.x: batch_x_vali,
                                                            self.net.y: batch_y_vali,
                                                            self.net.keep_prob: 1.})

        logging.info("training error = {:}, validation error = {:}, loss= {:.5f}".format(rmse_train[1], rmse_vali[1], loss))
        print(rmse_train[1], ' ', rmse_vali[1])
        img = util.combine_img_prediction(batch_x, batch_y, prediction)
        util.save_image(img, "%s/%s.jpg" % (self.prediction_path, name))

        return pred_shape

    def output_epoch_stats(self, epoch, total_loss, training_iters, lr):
        logging.info(
            "Epoch {:}, Average loss: {:.4f}, learning rate: {:.12f}".format(epoch, (total_loss / training_iters), lr))

    def output_minibatch_stats(self, sess, summary_writer, step, batch_x, batch_y):
        # Calculate batch loss and accuracy
        summary_str, loss, acc, predictions = sess.run([self.summary_op,
                                                        self.net.cost,
                                                        self.net.accuracy,
                                                        self.net.predicter],
                                                       feed_dict={self.net.x: batch_x,
                                                                  self.net.y: batch_y,
                                                                  self.net.keep_prob: 1.})
        summary_writer.add_summary(summary_str, step)
        summary_writer.flush()
#       rmse_train2 = sess.run(self.net.rmse, feed_dict={self.net.x: batch_x,
#                                                        self.net.y: batch_y,
#                                                        self.net.keep_prob: 1.})
        logging.info(
            "Iter {:}, Minibatch Loss= {:.4f}".format(step, loss))


def error_rate(predictions, labels):
    """
    Return the error rate based on dense predictions and 1-hot labels.
    """

    return 100.0 - (
        100.0 *
        np.sum(np.argmax(predictions, 3) == np.argmax(labels, 3)) /
        (predictions.shape[0] * predictions.shape[1] * predictions.shape[2]))

def rmse(predictions, labels):
    """

    :param predictions:
    :param labels:
    :return: rmse
    """

    return tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(predictions, labels))))

def get_image_summary(img, idx=0):
    """
    Make an image summary for 4d tensor image with index idx
    """

    V = tf.slice(img, (0, 0, 0, idx), (1, -1, -1, 1))
    img_w = tf.shape(img)[1]
    img_h = tf.shape(img)[2]
    V = tf.reshape(V, tf.stack((img_w, img_h, 1)))
    V = tf.transpose(V, (2, 0, 1))
    V = tf.reshape(V, tf.stack((-1, img_w, img_h, 1)))
    return V
