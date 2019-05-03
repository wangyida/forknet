#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  3 17:41:29 2019

@author: shun-chengwu
"""

import argparse
from six import text_type as _text_type
import tensorflow as tf
from tensorflow.contrib.slim.python.slim.nets import vgg
from tensorflow.contrib.slim.python.slim.nets import inception
from tensorflow.contrib.slim.python.slim.nets import resnet_v1
from tensorflow.contrib.slim.python.slim.nets import resnet_v2
from mmdnn.conversion.examples.imagenet_test import TestKit
import tensorflow.contrib.slim as slim

from tensorflow.contrib import layers
from tensorflow.contrib.framework.python.ops import arg_scope
from tensorflow.contrib.layers.python.layers import layers as layers_lib
from tensorflow.contrib.layers.python.layers import regularizers
from tensorflow.contrib.layers.python.layers import utils
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import variable_scope
from model import depvox_gan

def depvox_arg_scope(weight_decay=0.0005):
    """Defines the VGG arg scope.

    Args:
    weight_decay: The l2 regularization coefficient.

    Returns:
    An arg_scope.
    """
    with arg_scope(
        [layers.conv3d, layers_lib.fully_connected],
        activation_fn=nn_ops.relu,
        weights_regularizer=regularizers.l2_regularizer(weight_decay),
        biases_initializer=init_ops.zeros_initializer()):
        with arg_scope([layers.conv3d], padding='SAME') as arg_sc:
            return arg_sc

def _main():
    parser = argparse.ArgumentParser()

    """
    parser.add_argument('-n', '--network', type=_text_type, help='Model Type', required=True,
        choices = input_layer_map.keys())
    """

    parser.add_argument('-i', '--image',
        type=_text_type, help='Test Image Path')

    parser.add_argument('-ckpt', '--checkpoint',
        type=_text_type, help='Tensorflow Checkpoint file name', required=True)

    args = parser.parse_args()


    with slim.arg_scope(depvox_arg_scope()):
        # data_input = input_layer_map[args.network]()
        kernel = [[3, 3, 3, 3, 3], [3, 3, 3, 3, 3], [3, 3, 3, 3, 3]]
        stride = [1, 2, 2, 2, 1]
        dilations = [1, 1, 1, 1, 1]
        depvox_gan_model = depvox_gan(
            batch_size=8,
            vox_shape=[80, 48, 80, 12],
            part_shape=[80, 48, 80, 1],
            dim_z=16,
            dim=[512, 256, 128, 32, 12],
            start_vox_size=[5, 3, 5],
            kernel=kernel,
            stride=stride,
            dilations=dilations,
            discriminative=True,
            is_train=True)
        Z_tf, z_part_enc_tf, full_tf, full_gen_tf, full_gen_decode_tf,\
        recon_vae_loss_tf, recon_gen_loss_tf, gen_loss_tf, discrim_loss_tf,\
        cost_pred_tf, cost_code_encode_tf, cost_code_discrim_tf, cost_encode_tf, cost_gen_tf, summary_tf,\
        part_tf, part_gen_tf, part_vae_decode_tf = depvox_gan_model.build_model()
        # logits, endpoints = .build_model() #your network
        # labels = tf.squeeze(logits)

    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        writer = tf.summary.FileWriter('./graphs', sess.graph)
        writer.close()
        sess.run(init)
        saver = tf.train.Saver()
        saver.restore(sess, args.checkpoint)
        save_path = saver.save(sess, "./shitnet.ckpt")
        import ipdb; ipdb.set_trace()
        print("Model saved in file: %s" % save_path)

        if args.image:
            import numpy as np
            func = TestKit.preprocess_func['tensorflow'][args.network]
            img = func(args.image)
            img = np.expand_dims(img, axis = 0)
            predict = sess.run(logits, feed_dict = {data_input : img})
            predict = np.squeeze(predict)
            top_indices = predict.argsort()[-5:][::-1]
            result = [(i, predict[i]) for i in top_indices]
            print (result)


if __name__=='__main__':
    _main()
