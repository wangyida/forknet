import numpy as np

from config import cfg
import tensorflow as tf
from util import *


def batchnormalize(X, eps=1e-5, g=None, b=None, batch_size=10):
    if X.get_shape().ndims == 5:
        if batch_size == 1:
            mean = 0
            std = 1 - eps
        else:
            mean = tf.reduce_mean(X, [0, 1, 2, 3])
            std = tf.reduce_mean(tf.square(X - mean), [0, 1, 2, 3])
        X = (X - mean) / tf.sqrt(std + eps)

        if g is not None and b is not None:
            g = tf.reshape(g, [1, 1, 1, 1, -1])
            b = tf.reshape(b, [1, 1, 1, 1, -1])
            X = X * g + b

    elif X.get_shape().ndims == 2:
        if batch_size == 1:
            mean = 0
            std = 1 - eps
        else:
            mean = tf.reduce_mean(X, 0)
            std = tf.reduce_mean(tf.square(X - mean), 0)
        X = (X - mean) / tf.sqrt(std + eps)  #std

        if g is not None and b is not None:
            g = tf.reshape(g, [1, -1])
            b = tf.reshape(b, [1, -1])
            X = X * g + b

    else:
        raise NotImplementedError

    return X


def layernormalize(X, eps=1e-5, g=None, b=None):
    if X.get_shape().ndims == 5:
        mean, std = tf.nn.moments(X, [1, 2, 3, 4], keep_dims=True)
        X = (X - mean) / tf.sqrt(std + eps)

        if g is not None and b is not None:
            X = X * g + b

    elif X.get_shape().ndims == 2:
        mean = tf.reduce_mean(X, 1)
        std = tf.reduce_mean(tf.square(X - mean), 1)
        X = (X - mean) / tf.sqrt(std + eps)  #std

        if g is not None and b is not None:
            X = X * g + b

    else:
        raise NotImplementedError

    return X


def lrelu(X, leak=0.2):
    return tf.maximum(X, leak * X)


def softmax(X, batch_size, vox_shape):
    c = tf.reduce_max(X, 4)
    c = tf.reshape(c,
                   [batch_size, vox_shape[0], vox_shape[1], vox_shape[2], 1])
    exp = tf.exp(tf.subtract(X, c))
    expsum = tf.reduce_sum(exp, 4)
    expsum = tf.reshape(
        expsum, [batch_size, vox_shape[0], vox_shape[1], vox_shape[2], 1])
    soft = tf.div(exp, expsum)

    return soft


class depvox_gan():
    def __init__(self,
                 batch_size=16,
                 vox_shape=[80, 48, 80, 12],
                 part_shape=[80, 48, 80, 1],
                 dim_z=16,
                 dim=[512, 256, 192, 64, 12],
                 start_vox_size=[5, 3, 5],
                 kernel=[[3, 3, 3, 3, 3], [3, 3, 3, 3, 3], [3, 3, 3, 3, 3]],
                 stride=[1, 2, 2, 2, 1],
                 dilations=[1, 1, 1, 1, 1],
                 dim_code=512,
                 generative=True,
                 discriminative=True,
                 is_train=True):

        self.batch_size = batch_size
        self.vox_shape = vox_shape
        self.part_shape = part_shape
        self.n_class = vox_shape[3]
        self.dim_z = dim_z
        self.dim_W1 = dim[0]
        self.dim_W2 = dim[1]
        self.dim_W3 = dim[2]
        self.dim_W4 = dim[3]
        self.dim_W5 = dim[4]
        self.start_vox_size = np.array(start_vox_size)
        self.kernel = np.array(kernel)
        self.kernel1 = self.kernel[:, 0]
        self.kernel2 = self.kernel[:, 1]
        self.kernel3 = self.kernel[:, 2]
        self.kernel4 = self.kernel[:, 3]
        self.kernel5 = self.kernel[:, 4]
        self.stride = stride
        self.dilations = dilations

        self.lamda_recons = cfg.LAMDA_RECONS
        self.lamda_gamma = cfg.LAMDA_GAMMA

        self.dim_code = dim_code
        self.generative = generative
        self.discriminative = discriminative
        self.is_train = is_train

        # parameters of generator y
        self.gen_y_W1 = tf.Variable(
            tf.random_normal([
                self.dim_z * self.start_vox_size[0] * self.start_vox_size[1] *
                self.start_vox_size[2], self.dim_W1 * self.start_vox_size[0] *
                self.start_vox_size[1] * self.start_vox_size[2]
            ],
                             stddev=0.02),
            name='gen_y_W1')
        self.gen_y_bn_g1 = tf.Variable(
            tf.random_normal([
                self.dim_W1 * self.start_vox_size[0] * self.start_vox_size[1] *
                self.start_vox_size[2]
            ],
                             mean=1.0,
                             stddev=0.02),
            name='gen_y_bn_g1')
        self.gen_y_bn_b1 = tf.Variable(
            tf.zeros([
                self.dim_W1 * self.start_vox_size[0] * self.start_vox_size[1] *
                self.start_vox_size[2]
            ]),
            name='gen_y_bn_b1')

        self.gen_y_W2 = tf.Variable(
            tf.random_normal([
                self.kernel2[0], self.kernel2[1], self.kernel2[2], self.dim_W2,
                self.dim_W1
            ],
                             stddev=0.02),
            name='gen_y_W2')
        self.gen_y_bn_g2 = tf.Variable(
            tf.random_normal([self.dim_W2], mean=1.0, stddev=0.02),
            name='gen_y_bn_g2')
        self.gen_y_bn_b2 = tf.Variable(
            tf.zeros([self.dim_W2]), name='gen_y_bn_b2')

        self.gen_y_W3 = tf.Variable(
            tf.random_normal([
                self.kernel3[0], self.kernel3[1], self.kernel3[2], self.dim_W3,
                self.dim_W2 * 2
            ],
                             stddev=0.02),
            name='gen_y_W3')
        self.gen_y_bn_g3 = tf.Variable(
            tf.random_normal([self.dim_W3], mean=1.0, stddev=0.02),
            name='gen_y_bn_g3')
        self.gen_y_bn_b3 = tf.Variable(
            tf.zeros([self.dim_W3]), name='gen_y_bn_b3')

        self.gen_y_W4 = tf.Variable(
            tf.random_normal([
                self.kernel4[0], self.kernel4[1], self.kernel4[2], self.dim_W4,
                self.dim_W3 * 2
            ],
                             stddev=0.02),
            name='gen_y_W4')
        self.gen_y_bn_g4 = tf.Variable(
            tf.random_normal([self.dim_W4], mean=1.0, stddev=0.02),
            name='gen_y_bn_g4')
        self.gen_y_bn_b4 = tf.Variable(
            tf.zeros([self.dim_W4]), name='gen_y_bn_b4')

        self.gen_y_W5 = tf.Variable(
            tf.random_normal([
                self.kernel5[0], self.kernel5[1], self.kernel5[2], self.dim_W5,
                self.dim_W4 * 2
            ],
                             stddev=0.02),
            name='gen_y_W5')
        self.gen_y_bn_g5 = tf.Variable(
            tf.random_normal([self.dim_W5], mean=1.0, stddev=0.02),
            name='gen_y_bn_g5')
        self.gen_y_bn_b5 = tf.Variable(
            tf.zeros([self.dim_W5]), name='gen_y_bn_b5')

        # parameters of encoder x
        self.encode_x_W1 = tf.Variable(
            tf.random_normal([
                self.kernel5[0], self.kernel5[1], self.kernel5[2],
                self.part_shape[-1], self.dim_W4
            ],
                             stddev=0.02),
            name='encode_x_W1')
        self.encode_x_bn_g1 = tf.Variable(
            tf.random_normal([self.dim_W4], mean=1.0, stddev=0.02),
            name='encode_x_bn_g1')
        self.encode_x_bn_b1 = tf.Variable(
            tf.zeros([self.dim_W4]), name='encode_x_bn_b1')

        self.encode_x_W2 = tf.Variable(
            tf.random_normal([
                self.kernel4[0], self.kernel4[1], self.kernel4[2],
                self.dim_W4 * 3, self.dim_W3
            ],
                             stddev=0.02),
            name='encode_x_W2')
        self.encode_x_bn_g2 = tf.Variable(
            tf.random_normal([self.dim_W3], mean=1.0, stddev=0.02),
            name='encode_x_bn_g2')
        self.encode_x_bn_b2 = tf.Variable(
            tf.zeros([self.dim_W3]), name='encode_x_bn_b2')

        self.encode_x_W3 = tf.Variable(
            tf.random_normal([
                self.kernel3[0], self.kernel3[1], self.kernel3[2], self.dim_W3,
                self.dim_W2
            ],
                             stddev=0.02),
            name='encode_x_W3')
        self.encode_x_bn_g3 = tf.Variable(
            tf.random_normal([self.dim_W2], mean=1.0, stddev=0.02),
            name='encode_x_bn_g3')
        self.encode_x_bn_b3 = tf.Variable(
            tf.zeros([self.dim_W2]), name='encode_x_bn_b3')

        self.discrim_y_W1 = tf.Variable(
            tf.random_normal([
                self.kernel5[0], self.kernel5[1], self.kernel5[2], self.dim_W5+self.part_shape[-1],
                self.dim_W4
            ],
                             stddev=0.02),
            name='discrim_y_vox_W1')
        self.discrim_y_bn_g1 = tf.Variable(
            tf.random_normal([1], mean=1.0, stddev=0.02),
            name='discrim_y_vox_bn_g1')
        self.discrim_y_bn_b1 = tf.Variable(
            tf.zeros([1]), name='discrim_y_vox_bn_b1')

        # parameters of discriminator
        self.discrim_y_W2 = tf.Variable(
            tf.random_normal([
                self.kernel4[0], self.kernel4[1], self.kernel4[2], self.dim_W4,
                self.dim_W3
            ],
                             stddev=0.02),
            name='discrim_y_vox_W2')
        self.discrim_y_bn_g2 = tf.Variable(
            tf.random_normal([1], mean=1.0, stddev=0.02),
            name='discrim_y_vox_bn_g2')
        self.discrim_y_bn_b2 = tf.Variable(
            tf.zeros([1]), name='discrim_y_vox_bn_b2')

        self.discrim_y_W3 = tf.Variable(
            tf.random_normal([
                self.kernel3[0], self.kernel3[1], self.kernel3[2], self.dim_W3,
                self.dim_W2
            ],
                             stddev=0.02),
            name='discrim_y_vox_W3')
        self.discrim_y_bn_g3 = tf.Variable(
            tf.random_normal([1], mean=1.0, stddev=0.02),
            name='discrim_y_vox_bn_g3')
        self.discrim_y_bn_b3 = tf.Variable(
            tf.zeros([1]), name='discrim_y_vox_bn_b3')

        self.discrim_y_W4 = tf.Variable(
            tf.random_normal([
                self.kernel2[0], self.kernel2[1], self.kernel2[2], self.dim_W2,
                self.dim_W1
            ],
                             stddev=0.02),
            name='discrim_y_vox_W4')
        self.discrim_y_bn_g4 = tf.Variable(
            tf.random_normal([1], mean=1.0, stddev=0.02),
            name='discrim_y_vox_bn_g4')
        self.discrim_y_bn_b4 = tf.Variable(
            tf.zeros([1]), name='discrim_y_vox_bn_b4')

        # patch GAN
        """
        self.discrim_y_W5 = tf.Variable(
            tf.random_normal([1, 1, 1, self.dim_W1, self.dim_z], stddev=0.02),
            name='discrim_y_vox_W5')
        """
        self.discrim_y_W5 = tf.Variable(
            tf.random_normal([
                self.start_vox_size[0] * self.start_vox_size[1] *
                self.start_vox_size[2] * self.dim_W1, 1
            ],
                             stddev=0.02),
            name='discrim_y_vox_W5')

        # parameters of generator x
        self.gen_x_W1 = tf.Variable(
            tf.random_normal([
                self.dim_z * self.start_vox_size[0] * self.start_vox_size[1] *
                self.start_vox_size[2], self.dim_W1 * self.start_vox_size[0] *
                self.start_vox_size[1] * self.start_vox_size[2]
            ],
                             stddev=0.02),
            name='gen_x_W1')
        self.gen_x_bn_g1 = tf.Variable(
            tf.random_normal([
                self.dim_W1 * self.start_vox_size[0] * self.start_vox_size[1] *
                self.start_vox_size[2]
            ],
                             mean=1.0,
                             stddev=0.02),
            name='gen_x_bn_g1')
        self.gen_x_bn_b1 = tf.Variable(
            tf.zeros([
                self.dim_W1 * self.start_vox_size[0] * self.start_vox_size[1] *
                self.start_vox_size[2]
            ]),
            name='gen_x_bn_b1')

        self.gen_x_W2 = tf.Variable(
            tf.random_normal([
                self.kernel2[0], self.kernel2[1], self.kernel2[2], self.dim_W2,
                self.dim_W1
            ],
                             stddev=0.02),
            name='gen_x_W2')
        self.gen_x_bn_g2 = tf.Variable(
            tf.random_normal([self.dim_W2], mean=1.0, stddev=0.02),
            name='gen_x_bn_g2')
        self.gen_x_bn_b2 = tf.Variable(
            tf.zeros([self.dim_W2]), name='gen_x_bn_b2')

        self.gen_x_W3 = tf.Variable(
            tf.random_normal([
                self.kernel3[0], self.kernel3[1], self.kernel3[2], self.dim_W3,
                self.dim_W2
            ],
                             stddev=0.02),
            name='gen_x_W3')
        self.gen_x_bn_g3 = tf.Variable(
            tf.random_normal([self.dim_W3], mean=1.0, stddev=0.02),
            name='gen_x_bn_g3')
        self.gen_x_bn_b3 = tf.Variable(
            tf.zeros([self.dim_W3]), name='gen_x_bn_b3')

        self.gen_x_W4 = tf.Variable(
            tf.random_normal([
                self.kernel4[0], self.kernel4[1], self.kernel4[2], self.dim_W4,
                self.dim_W3
            ],
                             stddev=0.02),
            name='gen_x_W4')
        self.gen_x_bn_g4 = tf.Variable(
            tf.random_normal([self.dim_W4], mean=1.0, stddev=0.02),
            name='gen_x_bn_g4')
        self.gen_x_bn_b4 = tf.Variable(
            tf.zeros([self.dim_W4]), name='gen_x_bn_b4')

        self.gen_x_W5 = tf.Variable(
            tf.random_normal([
                self.kernel5[0], self.kernel5[1], self.kernel5[2],
                self.part_shape[-1], self.dim_W4
            ],
                             stddev=0.02),
            name='gen_x_W5')
        self.gen_x_bn_g5 = tf.Variable(
            tf.random_normal([self.part_shape[-1]], mean=1.0, stddev=0.02),
            name='gen_x_bn_g5')
        self.gen_x_bn_b5 = tf.Variable(
            tf.zeros([self.part_shape[-1]]), name='gen_x_bn_b5')

        # parameters of encoder y
        self.encode_y_W1 = tf.Variable(
            tf.random_normal([
                self.kernel5[0], self.kernel5[1], self.kernel5[2],
                self.vox_shape[-1], self.dim_W4
            ],
                             stddev=0.02),
            name='encode_y_W1')
        self.encode_y_bn_g1 = tf.Variable(
            tf.random_normal([self.dim_W4], mean=1.0, stddev=0.02),
            name='encode_y_bn_g1')
        self.encode_y_bn_b1 = tf.Variable(
            tf.zeros([self.dim_W4]), name='encode_y_bn_b1')

        self.encode_y_W2 = tf.Variable(
            tf.random_normal([
                self.kernel4[0], self.kernel4[1], self.kernel4[2],
                self.dim_W4 * 3, self.dim_W3
            ],
                             stddev=0.02),
            name='encode_y_W2')
        self.encode_y_bn_g2 = tf.Variable(
            tf.random_normal([self.dim_W3], mean=1.0, stddev=0.02),
            name='encode_y_bn_g2')
        self.encode_y_bn_b2 = tf.Variable(
            tf.zeros([self.dim_W3]), name='encode_y_bn_b2')

        self.encode_y_W3 = tf.Variable(
            tf.random_normal([
                self.kernel3[0], self.kernel3[1], self.kernel3[2], self.dim_W3,
                self.dim_W2
            ],
                             stddev=0.02),
            name='encode_y_W3')
        self.encode_y_bn_g3 = tf.Variable(
            tf.random_normal([self.dim_W2], mean=1.0, stddev=0.02),
            name='encode_y_bn_g3')
        self.encode_y_bn_b3 = tf.Variable(
            tf.zeros([self.dim_W2]), name='encode_y_bn_b3')

        self.encode_y_W4 = tf.Variable(
            tf.random_normal([
                self.kernel2[0], self.kernel2[1], self.kernel2[2], self.dim_W2,
                self.dim_W1
            ],
                             stddev=0.02),
            name='encode_y_W4')
        self.encode_y_bn_g4 = tf.Variable(
            tf.random_normal([self.dim_W1], mean=1.0, stddev=0.02),
            name='encode_y_bn_g4')
        self.encode_y_bn_b4 = tf.Variable(
            tf.zeros([self.dim_W1]), name='encode_y_bn_b4')

        self.encode_y_W5 = tf.Variable(
            tf.random_normal([1, 1, 1, self.dim_W1, self.dim_z], stddev=0.02),
            name='encode_y_W5')
        self.encode_y_W5_sigma = tf.Variable(
            tf.random_normal([1, 1, 1, self.dim_W1, self.dim_z], stddev=0.02),
            name='encode_y_W5_sigma')

        # parameters of discriminator
        self.discrim_x_W1 = tf.Variable(
            tf.random_normal([
                self.kernel5[0], self.kernel5[1], self.kernel5[2],
                self.part_shape[-1], self.dim_W4
            ],
                             stddev=0.02),
            name='discrim_x_vox_W1')
        self.discrim_x_bn_g1 = tf.Variable(
            tf.random_normal([1], mean=1.0, stddev=0.02),
            name='discrim_x_vox_bn_g1')
        self.discrim_x_bn_b1 = tf.Variable(
            tf.zeros([1]), name='discrim_x_vox_bn_b1')

        self.discrim_x_W2 = tf.Variable(
            tf.random_normal([
                self.kernel4[0], self.kernel4[1], self.kernel4[2], self.dim_W4,
                self.dim_W3
            ],
                             stddev=0.02),
            name='discrim_x_vox_W2')
        self.discrim_x_bn_g2 = tf.Variable(
            tf.random_normal([1], mean=1.0, stddev=0.02),
            name='discrim_x_vox_bn_g2')
        self.discrim_x_bn_b2 = tf.Variable(
            tf.zeros([1]), name='discrim_x_vox_bn_b2')

        self.discrim_x_W3 = tf.Variable(
            tf.random_normal([
                self.kernel3[0], self.kernel3[1], self.kernel3[2], self.dim_W3,
                self.dim_W2
            ],
                             stddev=0.02),
            name='discrim_x_vox_W3')
        self.discrim_x_bn_g3 = tf.Variable(
            tf.random_normal([1], mean=1.0, stddev=0.02),
            name='discrim_x_vox_bn_g3')
        self.discrim_x_bn_b3 = tf.Variable(
            tf.zeros([1]), name='discrim_x_vox_bn_b3')

        self.discrim_x_W4 = tf.Variable(
            tf.random_normal([
                self.kernel2[0], self.kernel2[1], self.kernel2[2], self.dim_W2,
                self.dim_W1
            ],
                             stddev=0.02),
            name='discrim_x_vox_W4')
        self.discrim_x_bn_g4 = tf.Variable(
            tf.random_normal([1], mean=1.0, stddev=0.02),
            name='discrim_x_vox_bn_g4')
        self.discrim_x_bn_b4 = tf.Variable(
            tf.zeros([1]), name='discrim_x_vox_bn_b4')

        # patch GAN
        """
        self.discrim_x_W5 = tf.Variable(
            tf.random_normal([1, 1, 1, self.dim_W1, self.dim_z], stddev=0.02),
            name='discrim_x_vox_W5')
        """
        self.discrim_x_W5 = tf.Variable(
            tf.random_normal([
                self.start_vox_size[0] * self.start_vox_size[1] *
                self.start_vox_size[2] * self.dim_W1, 1
            ],
                             stddev=0.02),
            name='discrim_x_vox_W5')

        # parameters of codes discriminator
        self.cod_x_W1 = tf.Variable(
            tf.random_normal([
                self.dim_z * self.start_vox_size[0] * self.start_vox_size[1] *
                self.start_vox_size[2], self.dim_code
            ],
                             stddev=0.02),
            name='cod_x_W1')
        self.cod_x_bn_g1 = tf.Variable(
            tf.random_normal([dim_code], mean=1.0, stddev=0.02),
            name='cod_x_bn_g1')
        self.cod_x_bn_b1 = tf.Variable(
            tf.zeros([dim_code]), name='cod_x_bn_b1')

        self.cod_x_W2 = tf.Variable(
            tf.random_normal([dim_code, dim_code], stddev=0.02),
            name='cod_x_W2')
        self.cod_x_bn_g2 = tf.Variable(
            tf.random_normal([dim_code], mean=1.0, stddev=0.02),
            name='cod_x_bn_g2')
        self.cod_x_bn_b2 = tf.Variable(
            tf.zeros([dim_code]), name='cod_x_bn_b2')

        self.cod_x_W3 = tf.Variable(
            tf.random_normal([dim_code, 1], stddev=0.02), name='cod_x_W3')

        self.cod_y_W1 = tf.Variable(
            tf.random_normal([
                self.dim_z * self.start_vox_size[0] * self.start_vox_size[1] *
                self.start_vox_size[2], self.dim_code
            ],
                             stddev=0.02),
            name='cod_y_W1')
        self.cod_y_bn_g1 = tf.Variable(
            tf.random_normal([dim_code], mean=1.0, stddev=0.02),
            name='cod_y_bn_g1')
        self.cod_y_bn_b1 = tf.Variable(
            tf.zeros([dim_code]), name='cod_y_bn_b1')

        self.cod_y_W2 = tf.Variable(
            tf.random_normal([dim_code, dim_code], stddev=0.02),
            name='cod_y_W2')
        self.cod_y_bn_g2 = tf.Variable(
            tf.random_normal([dim_code], mean=1.0, stddev=0.02),
            name='cod_y_bn_g2')
        self.cod_y_bn_b2 = tf.Variable(
            tf.zeros([dim_code]), name='cod_y_bn_b2')

        self.cod_y_W3 = tf.Variable(
            tf.random_normal([dim_code, 1], stddev=0.02), name='cod_y_W3')

        self.saver = tf.train.Saver()

    def build_model(self):

        full_gt_ = tf.placeholder(tf.int32, [
            self.batch_size, self.vox_shape[0], self.vox_shape[1],
            self.vox_shape[2]
        ])
        full_gt = tf.one_hot(full_gt_, self.n_class)
        full_gt = tf.cast(full_gt, tf.float32)

        complete_gt = tf.one_hot(tf.clip_by_value(full_gt_, clip_value_min=0, clip_value_max=1), 2)
        complete_gt = tf.cast(complete_gt, tf.float32)

        # tsdf--start
        part_gt_ = tf.placeholder(tf.float32, [
            self.batch_size, self.vox_shape[0], self.vox_shape[1],
            self.vox_shape[2]
        ])  
        part_gt = tf.abs(tf.expand_dims(part_gt_, -1))
        # tsdf--end

        Z = tf.placeholder(tf.float32, [
            self.batch_size, self.start_vox_size[0], self.start_vox_size[1],
            self.start_vox_size[2], self.dim_z
        ])

        # weights for balancing training
        batch_mean_full_gt = tf.reduce_mean(full_gt, [0, 1, 2, 3])
        ones = tf.ones_like(batch_mean_full_gt)
        inverse = tf.div(ones, tf.add(batch_mean_full_gt, ones))
        weight_full = inverse * tf.div(1., tf.reduce_sum(inverse))

        # weights for balancing training
        batch_mean_complete_gt = tf.reduce_mean(complete_gt, [0, 1, 2, 3])
        ones = tf.ones_like(batch_mean_complete_gt)
        inverse = tf.div(ones, tf.add(batch_mean_complete_gt, ones))
        weight_complete = inverse * tf.div(1., tf.reduce_sum(inverse))

        # encode from tsdf and vox
        Z_encode_part = self.encoder_part(part_gt)
        scene_mask_1 = tf.random_uniform([self.batch_size,1,1,1,self.n_class - 1], minval=0, maxval=1, dtype=tf.int32)
        scene_mask_2 = 1 - scene_mask_1
        scene_mask_1 = tf.concat([tf.ones([self.batch_size,1,1,1,1], dtype=tf.int32), scene_mask_1], -1)
        scene_mask_2 = tf.concat([tf.ones([self.batch_size,1,1,1,1], dtype=tf.int32), scene_mask_2], -1)
        feature_mask_1 = tf.concat([scene_mask_1, tf.ones([self.batch_size,1,1,1,self.dim_z-self.n_class], dtype=tf.int32)], -1)
        feature_mask_2 = tf.concat([scene_mask_2, tf.ones([self.batch_size,1,1,1,self.dim_z-self.n_class], dtype=tf.int32)], -1) 
        scene_mask_1 = tf.cast(scene_mask_1, dtype=tf.float32)
        feature_mask_1 = tf.cast(feature_mask_1, dtype=tf.float32)
        scene_mask_2 = tf.cast(scene_mask_2, dtype=tf.float32)
        feature_mask_2 = tf.cast(feature_mask_2, dtype=tf.float32)
        # test_mask = tf.ones([1,1,1,1,self.dim_z]) * [1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0]
        # test_mask = tf.ones([1,1,1,1,self.dim_z]) * [1,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1]
        # test_mask = tf.ones([1,1,1,1,self.dim_z]) * [1,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1]
        """
        Z_encode_full = self.encoder_full(full_gt)
        """

        part_vae_dec, h2_t, h3_t, h4_t = self.generate_part(Z_encode_part)
        # h2_t = tf.zeros_like(h2_t)
        # h3_t = tf.zeros_like(h3_t)
        # h4_t = tf.zeros_like(h4_t)
        """
        part_gen_dec, h2_v, h3_v, h4_v = self.generate_part(Z_encode_full)
        """

        """
        full_vae_dec = self.generate_full(Z_encode_full, h2_v, h3_v, h4_v)
        """
        full_gen_dec = self.generate_full(Z_encode_part, h2_t, h3_t, h4_t)
        full_gen_dec_1 = self.generate_full(Z_encode_part * feature_mask_1, h2_t, h3_t, h4_t)
        full_gen_dec_2 = self.generate_full(Z_encode_part * feature_mask_2, h2_t, h3_t, h4_t)

        # complete = self.complete(full_gen_dec)
        # encode again from loops
        """
        full_gen_dec_o = tf.one_hot(
            tf.argmax(full_gen_dec, axis=4, output_type=tf.int32),
            self.n_class)
        full_gen_dec_o = tf.cast(full_gen_dec_o, tf.float32)
        Z_encode_part_full = self.encoder_full(full_gen_dec_o)
        Z_encode_full_part = self.encoder_part(part_gen_dec)

        part_cc_dec, _, _, _ = self.generate_part(Z_encode_part_full)
        _, h2_vt, h3_vt, h4_vt = self.generate_part(Z_encode_full_part)
        full_cc_dec = self.generate_full(Z_encode_full_part, h2_vt, h3_vt,
                                         h4_vt)
         """

        # code_discriminator
        h_code_part_gt = self.code_discriminator_x(Z)
        """
        h_code_full_gt = self.code_discriminator_y(Z)
        """

        h_code_encode_part = self.code_discriminator_x(Z_encode_part)
        """
        h_code_encode_full = self.code_discriminator_y(Z_encode_full)

        h_code_encode_part_full = self.code_discriminator_y(Z_encode_part_full)
        h_code_encode_full_part = self.code_discriminator_x(Z_encode_full_part)
        """

        cost_code_encode = tf.reduce_mean(
            tf.reduce_sum(
                tf.nn.sigmoid_cross_entropy_with_logits(
                    logits=h_code_encode_part,
                    labels=tf.ones_like(h_code_encode_part)), [1]))
        """
        cost_code_encode += tf.reduce_mean(
            tf.reduce_sum(
                tf.nn.sigmoid_cross_entropy_with_logits(
                    logits=h_code_encode_full,
                    labels=tf.ones_like(h_code_encode_full)), [1]))
        cost_code_encode += tf.reduce_mean(
            tf.reduce_sum(
                tf.nn.sigmoid_cross_entropy_with_logits(
                    logits=h_code_encode_part_full,
                    labels=tf.ones_like(h_code_encode_part_full)), [1]))
        cost_code_encode += tf.reduce_mean(
            tf.reduce_sum(
                tf.nn.sigmoid_cross_entropy_with_logits(
                    logits=h_code_encode_full_part,
                    labels=tf.ones_like(h_code_encode_full_part)), [1]))
        """

        cost_code_discrim = tf.reduce_mean(
            tf.reduce_sum(
                tf.nn.sigmoid_cross_entropy_with_logits(
                    logits=h_code_part_gt,
                    labels=tf.ones_like(h_code_part_gt)),
                [1])) + tf.reduce_mean(
                    tf.reduce_sum(
                        tf.nn.sigmoid_cross_entropy_with_logits(
                            logits=h_code_encode_part,
                            labels=tf.zeros_like(h_code_encode_part)), [1]))
        """
        cost_code_discrim += tf.reduce_mean(
            tf.reduce_sum(
                tf.nn.sigmoid_cross_entropy_with_logits(
                    logits=h_code_full_gt,
                    labels=tf.ones_like(h_code_full_gt)),
                [1])) + tf.reduce_mean(
                    tf.reduce_sum(
                        tf.nn.sigmoid_cross_entropy_with_logits(
                            logits=h_code_encode_full,
                            labels=tf.zeros_like(h_code_encode_full)), [1]))
        cost_code_discrim += tf.reduce_mean(
            tf.reduce_sum(
                tf.nn.sigmoid_cross_entropy_with_logits(
                    logits=h_code_full_gt,
                    labels=tf.ones_like(h_code_full_gt)),
                [1])) + tf.reduce_mean(
                    tf.reduce_sum(
                        tf.nn.sigmoid_cross_entropy_with_logits(
                            logits=h_code_encode_part_full,
                            labels=tf.zeros_like(h_code_encode_part_full)),
                        [1]))
        cost_code_discrim += tf.reduce_mean(
            tf.reduce_sum(
                tf.nn.sigmoid_cross_entropy_with_logits(
                    logits=h_code_part_gt,
                    labels=tf.ones_like(h_code_part_gt)),
                [1])) + tf.reduce_mean(
                    tf.reduce_sum(
                        tf.nn.sigmoid_cross_entropy_with_logits(
                            logits=h_code_encode_full_part,
                            labels=tf.zeros_like(h_code_encode_full_part)),
                        [1]))
        """

        # Completing from depth and semantic depth
        """
        recons_vae_loss = tf.reduce_mean(
            tf.reduce_sum(
                -tf.reduce_sum(
                    self.lamda_gamma * full_gt * tf.log(1e-6 + full_vae_dec) +
                    (1 - self.lamda_gamma) *
                    (1 - full_gt) * tf.log(1e-6 + 1 - full_vae_dec), [1, 2, 3])
                * weight_full, 1))
        """
        recons_vae_loss = tf.reduce_mean(
            tf.reduce_sum(
                tf.squared_difference(part_gt, part_vae_dec), [1, 2, 3, 4]))
        # latent consistency
        """
        recons_vae_loss += tf.reduce_mean(
            tf.reduce_sum(
                tf.squared_difference(Z_encode_part, Z_encode_full_part),
                [1, 2, 3, 4]))
        recons_vae_loss += tf.reduce_mean(
            tf.reduce_sum(
                tf.squared_difference(Z_encode_part, Z_encode_part_full),
                [1, 2, 3, 4]))
        recons_vae_loss += tf.reduce_mean(
            tf.reduce_sum(
                tf.squared_difference(Z_encode_full, Z_encode_part_full),
                [1, 2, 3, 4]))
        recons_vae_loss += tf.reduce_mean(
            tf.reduce_sum(
                tf.squared_difference(Z_encode_full, Z_encode_full_part),
                [1, 2, 3, 4]))
        """
        # latent consistency

        # Cycle consistencies
        """
        recons_cc_loss = tf.reduce_mean(
            tf.reduce_sum(
                -tf.reduce_sum(
                    self.lamda_gamma * full_gt * tf.log(1e-6 + full_cc_dec) +
                    (1 - self.lamda_gamma) *
                    (1 - full_gt) * tf.log(1e-6 + 1 - full_cc_dec), [1, 2, 3])
                * weight_full, 1))
        recons_cc_loss += tf.reduce_mean(
            tf.reduce_sum(
                tf.squared_difference(part_gt, part_cc_dec), [1, 2, 3, 4]))
        """

        # latent consistency
        """
        recons_cc_loss += tf.reduce_mean(
            tf.reduce_sum(
                tf.squared_difference(Z_encode_part_full, Z_encode_full_part),
                [1, 2, 3, 4]))
        """
        # latent consistency
        # SUPERVISED (paired data)
        recons_gen_loss = tf.reduce_mean(
            tf.reduce_sum(
                -tf.reduce_sum(
                    self.lamda_gamma * full_gt * tf.log(1e-6 + full_gen_dec) +
                    (1 - self.lamda_gamma) *
                    (1 - full_gt) * tf.log(1e-6 + 1 - full_gen_dec), [1, 2, 3])
                * weight_full, 1))
        """
        recons_gen_loss += tf.reduce_mean(
            tf.reduce_sum(
                -tf.reduce_sum(
                    self.lamda_gamma * full_gt * tf.log(1e-6 + full_gen_dec_1) * scene_mask_1 +
                    (1 - self.lamda_gamma) *
                    (1 - full_gt * scene_mask_1) * tf.log(1e-6 + 1 - full_gen_dec_1), [1, 2, 3])
                * weight_full, 1))
        recons_gen_loss += tf.reduce_mean(
            tf.reduce_sum(
                -tf.reduce_sum(
                    self.lamda_gamma * full_gt * tf.log(1e-6 + full_gen_dec_2) * scene_mask_2 +
                    (1 - self.lamda_gamma) *
                    (1 - full_gt * scene_mask_2) * tf.log(1e-6 + 1 - full_gen_dec_2), [1, 2, 3])
                * weight_full, 1))
        """
        # complete 
        """
        recons_gen_loss += tf.reduce_mean(
            tf.reduce_sum(
                -tf.reduce_sum(
                    self.lamda_gamma * complete_gt * tf.log(1e-6 + complete) +
                    (1 - self.lamda_gamma) *
                    (1 - complete_gt) * tf.log(1e-6 + 1 - complete), [1, 2, 3])
                * weight_complete, 1))
        """
        """
        recons_gen_loss += tf.reduce_mean(
            tf.reduce_sum(
                tf.squared_difference(part_gt, part_gen_dec), [1, 2, 3, 4]))
        """
        # from scene, the observed surface can also be produced
        # latent consistency
        """
        recons_gen_loss += tf.reduce_mean(
            tf.reduce_sum(
                tf.squared_difference(Z_encode_part, Z_encode_full),
                [1, 2, 3, 4]))
        """
        # latent consistency

        # GAN_generate
        part_gen, h2_z, h3_z, h4_z = self.generate_part(Z)
        full_gen = self.generate_full(Z, h2_z, h3_z, h4_z)

        if self.discriminative is True:
            h_full_gt = self.discriminate_full(tf.concat([full_gt, part_gt], -1))
            h_full_gen = self.discriminate_full(tf.concat([full_gen, part_gen], -1))
            """
            h_full_gen_dec = self.discriminate_full(full_gen_dec)
            """

            h_part_gt = self.discriminate_part(part_gt)
            h_part_gen = self.discriminate_part(part_gen)
            """
            h_part_gen_dec = self.discriminate_part(part_gen_dec)
            """
            # Standard_GAN_Loss
            discrim_loss = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(
                    logits=h_full_gt,
                    labels=tf.ones_like(h_full_gt))) + tf.reduce_mean(
                        tf.nn.sigmoid_cross_entropy_with_logits(
                            logits=h_full_gen,
                            labels=tf.zeros_like(h_full_gen)))

            discrim_loss += tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(
                    logits=h_part_gt,
                    labels=tf.ones_like(h_part_gt))) + tf.reduce_mean(
                        tf.nn.sigmoid_cross_entropy_with_logits(
                            logits=h_part_gen,
                            labels=tf.zeros_like(h_part_gen)))

            gen_loss = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(
                    logits=h_full_gen, labels=tf.ones_like(h_full_gen)))

            gen_loss += tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(
                    logits=h_part_gen, labels=tf.ones_like(h_part_gen)))
        else:
            gen_loss = tf.zeros([1])
            discrim_loss = tf.zeros([1])

        # main cost
        """
        cost_pred = self.lamda_recons * (
            recons_vae_loss + recons_cc_loss + recons_gen_loss)
        """
        cost_pred = self.lamda_recons * (
            recons_vae_loss + recons_gen_loss)

        # variational cost
        cost_code_encode = cost_code_encode
        cost_code_discrim = cost_code_discrim

        tf.summary.scalar("recons_vae_loss", tf.reduce_mean(recons_vae_loss))
        # tf.summary.scalar("recons_cc_loss", tf.reduce_mean(recons_cc_loss))
        tf.summary.scalar("gen_loss", tf.reduce_mean(gen_loss))
        tf.summary.scalar("discrim_loss", tf.reduce_mean(discrim_loss))
        tf.summary.scalar("cost_code_encode", tf.reduce_mean(cost_code_encode))
        tf.summary.scalar("cost_code_discrim",
                          tf.reduce_mean(cost_code_discrim))

        summary_op = tf.summary.merge_all()

        return Z, Z_encode_part, full_gt_, full_gen, full_gen_dec,\
        recons_vae_loss, recons_gen_loss, gen_loss, discrim_loss,\
        cost_pred, cost_code_encode, cost_code_discrim, summary_op,\
        part_gt_, part_gen, part_vae_dec

    def encoder_part(self, vox):

        h1 = lrelu(
            tf.layers.conv3d(
                vox,
                filters=self.dim_W4,
                kernel_size=(self.kernel5[0], self.kernel5[1],
                             self.kernel5[2]),
                strides=(self.stride[1], self.stride[2], self.stride[3]),
                padding='same',
                name='encode_x_1',
                reuse=tf.AUTO_REUSE))

        base_5 = tf.layers.conv3d(
            h1,
            filters=16,
            kernel_size=(3, 3, 3),
            strides=(1, 1, 1),
            padding='same',
            dilation_rate=(2, 2, 2),
            name='encode_x_sscnet_1',
            reuse=tf.AUTO_REUSE)

        base_6 = base_5 + tf.layers.conv3d(
            base_5,
            filters=16,
            kernel_size=(3, 3, 3),
            strides=(1, 1, 1),
            padding='same',
            dilation_rate=(2, 2, 2),
            name='encode_x_sscnet_2',
            reuse=tf.AUTO_REUSE)

        base_7 = tf.layers.conv3d(
            base_6,
            filters=16,
            kernel_size=(3, 3, 3),
            strides=(1, 1, 1),
            padding='same',
            dilation_rate=(2, 2, 2),
            name='encode_x_sscnet_3',
            reuse=tf.AUTO_REUSE)

        base_8 = base_7 + tf.layers.conv3d(
            base_7,
            filters=16,
            kernel_size=(3, 3, 3),
            strides=(1, 1, 1),
            padding='same',
            dilation_rate=(2, 2, 2),
            name='encode_x_sscnet_4',
            reuse=tf.AUTO_REUSE)
        base_9 = tf.concat([h1, base_6, base_8], -1)

        h2 = lrelu(
            tf.layers.batch_normalization(
                tf.layers.conv3d(
                    base_9,
                    filters=self.dim_W3,
                    kernel_size=(self.kernel4[0], self.kernel4[1],
                                 self.kernel4[2]),
                    strides=(self.stride[1], self.stride[2], self.stride[3]),
                    padding='same',
                    name='encode_x_2',
                    reuse=tf.AUTO_REUSE),
                name='encode_x_bn_2',
                reuse=tf.AUTO_REUSE,
                training=self.is_train))

        h3 = lrelu(
            tf.layers.batch_normalization(
                tf.layers.conv3d(
                    h2,
                    filters=self.dim_W2,
                    kernel_size=(self.kernel3[0], self.kernel3[1],
                                 self.kernel3[2]),
                    strides=(self.stride[1], self.stride[2], self.stride[3]),
                    padding='same',
                    name='encode_x_3',
                    reuse=tf.AUTO_REUSE),
                name='encode_x_bn_3',
                reuse=tf.AUTO_REUSE,
                training=self.is_train))

        h4 = lrelu(
            tf.layers.batch_normalization(
                tf.layers.conv3d(
                    h3,
                    filters=self.dim_W1,
                    kernel_size=(self.kernel2[0], self.kernel2[1],
                                 self.kernel2[2]),
                    strides=(self.stride[1], self.stride[2], self.stride[3]),
                    padding='same',
                    name='encode_x_4',
                    reuse=tf.AUTO_REUSE),
                name='encode_x_bn_4',
                reuse=tf.AUTO_REUSE,
                training=self.is_train))

        h5 = tf.layers.conv3d(
            h4,
            filters=self.dim_z,
            kernel_size=(1, 1, 1),
            strides=(1, 1, 1),
            padding='same',
            name='encode_x_5',
            reuse=tf.AUTO_REUSE)

        return h5

    def discriminate_full(self, vox):

        h1 = lrelu(
            tf.nn.conv3d(
                vox,
                self.discrim_y_W1,
                strides=self.stride,
                dilations=self.dilations,
                padding='SAME'))
        h2 = lrelu(
            layernormalize(
                tf.nn.conv3d(
                    h1,
                    self.discrim_y_W2,
                    strides=self.stride,
                    dilations=self.dilations,
                    padding='SAME'),
                g=self.discrim_y_bn_g2,
                b=self.discrim_y_bn_b2))
        h3 = lrelu(
            layernormalize(
                tf.nn.conv3d(
                    h2,
                    self.discrim_y_W3,
                    strides=self.stride,
                    dilations=self.dilations,
                    padding='SAME'),
                g=self.discrim_y_bn_g3,
                b=self.discrim_y_bn_b3))
        h4 = lrelu(
            layernormalize(
                tf.nn.conv3d(
                    h3,
                    self.discrim_y_W4,
                    strides=self.stride,
                    dilations=self.dilations,
                    padding='SAME'),
                g=self.discrim_y_bn_g4,
                b=self.discrim_y_bn_b4))
        h4 = tf.reshape(h4, [self.batch_size, -1])
        h5 = tf.matmul(h4, self.discrim_y_W5)
        y = tf.nn.sigmoid(h5)

        return h5

    def encoder_full(self, vox):

        h1 = lrelu(
            tf.layers.conv3d(
                vox,
                filters=self.dim_W4,
                kernel_size=(self.kernel5[0], self.kernel5[1],
                             self.kernel5[2]),
                strides=(self.stride[1], self.stride[2], self.stride[3]),
                padding='same',
                name='encode_y_1',
                reuse=tf.AUTO_REUSE))

        base_5 = tf.layers.conv3d(
            h1,
            filters=16,
            kernel_size=(3, 3, 3),
            strides=(1, 1, 1),
            padding='same',
            dilation_rate=(2, 2, 2),
            name='encode_y_sscnet_1',
            reuse=tf.AUTO_REUSE)

        base_6 = base_5 + tf.layers.conv3d(
            base_5,
            filters=16,
            kernel_size=(3, 3, 3),
            strides=(1, 1, 1),
            padding='same',
            dilation_rate=(2, 2, 2),
            name='encode_y_sscnet_2',
            reuse=tf.AUTO_REUSE)

        base_7 = tf.layers.conv3d(
            base_6,
            filters=16,
            kernel_size=(3, 3, 3),
            strides=(1, 1, 1),
            padding='same',
            dilation_rate=(2, 2, 2),
            name='encode_y_sscnet_3',
            reuse=tf.AUTO_REUSE)

        base_8 = base_7 + tf.layers.conv3d(
            base_7,
            filters=16,
            kernel_size=(3, 3, 3),
            strides=(1, 1, 1),
            padding='same',
            dilation_rate=(2, 2, 2),
            name='encode_y_sscnet_4',
            reuse=tf.AUTO_REUSE)
        base_9 = tf.concat([h1, base_6, base_8], -1)

        h2 = lrelu(
            tf.layers.batch_normalization(
                tf.layers.conv3d(
                    base_9,
                    filters=self.dim_W3,
                    kernel_size=(self.kernel4[0], self.kernel4[1],
                                 self.kernel4[2]),
                    strides=(self.stride[1], self.stride[2], self.stride[3]),
                    padding='same',
                    name='encode_y_2',
                    reuse=tf.AUTO_REUSE),
                name='encode_y_bn_2',
                reuse=tf.AUTO_REUSE,
                training=self.is_train))

        h3 = lrelu(
            tf.layers.batch_normalization(
                tf.layers.conv3d(
                    h2,
                    filters=self.dim_W2,
                    kernel_size=(self.kernel3[0], self.kernel3[1],
                                 self.kernel3[2]),
                    strides=(self.stride[1], self.stride[2], self.stride[3]),
                    padding='same',
                    name='encode_y_3',
                    reuse=tf.AUTO_REUSE),
                name='encode_y_bn_3',
                reuse=tf.AUTO_REUSE,
                training=self.is_train))

        h4 = lrelu(
            tf.layers.batch_normalization(
                tf.layers.conv3d(
                    h3,
                    filters=self.dim_W1,
                    kernel_size=(self.kernel2[0], self.kernel2[1],
                                 self.kernel2[2]),
                    strides=(self.stride[1], self.stride[2], self.stride[3]),
                    padding='same',
                    name='encode_y_4',
                    reuse=tf.AUTO_REUSE),
                name='encode_y_bn_4',
                reuse=tf.AUTO_REUSE,
                training=self.is_train))

        h5 = tf.layers.conv3d(
            h4,
            filters=self.dim_z,
            kernel_size=(1, 1, 1),
            strides=(1, 1, 1),
            padding='same',
            name='encode_y_5',
            reuse=tf.AUTO_REUSE)

        return h5

    def discriminate_part(self, vox):

        h1 = lrelu(
            tf.nn.conv3d(
                vox,
                self.discrim_x_W1,
                strides=self.stride,
                dilations=self.dilations,
                padding='SAME'))
        h2 = lrelu(
            layernormalize(
                tf.nn.conv3d(
                    h1,
                    self.discrim_x_W2,
                    strides=self.stride,
                    dilations=self.dilations,
                    padding='SAME'),
                g=self.discrim_x_bn_g2,
                b=self.discrim_x_bn_b2))
        h3 = lrelu(
            layernormalize(
                tf.nn.conv3d(
                    h2,
                    self.discrim_x_W3,
                    strides=self.stride,
                    dilations=self.dilations,
                    padding='SAME'),
                g=self.discrim_x_bn_g3,
                b=self.discrim_x_bn_b3))
        h4 = lrelu(
            layernormalize(
                tf.nn.conv3d(
                    h3,
                    self.discrim_x_W4,
                    strides=self.stride,
                    dilations=self.dilations,
                    padding='SAME'),
                g=self.discrim_x_bn_g4,
                b=self.discrim_x_bn_b4))
        h4 = tf.reshape(h4, [self.batch_size, -1])
        h5 = tf.matmul(h4, self.discrim_x_W5)
        y = tf.nn.sigmoid(h5)

        return h5

    def code_discriminator_x(self, Z):
        Z_ = tf.reshape(Z, [self.batch_size, -1])
        h1 = tf.nn.relu(
            batchnormalize(
                tf.matmul(Z_, self.cod_x_W1),
                g=self.cod_x_bn_g1,
                b=self.cod_x_bn_b1))
        h2 = tf.nn.relu(
            batchnormalize(
                tf.matmul(h1, self.cod_x_W2),
                g=self.cod_x_bn_g2,
                b=self.cod_x_bn_b2))
        h3 = tf.matmul(h2, self.cod_x_W3)
        y = tf.nn.sigmoid(h3)
        return h3

    def code_discriminator_y(self, Z):
        Z_ = tf.reshape(Z, [self.batch_size, -1])
        h1 = tf.nn.relu(
            batchnormalize(
                tf.matmul(Z_, self.cod_y_W1),
                g=self.cod_y_bn_g1,
                b=self.cod_y_bn_b1))
        h2 = tf.nn.relu(
            batchnormalize(
                tf.matmul(h1, self.cod_y_W2),
                g=self.cod_y_bn_g2,
                b=self.cod_y_bn_b2))
        h3 = tf.matmul(h2, self.cod_y_W3)
        y = tf.nn.sigmoid(h3)
        return h3

    def generate_full(self, Z, h2_, h3_, h4_):

        Z_ = tf.reshape(Z, [self.batch_size, -1])
        h1 = tf.nn.relu(
            batchnormalize(
                tf.matmul(Z_, self.gen_y_W1),
                g=self.gen_y_bn_g1,
                b=self.gen_y_bn_b1))
        h1 = tf.reshape(h1, [
            self.batch_size, self.start_vox_size[0], self.start_vox_size[1],
            self.start_vox_size[2], self.dim_W1
        ])

        vox_size_l2 = self.start_vox_size * 2
        output_shape_l2 = [
            self.batch_size, vox_size_l2[0], vox_size_l2[1], vox_size_l2[2],
            self.dim_W2
        ]
        h2 = tf.nn.conv3d_transpose(
            h1,
            self.gen_y_W2,
            output_shape=output_shape_l2,
            strides=self.stride)
        h2 = tf.nn.relu(
            batchnormalize(
                h2,
                g=self.gen_y_bn_g2,
                b=self.gen_y_bn_b2,
                batch_size=self.batch_size))

        vox_size_l3 = self.start_vox_size * 4
        output_shape_l3 = [
            self.batch_size, vox_size_l3[0], vox_size_l3[1], vox_size_l3[2],
            self.dim_W3
        ]
        h3 = tf.nn.conv3d_transpose(
            tf.concat([h2, h2_], -1),
            self.gen_y_W3,
            output_shape=output_shape_l3,
            strides=self.stride)
        h3 = tf.nn.relu(
            batchnormalize(
                h3,
                g=self.gen_y_bn_g3,
                b=self.gen_y_bn_b3,
                batch_size=self.batch_size))

        vox_size_l4 = self.start_vox_size * 8
        output_shape_l4 = [
            self.batch_size, vox_size_l4[0], vox_size_l4[1], vox_size_l4[2],
            self.dim_W4
        ]
        h4 = tf.nn.conv3d_transpose(
            tf.concat([h3, h3_], -1),
            self.gen_y_W4,
            output_shape=output_shape_l4,
            strides=self.stride)
        h4 = tf.nn.relu(
            batchnormalize(
                h4,
                g=self.gen_y_bn_g4,
                b=self.gen_y_bn_b4,
                batch_size=self.batch_size))

        vox_size_l5 = self.start_vox_size * 16
        output_shape_l5 = [
            self.batch_size, vox_size_l5[0], vox_size_l5[1], vox_size_l5[2],
            self.dim_W5
        ]
        h5 = tf.nn.conv3d_transpose(
            tf.concat([h4, h4_], -1),
            self.gen_y_W5,
            output_shape=output_shape_l5,
            strides=self.stride)

        x = softmax(h5, self.batch_size, self.vox_shape)
        return x

    def generate_part(self, Z):

        Z_ = tf.reshape(Z, [self.batch_size, -1])
        h1 = tf.nn.relu(
            batchnormalize(
                tf.matmul(Z_, self.gen_x_W1),
                g=self.gen_x_bn_g1,
                b=self.gen_x_bn_b1))
        h1 = tf.reshape(h1, [
            self.batch_size, self.start_vox_size[0], self.start_vox_size[1],
            self.start_vox_size[2], self.dim_W1
        ])

        vox_size_l2 = self.start_vox_size * 2
        output_shape_l2 = [
            self.batch_size, vox_size_l2[0], vox_size_l2[1], vox_size_l2[2],
            self.dim_W2
        ]
        h2 = tf.nn.conv3d_transpose(
            h1,
            self.gen_x_W2,
            output_shape=output_shape_l2,
            strides=self.stride)
        h2 = tf.nn.relu(
            batchnormalize(
                h2,
                g=self.gen_x_bn_g2,
                b=self.gen_x_bn_b2,
                batch_size=self.batch_size))

        vox_size_l3 = self.start_vox_size * 4
        output_shape_l3 = [
            self.batch_size, vox_size_l3[0], vox_size_l3[1], vox_size_l3[2],
            self.dim_W3
        ]
        h3 = tf.nn.conv3d_transpose(
            h2,
            self.gen_x_W3,
            output_shape=output_shape_l3,
            strides=self.stride)
        h3 = tf.nn.relu(
            batchnormalize(
                h3,
                g=self.gen_x_bn_g3,
                b=self.gen_x_bn_b3,
                batch_size=self.batch_size))

        vox_size_l4 = self.start_vox_size * 8
        output_shape_l4 = [
            self.batch_size, vox_size_l4[0], vox_size_l4[1], vox_size_l4[2],
            self.dim_W4
        ]
        h4 = tf.nn.conv3d_transpose(
            h3,
            self.gen_x_W4,
            output_shape=output_shape_l4,
            strides=self.stride)
        h4 = tf.nn.relu(
            batchnormalize(
                h4,
                g=self.gen_x_bn_g4,
                b=self.gen_x_bn_b4,
                batch_size=self.batch_size))

        vox_size_l5 = self.start_vox_size * 16
        output_shape_l5 = [
            self.batch_size, vox_size_l5[0], vox_size_l5[1], vox_size_l5[2],
            self.part_shape[-1]
        ]
        h5 = tf.nn.conv3d_transpose(
            h4,
            self.gen_x_W5,
            output_shape=output_shape_l5,
            strides=self.stride)

        # x = softmax(h5, self.batch_size, self.part_shape)
        return h5, h2, h3, h4

    def complete(self, vox):
        base = tf.nn.relu(
            tf.layers.conv3d(
                vox,
                filters=32,
                kernel_size=(3, 3, 3),
                strides=(1, 1, 1),
                padding='same',
                dilation_rate=(1, 1, 1),
                name='gen_com_1',
                reuse=tf.AUTO_REUSE))

        #res1
        res1_1 = tf.nn.relu(
            tf.layers.conv3d(
                base,
                filters=32,
                kernel_size=(3, 3, 3),
                strides=(1, 1, 1),
                padding='same',
                dilation_rate=(1, 1, 1),
                name='gen_com_res1_1',
                reuse=tf.AUTO_REUSE))
        res1_2 = tf.layers.conv3d(
                res1_1,
                filters=32,
                kernel_size=(3, 3, 3),
                strides=(1, 1, 1),
                padding='same',
                dilation_rate=(1, 1, 1),
                name='gen_com_res1_2',
                reuse=tf.AUTO_REUSE)

        res1 = tf.nn.relu(tf.add(base, res1_2))

        #res2
        res2_1 = tf.nn.relu(
            tf.layers.conv3d(
                res1,
                filters=32,
                kernel_size=(3, 3, 3),
                strides=(1, 1, 1),
                padding='same',
                dilation_rate=(1, 1, 1),
                name='gen_com_res2_1',
                reuse=tf.AUTO_REUSE))
        res2_2 = tf.layers.conv3d(
                res2_1,
                filters=32,
                kernel_size=(3, 3, 3),
                strides=(1, 1, 1),
                padding='same',
                dilation_rate=(1, 1, 1),
                name='gen_com_res2_2',
                reuse=tf.AUTO_REUSE)

        res2 = tf.nn.relu(tf.add(res1, res2_2))

        """ 
        #res3
        res3_1 = tf.nn.relu(
            tf.layers.conv3d(
                res2,
                filters=32,
                kernel_size=(3, 3, 3),
                strides=(1, 1, 1),
                padding='same',
                dilation_rate=(1, 1, 1),
                name='gen_com_res3_1',
                reuse=tf.AUTO_REUSE))
        res3_2 = tf.layers.conv3d(
                res3_1,
                filters=32,
                kernel_size=(3, 3, 3),
                strides=(1, 1, 1),
                padding='same',
                dilation_rate=(1, 1, 1),
                name='gen_com_res3_2',
                reuse=tf.AUTO_REUSE)

        res3 = tf.nn.relu(tf.add(res2, res3_2))
        """
        out = tf.layers.conv3d(
                res2,
                filters=2,
                kernel_size=(3, 3, 3),
                strides=(1, 1, 1),
                padding='same',
                dilation_rate=(1, 1, 1),
                name='gen_com_final',
                reuse=tf.AUTO_REUSE)

        complete = softmax(out, self.batch_size, [self.vox_shape[0], self.vox_shape[1], self.vox_shape[2], 2])

        return complete 

    def samples_generator(self, visual_size):

        Z = tf.placeholder(tf.float32, [
            visual_size, self.start_vox_size[0], self.start_vox_size[1],
            self.start_vox_size[2], self.dim_z
        ])

        Z_ = tf.reshape(Z, [visual_size, -1])
        part, h2_t, h3_t, h4_t = self.generate_part(Z_)
        # h2_t = tf.zeros_like(h2_t)
        # h3_t = tf.zeros_like(h3_t)
        # h4_t = tf.zeros_like(h4_t)
        full = self.generate_full(Z_, h2_t, h3_t, h4_t)
        # complete = self.complete(full)
        """
        h1 = tf.nn.relu(
            batchnormalize(
                tf.matmul(Z_, self.gen_y_W1),
                g=self.gen_y_bn_g1,
                b=self.gen_y_bn_b1))
        h1 = tf.reshape(h1, [
            visual_size, self.start_vox_size[0], self.start_vox_size[1],
            self.start_vox_size[2], self.dim_W1
        ])

        vox_size_l2 = self.start_vox_size * 2
        output_shape_l2 = [
            visual_size, vox_size_l2[0], vox_size_l2[1], vox_size_l2[2],
            self.dim_W2
        ]
        h2 = tf.nn.conv3d_transpose(
            h1,
            self.gen_y_W2,
            output_shape=output_shape_l2,
            strides=self.stride)
        h2 = tf.nn.relu(
            batchnormalize(
                h2,
                g=self.gen_y_bn_g2,
                b=self.gen_y_bn_b2,
                batch_size=self.batch_size))

        h2_ = tf.nn.conv3d_transpose(
            h1,
            self.gen_x_W2,
            output_shape=output_shape_l2,
            strides=self.stride)
        h2_ = tf.nn.relu(
            batchnormalize(
                h2_,
                g=self.gen_x_bn_g2,
                b=self.gen_x_bn_b2,
                batch_size=self.batch_size))

        vox_size_l3 = self.start_vox_size * 4
        output_shape_l3 = [
            visual_size, vox_size_l3[0], vox_size_l3[1], vox_size_l3[2],
            self.dim_W3
        ]
        h3 = tf.nn.conv3d_transpose(
            tf.concat([h2, h2_], -1),
            self.gen_y_W3,
            output_shape=output_shape_l3,
            strides=self.stride)
        h3 = tf.nn.relu(
            batchnormalize(
                h3,
                g=self.gen_x_bn_g3,
                b=self.gen_x_bn_b3,
                batch_size=self.batch_size))

        h3_ = tf.nn.conv3d_transpose(
            h2_,
            self.gen_x_W3,
            output_shape=output_shape_l3,
            strides=self.stride)
        h3_ = tf.nn.relu(
            batchnormalize(
                h3_,
                g=self.gen_y_bn_g3,
                b=self.gen_y_bn_b3,
                batch_size=self.batch_size))

        vox_size_l4 = self.start_vox_size * 8
        output_shape_l4 = [
            visual_size, vox_size_l4[0], vox_size_l4[1], vox_size_l4[2],
            self.dim_W4
        ]
        h4 = tf.nn.conv3d_transpose(
            tf.concat([h3, h3_], -1),
            self.gen_y_W4,
            output_shape=output_shape_l4,
            strides=self.stride)
        h4 = tf.nn.relu(
            batchnormalize(
                h4,
                g=self.gen_y_bn_g4,
                b=self.gen_y_bn_b4,
                batch_size=self.batch_size))

        h4_ = tf.nn.conv3d_transpose(
            h3_,
            self.gen_x_W4,
            output_shape=output_shape_l4,
            strides=self.stride)
        h4_ = tf.nn.relu(
            batchnormalize(
                h4_,
                g=self.gen_x_bn_g4,
                b=self.gen_x_bn_b4,
                batch_size=self.batch_size))

        vox_size_l5 = self.start_vox_size * 16
        output_shape_l5 = [
            visual_size, vox_size_l5[0], vox_size_l5[1], vox_size_l5[2],
            self.dim_W5
        ]
        h5 = tf.nn.conv3d_transpose(
            tf.concat([h4, h4_], -1),
            self.gen_y_W5,
            output_shape=output_shape_l5,
            strides=self.stride)

        x = softmax(h5, visual_size, self.vox_shape)
        """
        return Z, full, part
