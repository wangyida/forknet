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
                 vox_shape=[80, 48, 80, 13],
                 tsdf_shape=[80, 48, 80, 3],
                 dim_z=16,
                 dim=[512, 256, 192, 64, 13],
                 start_vox_size=[5, 3, 5],
                 kernel=[[3, 3, 3, 3, 3], [3, 3, 3, 3, 3], [3, 3, 3, 3, 3]],
                 stride=[1, 2, 2, 2, 1],
                 dilations=[1, 1, 1, 1, 1],
                 dim_code=512,
                 generative=True):

        self.batch_size = batch_size
        self.vox_shape = vox_shape
        self.tsdf_shape = tsdf_shape
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
                self.dim_W2
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
                self.dim_W3
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
                self.dim_W4
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
                self.tsdf_shape[-1], self.dim_W4
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
                self.kernel5[0], self.kernel5[1], self.kernel5[2], self.dim_W5,
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
                self.tsdf_shape[-1], self.dim_W4
            ],
                             stddev=0.02),
            name='gen_x_W5')
        self.gen_x_bn_g5 = tf.Variable(
            tf.random_normal([self.tsdf_shape[-1]], mean=1.0, stddev=0.02),
            name='gen_x_bn_g5')
        self.gen_x_bn_b5 = tf.Variable(
            tf.zeros([self.tsdf_shape[-1]]), name='gen_x_bn_b5')

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
                self.tsdf_shape[-1], self.dim_W4
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

        com_seg_gt_ = tf.placeholder(tf.int32, [
            self.batch_size, self.vox_shape[0], self.vox_shape[1],
            self.vox_shape[2]
        ])
        com_seg_gt = tf.one_hot(com_seg_gt_, self.n_class)
        com_seg_gt = tf.cast(com_seg_gt, tf.float32)

        # tsdf--start
        tsdf_gt_ = tf.placeholder(tf.int32, [
            self.batch_size, self.vox_shape[0], self.vox_shape[1],
            self.vox_shape[2]
        ])
        tsdf_gt = tf.one_hot(tsdf_gt_, self.tsdf_shape[-1])
        tsdf_gt = tf.cast(tsdf_gt, tf.float32)
        # tsdf--end

        Z = tf.placeholder(tf.float32, [
            self.batch_size, self.start_vox_size[0], self.start_vox_size[1],
            self.start_vox_size[2], self.dim_z
        ])

        # encode from tsdf and vox
        Z_encode_tsdf, sigma_tsdf = self.encoder_tsdf(tsdf_gt)
        tsdf_vae_dec = self.generate_tsdf(Z_encode_tsdf)

        # reconstructed surface
        # surface_dec = tf.placeholder(tf.float32, self.tsdf_shape)
        # surface_dec = self.surface_net(tsdf_vae_dec + tsdf_gt)

        # reconstruction
        """
        com_gt = tf.stack([
            com_seg_gt[:, :, :, :, 0],
            tf.reduce_sum(com_seg_gt[:, :, :, :, 1:], 4)
        ], 4)
        """

        batch_mean_com_seg_gt = tf.reduce_mean(com_seg_gt, [0, 1, 2, 3])
        ones = tf.ones_like(batch_mean_com_seg_gt)
        inverse = tf.div(ones, tf.add(batch_mean_com_seg_gt, ones))
        weight_com_seg = inverse * tf.div(1., tf.reduce_sum(inverse))

        batch_mean_tsdf_gt = tf.reduce_mean(tsdf_gt, [0, 1, 2, 3])
        ones = tf.ones_like(batch_mean_tsdf_gt)
        inverse = tf.div(ones, tf.add(batch_mean_tsdf_gt, ones))
        weight_tsdf = inverse * tf.div(1., tf.reduce_sum(inverse))

        Z_encode_seg, sigma_seg = self.encoder_vox(com_seg_gt)
        com_seg_vae_dec = self.generate_vox(Z_encode_seg)

        # cross generators
        com_seg_gen_dec = self.generate_vox(Z_encode_tsdf)
        tsdf_gen_dec = self.generate_tsdf(Z_encode_seg)

        # encode again from the bridge
        com_seg_gen_dec = tf.argmax(com_seg_gen_dec, axis=4, output_type=tf.int32)
        com_seg_gen_dec = tf.one_hot(com_seg_gen_dec, self.n_class)
        com_seg_gen_dec = tf.cast(com_seg_gen_dec, tf.float32)
        tsdf_gen_dec = tf.argmax(tsdf_gen_dec, axis=4, output_type=tf.int32)
        tsdf_gen_dec = tf.one_hot(tsdf_gen_dec, self.tsdf_shape[-1])
        tsdf_gen_dec = tf.cast(tsdf_gen_dec, tf.float32)
        Z_encode_tsdf_seg, sigma_tsdf_seg = self.encoder_vox(com_seg_gen_dec)
        Z_encode_seg_tsdf, sigma_seg_tsdf = self.encoder_tsdf(tsdf_gen_dec)

        com_seg_cc_dec = self.generate_vox(Z_encode_seg_tsdf)
        tsdf_cc_dec = self.generate_tsdf(Z_encode_tsdf_seg)

        # code_discriminator
        h_code_tsdf_gt = self.code_discriminator_x(Z)
        h_code_com_seg_gt = self.code_discriminator_y(Z)
        h_code_encode_tsdf = self.code_discriminator_x(Z_encode_tsdf)
        h_code_encode_vox = self.code_discriminator_y(Z_encode_seg)
        h_code_encode_tsdf_seg = self.code_discriminator_y(Z_encode_tsdf_seg)
        h_code_encode_seg_tsdf = self.code_discriminator_x(Z_encode_seg_tsdf)
        """
        recons_loss_surface = tf.reduce_mean(
            tf.reduce_sum(
                -tf.reduce_sum(
                    self.lamda_gamma * tsdf_gt * tf.log(1e-6 + surface_dec) +
                    (1 - self.lamda_gamma) *
                    (1 - tsdf_gt) * tf.log(1e-6 + 1 - surface_dec), [1, 2, 3])
                * weight_tsdf, 1))
        """

        code_encode_loss = tf.reduce_mean(
            tf.reduce_sum(
                tf.nn.sigmoid_cross_entropy_with_logits(
                    logits=h_code_encode_tsdf,
                    labels=tf.ones_like(h_code_encode_tsdf)), [1]))
        code_encode_loss += tf.reduce_mean(
            tf.reduce_sum(
                tf.nn.sigmoid_cross_entropy_with_logits(
                    logits=h_code_encode_vox,
                    labels=tf.ones_like(h_code_encode_vox)), [1]))
        code_encode_loss += tf.reduce_mean(
            tf.reduce_sum(
                tf.nn.sigmoid_cross_entropy_with_logits(
                    logits=h_code_encode_tsdf_seg,
                    labels=tf.ones_like(h_code_encode_tsdf_seg)), [1]))
        code_encode_loss += tf.reduce_mean(
            tf.reduce_sum(
                tf.nn.sigmoid_cross_entropy_with_logits(
                    logits=h_code_encode_seg_tsdf,
                    labels=tf.ones_like(h_code_encode_seg_tsdf)), [1]))

        code_discrim_loss = tf.reduce_mean(
            tf.reduce_sum(
                tf.nn.sigmoid_cross_entropy_with_logits(
                    logits=h_code_tsdf_gt,
                    labels=tf.ones_like(h_code_tsdf_gt)),
                [1])) + tf.reduce_mean(
                    tf.reduce_sum(
                        tf.nn.sigmoid_cross_entropy_with_logits(
                            logits=h_code_encode_tsdf,
                            labels=tf.zeros_like(h_code_encode_tsdf)), [1]))
        code_discrim_loss += tf.reduce_mean(
            tf.reduce_sum(
                tf.nn.sigmoid_cross_entropy_with_logits(
                    logits=h_code_com_seg_gt, labels=tf.ones_like(h_code_com_seg_gt)),
                [1])) + tf.reduce_mean(
                    tf.reduce_sum(
                        tf.nn.sigmoid_cross_entropy_with_logits(
                            logits=h_code_encode_vox,
                            labels=tf.zeros_like(h_code_encode_vox)), [1]))
        code_discrim_loss += tf.reduce_mean(
            tf.reduce_sum(
                tf.nn.sigmoid_cross_entropy_with_logits(
                    logits=h_code_com_seg_gt, labels=tf.ones_like(h_code_com_seg_gt)),
                [1])) + tf.reduce_mean(
                    tf.reduce_sum(
                        tf.nn.sigmoid_cross_entropy_with_logits(
                            logits=h_code_encode_tsdf_seg,
                            labels=tf.zeros_like(h_code_encode_tsdf_seg)),
                        [1]))
        code_discrim_loss += tf.reduce_mean(
            tf.reduce_sum(
                tf.nn.sigmoid_cross_entropy_with_logits(
                    logits=h_code_tsdf_gt,
                    labels=tf.ones_like(h_code_tsdf_gt)),
                [1])) + tf.reduce_mean(
                    tf.reduce_sum(
                        tf.nn.sigmoid_cross_entropy_with_logits(
                            logits=h_code_encode_seg_tsdf,
                            labels=tf.zeros_like(h_code_encode_seg_tsdf)),
                        [1]))

        # Completing from depth and semantic depth
        recons_vae_loss = tf.reduce_mean(
            tf.reduce_sum(
                -tf.reduce_sum(
                    self.lamda_gamma * com_seg_gt *
                    tf.log(1e-6 + com_seg_vae_dec) + (1 - self.lamda_gamma) *
                    (1 - com_seg_gt) * tf.log(1e-6 + 1 - com_seg_vae_dec),
                    [1, 2, 3]) * weight_com_seg, 1))

        recons_vae_loss += tf.reduce_mean(
            tf.reduce_sum(
                -tf.reduce_sum(
                    self.lamda_gamma * tsdf_gt * tf.log(1e-6 + tsdf_vae_dec) +
                    (1 - self.lamda_gamma) *
                    (1 - tsdf_gt) * tf.log(1e-6 + 1 - tsdf_vae_dec), [1, 2, 3])
                * weight_tsdf, 1))
        # latent consistency
        """
        recons_vae_loss += tf.reduce_mean(
            tf.reduce_sum(
                tf.squared_difference(Z_encode_tsdf, Z_encode_seg_tsdf),
                [1, 2, 3, 4]))
        recons_vae_loss += tf.reduce_mean(
            tf.reduce_sum(
                tf.squared_difference(Z_encode_tsdf, Z_encode_tsdf_seg),
                [1, 2, 3, 4]))
        recons_vae_loss += tf.reduce_mean(
            tf.reduce_sum(
                tf.squared_difference(Z_encode_seg, Z_encode_tsdf_seg),
                [1, 2, 3, 4]))
        recons_vae_loss += tf.reduce_mean(
            tf.reduce_sum(
                tf.squared_difference(Z_encode_seg, Z_encode_seg_tsdf),
                [1, 2, 3, 4]))
        """
        # latent consistency

        # Cycle consistencies
        recons_cc_loss = tf.reduce_mean(
            tf.reduce_sum(
                -tf.reduce_sum(
                    self.lamda_gamma * com_seg_gt *
                    tf.log(1e-6 + com_seg_cc_dec) + (1 - self.lamda_gamma) *
                    (1 - com_seg_gt) * tf.log(1e-6 + 1 - com_seg_cc_dec),
                    [1, 2, 3]) * weight_com_seg, 1))

        recons_cc_loss += tf.reduce_mean(
            tf.reduce_sum(
                -tf.reduce_sum(
                    self.lamda_gamma * tsdf_gt * tf.log(1e-6 + tsdf_cc_dec) +
                    (1 - self.lamda_gamma) *
                    (1 - tsdf_gt) * tf.log(1e-6 + 1 - tsdf_cc_dec), [1, 2, 3]) *
                weight_tsdf, 1))
        # latent consistency
        """
        recons_cc_loss += tf.reduce_mean(
            tf.reduce_sum(
                tf.squared_difference(Z_encode_tsdf_seg, Z_encode_seg_tsdf),
                [1, 2, 3, 4]))
        """
        # latent consistency
        # SUPERVISED (paired data)
        recons_gen_loss = tf.reduce_mean(
            tf.reduce_sum(
                -tf.reduce_sum(
                    self.lamda_gamma * com_seg_gt *
                    tf.log(1e-6 + com_seg_gen_dec) + (1 - self.lamda_gamma) *
                    (1 - com_seg_gt) * tf.log(1e-6 + 1 - com_seg_gen_dec),
                    [1, 2, 3]) * weight_com_seg, 1))

        # from scene, the observed surface can also be produced
        recons_gen_loss += tf.reduce_mean(
            tf.reduce_sum(
                -tf.reduce_sum(
                    self.lamda_gamma * tsdf_gt * tf.log(1e-6 + tsdf_gen_dec) +
                    (1 - self.lamda_gamma) *
                    (1 - tsdf_gt) * tf.log(1e-6 + 1 - tsdf_gen_dec), [1, 2, 3])
                * weight_tsdf, 1))
        # latent consistency
        recons_gen_loss += tf.reduce_mean(
            tf.reduce_sum(
                tf.squared_difference(Z_encode_tsdf, Z_encode_seg),
                [1, 2, 3, 4]))
        # latent consistency

        # GAN_generate
        com_seg_gen = self.generate_vox(Z)
        tsdf_gen = self.generate_tsdf(Z)

        # surface_gen = self.surface_net(tsdf_gen + tsdf_gt)

        h_com_seg_gt = self.discriminate_vox(com_seg_gt)
        h_com_seg_gen = self.discriminate_vox(com_seg_gen)
        h_com_seg_gen_dec = self.discriminate_vox(com_seg_gen_dec)

        h_tsdf_gt = self.discriminate_tsdf(tsdf_gt)
        h_tsdf_gen = self.discriminate_tsdf(tsdf_gen)
        h_tsdf_gen_dec = self.discriminate_tsdf(tsdf_gen_dec)

        # Standard_GAN_Loss
        discrim_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(
                logits=h_com_seg_gt,
                labels=tf.ones_like(h_com_seg_gt))) + tf.reduce_mean(
                    tf.nn.sigmoid_cross_entropy_with_logits(
                        logits=h_com_seg_gen_dec,
                        labels=tf.zeros_like(h_com_seg_gen_dec)))

        discrim_loss += tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(
                logits=h_tsdf_gt,
                labels=tf.ones_like(h_tsdf_gt))) + tf.reduce_mean(
                    tf.nn.sigmoid_cross_entropy_with_logits(
                        logits=h_tsdf_gen_dec,
                        labels=tf.zeros_like(h_tsdf_gen_dec)))

        gen_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(
                logits=h_com_seg_gen_dec,
                labels=tf.ones_like(h_com_seg_gen_dec)))

        gen_loss += tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(
                logits=h_tsdf_gen_dec, labels=tf.ones_like(h_tsdf_gen_dec)))

        if self.generative is True:
            discrim_loss += tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(
                    logits=h_com_seg_gen, labels=tf.zeros_like(h_com_seg_gen)))
            discrim_loss += tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(
                    logits=h_tsdf_gen, labels=tf.zeros_like(h_tsdf_gen)))
            gen_loss += tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(
                    logits=h_com_seg_gen, labels=tf.ones_like(h_com_seg_gen)))
            gen_loss += tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(
                    logits=h_tsdf_gen, labels=tf.ones_like(h_tsdf_gen)))

        # Cost
        cost_enc = code_encode_loss + self.lamda_recons * (
            recons_vae_loss + recons_cc_loss + recons_gen_loss
        )  # + recons_loss_surface)
        cost_gen = self.lamda_recons * (
            recons_vae_loss + recons_cc_loss +
            recons_gen_loss) + 10 * gen_loss  # + recons_loss_surface)
        cost_discrim = 10 * discrim_loss
        cost_code = 10 * code_discrim_loss

        tf.summary.scalar("recons_vae_loss", tf.reduce_mean(recons_vae_loss))
        tf.summary.scalar("recons_cc_loss", tf.reduce_mean(recons_cc_loss))
        tf.summary.scalar("gen_loss", tf.reduce_mean(gen_loss))
        tf.summary.scalar("discrim_loss", tf.reduce_mean(discrim_loss))
        tf.summary.scalar("code_encode_loss", tf.reduce_mean(code_encode_loss))
        tf.summary.scalar("code_discrim_loss",
                          tf.reduce_mean(code_discrim_loss))

        summary_op = tf.summary.merge_all()

        return Z, Z_encode_tsdf, Z_encode_seg, com_seg_gt_, com_seg_gen, com_seg_gen_dec, com_seg_vae_dec, com_seg_cc_dec,\
        recons_vae_loss, recons_cc_loss, recons_gen_loss, code_encode_loss, gen_loss, discrim_loss,\
        cost_enc, cost_code, cost_gen, cost_discrim, summary_op,\
        tsdf_gt_, tsdf_gen, tsdf_gen_dec, tsdf_vae_dec, tsdf_cc_dec

    def encoder_tsdf(self, vox):

        h1 = lrelu(
            tf.nn.conv3d(
                vox,
                self.encode_x_W1,
                strides=self.stride,
                dilations=self.dilations,
                padding='SAME'))

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
            batchnormalize(
                tf.nn.conv3d(
                    base_9,
                    self.encode_x_W2,
                    strides=self.stride,
                    dilations=self.dilations,
                    padding='SAME'),
                g=self.encode_x_bn_g2,
                b=self.encode_x_bn_b2,
                batch_size=self.batch_size))
        h3 = lrelu(
            batchnormalize(
                tf.nn.conv3d(
                    h2,
                    self.encode_x_W3,
                    strides=self.stride,
                    dilations=self.dilations,
                    padding='SAME'),
                g=self.encode_x_bn_g3,
                b=self.encode_x_bn_b3,
                batch_size=self.batch_size))
        h4 = lrelu(
            batchnormalize(
                tf.nn.conv3d(
                    h3,
                    self.encode_y_W4,
                    strides=self.stride,
                    dilations=self.dilations,
                    padding='SAME'),
                g=self.encode_y_bn_g4,
                b=self.encode_y_bn_b4,
                batch_size=self.batch_size))
        h5 = tf.nn.conv3d(
            h4, self.encode_y_W5, strides=[1, 1, 1, 1, 1], padding='SAME')
        h5_sigma = tf.nn.conv3d(
            h4,
            self.encode_y_W5_sigma,
            strides=[1, 1, 1, 1, 1],
            padding='SAME')

        return h5, h5_sigma

    def discriminate_vox(self, vox):

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

    def encoder_vox(self, vox):

        h1 = lrelu(
            tf.nn.conv3d(
                vox,
                self.encode_y_W1,
                strides=self.stride,
                dilations=self.dilations,
                padding='SAME'))

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
            batchnormalize(
                tf.nn.conv3d(
                    base_9,
                    self.encode_y_W2,
                    strides=self.stride,
                    dilations=self.dilations,
                    padding='SAME'),
                g=self.encode_y_bn_g2,
                b=self.encode_y_bn_b2,
                batch_size=self.batch_size))
        h3 = lrelu(
            batchnormalize(
                tf.nn.conv3d(
                    h2,
                    self.encode_y_W3,
                    strides=self.stride,
                    dilations=self.dilations,
                    padding='SAME'),
                g=self.encode_y_bn_g3,
                b=self.encode_y_bn_b3,
                batch_size=self.batch_size))
        h4 = lrelu(
            batchnormalize(
                tf.nn.conv3d(
                    h3,
                    self.encode_y_W4,
                    strides=self.stride,
                    dilations=self.dilations,
                    padding='SAME'),
                g=self.encode_y_bn_g4,
                b=self.encode_y_bn_b4,
                batch_size=self.batch_size))
        h5 = tf.nn.conv3d(
            h4, self.encode_y_W5, strides=[1, 1, 1, 1, 1], padding='SAME')
        h5_sigma = tf.nn.conv3d(
            h4,
            self.encode_y_W5_sigma,
            strides=[1, 1, 1, 1, 1],
            padding='SAME')

        return h5, h5_sigma

    def discriminate_tsdf(self, vox):

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

    def generate_vox(self, Z):

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
            h2,
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
            h3,
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
            h4,
            self.gen_y_W5,
            output_shape=output_shape_l5,
            strides=self.stride)

        x = softmax(h5, self.batch_size, self.vox_shape)
        return x

    def generate_tsdf(self, Z):

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
            self.tsdf_shape[-1]
        ]
        h5 = tf.nn.conv3d_transpose(
            h4,
            self.gen_x_W5,
            output_shape=output_shape_l5,
            strides=self.stride)

        x = softmax(h5, self.batch_size, self.tsdf_shape)
        return x

    def surface_net(self, vox):
        base = tf.nn.relu(
            tf.nn.conv3d(
                vox,
                self.gen_surface_W1,
                strides=[1, 1, 1, 1, 1],
                padding='SAME'))

        #res1
        res1_1 = tf.nn.relu(
            tf.nn.conv3d(
                base,
                self.gen_surface_res1_W1,
                strides=[1, 1, 1, 1, 1],
                padding='SAME'))
        res1_2 = tf.nn.conv3d(
            res1_1,
            self.gen_surface_res1_W2,
            strides=[1, 1, 1, 1, 1],
            padding='SAME')

        res1 = tf.nn.relu(tf.add(base, res1_2))

        #res2
        res2_1 = tf.nn.relu(
            tf.nn.conv3d(
                res1,
                self.gen_surface_res2_W1,
                strides=[1, 1, 1, 1, 1],
                padding='SAME'))
        res2_2 = tf.nn.conv3d(
            res2_1,
            self.gen_surface_res2_W2,
            strides=[1, 1, 1, 1, 1],
            padding='SAME')

        res2 = tf.nn.relu(tf.add(res1, res2_2))

        #res3
        res3_1 = tf.nn.relu(
            tf.nn.conv3d(
                res2,
                self.gen_surface_res3_W1,
                strides=[1, 1, 1, 1, 1],
                padding='SAME'))
        res3_2 = tf.nn.conv3d(
            res3_1,
            self.gen_surface_res3_W2,
            strides=[1, 1, 1, 1, 1],
            padding='SAME')

        res3 = tf.nn.relu(tf.add(res2, res3_2))

        #res4
        res4_1 = tf.nn.relu(
            tf.nn.conv3d(
                res3,
                self.gen_surface_res4_W1,
                strides=[1, 1, 1, 1, 1],
                padding='SAME'))
        res4_2 = tf.nn.conv3d(
            res4_1,
            self.gen_surface_res4_W2,
            strides=[1, 1, 1, 1, 1],
            padding='SAME')

        res4 = tf.nn.relu(tf.add(res3, res4_2))

        out = tf.nn.conv3d(
            res4, self.gen_surface_W2, strides=[1, 1, 1, 1, 1], padding='SAME')
        x_gen_surface = softmax(out, self.batch_size, self.tsdf_shape)

        return x_gen_surface

    def samples_generator(self, visual_size):

        Z = tf.placeholder(tf.float32, [
            visual_size, self.start_vox_size[0], self.start_vox_size[1],
            self.start_vox_size[2], self.dim_z
        ])

        Z_ = tf.reshape(Z, [visual_size, -1])
        h1 = tf.nn.relu(
            batchnormalize(
                tf.matmul(Z_, self.gen_x_W1),
                g=self.gen_x_bn_g1,
                b=self.gen_x_bn_b1))
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
            visual_size, vox_size_l3[0], vox_size_l3[1], vox_size_l3[2],
            self.dim_W3
        ]
        h3 = tf.nn.conv3d_transpose(
            h2,
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
            visual_size, vox_size_l4[0], vox_size_l4[1], vox_size_l4[2],
            self.dim_W4
        ]
        h4 = tf.nn.conv3d_transpose(
            h3,
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
            visual_size, vox_size_l5[0], vox_size_l5[1], vox_size_l5[2],
            self.dim_W5
        ]
        h5 = tf.nn.conv3d_transpose(
            h4,
            self.gen_y_W5,
            output_shape=output_shape_l5,
            strides=self.stride)

        x = softmax(h5, visual_size, self.vox_shape)
        return Z, x
