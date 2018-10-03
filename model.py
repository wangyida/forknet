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

    # depth--start
    elif X.get_shape().ndims == 4:
        if batch_size == 1:
            mean = 0
            std = 1 - eps
        else:
            mean = tf.reduce_mean(X, [0, 1, 2])
            std = tf.reduce_mean(tf.square(X - mean), [0, 1, 2])
        X = (X - mean) / tf.sqrt(std + eps)

        if g is not None and b is not None:
            g = tf.reshape(g, [1, 1, 1, -1])
            b = tf.reshape(b, [1, 1, 1, -1])
            X = X * g + b
    # depth--end

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


class FCR_aGAN():
    def __init__(self,
                 batch_size=16,
                 vox_shape=[80, 48, 80, 12],
                 tsdf_shape=[80, 48, 80, 3],
                 dim_z=16,
                 dim=[512, 256, 128, 64, 12],
                 start_vox_size=[5, 3, 5],
                 kernel=[[3, 3, 3, 3, 3], [3, 3, 3, 3, 3], [3, 3, 3, 3, 3]],
                 stride=[1, 2, 2, 2, 1],
                 dilations=[1, 1, 1, 1, 1],
                 dim_code=750,
                 refine_ch=32,
                 refine_kernel=3,
                 refiner="resnet"):

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
        self.refine_ch = refine_ch
        self.refine_kernel = refine_kernel
        self.refiner = refiner

        # parameters of generator
        self.gen_W1 = tf.Variable(
            tf.random_normal([
                self.dim_z * self.start_vox_size[0] * self.start_vox_size[1] *
                self.start_vox_size[2], self.dim_W1 * self.start_vox_size[0] *
                self.start_vox_size[1] * self.start_vox_size[2]
            ],
                             stddev=0.02),
            name='gen_W1')
        self.gen_bn_g1 = tf.Variable(
            tf.random_normal([
                self.dim_W1 * self.start_vox_size[0] * self.start_vox_size[1] *
                self.start_vox_size[2]
            ],
                             mean=1.0,
                             stddev=0.02),
            name='gen_bn_g1')
        self.gen_bn_b1 = tf.Variable(
            tf.zeros([
                self.dim_W1 * self.start_vox_size[0] * self.start_vox_size[1] *
                self.start_vox_size[2]
            ]),
            name='gen_bn_b1')

        self.gen_W2 = tf.Variable(
            tf.random_normal([
                self.kernel2[0], self.kernel2[1], self.kernel2[2], self.dim_W2,
                self.dim_W1
            ],
                             stddev=0.02),
            name='gen_W2')
        self.gen_bn_g2 = tf.Variable(
            tf.random_normal([self.dim_W2], mean=1.0, stddev=0.02),
            name='gen_bn_g2')
        self.gen_bn_b2 = tf.Variable(tf.zeros([self.dim_W2]), name='gen_bn_b2')

        self.gen_W3 = tf.Variable(
            tf.random_normal([
                self.kernel3[0], self.kernel3[1], self.kernel3[2], self.dim_W3,
                self.dim_W2
            ],
                             stddev=0.02),
            name='gen_W3')
        self.gen_bn_g3 = tf.Variable(
            tf.random_normal([self.dim_W3], mean=1.0, stddev=0.02),
            name='gen_bn_g3')
        self.gen_bn_b3 = tf.Variable(tf.zeros([self.dim_W3]), name='gen_bn_b3')

        self.gen_W4 = tf.Variable(
            tf.random_normal([
                self.kernel4[0], self.kernel4[1], self.kernel4[2], self.dim_W4,
                self.dim_W3
            ],
                             stddev=0.02),
            name='gen_W4')
        self.gen_bn_g4 = tf.Variable(
            tf.random_normal([self.dim_W4], mean=1.0, stddev=0.02),
            name='gen_bn_g4')
        self.gen_bn_b4 = tf.Variable(tf.zeros([self.dim_W4]), name='gen_bn_b4')

        self.gen_W5 = tf.Variable(
            tf.random_normal([
                self.kernel5[0], self.kernel5[1], self.kernel5[2], self.dim_W5,
                self.dim_W4
            ],
                             stddev=0.02),
            name='gen_W5')
        self.gen_bn_g5 = tf.Variable(
            tf.random_normal([self.dim_W5], mean=1.0, stddev=0.02),
            name='gen_bn_g5')
        self.gen_bn_b5 = tf.Variable(tf.zeros([self.dim_W5]), name='gen_bn_b5')

        # parameters of encoder
        self.encode_W1 = tf.Variable(
            tf.random_normal([
                self.kernel5[0], self.kernel5[1], self.kernel5[2],
                self.tsdf_shape[-1], self.dim_W4
            ],
                             stddev=0.02),
            name='encode_W1')
        self.encode_bn_g1 = tf.Variable(
            tf.random_normal([self.dim_W4], mean=1.0, stddev=0.02),
            name='encode_bn_g1')
        self.encode_bn_b1 = tf.Variable(
            tf.zeros([self.dim_W4]), name='encode_bn_b1')

        self.encode_W2 = tf.Variable(
            tf.random_normal([
                self.kernel4[0], self.kernel4[1], self.kernel4[2], self.dim_W4,
                self.dim_W3
            ],
                             stddev=0.02),
            name='encode_W2')
        self.encode_bn_g2 = tf.Variable(
            tf.random_normal([self.dim_W3], mean=1.0, stddev=0.02),
            name='encode_bn_g2')
        self.encode_bn_b2 = tf.Variable(
            tf.zeros([self.dim_W3]), name='encode_bn_b2')

        self.encode_W3 = tf.Variable(
            tf.random_normal([
                self.kernel3[0], self.kernel3[1], self.kernel3[2], self.dim_W3,
                self.dim_W2
            ],
                             stddev=0.02),
            name='encode_W3')
        self.encode_bn_g3 = tf.Variable(
            tf.random_normal([self.dim_W2], mean=1.0, stddev=0.02),
            name='encode_bn_g3')
        self.encode_bn_b3 = tf.Variable(
            tf.zeros([self.dim_W2]), name='encode_bn_b3')

        self.encode_W4 = tf.Variable(
            tf.random_normal([
                self.kernel2[0], self.kernel2[1], self.kernel2[2], self.dim_W2,
                self.dim_W1
            ],
                             stddev=0.02),
            name='encode_W4')
        self.encode_bn_g4 = tf.Variable(
            tf.random_normal([self.dim_W1], mean=1.0, stddev=0.02),
            name='encode_bn_g4')
        self.encode_bn_b4 = tf.Variable(
            tf.zeros([self.dim_W1]), name='encode_bn_b4')

        self.encode_W5 = tf.Variable(
            tf.random_normal([1, 1, 1, self.dim_W1, self.dim_z], stddev=0.02),
            name='encode_W5')
        self.encode_W5_sigma = tf.Variable(
            tf.random_normal([1, 1, 1, self.dim_W1, self.dim_z], stddev=0.02),
            name='encode_W5_sigma')

        self.discrim_W1 = tf.Variable(
            tf.random_normal([
                self.kernel5[0], self.kernel5[1], self.kernel5[2], self.dim_W5,
                self.dim_W4
            ],
                             stddev=0.02),
            name='discrim_vox_W1')
        self.discrim_bn_g1 = tf.Variable(
            tf.random_normal([1], mean=1.0, stddev=0.02),
            name='discrim_vox_bn_g1')
        self.discrim_bn_b1 = tf.Variable(
            tf.zeros([1]), name='discrim_vox_bn_b1')

        # parameters of discriminator
        self.discrim_W2 = tf.Variable(
            tf.random_normal([
                self.kernel4[0], self.kernel4[1], self.kernel4[2], self.dim_W4,
                self.dim_W3
            ],
                             stddev=0.02),
            name='discrim_vox_W2')
        self.discrim_bn_g2 = tf.Variable(
            tf.random_normal([1], mean=1.0, stddev=0.02),
            name='discrim_vox_bn_g2')
        self.discrim_bn_b2 = tf.Variable(
            tf.zeros([1]), name='discrim_vox_bn_b2')

        self.discrim_W3 = tf.Variable(
            tf.random_normal([
                self.kernel3[0], self.kernel3[1], self.kernel3[2], self.dim_W3,
                self.dim_W2
            ],
                             stddev=0.02),
            name='discrim_vox_W3')
        self.discrim_bn_g3 = tf.Variable(
            tf.random_normal([1], mean=1.0, stddev=0.02),
            name='discrim_vox_bn_g3')
        self.discrim_bn_b3 = tf.Variable(
            tf.zeros([1]), name='discrim_vox_bn_b3')

        self.discrim_W4 = tf.Variable(
            tf.random_normal([
                self.kernel2[0], self.kernel2[1], self.kernel2[2], self.dim_W2,
                self.dim_W1
            ],
                             stddev=0.02),
            name='discrim_vox_W4')
        self.discrim_bn_g4 = tf.Variable(
            tf.random_normal([1], mean=1.0, stddev=0.02),
            name='discrim_vox_bn_g4')
        self.discrim_bn_b4 = tf.Variable(
            tf.zeros([1]), name='discrim_vox_bn_b4')

        # patch GAN
        self.discrim_W5 = tf.Variable(
            tf.random_normal([1, 1, 1, self.dim_W1, self.dim_z], stddev=0.02),
            name='discrim_vox_W5')
        """ original GAN
        self.discrim_W5 = tf.Variable(
            tf.random_normal([
                self.start_vox_size[0] * self.start_vox_size[1] *
                self.start_vox_size[2] * self.dim_W1, 1
            ],
                             stddev=0.02),
            name='discrim_vox_W5')
        """

        # parameters of codes discriminator
        self.cod_W1 = tf.Variable(
            tf.random_normal([
                self.dim_z * self.start_vox_size[0] * self.start_vox_size[1] *
                self.start_vox_size[2], self.dim_code
            ],
                             stddev=0.02),
            name='cod_W1')
        self.cod_bn_g1 = tf.Variable(
            tf.random_normal([dim_code], mean=1.0, stddev=0.02),
            name='cod_bn_g1')
        self.cod_bn_b1 = tf.Variable(tf.zeros([dim_code]), name='cod_bn_b1')

        self.cod_W2 = tf.Variable(
            tf.random_normal([dim_code, dim_code], stddev=0.02), name='cod_W2')
        self.cod_bn_g2 = tf.Variable(
            tf.random_normal([dim_code], mean=1.0, stddev=0.02),
            name='cod_bn_g2')
        self.cod_bn_b2 = tf.Variable(tf.zeros([dim_code]), name='cod_bn_b2')

        self.cod_W3 = tf.Variable(
            tf.random_normal([dim_code, 1], stddev=0.02), name='cod_W3')

        # parameters of refiner
        self.refine_W1 = tf.Variable(
            tf.random_normal([
                self.refine_kernel, self.refine_kernel, self.refine_kernel,
                self.dim_W5, self.refine_ch
            ],
                             stddev=0.02),
            name='refine_W1')
        self.refine_res1_W1 = tf.Variable(
            tf.random_normal([
                self.refine_kernel, self.refine_kernel, self.refine_kernel,
                self.refine_ch, self.refine_ch
            ],
                             stddev=0.02),
            name='refine__res1_W1')
        self.refine_res1_W2 = tf.Variable(
            tf.random_normal([
                self.refine_kernel, self.refine_kernel, self.refine_kernel,
                self.refine_ch, self.refine_ch
            ],
                             stddev=0.02),
            name='refine__res1_W2')

        self.refine_res2_W1 = tf.Variable(
            tf.random_normal([
                self.refine_kernel, self.refine_kernel, self.refine_kernel,
                self.refine_ch, self.refine_ch
            ],
                             stddev=0.02),
            name='refine__res2_W1')
        self.refine_res2_W2 = tf.Variable(
            tf.random_normal([
                self.refine_kernel, self.refine_kernel, self.refine_kernel,
                self.refine_ch, self.refine_ch
            ],
                             stddev=0.02),
            name='refine__res2_W2')

        self.refine_res3_W1 = tf.Variable(
            tf.random_normal([
                self.refine_kernel, self.refine_kernel, self.refine_kernel,
                self.refine_ch, self.refine_ch
            ],
                             stddev=0.02),
            name='refine__res3_W1')
        self.refine_res3_W2 = tf.Variable(
            tf.random_normal([
                self.refine_kernel, self.refine_kernel, self.refine_kernel,
                self.refine_ch, self.refine_ch
            ],
                             stddev=0.02),
            name='refine__res3_W2')

        self.refine_res4_W1 = tf.Variable(
            tf.random_normal([
                self.refine_kernel, self.refine_kernel, self.refine_kernel,
                self.refine_ch, self.refine_ch
            ],
                             stddev=0.02),
            name='refine__res4_W1')
        self.refine_res4_W2 = tf.Variable(
            tf.random_normal([
                self.refine_kernel, self.refine_kernel, self.refine_kernel,
                self.refine_ch, self.refine_ch
            ],
                             stddev=0.02),
            name='refine__res4_W2')

        self.refine_W2 = tf.Variable(
            tf.random_normal([
                self.refine_kernel, self.refine_kernel, self.refine_kernel,
                self.refine_ch, self.dim_W5
            ],
                             stddev=0.02),
            name='refine_W2')

        # parameters of refiner (sscnet)
        self.refine_ssc_W1 = tf.Variable(
            tf.random_normal([7, 7, 7, self.dim_W5, 16], stddev=0.02),
            name='refine_ssc_W1')
        self.refine_ssc_W2 = tf.Variable(
            tf.random_normal([3, 3, 3, 16, 32], stddev=0.02),
            name='refine_ssc_W2')
        self.refine_ssc_W3 = tf.Variable(
            tf.random_normal([3, 3, 3, 32, 32], stddev=0.02),
            name='refine_ssc_W3')
        self.refine_ssc_W4 = tf.Variable(
            tf.random_normal([1, 1, 1, 16, 32], stddev=0.02),
            name='refine_ssc_W4')

        self.refine_ssc_W5 = tf.Variable(
            tf.random_normal([3, 3, 3, 32, 64], stddev=0.02),
            name='refine_ssc_W5')
        self.refine_ssc_W6 = tf.Variable(
            tf.random_normal([3, 3, 3, 64, 64], stddev=0.02),
            name='refine_ssc_W6')
        self.refine_ssc_W7 = tf.Variable(
            tf.random_normal([1, 1, 1, 64, 64], stddev=0.02),
            name='refine_ssc_W7')

        self.refine_ssc_W8 = tf.Variable(
            tf.random_normal([3, 3, 3, 64, 64], stddev=0.02),
            name='refine_ssc_W8')
        self.refine_ssc_W9 = tf.Variable(
            tf.random_normal([3, 3, 3, 64, 64], stddev=0.02),
            name='refine_ssc_W9')

        self.refine_ssc_W10 = tf.Variable(
            tf.random_normal([3, 3, 3, 64, 64], stddev=0.02),
            name='refine_ssc_W10')
        self.refine_ssc_W11 = tf.Variable(
            tf.random_normal([3, 3, 3, 64, 64], stddev=0.02),
            name='refine_ssc_W11')

        self.refine_ssc_W12 = tf.Variable(
            tf.random_normal([3, 3, 3, 64, 64], stddev=0.02),
            name='refine_ssc_W12')
        self.refine_ssc_W13 = tf.Variable(
            tf.random_normal([3, 3, 3, 64, 64], stddev=0.02),
            name='refine_ssc_W13')

        self.refine_ssc_W14 = tf.Variable(
            tf.random_normal([1, 1, 1, 64 * 3, 128], stddev=0.02),
            name='refine_ssc_W14')
        self.refine_ssc_W15 = tf.Variable(
            tf.random_normal([1, 1, 1, 128, 128], stddev=0.02),
            name='refine_ssc_W15')
        self.refine_ssc_W16 = tf.Variable(
            tf.random_normal([1, 1, 1, 128, 128], stddev=0.02),
            name='refine_ssc_W16')

        self.refine_ssc_W17 = tf.Variable(
            tf.random_normal([3, 3, 3, 128, 128], stddev=0.02),
            name='refine_ssc_W17')
        self.refine_ssc_W18 = tf.Variable(
            tf.random_normal([3, 3, 3, 12, 128], stddev=0.02),
            name='refine_ssc_W18')

        self.saver = tf.train.Saver()

    def build_model(self):

        vox_real_ = tf.placeholder(tf.int32, [
            self.batch_size, self.vox_shape[0], self.vox_shape[1],
            self.vox_shape[2]
        ])
        vox_real = tf.one_hot(vox_real_, self.n_class)
        vox_real = tf.cast(vox_real, tf.float32)
        # tsdf--start
        tsdf_real_ = tf.placeholder(tf.int32, [
            self.batch_size, self.vox_shape[0], self.vox_shape[1],
            self.vox_shape[2]
        ])
        tsdf_real = tf.one_hot(tsdf_real_, self.tsdf_shape[-1])
        tsdf_real = tf.cast(tsdf_real, tf.float32)
        # tsdf--end
        Z = tf.placeholder(tf.float32, [
            self.batch_size, self.start_vox_size[0], self.start_vox_size[1],
            self.start_vox_size[2], self.dim_z
        ])

        filter_bilateral = tf.placeholder(
            tf.float32, [self.batch_size] +
            [self.vox_shape[0], self.vox_shape[1], self.vox_shape[2], 4])
        mean, sigma = self.encoder(tsdf_real)
        Z_encode = mean

        # code_discriminator
        h_code_encode = self.code_discriminator(Z_encode)
        h_code_real = self.code_discriminator(Z)

        # empty space mask
        """
        empty_mask = tf.ones_like(vox_real)
        empty_mask[:, :, :, :, 0] = tf.to_float(
            tf.nn.relu(
                tf.random_uniform(
                    empty_mask[:, :, :, :, 0].get_shape(),
                    minval=-4,
                    maxval=1,
                    dtype=tf.int32)))
        """

        code_encode_loss = tf.reduce_mean(
            tf.reduce_sum(
                tf.nn.sigmoid_cross_entropy_with_logits(
                    logits=h_code_encode, labels=tf.ones_like(h_code_encode)),
                [1]))
        code_discrim_loss = tf.reduce_mean(
            tf.reduce_sum(
                tf.nn.sigmoid_cross_entropy_with_logits(
                    logits=h_code_real, labels=tf.ones_like(h_code_real)),
                [1])) + tf.reduce_mean(
                    tf.reduce_sum(
                        tf.nn.sigmoid_cross_entropy_with_logits(
                            logits=h_code_encode,
                            labels=tf.zeros_like(h_code_encode)), [1]))

        # reconstruction
        vox_gen_decode = self.generate(Z_encode)
        batch_mean_vox_real = tf.reduce_mean(vox_real, [0, 1, 2, 3])
        # batch_mean_vox_real ranges from 0 to 1
        # inverse ranges from 0.5 to 1
        inverse = tf.div(
            tf.ones_like(batch_mean_vox_real),
            tf.add(batch_mean_vox_real, tf.ones_like(batch_mean_vox_real)))
        weight = inverse * tf.div(1., tf.reduce_sum(inverse))
        recons_loss = tf.reduce_mean(
            tf.reduce_sum(
                -tf.reduce_sum(
                    self.lamda_gamma * vox_real * tf.log(1e-6 + vox_gen_decode)
                    + (1 - self.lamda_gamma) *
                    (1 - vox_real) * tf.log(1e-6 + 1 - vox_gen_decode),
                    [1, 2, 3]) * weight, 1))

        # completion loss
        vox_real_complete = tf.stack([
            vox_real[:, :, :, :, 0],
            tf.reduce_sum(vox_real[:, :, :, :, 1:], 4)
        ], 4)
        """
        vox_real_complete = softmax(
            vox_real_complete, self.batch_size,
            [self.vox_shape[0], self.vox_shape[1], self.vox_shape[2], 2])
        """

        vox_gen_complete = tf.stack([
            vox_gen_decode[:, :, :, :, 0],
            tf.reduce_sum(vox_gen_decode[:, :, :, :, 1:], 4)
        ], 4)
        """
        vox_gen_complete = softmax(
            vox_gen_complete, self.batch_size,
            [self.vox_shape[0], self.vox_shape[1], self.vox_shape[2], 2])
        """
        # inverse ranges from 1/1.1 to 10
        inverse = tf.div(
            tf.ones_like(batch_mean_vox_real), batch_mean_vox_real + 0.1)
        weight = inverse * tf.div(1., tf.reduce_sum(inverse))
        complete_loss = -tf.reduce_sum(
            self.lamda_gamma * vox_real_complete *
            tf.log(1e-6 + vox_gen_complete) + (1 - self.lamda_gamma) *
            (1 - vox_real_complete) * tf.log(1e-6 + 1 - vox_gen_complete),
            [1, 2, 3])
        weight_complete = tf.stack([weight[0], tf.reduce_sum(weight[1:])])
        complete_loss = tf.reduce_mean(
            tf.reduce_sum(complete_loss * weight_complete, 1))

        # depth segmentation
        tsdf_seg = tf.multiply(vox_gen_decode,
                               tf.expand_dims(tsdf_real[:, :, :, :, 1], -1))
        tsdf_seg_loss = tf.reduce_mean(
            tf.reduce_sum(
                -tf.reduce_sum(
                    self.lamda_gamma * tf.multiply(
                        vox_real, tf.expand_dims(tsdf_real[:, :, :, :, 1], -1))
                    * tf.log(1e-6 + tsdf_seg) + (1 - self.lamda_gamma) *
                    (1 - tf.multiply(
                        vox_real, tf.expand_dims(tsdf_real[:, :, :, :, 1], -1))
                     ) * tf.log(1e-6 + 1 - tsdf_seg), [1, 2, 3]) * weight, 1))

        recons_loss = recons_loss + complete_loss + tsdf_seg_loss

        # refiner
        vox_after_refine_dec = tf.placeholder(tf.float32, [
            self.batch_size, self.vox_shape[0], self.vox_shape[1],
            self.vox_shape[2], self.n_class
        ])
        if self.refiner is 'sscnet':
            vox_after_refine_dec = self.refine_sscnet(vox_gen_decode)
        else:
            vox_after_refine_dec = self.refine_resnet(vox_gen_decode)

        recons_loss_refine = tf.reduce_mean(
            tf.reduce_sum(
                -tf.reduce_sum(
                    self.lamda_gamma * vox_real *
                    tf.log(1e-6 + vox_after_refine_dec) +
                    (1 - self.lamda_gamma) *
                    (1 - vox_real) * tf.log(1e-6 + 1 - vox_after_refine_dec),
                    [1, 2, 3]) * weight, 1))

        # GAN_generate
        vox_gen = self.generate(Z)
        vox_after_refine_gen = tf.placeholder(tf.float32, [
            self.batch_size, self.vox_shape[0], self.vox_shape[1],
            self.vox_shape[2], self.n_class
        ])
        if self.refiner is 'sscnet':
            vox_after_refine_gen = self.refine_sscnet(vox_gen)
        else:
            vox_after_refine_gen = self.refine_resnet(vox_gen)

        h_real = self.discriminate(vox_real)
        h_gen = self.discriminate(vox_gen)
        h_gen_dec = self.discriminate(vox_gen_decode)
        h_gen_ref = self.discriminate(vox_after_refine_gen)
        h_gen_dec_ref = self.discriminate(vox_after_refine_dec)

        # Standard_GAN_Loss
        discrim_loss = tf.reduce_mean(
            tf.reduce_sum(
                tf.nn.sigmoid_cross_entropy_with_logits(
                    logits=h_real, labels=tf.ones_like(h_real)),
                1)) + tf.reduce_mean(
                    tf.reduce_sum(
                        tf.nn.sigmoid_cross_entropy_with_logits(
                            logits=h_gen, labels=tf.zeros_like(h_gen)),
                        1)) + tf.reduce_mean(
                            tf.reduce_sum(
                                tf.nn.sigmoid_cross_entropy_with_logits(
                                    logits=h_gen_dec,
                                    labels=tf.zeros_like(h_gen_dec)), 1))

        gen_loss = tf.reduce_mean(
            tf.reduce_sum(
                tf.nn.sigmoid_cross_entropy_with_logits(
                    logits=h_gen,
                    labels=tf.ones_like(h_gen)), 1)) + tf.reduce_mean(
                        tf.reduce_sum(
                            tf.nn.sigmoid_cross_entropy_with_logits(
                                logits=h_gen_dec,
                                labels=tf.ones_like(h_gen_dec)), 1))

        # for refine
        discrim_loss_refine = tf.reduce_mean(
            tf.reduce_sum(
                tf.nn.sigmoid_cross_entropy_with_logits(
                    logits=h_real, labels=tf.ones_like(h_real)),
                1)) + tf.reduce_mean(
                    tf.reduce_sum(
                        tf.nn.sigmoid_cross_entropy_with_logits(
                            logits=h_gen_ref, labels=tf.zeros_like(h_gen_ref)),
                        1)) + tf.reduce_mean(
                            tf.reduce_sum(
                                tf.nn.sigmoid_cross_entropy_with_logits(
                                    logits=h_gen_dec_ref,
                                    labels=tf.zeros_like(h_gen_dec_ref)), 1))

        gen_loss_refine = tf.reduce_mean(
            tf.reduce_sum(
                tf.nn.sigmoid_cross_entropy_with_logits(
                    logits=h_gen_ref, labels=tf.ones_like(h_gen_ref)),
                1)) + tf.reduce_mean(
                    tf.reduce_sum(
                        tf.nn.sigmoid_cross_entropy_with_logits(
                            logits=h_gen_dec_ref,
                            labels=tf.ones_like(h_gen_dec_ref)), 1))
        """
        #LS_GAN_Loss
        a=-1
        b=1
        c=0

        discrim_loss = tf.reduce_mean(0.5*((h_real-b)**2) + 0.5*((h_gen-a)**2) + 0.5*((h_gen_dec-a)**2))
        gen_loss = tf.reduce_mean(0.5*((h_gen-c)**2) + 0.5*((h_gen_dec-c)**2))
        """

        # Cost
        cost_enc = code_encode_loss + self.lamda_recons * recons_loss
        cost_gen = self.lamda_recons * recons_loss + gen_loss
        cost_discrim = discrim_loss
        cost_code = code_discrim_loss
        cost_gen_ref = self.lamda_recons * recons_loss_refine + gen_loss_refine
        cost_discrim_ref = discrim_loss_refine

        tf.summary.scalar("recons_loss", tf.reduce_mean(recons_loss))
        tf.summary.scalar("gen_loss", tf.reduce_mean(gen_loss))
        tf.summary.scalar("discrim_loss", tf.reduce_mean(discrim_loss))
        tf.summary.scalar("code_encode_loss", tf.reduce_mean(code_encode_loss))
        tf.summary.scalar("code_discrim_loss",
                          tf.reduce_mean(code_discrim_loss))

        summary_op = tf.summary.merge_all()

        return Z, Z_encode, vox_real_, vox_gen, vox_gen_decode, vox_gen_complete, tsdf_seg, vox_after_refine_dec, vox_after_refine_gen,\
        recons_loss, code_encode_loss, gen_loss, discrim_loss, recons_loss_refine, gen_loss_refine, discrim_loss_refine,\
        cost_enc, cost_code, cost_gen, cost_discrim, cost_gen_ref, cost_discrim_ref, summary_op,\
        tsdf_real_

    def encoder(self, vox):

        h1 = lrelu(
            tf.nn.conv3d(
                vox,
                self.encode_W1,
                strides=self.stride,
                dilations=self.dilations,
                padding='SAME'))
        h2 = lrelu(
            batchnormalize(
                tf.nn.conv3d(
                    h1,
                    self.encode_W2,
                    strides=self.stride,
                    dilations=self.dilations,
                    padding='SAME'),
                g=self.encode_bn_g2,
                b=self.encode_bn_b2,
                batch_size=self.batch_size))
        h3 = lrelu(
            batchnormalize(
                tf.nn.conv3d(
                    h2,
                    self.encode_W3,
                    strides=self.stride,
                    dilations=self.dilations,
                    padding='SAME'),
                g=self.encode_bn_g3,
                b=self.encode_bn_b3,
                batch_size=self.batch_size))
        h4 = lrelu(
            batchnormalize(
                tf.nn.conv3d(
                    h3,
                    self.encode_W4,
                    strides=self.stride,
                    dilations=self.dilations,
                    padding='SAME'),
                g=self.encode_bn_g4,
                b=self.encode_bn_b4,
                batch_size=self.batch_size))
        h5 = tf.nn.conv3d(
            h4, self.encode_W5, strides=[1, 1, 1, 1, 1], padding='SAME')
        h5_sigma = tf.nn.conv3d(
            h4, self.encode_W5_sigma, strides=[1, 1, 1, 1, 1], padding='SAME')

        return h5, h5_sigma

    def discriminate(self, vox):

        h1 = lrelu(
            tf.nn.conv3d(
                vox,
                self.discrim_W1,
                strides=self.stride,
                dilations=self.dilations,
                padding='SAME'))
        h2 = lrelu(
            layernormalize(
                tf.nn.conv3d(
                    h1,
                    self.discrim_W2,
                    strides=self.stride,
                    dilations=self.dilations,
                    padding='SAME'),
                g=self.discrim_bn_g2,
                b=self.discrim_bn_b2))
        h3 = lrelu(
            layernormalize(
                tf.nn.conv3d(
                    h2,
                    self.discrim_W3,
                    strides=self.stride,
                    dilations=self.dilations,
                    padding='SAME'),
                g=self.discrim_bn_g3,
                b=self.discrim_bn_b3))
        h4 = lrelu(
            layernormalize(
                tf.nn.conv3d(
                    h3,
                    self.discrim_W4,
                    strides=self.stride,
                    dilations=self.dilations,
                    padding='SAME'),
                g=self.discrim_bn_g4,
                b=self.discrim_bn_b4))
        # this is added for patch GAN
        h4 = tf.nn.conv3d(
            h4,
            self.discrim_W5,
            strides=self.stride,
            dilations=self.dilations,
            padding='SAME')
        # end of patch GAN
        h4 = tf.reshape(h4, [self.batch_size, -1])
        """ original final layer
        h5 = tf.matmul(h4, self.discrim_W5)
        y = tf.nn.sigmoid(h5)
        """

        return h4

    def code_discriminator(self, Z):
        Z_ = tf.reshape(Z, [self.batch_size, -1])
        h1 = tf.nn.relu(
            batchnormalize(
                tf.matmul(Z_, self.cod_W1), g=self.cod_bn_g1,
                b=self.cod_bn_b1))
        h2 = tf.nn.relu(
            batchnormalize(
                tf.matmul(h1, self.cod_W2), g=self.cod_bn_g2,
                b=self.cod_bn_b2))
        h3 = tf.matmul(h2, self.cod_W3)
        y = tf.nn.sigmoid(h3)
        return h3

    def generate(self, Z):

        Z_ = tf.reshape(Z, [self.batch_size, -1])
        h1 = tf.nn.relu(
            batchnormalize(
                tf.matmul(Z_, self.gen_W1), g=self.gen_bn_g1,
                b=self.gen_bn_b1))
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
            h1, self.gen_W2, output_shape=output_shape_l2, strides=self.stride)
        h2 = tf.nn.relu(
            batchnormalize(
                h2,
                g=self.gen_bn_g2,
                b=self.gen_bn_b2,
                batch_size=self.batch_size))

        vox_size_l3 = self.start_vox_size * 4
        output_shape_l3 = [
            self.batch_size, vox_size_l3[0], vox_size_l3[1], vox_size_l3[2],
            self.dim_W3
        ]
        h3 = tf.nn.conv3d_transpose(
            h2, self.gen_W3, output_shape=output_shape_l3, strides=self.stride)
        h3 = tf.nn.relu(
            batchnormalize(
                h3,
                g=self.gen_bn_g3,
                b=self.gen_bn_b3,
                batch_size=self.batch_size))

        vox_size_l4 = self.start_vox_size * 8
        output_shape_l4 = [
            self.batch_size, vox_size_l4[0], vox_size_l4[1], vox_size_l4[2],
            self.dim_W4
        ]
        h4 = tf.nn.conv3d_transpose(
            h3, self.gen_W4, output_shape=output_shape_l4, strides=self.stride)
        h4 = tf.nn.relu(
            batchnormalize(
                h4,
                g=self.gen_bn_g4,
                b=self.gen_bn_b4,
                batch_size=self.batch_size))

        vox_size_l5 = self.start_vox_size * 16
        output_shape_l5 = [
            self.batch_size, vox_size_l5[0], vox_size_l5[1], vox_size_l5[2],
            self.dim_W5
        ]
        h5 = tf.nn.conv3d_transpose(
            h4, self.gen_W5, output_shape=output_shape_l5, strides=self.stride)

        x = softmax(h5, self.batch_size, self.vox_shape)
        return x

    def refine_resnet(self, vox):
        base = tf.nn.relu(
            tf.nn.conv3d(
                vox, self.refine_W1, strides=[1, 1, 1, 1, 1], padding='SAME'))

        #res1
        res1_1 = tf.nn.relu(
            tf.nn.conv3d(
                base,
                self.refine_res1_W1,
                strides=[1, 1, 1, 1, 1],
                padding='SAME'))
        res1_2 = tf.nn.conv3d(
            res1_1,
            self.refine_res1_W2,
            strides=[1, 1, 1, 1, 1],
            padding='SAME')

        res1 = tf.nn.relu(tf.add(base, res1_2))

        #res2
        res2_1 = tf.nn.relu(
            tf.nn.conv3d(
                res1,
                self.refine_res2_W1,
                strides=[1, 1, 1, 1, 1],
                padding='SAME'))
        res2_2 = tf.nn.conv3d(
            res2_1,
            self.refine_res2_W2,
            strides=[1, 1, 1, 1, 1],
            padding='SAME')

        res2 = tf.nn.relu(tf.add(res1, res2_2))

        #res3
        res3_1 = tf.nn.relu(
            tf.nn.conv3d(
                res2,
                self.refine_res3_W1,
                strides=[1, 1, 1, 1, 1],
                padding='SAME'))
        res3_2 = tf.nn.conv3d(
            res3_1,
            self.refine_res3_W2,
            strides=[1, 1, 1, 1, 1],
            padding='SAME')

        res3 = tf.nn.relu(tf.add(res2, res3_2))

        #res4
        res4_1 = tf.nn.relu(
            tf.nn.conv3d(
                res3,
                self.refine_res4_W1,
                strides=[1, 1, 1, 1, 1],
                padding='SAME'))
        res4_2 = tf.nn.conv3d(
            res4_1,
            self.refine_res4_W2,
            strides=[1, 1, 1, 1, 1],
            padding='SAME')

        res4 = tf.nn.relu(tf.add(res3, res4_2))

        out = tf.nn.conv3d(
            res4, self.refine_W2, strides=[1, 1, 1, 1, 1], padding='SAME')
        x_refine = softmax(out, self.batch_size, self.vox_shape)

        return x_refine

    def refine_sscnet(self, vox):
        """
        ssc1 = tf.nn.relu(
            tf.nn.conv3d(
                tf.concat([vox, tsdf], -1),
                self.refine_ssc_W1,
                strides=[1, 2, 2, 2, 1],
                padding='SAME'))
        ssc2 = tf.nn.relu(
            tf.nn.conv3d(
                ssc1,
                self.refine_ssc_W2,
                strides=[1, 1, 1, 1, 1],
                padding='SAME'))
        ssc3 = tf.nn.relu(
            tf.nn.conv3d(
                ssc2,
                self.refine_ssc_W3,
                strides=[1, 1, 1, 1, 1],
                padding='SAME'))

        
        ssc_add1 = tf.nn.max_pool3d(ssc3 + tf.nn.relu(
            tf.nn.conv3d(
                ssc1,
                self.refine_ssc_W4,
                strides=[1, 1, 1, 1, 1],
                padding='SAME')),
            ksize = [1,3,3,3,1],
            strides = [1,2,2,2,1],
            padding='SAME')

        ssc5 = tf.nn.relu(
            tf.nn.conv3d(
                ssc_add1,
                self.refine_ssc_W5,
                strides=[1, 1, 1, 1, 1],
                padding='SAME'))
        ssc6 = tf.nn.relu(
            tf.nn.conv3d(
                ssc5,
                self.refine_ssc_W6,
                strides=[1, 1, 1, 1, 1],
                padding='SAME'))

        ssc_add2 = ssc6 + tf.nn.relu(
            tf.nn.conv3d(
                ssc5,
                self.refine_ssc_W7,
                strides=[1, 1, 1, 1, 1],
                padding='SAME'))

        ssc8 = tf.nn.relu(
            tf.nn.conv3d(
                ssc_add2,
                self.refine_ssc_W8,
                strides=[1, 1, 1, 1, 1],
                padding='SAME'))
        ssc9 = tf.nn.relu(
            tf.nn.conv3d(
                ssc8,
                self.refine_ssc_W9,
                strides=[1, 1, 1, 1, 1],
                padding='SAME'))

        ssc_add3 = ssc8 + ssc9 

        ssc10 = tf.nn.relu(
            tf.nn.conv3d(
                ssc_add3,
                self.refine_ssc_W10,
                strides=[1, 1, 1, 1, 1],
                dilations=[1, 1, 1, 1, 1],
                padding='SAME'))
        ssc11 = tf.nn.relu(
            tf.nn.conv3d(
                ssc10,
                self.refine_ssc_W11,
                strides=[1, 1, 1, 1, 1],
                dilations=[1, 1, 1, 1, 1],
                padding='SAME'))
            # Should be 1 2 2 2 1
            #
            #
            #
            #

        ssc_add4 = ssc10 + ssc11 

        ssc12 = tf.nn.relu(
            tf.nn.conv3d(
                ssc_add4,
                self.refine_ssc_W12,
                strides=[1, 1, 1, 1, 1],
                dilations=[1, 1, 1, 1, 1],
                padding='SAME'))
        ssc13 = tf.nn.relu(
            tf.nn.conv3d(
                ssc12,
                self.refine_ssc_W13,
                strides=[1, 1, 1, 1, 1],
                dilations=[1, 1, 1, 1, 1],
                padding='SAME'))

        ssc_add5 = ssc12 + ssc13 

        ssc_concat = tf.concat([ssc_add3, ssc_add4, ssc_add5], -1)

        ssc14 = tf.nn.relu(
            tf.nn.conv3d(
                ssc_concat,
                self.refine_ssc_W14,
                strides=[1, 1, 1, 1, 1],
                padding='SAME'))
        ssc15 = tf.nn.relu(
            tf.nn.conv3d(
                ssc14,
                self.refine_ssc_W15,
                strides=[1, 1, 1, 1, 1],
                padding='SAME'))
        ssc16 = tf.nn.relu(
            tf.nn.conv3d(
                ssc15,
                self.refine_ssc_W16,
                strides=[1, 1, 1, 1, 1],
                padding='SAME'))

        ssc17 = tf.nn.relu(
            tf.nn.conv3d_transpose(
                ssc16,
                self.refine_ssc_W17,
                output_shape=[self.batch_size, 40, 24, 40, 128],
                strides=[1, 2, 2, 2, 1],
                padding='SAME'))
        ssc18 = tf.nn.relu(
            tf.nn.conv3d_transpose(
                ssc17,
                self.refine_ssc_W18,
                output_shape=[self.batch_size, 80, 48, 80, 12],
                strides=[1, 2, 2, 2, 1],
                padding='SAME'))

        x_refine = softmax(ssc18, self.batch_size, self.vox_shape)
        """
        base_1 = tf.layers.conv3d(
            vox,
            filters=16,
            kernel_size=(7, 7, 7),
            strides=(2, 2, 2),
            padding='same',
            dilation_rate=(1, 1, 1),
            name='refine_sscnet_1',
            reuse=tf.AUTO_REUSE)

        h = tf.layers.conv3d(
            base_1,
            filters=32,
            kernel_size=(3, 3, 3),
            strides=(1, 1, 1),
            padding='same',
            dilation_rate=(1, 1, 1),
            name='refine_sscnet_2',
            reuse=tf.AUTO_REUSE)
        h = tf.layers.conv3d(
            h,
            filters=32,
            kernel_size=(3, 3, 3),
            strides=(1, 1, 1),
            padding='same',
            dilation_rate=(1, 1, 1),
            name='refine_sscnet_3',
            reuse=tf.AUTO_REUSE)
        h = h + tf.layers.conv3d(
            base_1,
            filters=32,
            kernel_size=(1, 1, 1),
            strides=(1, 1, 1),
            padding='same',
            dilation_rate=(1, 1, 1),
            name='refine_sscnet_4',
            reuse=tf.AUTO_REUSE)
        h = tf.layers.max_pooling3d(
            h,
            pool_size=(3, 3, 3),
            strides=(2, 2, 2),
            padding='same',
            name='refine_sscnet_5')

        base_2 = tf.layers.conv3d(
            h,
            filters=64,
            kernel_size=(3, 3, 3),
            strides=(1, 1, 1),
            padding='same',
            dilation_rate=(1, 1, 1),
            name='refine_sscnet_6',
            reuse=tf.AUTO_REUSE)

        h = tf.layers.conv3d(
            base_2,
            filters=64,
            kernel_size=(3, 3, 3),
            strides=(1, 1, 1),
            padding='same',
            dilation_rate=(1, 1, 1),
            name='refine_sscnet_7',
            reuse=tf.AUTO_REUSE)

        h = h + tf.layers.conv3d(
            base_2,
            filters=64,
            kernel_size=(1, 1, 1),
            strides=(1, 1, 1),
            padding='same',
            dilation_rate=(1, 1, 1),
            name='refine_sscnet_8',
            reuse=tf.AUTO_REUSE)

        base_3 = tf.layers.conv3d(
            h,
            filters=64,
            kernel_size=(3, 3, 3),
            strides=(1, 1, 1),
            padding='same',
            dilation_rate=(1, 1, 1),
            name='refine_sscnet_9',
            reuse=tf.AUTO_REUSE)

        base_4 = base_3 + tf.layers.conv3d(
            base_3,
            filters=64,
            kernel_size=(3, 3, 3),
            strides=(1, 1, 1),
            padding='same',
            dilation_rate=(1, 1, 1),
            name='refine_sscnet_10',
            reuse=tf.AUTO_REUSE)

        base_5 = tf.layers.conv3d(
            base_4,
            filters=64,
            kernel_size=(3, 3, 3),
            strides=(1, 1, 1),
            padding='same',
            dilation_rate=(2, 2, 2),
            name='refine_sscnet_11',
            reuse=tf.AUTO_REUSE)

        base_6 = base_5 + tf.layers.conv3d(
            base_5,
            filters=64,
            kernel_size=(3, 3, 3),
            strides=(1, 1, 1),
            padding='same',
            dilation_rate=(2, 2, 2),
            name='refine_sscnet_12',
            reuse=tf.AUTO_REUSE)

        base_7 = tf.layers.conv3d(
            base_6,
            filters=64,
            kernel_size=(3, 3, 3),
            strides=(1, 1, 1),
            padding='same',
            dilation_rate=(2, 2, 2),
            name='refine_sscnet_13',
            reuse=tf.AUTO_REUSE)

        base_8 = base_7 + tf.layers.conv3d(
            base_7,
            filters=64,
            kernel_size=(3, 3, 3),
            strides=(1, 1, 1),
            padding='same',
            dilation_rate=(2, 2, 2),
            name='refine_sscnet_14',
            reuse=tf.AUTO_REUSE)

        base_9 = tf.concat([base_4, base_6, base_8], -1)
        base_9 = tf.layers.conv3d(
            base_9,
            filters=128,
            kernel_size=(1, 1, 1),
            strides=(1, 1, 1),
            padding='same',
            dilation_rate=(1, 1, 1),
            name='refine_sscnet_15',
            reuse=tf.AUTO_REUSE)
        base_9 = tf.layers.conv3d(
            base_9,
            filters=128,
            kernel_size=(1, 1, 1),
            strides=(1, 1, 1),
            padding='same',
            dilation_rate=(1, 1, 1),
            name='refine_sscnet_16',
            reuse=tf.AUTO_REUSE)
        base_9 = tf.layers.conv3d(
            base_9,
            filters=128,
            kernel_size=(1, 1, 1),
            strides=(1, 1, 1),
            padding='same',
            dilation_rate=(1, 1, 1),
            name='refine_sscnet_17',
            reuse=tf.AUTO_REUSE)

        base_9 = tf.layers.conv3d_transpose(
            base_9,
            filters=128,
            kernel_size=(3, 3, 3),
            strides=(2, 2, 2),
            padding='same',
            name='refine_sscnet_18',
            reuse=tf.AUTO_REUSE)
        base_9 = tf.layers.conv3d_transpose(
            base_9,
            filters=12,
            kernel_size=(3, 3, 3),
            strides=(2, 2, 2),
            padding='same',
            name='refine_sscnet_19',
            reuse=tf.AUTO_REUSE)

        x_refine = softmax(base_9, self.batch_size, self.vox_shape)

        return x_refine

    def samples_generator(self, visual_size):

        Z = tf.placeholder(tf.float32, [
            visual_size, self.start_vox_size[0], self.start_vox_size[1],
            self.start_vox_size[2], self.dim_z
        ])

        Z_ = tf.reshape(Z, [visual_size, -1])
        h1 = tf.nn.relu(
            batchnormalize(
                tf.matmul(Z_, self.gen_W1), g=self.gen_bn_g1,
                b=self.gen_bn_b1))
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
            h1, self.gen_W2, output_shape=output_shape_l2, strides=self.stride)
        h2 = tf.nn.relu(
            batchnormalize(
                h2,
                g=self.gen_bn_g2,
                b=self.gen_bn_b2,
                batch_size=self.batch_size))

        vox_size_l3 = self.start_vox_size * 4
        output_shape_l3 = [
            visual_size, vox_size_l3[0], vox_size_l3[1], vox_size_l3[2],
            self.dim_W3
        ]
        h3 = tf.nn.conv3d_transpose(
            h2, self.gen_W3, output_shape=output_shape_l3, strides=self.stride)
        h3 = tf.nn.relu(
            batchnormalize(
                h3,
                g=self.gen_bn_g3,
                b=self.gen_bn_b3,
                batch_size=self.batch_size))

        vox_size_l4 = self.start_vox_size * 8
        output_shape_l4 = [
            visual_size, vox_size_l4[0], vox_size_l4[1], vox_size_l4[2],
            self.dim_W4
        ]
        h4 = tf.nn.conv3d_transpose(
            h3, self.gen_W4, output_shape=output_shape_l4, strides=self.stride)
        h4 = tf.nn.relu(
            batchnormalize(
                h4,
                g=self.gen_bn_g4,
                b=self.gen_bn_b4,
                batch_size=self.batch_size))

        vox_size_l5 = self.start_vox_size * 16
        output_shape_l5 = [
            visual_size, vox_size_l5[0], vox_size_l5[1], vox_size_l5[2],
            self.dim_W5
        ]
        h5 = tf.nn.conv3d_transpose(
            h4, self.gen_W5, output_shape=output_shape_l5, strides=self.stride)

        x = softmax(h5, visual_size, self.vox_shape)
        return Z, x

    def refine_generator_resnet(self, visual_size):
        vox = tf.placeholder(tf.float32, [
            visual_size, self.vox_shape[0], self.vox_shape[1],
            self.vox_shape[2], self.vox_shape[3]
        ])

        base = tf.nn.relu(
            tf.nn.conv3d(
                vox, self.refine_W1, strides=[1, 1, 1, 1, 1], padding='SAME'))

        #res1
        res1_1 = tf.nn.relu(
            tf.nn.conv3d(
                base,
                self.refine_res1_W1,
                strides=[1, 1, 1, 1, 1],
                padding='SAME'))
        res1_2 = tf.nn.conv3d(
            res1_1,
            self.refine_res1_W2,
            strides=[1, 1, 1, 1, 1],
            padding='SAME')

        res1 = tf.nn.relu(tf.add(base, res1_2))

        #res2
        res2_1 = tf.nn.relu(
            tf.nn.conv3d(
                res1,
                self.refine_res2_W1,
                strides=[1, 1, 1, 1, 1],
                padding='SAME'))
        res2_2 = tf.nn.conv3d(
            res2_1,
            self.refine_res2_W2,
            strides=[1, 1, 1, 1, 1],
            padding='SAME')

        res2 = tf.nn.relu(tf.add(res1, res2_2))

        #res3
        res3_1 = tf.nn.relu(
            tf.nn.conv3d(
                res2,
                self.refine_res3_W1,
                strides=[1, 1, 1, 1, 1],
                padding='SAME'))
        res3_2 = tf.nn.conv3d(
            res3_1,
            self.refine_res3_W2,
            strides=[1, 1, 1, 1, 1],
            padding='SAME')

        res3 = tf.nn.relu(tf.add(res2, res3_2))

        #res4
        res4_1 = tf.nn.relu(
            tf.nn.conv3d(
                res3,
                self.refine_res4_W1,
                strides=[1, 1, 1, 1, 1],
                padding='SAME'))
        res4_2 = tf.nn.conv3d(
            res4_1,
            self.refine_res4_W2,
            strides=[1, 1, 1, 1, 1],
            padding='SAME')

        res4 = tf.nn.relu(tf.add(res3, res4_2))

        out = tf.nn.conv3d(
            res4, self.refine_W2, strides=[1, 1, 1, 1, 1], padding='SAME')
        x_refine = softmax(out, self.batch_size, self.vox_shape)

        return vox, x_refine

    def refine_generator_sscnet(self, visual_size):
        vox = tf.placeholder(tf.float32, [
            visual_size, self.vox_shape[0], self.vox_shape[1],
            self.vox_shape[2], self.vox_shape[3]
        ])
        """
        ssc1 = tf.nn.relu(
            tf.nn.conv3d(
                tf.concat([vox, tsdf], -1),
                self.refine_ssc_W1,
                strides=[1, 2, 2, 2, 1],
                padding='SAME'))
        ssc2 = tf.nn.relu(
            tf.nn.conv3d(
                ssc1,
                self.refine_ssc_W2,
                strides=[1, 1, 1, 1, 1],
                padding='SAME'))
        ssc3 = tf.nn.relu(
            tf.nn.conv3d(
                ssc2,
                self.refine_ssc_W3,
                strides=[1, 1, 1, 1, 1],
                padding='SAME'))

        
        ssc_add1 = tf.nn.max_pool3d(ssc3 + tf.nn.relu(
            tf.nn.conv3d(
                ssc1,
                self.refine_ssc_W4,
                strides=[1, 1, 1, 1, 1],
                padding='SAME')),
            ksize = [1,3,3,3,1],
            strides = [1,2,2,2,1],
            padding='SAME')

        ssc5 = tf.nn.relu(
            tf.nn.conv3d(
                ssc_add1,
                self.refine_ssc_W5,
                strides=[1, 1, 1, 1, 1],
                padding='SAME'))
        ssc6 = tf.nn.relu(
            tf.nn.conv3d(
                ssc5,
                self.refine_ssc_W6,
                strides=[1, 1, 1, 1, 1],
                padding='SAME'))

        ssc_add2 = ssc6 + tf.nn.relu(
            tf.nn.conv3d(
                ssc5,
                self.refine_ssc_W7,
                strides=[1, 1, 1, 1, 1],
                padding='SAME'))

        ssc8 = tf.nn.relu(
            tf.nn.conv3d(
                ssc_add2,
                self.refine_ssc_W8,
                strides=[1, 1, 1, 1, 1],
                padding='SAME'))
        ssc9 = tf.nn.relu(
            tf.nn.conv3d(
                ssc8,
                self.refine_ssc_W9,
                strides=[1, 1, 1, 1, 1],
                padding='SAME'))

        ssc_add3 = ssc8 + ssc9 

        ssc10 = tf.nn.relu(
            tf.nn.conv3d(
                ssc_add3,
                self.refine_ssc_W10,
                strides=[1, 1, 1, 1, 1],
                dilations=[1, 2, 2, 2, 1],
                padding='SAME'))
        ssc11 = tf.nn.relu(
            tf.nn.conv3d(
                ssc10,
                self.refine_ssc_W11,
                strides=[1, 1, 1, 1, 1],
                dilations=[1, 2, 2, 2, 1],
                padding='SAME'))

        ssc_add4 = ssc10 + ssc11 

        ssc12 = tf.nn.relu(
            tf.nn.conv3d(
                ssc_add4,
                self.refine_ssc_W12,
                strides=[1, 1, 1, 1, 1],
                dilations=[1, 2, 2, 2, 1],
                padding='SAME'))
        ssc13 = tf.nn.relu(
            tf.nn.conv3d(
                ssc12,
                self.refine_ssc_W13,
                strides=[1, 1, 1, 1, 1],
                dilations=[1, 2, 2, 2, 1],
                padding='SAME'))

        ssc_add5 = ssc12 + ssc13 

        ssc_concat = tf.concat([ssc_add3, ssc_add4, ssc_add5], -1)

        ssc14 = tf.nn.relu(
            tf.nn.conv3d(
                ssc_concat,
                self.refine_ssc_W14,
                strides=[1, 1, 1, 1, 1],
                padding='SAME'))
        ssc15 = tf.nn.relu(
            tf.nn.conv3d(
                ssc14,
                self.refine_ssc_W15,
                strides=[1, 1, 1, 1, 1],
                padding='SAME'))
        ssc16 = tf.nn.relu(
            tf.nn.conv3d(
                ssc15,
                self.refine_ssc_W16,
                strides=[1, 1, 1, 1, 1],
                padding='SAME'))

        ssc17 = tf.nn.relu(
            tf.nn.conv3d_transpose(
                ssc16,
                self.refine_ssc_W17,
                output_shape=[self.batch_size, 40, 24, 40, 128],
                strides=[1, 2, 2, 2, 1],
                padding='SAME'))
        ssc18 = tf.nn.relu(
            tf.nn.conv3d_transpose(
                ssc17,
                self.refine_ssc_W18,
                output_shape=[self.batch_size, 80, 48, 80, 12],
                strides=[1, 2, 2, 2, 1],
                padding='SAME'))

        x_refine = softmax(ssc18, self.batch_size, self.vox_shape)
        """
        base_1 = tf.layers.conv3d(
            vox,
            filters=16,
            kernel_size=(7, 7, 7),
            strides=(2, 2, 2),
            padding='same',
            dilation_rate=(1, 1, 1),
            name='refine_sscnet_1',
            reuse=tf.AUTO_REUSE)

        h = tf.layers.conv3d(
            base_1,
            filters=32,
            kernel_size=(3, 3, 3),
            strides=(1, 1, 1),
            padding='same',
            dilation_rate=(1, 1, 1),
            name='refine_sscnet_2',
            reuse=tf.AUTO_REUSE)
        h = tf.layers.conv3d(
            h,
            filters=32,
            kernel_size=(3, 3, 3),
            strides=(1, 1, 1),
            padding='same',
            dilation_rate=(1, 1, 1),
            name='refine_sscnet_3',
            reuse=tf.AUTO_REUSE)
        h = h + tf.layers.conv3d(
            base_1,
            filters=32,
            kernel_size=(1, 1, 1),
            strides=(1, 1, 1),
            padding='same',
            dilation_rate=(1, 1, 1),
            name='refine_sscnet_4',
            reuse=tf.AUTO_REUSE)
        h = tf.layers.max_pooling3d(
            h,
            pool_size=(3, 3, 3),
            strides=(2, 2, 2),
            padding='same',
            name='refine_sscnet_5')

        base_2 = tf.layers.conv3d(
            h,
            filters=64,
            kernel_size=(3, 3, 3),
            strides=(1, 1, 1),
            padding='same',
            dilation_rate=(1, 1, 1),
            name='refine_sscnet_6',
            reuse=tf.AUTO_REUSE)

        h = tf.layers.conv3d(
            base_2,
            filters=64,
            kernel_size=(3, 3, 3),
            strides=(1, 1, 1),
            padding='same',
            dilation_rate=(1, 1, 1),
            name='refine_sscnet_7',
            reuse=tf.AUTO_REUSE)

        h = h + tf.layers.conv3d(
            base_2,
            filters=64,
            kernel_size=(1, 1, 1),
            strides=(1, 1, 1),
            padding='same',
            dilation_rate=(1, 1, 1),
            name='refine_sscnet_8',
            reuse=tf.AUTO_REUSE)

        base_3 = tf.layers.conv3d(
            h,
            filters=64,
            kernel_size=(3, 3, 3),
            strides=(1, 1, 1),
            padding='same',
            dilation_rate=(1, 1, 1),
            name='refine_sscnet_9',
            reuse=tf.AUTO_REUSE)

        base_4 = base_3 + tf.layers.conv3d(
            base_3,
            filters=64,
            kernel_size=(3, 3, 3),
            strides=(1, 1, 1),
            padding='same',
            dilation_rate=(1, 1, 1),
            name='refine_sscnet_10',
            reuse=tf.AUTO_REUSE)

        base_5 = tf.layers.conv3d(
            base_4,
            filters=64,
            kernel_size=(3, 3, 3),
            strides=(1, 1, 1),
            padding='same',
            dilation_rate=(2, 2, 2),
            name='refine_sscnet_11',
            reuse=tf.AUTO_REUSE)

        base_6 = base_5 + tf.layers.conv3d(
            base_5,
            filters=64,
            kernel_size=(3, 3, 3),
            strides=(1, 1, 1),
            padding='same',
            dilation_rate=(2, 2, 2),
            name='refine_sscnet_12',
            reuse=tf.AUTO_REUSE)

        base_7 = tf.layers.conv3d(
            base_6,
            filters=64,
            kernel_size=(3, 3, 3),
            strides=(1, 1, 1),
            padding='same',
            dilation_rate=(2, 2, 2),
            name='refine_sscnet_13',
            reuse=tf.AUTO_REUSE)

        base_8 = base_7 + tf.layers.conv3d(
            base_7,
            filters=64,
            kernel_size=(3, 3, 3),
            strides=(1, 1, 1),
            padding='same',
            dilation_rate=(2, 2, 2),
            name='refine_sscnet_14',
            reuse=tf.AUTO_REUSE)

        base_9 = tf.concat([base_4, base_6, base_8], -1)
        base_9 = tf.layers.conv3d(
            base_9,
            filters=128,
            kernel_size=(1, 1, 1),
            strides=(1, 1, 1),
            padding='same',
            dilation_rate=(1, 1, 1),
            name='refine_sscnet_15',
            reuse=tf.AUTO_REUSE)
        base_9 = tf.layers.conv3d(
            base_9,
            filters=128,
            kernel_size=(1, 1, 1),
            strides=(1, 1, 1),
            padding='same',
            dilation_rate=(1, 1, 1),
            name='refine_sscnet_16',
            reuse=tf.AUTO_REUSE)
        base_9 = tf.layers.conv3d(
            base_9,
            filters=128,
            kernel_size=(1, 1, 1),
            strides=(1, 1, 1),
            padding='same',
            dilation_rate=(1, 1, 1),
            name='refine_sscnet_17',
            reuse=tf.AUTO_REUSE)

        base_9 = tf.layers.conv3d_transpose(
            base_9,
            filters=128,
            kernel_size=(3, 3, 3),
            strides=(2, 2, 2),
            padding='same',
            name='refine_sscnet_18',
            reuse=tf.AUTO_REUSE)
        base_9 = tf.layers.conv3d_transpose(
            base_9,
            filters=12,
            kernel_size=(3, 3, 3),
            strides=(2, 2, 2),
            padding='same',
            name='refine_sscnet_19',
            reuse=tf.AUTO_REUSE)

        x_refine = softmax(base_9, self.batch_size, self.vox_shape)

        return vox, x_refine
