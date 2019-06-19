import numpy as np

from config import cfg
import tensorflow as tf
from util import *
from metric import sparse_ml


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
                 part_shape=[80, 48, 80, 2],
                 dim_z=16,
                 dim=[512, 256, 192, 64, 12],
                 start_vox_size=[5, 3, 5],
                 kernel=[[4, 4, 4, 4, 4], [4, 4, 4, 4, 4], [4, 4, 4, 4, 4]],
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
        """

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
        """

        self.discrim_y_W1 = tf.Variable(
            tf.random_normal([
                self.kernel5[0], self.kernel5[1], self.kernel5[2],
                self.dim_W5, self.dim_W4
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

        """
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
        """

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

        self.saver = tf.train.Saver()

    def build_model(self):

        full_gt_ = tf.placeholder(tf.int32, [
            None, self.vox_shape[0], self.vox_shape[1],
            self.vox_shape[2]
        ])
        full_gt = tf.one_hot(full_gt_, self.n_class)
        full_gt = tf.cast(full_gt, tf.float32)

        complete_gt = tf.one_hot(
            tf.clip_by_value(full_gt_, clip_value_min=0, clip_value_max=1), 2)
        complete_gt = tf.cast(complete_gt, tf.float32)

        # tsdf--start
        part_gt_ = tf.placeholder(tf.float32, [
            None, self.vox_shape[0], self.vox_shape[1],
            self.vox_shape[2]
        ])
        part_gt = tf.abs(tf.expand_dims(part_gt_, -1))
        # tsdf--end

        Z = tf.placeholder(tf.float32, [
            None, self.start_vox_size[0], self.start_vox_size[1],
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
        Z_encode_part, Z_mu, Z_log_sigma = self.encoder_part(part_gt)
        variation_loss = -0.5 * tf.reduce_sum(
            1.0 + 2.0 * Z_log_sigma - tf.square(Z_mu) - tf.exp(
                2.0 * Z_log_sigma), 1)
        dim_code = Z_mu.get_shape().as_list()
        """
        nebula3d = tf.Variable(tf.truncated_normal([4, dim_code[1]]), 'nebula3d')
        loss_0d, loss_1d, loss_2d, loss_3d, _, _, nebula3d, nebula_index = sparse_ml(4, dim_code[1], nebula3d, Z_mu, info_type='unsupervised')
        """

        """
        Z_encode_full = self.encoder_full(full_gt)
        """

        complete_gen_dec, h3_t, h4_t = self.generate_part(Z_encode_part)
        """
        part_gen_dec, h2_v, h3_v, h4_v = self.generate_part(Z_encode_full)
        """
        """
        full_vae_dec = self.generate_full(Z_encode_full, h3_v, h4_v)
        """
        full_gen_dec, full_gen_dec_ref = self.generate_full(Z_encode_part, h3_t, h4_t)

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
        full_cc_dec = self.generate_full(Z_encode_full_part, h3_vt,
                                         h4_vt)
         """

        # code_discriminator
        """
        h_code_part_gt = self.code_discriminator_x(Z)
        h_code_full_gt = self.code_discriminator_y(Z)

        h_code_encode_part = self.code_discriminator_x(Z_encode_part)
        h_code_encode_full = self.code_discriminator_y(Z_encode_full)

        h_code_encode_part_full = self.code_discriminator_y(Z_encode_part_full)
        h_code_encode_full_part = self.code_discriminator_x(Z_encode_full_part)

        cost_code_encode = tf.reduce_mean(
            tf.reduce_sum(
                tf.nn.sigmoid_cross_entropy_with_logits(
                    logits=h_code_encode_part,
                    labels=tf.ones_like(h_code_encode_part)), [1]))
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
        recons_vae_loss = tf.reduce_mean(
            tf.reduce_sum(
                tf.squared_difference(part_gt, part_vae_dec), [1, 2, 3, 4]))
        recons_vae_loss = tf.reduce_mean(
            -tf.reduce_sum(
                self.lamda_gamma * part_gt * tf.log(1e-6 + part_vae_dec) +
                (1 - self.lamda_gamma) * (1 - part_gt) * tf.log(1e-6 + 1 - part_vae_dec), [1, 2, 3, 4]))
        """
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
        recons_loss = tf.reduce_sum(
                -tf.reduce_sum(
                    self.lamda_gamma * full_gt * tf.log(1e-6 + full_gen_dec) +
                    (1 - self.lamda_gamma) *
                    (1 - full_gt) * tf.log(1e-6 + 1 - full_gen_dec), [1, 2, 3])
                * weight_full, 1)
        # complete
        recons_loss += tf.reduce_sum(
                -tf.reduce_sum(
                    self.lamda_gamma * complete_gt * tf.log(1e-6 + complete_gen_dec) +
                    (1 - self.lamda_gamma) *
                    (1 - complete_gt) * tf.log(1e-6 + 1 - complete_gen_dec), [1, 2, 3])
                * weight_complete, 1)
        # refine
        refine_loss = tf.reduce_mean(
            tf.reduce_sum(
                -tf.reduce_sum(
                    self.lamda_gamma * full_gt * tf.log(1e-6 + full_gen_dec_ref) +
                    (1 - self.lamda_gamma) *
                    (1 - full_gt) * tf.log(1e-6 + 1 - full_gen_dec_ref), [1, 2, 3])
                * weight_full, 1))
        """
        recons_loss += tf.reduce_sum(
                tf.squared_difference(part_gt, part_gen_dec), [1, 2, 3, 4])
        """
        # from scene, the observed surface can also be produced
        # latent consistency
        """
        recons_loss += tf.reduce_sum(
                tf.squared_difference(Z_encode_part, Z_encode_full),
                [1, 2, 3, 4])
        """
        # latent consistency

        # GAN_generate
        if self.discriminative is True:
            """
            complete_gen, h3_z, h4_z = self.generate_part(Z)
            full_gen, full_gen_ref = self.generate_full(Z, h3_z, h4_z)
            """
            complete_gen, h3_z, h4_z = self.generate_part(Z_encode_part)
            full_gen, full_gen_ref = self.generate_full(Z_encode_part, h3_z, h4_z)

            h_full_gt = self.discriminate_full(full_gt)
            h_full_gen = self.discriminate_full(full_gen)
            # h_full_gen_dec = self.discriminate_full(full_gen_dec)

            h_part_gt = self.discriminate_part(complete_gt)
            h_part_gen = self.discriminate_part(complete_gen)
            # h_part_gen_dec = self.discriminate_part(complete_gen_dec)
            # Standard_GAN_Loss
            discrim_loss = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(
                    logits=h_full_gt, labels=tf.ones_like(h_full_gt))
            ) + tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(
                    logits=h_full_gen,
                    labels=tf.zeros_like(h_full_gen))
                ) 
            """
            + tf.reduce_mean(
                        tf.nn.sigmoid_cross_entropy_with_logits(
                            logits=h_full_gen_dec,
                            labels=tf.zeros_like(h_full_gen_dec)))
            """

            discrim_loss += tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(
                    logits=h_part_gt, labels=tf.ones_like(h_part_gt))
            ) + tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(
                    logits=h_part_gen,
                    labels=tf.zeros_like(h_part_gen))
                ) 
            """
            + tf.reduce_mean(
                        tf.nn.sigmoid_cross_entropy_with_logits(
                            logits=h_part_gen_dec,
                            labels=tf.zeros_like(h_part_gen_dec)))
            """

            gen_loss = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(
                    logits=h_full_gen,
                    labels=tf.ones_like(h_full_gen))) 
            """
                + tf.reduce_mean(
                        tf.nn.sigmoid_cross_entropy_with_logits(
                            logits=h_full_gen_dec,
                            labels=tf.ones_like(h_full_gen_dec)))
            """

            gen_loss += tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(
                    logits=h_part_gen,
                    labels=tf.ones_like(h_part_gen))) 
            """
                + tf.reduce_mean(
                        tf.nn.sigmoid_cross_entropy_with_logits(
                            logits=h_part_gen_dec,
                            labels=tf.ones_like(h_part_gen_dec)))
            """

        else:
            complete_gen = complete_gen_dec
            full_gen = full_gen_dec
            gen_loss = tf.zeros([1])
            discrim_loss = tf.zeros([1])

        # variational cost
        variation_loss = tf.reduce_mean(variation_loss + recons_loss)

        # main cost
        recons_loss = self.lamda_recons * tf.reduce_mean(recons_loss)

        summary_op = tf.summary.merge_all()

        return Z, Z_encode_part, full_gt_, full_gen, full_gen_dec, full_gen_dec_ref,\
        gen_loss, discrim_loss, recons_loss, variation_loss, refine_loss, summary_op,\
        part_gt_, complete_gen, complete_gen_dec

    def encoder_part(self, vox):

        h1 = lrelu(
            tf.layers.conv3d(
                vox,
                filters=16,
                kernel_size=(7, 7, 7),
                strides=(2, 2, 2),
                padding='same',
                name='encode_x_sscnet_0',
                reuse=tf.AUTO_REUSE))
        h1_0 = lrelu(
            tf.layers.conv3d(
                h1,
                filters=32,
                kernel_size=(1, 1, 1),
                strides=(1, 1, 1),
                padding='same',
                name='encode_x_sscnet_0_0',
                reuse=tf.AUTO_REUSE))
        h1_1 = lrelu(
            tf.layers.conv3d(
                h1,
                filters=32,
                kernel_size=(3, 3, 3),
                strides=(1, 1, 1),
                padding='same',
                name='encode_x_sscnet_0_1',
                reuse=tf.AUTO_REUSE))
        h1_2 = lrelu(
            tf.layers.conv3d(
                h1_1,
                filters=32,
                kernel_size=(3, 3, 3),
                strides=(1, 1, 1),
                padding='same',
                name='encode_x_sscnet_0_2',
                reuse=tf.AUTO_REUSE))

        base_4 = h1_0 + h1_2
        base_5 = tf.layers.conv3d(
            base_4,
            filters=64,
            kernel_size=(3, 3, 3),
            strides=(1, 1, 1),
            padding='same',
            dilation_rate=(2, 2, 2),
            name='encode_x_sscnet_1',
            reuse=tf.AUTO_REUSE)

        base_6 = base_5 + tf.layers.conv3d(
            base_5,
            filters=64,
            kernel_size=(3, 3, 3),
            strides=(1, 1, 1),
            padding='same',
            dilation_rate=(2, 2, 2),
            name='encode_x_sscnet_2',
            reuse=tf.AUTO_REUSE)

        base_7 = tf.layers.conv3d(
            base_6,
            filters=64,
            kernel_size=(3, 3, 3),
            strides=(1, 1, 1),
            padding='same',
            dilation_rate=(2, 2, 2),
            name='encode_x_sscnet_3',
            reuse=tf.AUTO_REUSE)

        base_8 = base_7 + tf.layers.conv3d(
            base_7,
            filters=64,
            kernel_size=(3, 3, 3),
            strides=(1, 1, 1),
            padding='same',
            dilation_rate=(2, 2, 2),
            name='encode_x_sscnet_4',
            reuse=tf.AUTO_REUSE)
        base_9 = tf.concat([base_4, base_6, base_8], -1)

        h1_1 = tf.layers.conv3d(
            base_9,
            filters=128,
            kernel_size=(1, 1, 1),
            strides=(1, 1, 1),
            padding='same',
            dilation_rate=(1, 1, 1),
            name='encode_x_sscnet_5',
            reuse=tf.AUTO_REUSE)
        h1_2 = tf.layers.conv3d(
            h1_1,
            filters=128,
            kernel_size=(1, 1, 1),
            strides=(1, 1, 1),
            padding='same',
            dilation_rate=(1, 1, 1),
            name='encode_x_sscnet_6',
            reuse=tf.AUTO_REUSE)

        h2 = lrelu(
            tf.layers.batch_normalization(
                tf.layers.conv3d(
                    h1_2,
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

        if self.discriminative is True:
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

            # start to add hidden layers
            dims = h5.get_shape().as_list()
            n_code = dims[1] * dims[2] * dims[3] * dims[4]
            flattened = tf.contrib.layers.flatten(h5)
            epsilon = tf.random_normal(tf.stack([tf.shape(h5)[0], n_code]))
            z_hidden = tf.layers.dense(
                flattened, 
                256, 
                use_bias=True,
                name='encode_x_hidden', 
                reuse=tf.AUTO_REUSE)
            z_mu = tf.layers.dense(
                z_hidden, 
                n_code, 
                use_bias=False,
                name='encode_x_mu', 
                reuse=tf.AUTO_REUSE)
            z_log_sigma = 0.5 * tf.layers.dense(
                z_hidden, 
                n_code, 
                use_bias=False,
                name='encode_x_log_sigma', 
                reuse=tf.AUTO_REUSE)
            if self.is_train is True:
                z = tf.add(
                    z_mu,
                    tf.multiply(epsilon, tf.exp(z_log_sigma)),
                    name='encode_x_z')
            elif self.is_train is False:
                z = z_mu
            z = tf.reshape(z,
                           tf.stack([-1, dims[1], dims[2], dims[3], dims[4]]))
        elif self.discriminative is False:
            z = h3
            z_mu = tf.contrib.layers.flatten(z)
            z_log_sigma = tf.contrib.layers.flatten(z)
        return z, z_mu, z_log_sigma

# end of hidden layers
# return h5

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

    def generate_full(self, Z, h3_, h4_):

        if self.discriminative is True:
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
        elif self.discriminative is False:
            h2 = Z

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

        stage1 = softmax(h5, self.batch_size, self.vox_shape)

        # start to refine
        base = tf.nn.relu(tf.layers.conv3d(stage1, filters=16, kernel_size=3, padding='same', name='gen_y_ref_base', reuse=tf.AUTO_REUSE))

        #res1
        res1_1 = tf.nn.relu(tf.layers.conv3d(base, filters=16, kernel_size=3, padding='same', name='gen_y_ref_1_1', reuse=tf.AUTO_REUSE))
        res1_2 = tf.nn.relu(tf.layers.conv3d(res1_1, filters=16, kernel_size=3, padding='same', name='gen_y_ref_1_2', reuse=tf.AUTO_REUSE))

        res1 = tf.nn.relu(tf.add(base, res1_2))

        #res2
        res2_1 = tf.nn.relu(tf.layers.conv3d(res1, filters=16, kernel_size=3, padding='same', name='gen_y_ref_2_1', reuse=tf.AUTO_REUSE))
        res2_2 = tf.nn.relu(tf.layers.conv3d(res2_1, filters=16, kernel_size=3, padding='same', name='gen_y_ref_2_2', reuse=tf.AUTO_REUSE))

        res2 = tf.nn.relu(tf.add(res1, res2_2))

        #res3
        res3_1 = tf.nn.relu(tf.layers.conv3d(res2, filters=16, kernel_size=3, padding='same', name='gen_y_ref_3_1', reuse=tf.AUTO_REUSE))
        res3_2 = tf.nn.relu(tf.layers.conv3d(res3_1, filters=16, kernel_size=3, padding='same', name='gen_y_ref_3_2', reuse=tf.AUTO_REUSE))

        res3 = tf.nn.relu(tf.add(res2, res3_2))

        #res4
        res4_1 = tf.nn.relu(tf.layers.conv3d(res3, filters=16, kernel_size=3, padding='same', name='gen_y_ref_4_1', reuse=tf.AUTO_REUSE))
        res4_2 = tf.nn.relu(tf.layers.conv3d(res4_1, filters=16, kernel_size=3, padding='same', name='gen_y_ref_4_2', reuse=tf.AUTO_REUSE))

        res4 = tf.nn.relu(tf.add(res3, res4_2))

        res_end = tf.layers.conv3d(res4, filters=self.vox_shape[-1], kernel_size=3, padding='same', name='gen_y_ref_final', reuse=tf.AUTO_REUSE)
        stage2 = softmax(res_end, self.batch_size, self.vox_shape)

        return stage1, stage2

        """
        Z_ = tf.reshape(Z, [self.batch_size, -1])
        h1 = tf.nn.relu(
            tf.layers.batch_normalization(
                tf.layers.dense(
                    Z_, 
                    units=self.start_vox_size[0]*self.start_vox_size[1]*self.start_vox_size[2]*self.dim_W1,
                    use_bias=False,
                    name='gen_y_1',
                    reuse=tf.AUTO_REUSE),
                name='gen_y_bn_1',
                reuse=tf.AUTO_REUSE,
                training=self.is_train))
        h1 = tf.reshape(h1, [
            self.batch_size, self.start_vox_size[0], self.start_vox_size[1],
            self.start_vox_size[2], self.dim_W1
        ])

        h2 = tf.nn.relu(
            tf.layers.batch_normalization(
                tf.layers.conv3d_transpose(
                    h1,
                    filters=self.dim_W2,
                    kernel_size=(self.kernel4[0], self.kernel4[1],
                                 self.kernel4[2]),
                    strides=(self.stride[1], self.stride[2], self.stride[3]),
                    padding='same',
                    name='gen_y_2',
                    reuse=tf.AUTO_REUSE),
                name='gen_y_bn_2',
                reuse=tf.AUTO_REUSE,
                training=self.is_train))

        h3 = tf.nn.relu(
            tf.layers.batch_normalization(
                tf.layers.conv3d_transpose(
                    tf.concat([h2, h2_], -1),
                    filters=self.dim_W3,
                    kernel_size=(self.kernel4[0], self.kernel4[1],
                                 self.kernel4[2]),
                    strides=(self.stride[1], self.stride[2], self.stride[3]),
                    padding='same',
                    name='gen_y_3',
                    reuse=tf.AUTO_REUSE),
                name='gen_y_bn_3',
                reuse=tf.AUTO_REUSE,
                training=self.is_train))

        h4 = tf.nn.relu(
            tf.layers.batch_normalization(
                tf.layers.conv3d_transpose(
                    tf.concat([h3, h3_], -1),
                    filters=self.dim_W4,
                    kernel_size=(self.kernel4[0], self.kernel4[1],
                                 self.kernel4[2]),
                    strides=(self.stride[1], self.stride[2], self.stride[3]),
                    padding='same',
                    name='gen_y_4',
                    reuse=tf.AUTO_REUSE),
                name='gen_y_bn_4',
                reuse=tf.AUTO_REUSE,
                training=self.is_train))

        h5 = tf.layers.conv3d_transpose(
                    tf.concat([h4, h4_], -1),
                    filters=self.dim_W5,
                    kernel_size=(self.kernel4[0], self.kernel4[1],
                                 self.kernel4[2]),
                    strides=(self.stride[1], self.stride[2], self.stride[3]),
                    padding='same',
                    name='gen_y_5',
                    reuse=tf.AUTO_REUSE)

        x = softmax(h5, self.batch_size, self.vox_shape)
        return x
	"""

    def generate_part(self, Z):

        if self.discriminative is True:
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
        elif self.discriminative is False:
            h2 = Z

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

        x = softmax(h5, self.batch_size, self.part_shape)
        return x, h3, h4

        """
        Z_ = tf.reshape(Z, [self.batch_size, -1])
        h1 = tf.nn.relu(
            tf.layers.batch_normalization(
                tf.layers.dense(
                    Z_, 
                    units=self.start_vox_size[0]*self.start_vox_size[1]*self.start_vox_size[2]*self.dim_W1,
                    use_bias=False,
                    name='gen_x_1',
                    reuse=tf.AUTO_REUSE),
                name='gen_x_bn_1',
                reuse=tf.AUTO_REUSE,
                training=self.is_train))
        h1 = tf.reshape(h1, [
            self.batch_size, self.start_vox_size[0], self.start_vox_size[1],
            self.start_vox_size[2], self.dim_W1
        ])

        h2 = tf.nn.relu(
            tf.layers.batch_normalization(
                tf.layers.conv3d_transpose(
                    h1,
                    filters=self.dim_W2,
                    kernel_size=(self.kernel4[0], self.kernel4[1],
                                 self.kernel4[2]),
                    strides=(self.stride[1], self.stride[2], self.stride[3]),
                    padding='same',
                    name='gen_x_2',
                    reuse=tf.AUTO_REUSE),
                name='gen_x_bn_2',
                reuse=tf.AUTO_REUSE,
                training=self.is_train))

        h3 = tf.nn.relu(
            tf.layers.batch_normalization(
                tf.layers.conv3d_transpose(
                    h2,
                    filters=self.dim_W3,
                    kernel_size=(self.kernel4[0], self.kernel4[1],
                                 self.kernel4[2]),
                    strides=(self.stride[1], self.stride[2], self.stride[3]),
                    padding='same',
                    name='gen_x_3',
                    reuse=tf.AUTO_REUSE),
                name='gen_x_bn_3',
                reuse=tf.AUTO_REUSE,
                training=self.is_train))

        h4 = tf.nn.relu(
            tf.layers.batch_normalization(
                tf.layers.conv3d_transpose(
                    h3,
                    filters=self.dim_W4,
                    kernel_size=(self.kernel4[0], self.kernel4[1],
                                 self.kernel4[2]),
                    strides=(self.stride[1], self.stride[2], self.stride[3]),
                    padding='same',
                    name='gen_x_4',
                    reuse=tf.AUTO_REUSE),
                name='gen_x_bn_4',
                reuse=tf.AUTO_REUSE,
                training=self.is_train))

        h5 = tf.layers.conv3d_transpose(
                    h4,
                    filters=self.part_shape[-1],
                    kernel_size=(self.kernel4[0], self.kernel4[1],
                                 self.kernel4[2]),
                    strides=(self.stride[1], self.stride[2], self.stride[3]),
                    padding='same',
                    name='gen_x_5',
                    reuse=tf.AUTO_REUSE)

        # x = softmax(h5, self.batch_size, self.part_shape)
        return h5, h2, h3, h4
	"""

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

        complete = softmax(
            out, self.batch_size,
            [self.vox_shape[0], self.vox_shape[1], self.vox_shape[2], 2])

        return complete

    def samples_generator(self, visual_size):
        Z = tf.placeholder(tf.float32, [
            visual_size, self.start_vox_size[0], self.start_vox_size[1],
            self.start_vox_size[2], self.dim_z
        ])

        Z_ = tf.reshape(Z, [visual_size, -1])
        part, h3_t, h4_t = self.generate_part(Z_)
        full, full_ref = self.generate_full(Z_, h3_t, h4_t)
        # complete = self.complete(full)
        return Z, full, full_ref, part

