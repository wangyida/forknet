import numpy as np

from config import cfg
import tensorflow as tf
from util import *
from metric import sparse_ml


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
                 complete_shape=[80, 48, 80, 2],
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
        self.complete_shape = complete_shape
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
            tf.random_normal([1, 1, 1, self.dim_W1, self.dim_z], stddev=0.02),
            name='gen_y_W1')

        self.gen_y_W2 = tf.Variable(
            tf.random_normal([
                self.kernel2[0], self.kernel2[1], self.kernel2[2], self.dim_W2,
                self.dim_W1
            ],
                             stddev=0.02),
            name='gen_y_W2')

        self.gen_y_W3 = tf.Variable(
            tf.random_normal([
                self.kernel3[0], self.kernel3[1], self.kernel3[2], self.dim_W3,
                self.dim_W2
            ],
                             stddev=0.02),
            name='gen_y_W3')

        self.gen_y_W4 = tf.Variable(
            tf.random_normal([
                self.kernel4[0], self.kernel4[1], self.kernel4[2], self.dim_W4,
                self.dim_W3 * 2
            ],
                             stddev=0.02),
            name='gen_y_W4')

        self.gen_y_W5 = tf.Variable(
            tf.random_normal([
                self.kernel5[0], self.kernel5[1], self.kernel5[2], self.dim_W5,
                self.dim_W4 * 2
            ],
                             stddev=0.02),
            name='gen_y_W5')

        self.dis_y_W1 = tf.Variable(
            tf.random_normal([
                self.kernel5[0], self.kernel5[1], self.kernel5[2], self.dim_W5,
                self.dim_W4
            ],
                             stddev=0.02),
            name='dis_y_vox_W1')
        self.dis_y_bn_g1 = tf.Variable(
            tf.random_normal([1], mean=1.0, stddev=0.02),
            name='dis_y_vox_bn_g1')
        self.dis_y_bn_b1 = tf.Variable(tf.zeros([1]), name='dis_y_vox_bn_b1')

        # parameters of discriminator
        self.dis_y_W2 = tf.Variable(
            tf.random_normal([
                self.kernel4[0], self.kernel4[1], self.kernel4[2], self.dim_W4,
                self.dim_W3
            ],
                             stddev=0.02),
            name='dis_y_vox_W2')
        self.dis_y_bn_g2 = tf.Variable(
            tf.random_normal([1], mean=1.0, stddev=0.02),
            name='dis_y_vox_bn_g2')
        self.dis_y_bn_b2 = tf.Variable(tf.zeros([1]), name='dis_y_vox_bn_b2')

        self.dis_y_W3 = tf.Variable(
            tf.random_normal([
                self.kernel3[0], self.kernel3[1], self.kernel3[2], self.dim_W3,
                self.dim_W2
            ],
                             stddev=0.02),
            name='dis_y_vox_W3')
        self.dis_y_bn_g3 = tf.Variable(
            tf.random_normal([1], mean=1.0, stddev=0.02),
            name='dis_y_vox_bn_g3')
        self.dis_y_bn_b3 = tf.Variable(tf.zeros([1]), name='dis_y_vox_bn_b3')

        self.dis_y_W4 = tf.Variable(
            tf.random_normal([
                self.kernel2[0], self.kernel2[1], self.kernel2[2], self.dim_W2,
                self.dim_W1
            ],
                             stddev=0.02),
            name='dis_y_vox_W4')
        self.dis_y_bn_g4 = tf.Variable(
            tf.random_normal([1], mean=1.0, stddev=0.02),
            name='dis_y_vox_bn_g4')
        self.dis_y_bn_b4 = tf.Variable(tf.zeros([1]), name='dis_y_vox_bn_b4')

        # patch GAN
        self.dis_y_W5 = tf.Variable(
            tf.random_normal([
                self.start_vox_size[0] * self.start_vox_size[1] *
                self.start_vox_size[2] * self.dim_W1, 1
            ],
                             stddev=0.02),
            name='dis_y_vox_W5')

        self.dis_g_W1 = tf.Variable(
            tf.random_normal([
                self.kernel5[0], self.kernel5[1], self.kernel5[2],
                self.complete_shape[3], self.dim_W4
            ],
                             stddev=0.02),
            name='dis_g_vox_W1')
        self.dis_g_bn_g1 = tf.Variable(
            tf.random_normal([1], mean=1.0, stddev=0.02),
            name='dis_g_vox_bn_g1')
        self.dis_g_bn_b1 = tf.Variable(tf.zeros([1]), name='dis_g_vox_bn_b1')

        # parameters of discriminator
        self.dis_g_W2 = tf.Variable(
            tf.random_normal([
                self.kernel4[0], self.kernel4[1], self.kernel4[2], self.dim_W4,
                self.dim_W3
            ],
                             stddev=0.02),
            name='dis_g_vox_W2')
        self.dis_g_bn_g2 = tf.Variable(
            tf.random_normal([1], mean=1.0, stddev=0.02),
            name='dis_g_vox_bn_g2')
        self.dis_g_bn_b2 = tf.Variable(tf.zeros([1]), name='dis_g_vox_bn_b2')

        self.dis_g_W3 = tf.Variable(
            tf.random_normal([
                self.kernel3[0], self.kernel3[1], self.kernel3[2], self.dim_W3,
                self.dim_W2
            ],
                             stddev=0.02),
            name='dis_g_vox_W3')
        self.dis_g_bn_g3 = tf.Variable(
            tf.random_normal([1], mean=1.0, stddev=0.02),
            name='dis_g_vox_bn_g3')
        self.dis_g_bn_b3 = tf.Variable(tf.zeros([1]), name='dis_g_vox_bn_b3')

        self.dis_g_W4 = tf.Variable(
            tf.random_normal([
                self.kernel2[0], self.kernel2[1], self.kernel2[2], self.dim_W2,
                self.dim_W1
            ],
                             stddev=0.02),
            name='dis_g_vox_W4')
        self.dis_g_bn_g4 = tf.Variable(
            tf.random_normal([1], mean=1.0, stddev=0.02),
            name='dis_g_vox_bn_g4')
        self.dis_g_bn_b4 = tf.Variable(tf.zeros([1]), name='dis_g_vox_bn_b4')

        # patch GAN
        self.dis_g_W5 = tf.Variable(
            tf.random_normal([
                self.start_vox_size[0] * self.start_vox_size[1] *
                self.start_vox_size[2] * self.dim_W1, 1
            ],
                             stddev=0.02),
            name='dis_g_vox_W5')

        # parameters of generator x
        self.gen_x_W1 = tf.Variable(
            tf.random_normal([1, 1, 1, self.dim_W1, self.dim_z], stddev=0.02),
            name='gen_x_W1')

        self.gen_x_W2 = tf.Variable(
            tf.random_normal([
                self.kernel2[0], self.kernel2[1], self.kernel2[2], self.dim_W2,
                self.dim_W1
            ],
                             stddev=0.02),
            name='gen_x_W2')

        self.gen_x_W3 = tf.Variable(
            tf.random_normal([
                self.kernel3[0], self.kernel3[1], self.kernel3[2], self.dim_W3,
                self.dim_W2
            ],
                             stddev=0.02),
            name='gen_x_W3')

        self.gen_x_W4 = tf.Variable(
            tf.random_normal([
                self.kernel4[0], self.kernel4[1], self.kernel4[2], self.dim_W4,
                self.dim_W3
            ],
                             stddev=0.02),
            name='gen_x_W4')

        self.gen_x_W5 = tf.Variable(
            tf.random_normal([
                self.kernel5[0], self.kernel5[1], self.kernel5[2],
                self.complete_shape[-1], self.dim_W4
            ],
                             stddev=0.02),
            name='gen_x_W5')

        # parameters of generator sdf
        self.gen_z_W1 = tf.Variable(
            tf.random_normal([1, 1, 1, self.dim_W1, self.dim_z], stddev=0.02),
            name='gen_z_W1')

        self.gen_z_W2 = tf.Variable(
            tf.random_normal([
                self.kernel2[0], self.kernel2[1], self.kernel2[2], self.dim_W2,
                self.dim_W1
            ],
                             stddev=0.02),
            name='gen_z_W2')

        self.gen_z_W3 = tf.Variable(
            tf.random_normal([
                self.kernel3[0], self.kernel3[1], self.kernel3[2], self.dim_W3,
                self.dim_W2
            ],
                             stddev=0.02),
            name='gen_z_W3')

        self.gen_z_W4 = tf.Variable(
            tf.random_normal([
                self.kernel4[0], self.kernel4[1], self.kernel4[2], self.dim_W4,
                self.dim_W3
            ],
                             stddev=0.02),
            name='gen_z_W4')

        self.gen_z_W5 = tf.Variable(
            tf.random_normal([
                self.kernel5[0], self.kernel5[1], self.kernel5[2], 1,
                self.dim_W4
            ],
                             stddev=0.02),
            name='gen_z_W5')

        # parameters of discriminator
        self.dis_x_W1 = tf.Variable(
            tf.random_normal([
                self.kernel5[0], self.kernel5[1], self.kernel5[2], 1,
                self.dim_W4
            ],
                             stddev=0.02),
            name='dis_x_vox_W1')
        self.dis_x_bn_g1 = tf.Variable(
            tf.random_normal([1], mean=1.0, stddev=0.02),
            name='dis_x_vox_bn_g1')
        self.dis_x_bn_b1 = tf.Variable(tf.zeros([1]), name='dis_x_vox_bn_b1')

        self.dis_x_W2 = tf.Variable(
            tf.random_normal([
                self.kernel4[0], self.kernel4[1], self.kernel4[2], self.dim_W4,
                self.dim_W3
            ],
                             stddev=0.02),
            name='dis_x_vox_W2')
        self.dis_x_bn_g2 = tf.Variable(
            tf.random_normal([1], mean=1.0, stddev=0.02),
            name='dis_x_vox_bn_g2')
        self.dis_x_bn_b2 = tf.Variable(tf.zeros([1]), name='dis_x_vox_bn_b2')

        self.dis_x_W3 = tf.Variable(
            tf.random_normal([
                self.kernel3[0], self.kernel3[1], self.kernel3[2], self.dim_W3,
                self.dim_W2
            ],
                             stddev=0.02),
            name='dis_x_vox_W3')
        self.dis_x_bn_g3 = tf.Variable(
            tf.random_normal([1], mean=1.0, stddev=0.02),
            name='dis_x_vox_bn_g3')
        self.dis_x_bn_b3 = tf.Variable(tf.zeros([1]), name='dis_x_vox_bn_b3')

        self.dis_x_W4 = tf.Variable(
            tf.random_normal([
                self.kernel2[0], self.kernel2[1], self.kernel2[2], self.dim_W2,
                self.dim_W1
            ],
                             stddev=0.02),
            name='dis_x_vox_W4')
        self.dis_x_bn_g4 = tf.Variable(
            tf.random_normal([1], mean=1.0, stddev=0.02),
            name='dis_x_vox_bn_g4')
        self.dis_x_bn_b4 = tf.Variable(tf.zeros([1]), name='dis_x_vox_bn_b4')

        # patch GAN
        self.dis_x_W5 = tf.Variable(
            tf.random_normal([
                self.start_vox_size[0] * self.start_vox_size[1] *
                self.start_vox_size[2] * self.dim_W1, 1
            ],
                             stddev=0.02),
            name='dis_x_vox_W5')

        self.saver = tf.train.Saver()

    def build_model(self):

        part_gt_ = tf.placeholder(
            tf.float32,
            [None, self.vox_shape[0], self.vox_shape[1], self.vox_shape[2]])
        part_gt = tf.expand_dims(part_gt_, -1)

        full_gt_ = tf.placeholder(
            tf.int32,
            [None, self.vox_shape[0], self.vox_shape[1], self.vox_shape[2]])
        full_gt = tf.cast(tf.one_hot(full_gt_, self.n_class), tf.float32)

        surf_gt_ = tf.placeholder(
            tf.int32,
            [None, self.vox_shape[0], self.vox_shape[1], self.vox_shape[2]])
        surf_gt_tmp = tf.cast(tf.one_hot(surf_gt_, self.n_class), tf.float32)
        surf_gt = tf.concat([
            tf.expand_dims(full_gt[:, :, :, :, 0], -1),
            surf_gt_tmp[:, :, :, :, 1:]
        ], -1)

        comp_gt_ = tf.clip_by_value(
            full_gt_ + tf.dtypes.cast(tf.math.round(part_gt_), tf.int32),
            clip_value_min=0,
            clip_value_max=1)
        comp_gt = tf.one_hot(comp_gt_, 2)
        comp_gt = tf.cast(comp_gt, tf.float32)

        Z = tf.placeholder(tf.float32, [
            None, self.start_vox_size[0], self.start_vox_size[1],
            self.start_vox_size[2], self.dim_z
        ])

        # weights for balancing training
        batch_mean_full_gt = tf.reduce_mean(full_gt, [0, 1, 2, 3])
        ones = tf.ones_like(batch_mean_full_gt)
        inverse = tf.div(ones, tf.add(batch_mean_full_gt, ones))
        weight_full = inverse * tf.div(1., tf.reduce_sum(inverse))

        batch_mean_surf_gt = tf.reduce_mean(surf_gt, [0, 1, 2, 3])
        ones = tf.ones_like(batch_mean_surf_gt)
        inverse = tf.div(ones, tf.add(batch_mean_surf_gt, ones))
        weight_surf = inverse * tf.div(1., tf.reduce_sum(inverse))

        batch_mean_comp_gt = tf.reduce_mean(comp_gt, [0, 1, 2, 3])
        ones = tf.ones_like(batch_mean_comp_gt)
        inverse = tf.div(ones, tf.add(batch_mean_comp_gt, ones))
        weight_complete = inverse * tf.div(1., tf.reduce_sum(inverse))

        # encode from tsdf and vox
        Z_encode, Z_mu, Z_log_sigma, sscnet = self.encoder(part_gt)
        variation_loss = -0.5 * tf.reduce_sum(
            1.0 + 2.0 * Z_log_sigma - tf.square(Z_mu) - tf.exp(
                2.0 * Z_log_sigma), [1, 2, 3, 4])
        dim_code = Z_mu.get_shape().as_list()

        comp_dec, h3_t, h4_t, h5_t = self.generate_comp(Z_encode)
        surf_dec, full_dec = self.generate_full(Z_encode, h3_t, h4_t, h5_t)

        # complete = self.complete(full_dec)
        # encode again from loops
        """
        full_dec_o = tf.one_hot(
            tf.argmax(full_dec, axis=4, output_type=tf.int32),
            self.n_class)
        full_dec_o = tf.cast(full_dec_o, tf.float32)
        Z_encode_full_part = self.encoder(part_gen_dec)

        part_cc_dec, _, _, _ = self.generate_comp(Z_encode_full)
        _, h2_vt, h3_vt, h4_vt = self.generate_comp(Z_encode_full_part)
        full_cc_dec = self.generate_full(Z_encode_full_part, h3_vt,
                                         h4_vt)
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
            -tf.reduce_sum(
                self.lamda_gamma * part_gt * tf.log(1e-6 + part_vae_dec) +
                (1 - self.lamda_gamma) * (1 - part_gt) * tf.log(1e-6 + 1 - part_vae_dec), [1, 2, 3, 4]))
        """
        # latent consistency
        """
        recons_vae_loss += tf.reduce_mean(
            tf.reduce_sum(
                tf.squared_difference(Z_encode, Z_encode_full_part),
                [1, 2, 3, 4]))
        recons_vae_loss += tf.reduce_mean(
            tf.reduce_sum(
                tf.squared_difference(Z_encode, Z_encode_full),
                [1, 2, 3, 4]))
        recons_vae_loss += tf.reduce_mean(
            tf.reduce_sum(
                tf.squared_difference(Z_encode_full, Z_encode_full),
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
                tf.squared_difference(Z_encode_full, Z_encode_full_part),
                [1, 2, 3, 4]))
        """
        # latent consistency
        # SUPERVISED (paired data)
        # complete
        recons_com_loss = tf.reduce_sum(
            -tf.reduce_sum(
                self.lamda_gamma * comp_gt * tf.log(1e-6 + comp_dec) +
                (1 - self.lamda_gamma) *
                (1 - comp_gt) * tf.log(1e-6 + 1 - comp_dec), [1, 2, 3]) *
            weight_complete, 1)
        # geometric semantic scene completion (sscnet ops)
        recons_ssc_loss = tf.reduce_sum(
            -tf.reduce_sum(
                self.lamda_gamma * full_gt * tf.log(1e-6 + sscnet) +
                self.lamda_gamma * surf_gt * tf.log(1e-6 + sscnet) +
                (1 - self.lamda_gamma) *
                (1 - full_gt) * tf.log(1e-6 + 1 - sscnet), [1, 2, 3]) *
            weight_full, 1)
        # generative semantic scene completion (from latent features)
        recons_sem_loss = tf.reduce_sum(
            -tf.reduce_sum(
                self.lamda_gamma * full_gt * tf.log(1e-6 + surf_dec) +
                self.lamda_gamma * surf_gt * tf.log(1e-6 + surf_dec) +
                (1 - self.lamda_gamma) *
                (1 - surf_gt) * tf.log(1e-6 + 1 - surf_dec), [1, 2, 3]) *
            weight_surf, 1)
        # refine for segmentation
        refine_loss = tf.reduce_mean(
            tf.reduce_sum(
                -tf.reduce_sum(
                    self.lamda_gamma * full_gt * tf.log(1e-6 + full_dec) +
                    self.lamda_gamma * surf_gt * tf.log(1e-6 + full_dec) +
                    (1 - self.lamda_gamma) *
                    (1 - full_gt) * tf.log(1e-6 + 1 - full_dec), [1, 2, 3]) *
                weight_full, 1))
        """
        recons_loss += tf.reduce_sum(
                tf.squared_difference(part_gt, part_gen_dec), [1, 2, 3, 4])
        """
        # from scene, the observed surface can also be produced
        # latent consistency
        """
        recons_loss += tf.reduce_sum(
                tf.squared_difference(Z_encode, Z_encode_full),
                [1, 2, 3, 4])
        """
        # latent consistency

        # GAN_generate
        if self.discriminative is True:
            part_dec = self.generate_part(Z_encode)
            part_gen = self.generate_part(Z)
            comp_gen, h3_z, h4_z, h5_t = self.generate_comp(Z)
            full_gen, full_gen_ref = self.generate_full(Z, h3_z, h4_z, h5_t)

            recons_sdf_loss = tf.reduce_mean(
                tf.reduce_sum(
                    tf.squared_difference(part_gt, part_dec), [1, 2, 3, 4]))

            h_part_gt = self.discriminate_part(part_gt)
            h_part_gen = self.discriminate_part(part_gen)

            h_comp_gt = self.discriminate_comp(comp_gt)
            h_comp_gen = self.discriminate_comp(comp_gen)

            h_full_gt = self.discriminate_full(full_gt)
            h_full_gen = self.discriminate_full(full_gen)

            scores = tf.squeeze([
                tf.reduce_mean(tf.sigmoid(h_part_gt)),
                tf.reduce_mean(tf.sigmoid(h_part_gen)),
                tf.reduce_mean(tf.sigmoid(h_comp_gt)),
                tf.reduce_mean(tf.sigmoid(h_comp_gen)),
                tf.reduce_mean(tf.sigmoid(h_full_gt)),
                tf.reduce_mean(tf.sigmoid(h_full_gen)),
            ])

            # Standard_GAN_Loss
            dis_loss = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(
                    logits=h_full_gt,
                    labels=tf.ones_like(h_full_gt))) + tf.reduce_mean(
                        tf.nn.sigmoid_cross_entropy_with_logits(
                            logits=h_full_gen,
                            labels=tf.zeros_like(h_full_gen)))
            """
            dis_loss += tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(
                    logits=h_full_dec, labels=tf.zeros_like(h_full_dec)))
            """

            dis_loss += tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(
                    logits=h_part_gt,
                    labels=tf.ones_like(h_part_gt))) + tf.reduce_mean(
                        tf.nn.sigmoid_cross_entropy_with_logits(
                            logits=h_part_gen,
                            labels=tf.zeros_like(h_part_gen)))
            """
            dis_loss += tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(
                    logits=h_part_dec, labels=tf.zeros_like(h_part_dec)))
            """

            dis_loss += tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(
                    logits=h_comp_gt,
                    labels=tf.ones_like(h_comp_gt))) + tf.reduce_mean(
                        tf.nn.sigmoid_cross_entropy_with_logits(
                            logits=h_comp_gen,
                            labels=tf.zeros_like(h_comp_gen)))

            gen_loss = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(
                    logits=h_full_gen, labels=tf.ones_like(h_full_gen)))
            """
            gen_loss += tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(
                    logits=h_full_dec, labels=tf.ones_like(h_full_dec)))
            """

            gen_loss += tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(
                    logits=h_part_gen, labels=tf.ones_like(h_part_gen)))
            """
            gen_loss += tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(
                    logits=h_part_dec, labels=tf.ones_like(h_part_dec)))
            """

            gen_loss += tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(
                    logits=h_comp_gen, labels=tf.ones_like(h_comp_gen)))

        else:
            part_dec = part_gt
            scores = tf.zeros([6])
            comp_gen = comp_dec
            full_gen = surf_dec
            gen_loss = tf.zeros([1])
            dis_loss = tf.zeros([1])

        # variational cost
        # variation_loss = tf.reduce_mean(variation_loss)

        # main cost
        recons_ssc_loss = self.lamda_recons * tf.reduce_mean(recons_ssc_loss)
        if self.discriminative is True:
            recons_com_loss = self.lamda_recons * tf.reduce_mean(
                recons_com_loss +
                variation_loss) + self.lamda_recons * recons_sdf_loss
            recons_sem_loss = self.lamda_recons * tf.reduce_mean(
                recons_sem_loss +
                variation_loss) + self.lamda_recons * recons_sdf_loss
        elif self.discriminative is False:
            recons_com_loss = self.lamda_recons * tf.reduce_mean(
                recons_com_loss)
            recons_sem_loss = self.lamda_recons * tf.reduce_mean(
                recons_sem_loss)

        summary_op = tf.summary.merge_all()

        return Z, Z_encode, surf_gt_, full_gt_, full_gen, surf_dec, full_dec,\
        gen_loss, dis_loss, recons_ssc_loss, recons_com_loss, recons_sem_loss, variation_loss, refine_loss, summary_op,\
        part_gt_, part_dec, comp_gt, comp_gen, comp_dec, sscnet, scores

    def encoder(self, sdf):

        h1_base = lrelu(
            tf.layers.conv3d(
                sdf,
                filters=16,
                kernel_size=(7, 7, 7),
                strides=(2, 2, 2),
                padding='same',
                name='enc_ssc_1_base',
                reuse=tf.AUTO_REUSE))
        h1_0 = lrelu(
            tf.layers.conv3d(
                h1_base,
                filters=32,
                kernel_size=(1, 1, 1),
                strides=(1, 1, 1),
                padding='same',
                name='enc_ssc_1_0',
                reuse=tf.AUTO_REUSE))
        h1_1 = lrelu(
            tf.layers.conv3d(
                h1_base,
                filters=32,
                kernel_size=(3, 3, 3),
                strides=(1, 1, 1),
                padding='same',
                name='enc_ssc_1_1',
                reuse=tf.AUTO_REUSE))
        h1_2 = lrelu(
            tf.layers.conv3d(
                h1_1,
                filters=32,
                kernel_size=(3, 3, 3),
                strides=(1, 1, 1),
                padding='same',
                name='enc_ssc_1_2',
                reuse=tf.AUTO_REUSE))

        h2_0 = h1_0 + h1_2
        h2_1 = tf.layers.conv3d(
            h2_0,
            filters=64,
            kernel_size=(3, 3, 3),
            strides=(1, 1, 1),
            padding='same',
            dilation_rate=(1, 1, 1),
            name='enc_ssc_2_1',
            reuse=tf.AUTO_REUSE)

        h2_2 = tf.layers.conv3d(
            h2_1,
            filters=64,
            kernel_size=(3, 3, 3),
            strides=(1, 1, 1),
            padding='same',
            dilation_rate=(1, 1, 1),
            name='enc_ssc_2_2',
            reuse=tf.AUTO_REUSE)

        h3_0 = h2_2 + tf.layers.conv3d(
            h2_1,
            filters=64,
            kernel_size=(3, 3, 3),
            strides=(1, 1, 1),
            padding='same',
            dilation_rate=(1, 1, 1),
            name='enc_ssc_3_0',
            reuse=tf.AUTO_REUSE)

        h3_1 = tf.layers.conv3d(
            h3_0,
            filters=64,
            kernel_size=(3, 3, 3),
            strides=(1, 1, 1),
            padding='same',
            dilation_rate=(1, 1, 1),
            name='enc_ssc_3_1',
            reuse=tf.AUTO_REUSE)

        h4_0 = h3_1 + tf.layers.conv3d(
            h3_1,
            filters=64,
            kernel_size=(3, 3, 3),
            strides=(1, 1, 1),
            padding='same',
            dilation_rate=(1, 1, 1),
            name='enc_ssc_4_0',
            reuse=tf.AUTO_REUSE)

        h4_1 = tf.layers.conv3d(
            h4_0,
            filters=64,
            kernel_size=(3, 3, 3),
            strides=(1, 1, 1),
            padding='same',
            dilation_rate=(2, 2, 2),
            name='enc_ssc_4_1',
            reuse=tf.AUTO_REUSE)

        h5_0 = h4_1 + tf.layers.conv3d(
            h4_1,
            filters=64,
            kernel_size=(3, 3, 3),
            strides=(1, 1, 1),
            padding='same',
            dilation_rate=(2, 2, 2),
            name='enc_ssc_5_0',
            reuse=tf.AUTO_REUSE)

        h5_1 = tf.layers.conv3d(
            h5_0,
            filters=64,
            kernel_size=(3, 3, 3),
            strides=(1, 1, 1),
            padding='same',
            dilation_rate=(2, 2, 2),
            name='enc_ssc_5_1',
            reuse=tf.AUTO_REUSE)

        h6_0 = h5_1 + tf.layers.conv3d(
            h5_1,
            filters=64,
            kernel_size=(3, 3, 3),
            strides=(1, 1, 1),
            padding='same',
            dilation_rate=(2, 2, 2),
            name='enc_ssc_6_0',
            reuse=tf.AUTO_REUSE)
        h7_0 = tf.concat([h4_0, h5_0, h6_0], -1)

        h7_1 = tf.layers.conv3d_transpose(
            h7_0,
            filters=64,
            kernel_size=(4, 4, 4),
            strides=(2, 2, 2),
            padding='same',
            name='enc_ssc_7_1',
            reuse=tf.AUTO_REUSE)

        h7_2 = tf.layers.conv3d(
            tf.concat([h7_1, sdf], -1),
            filters=128,
            kernel_size=(1, 1, 1),
            strides=(1, 1, 1),
            padding='same',
            dilation_rate=(1, 1, 1),
            name='enc_ssc_7_2',
            reuse=tf.AUTO_REUSE)
        h7_3 = tf.layers.conv3d(
            h7_2,
            filters=128,
            kernel_size=(1, 1, 1),
            strides=(1, 1, 1),
            padding='same',
            dilation_rate=(1, 1, 1),
            name='enc_ssc_7_3',
            reuse=tf.AUTO_REUSE)
        h7_4 = tf.layers.conv3d(
            h7_3,
            filters=self.vox_shape[-1],
            kernel_size=(1, 1, 1),
            strides=(1, 1, 1),
            padding='same',
            dilation_rate=(1, 1, 1),
            name='enc_ssc_7_4',
            reuse=tf.AUTO_REUSE)

        ssc = softmax(h7_4, self.batch_size, self.vox_shape)

        h1 = tf.layers.conv3d(
            ssc,
            filters=self.dim_W4,
            kernel_size=(self.kernel5[0], self.kernel5[1], self.kernel5[2]),
            strides=(self.stride[1], self.stride[2], self.stride[3]),
            padding='same',
            name='enc_x_1',
            reuse=tf.AUTO_REUSE)

        h2 = lrelu(
            tf.layers.conv3d(
                h1,
                filters=self.dim_W3,
                kernel_size=(self.kernel4[0], self.kernel4[1],
                             self.kernel4[2]),
                strides=(self.stride[1], self.stride[2], self.stride[3]),
                padding='same',
                name='enc_x_2',
                reuse=tf.AUTO_REUSE))

        h3 = lrelu(
            tf.layers.conv3d(
                h2,
                filters=self.dim_W2,
                kernel_size=(self.kernel3[0], self.kernel3[1],
                             self.kernel3[2]),
                strides=(self.stride[1], self.stride[2], self.stride[3]),
                padding='same',
                name='enc_x_3',
                reuse=tf.AUTO_REUSE))

        if self.discriminative is True:
            h4 = lrelu(
                tf.layers.conv3d(
                    h3,
                    filters=self.dim_W1,
                    kernel_size=(self.kernel2[0], self.kernel2[1],
                                 self.kernel2[2]),
                    strides=(self.stride[1], self.stride[2], self.stride[3]),
                    padding='same',
                    name='enc_x_4',
                    reuse=tf.AUTO_REUSE))

            h5 = tf.layers.conv3d(
                h4,
                filters=self.dim_z,
                kernel_size=(1, 1, 1),
                strides=(1, 1, 1),
                padding='same',
                name='enc_x_5',
                reuse=tf.AUTO_REUSE)

            # start to add hidden layers
            dims = h5.get_shape().as_list()
            n_code = dims[1] * dims[2] * dims[3] * dims[4]
            # flattened = tf.contrib.layers.flatten(h5)
            # epsilon = tf.random_normal(tf.stack([tf.shape(h5)[0], n_code]))
            epsilon = tf.random_normal(
                [self.batch_size, dims[1], dims[2], dims[3], dims[4]])
            z_mu = tf.layers.conv3d(
                h5,
                filters=self.dim_z,
                kernel_size=(1, 1, 1),
                strides=(1, 1, 1),
                padding='same',
                name='enc_x_mu',
                reuse=tf.AUTO_REUSE)
            z_log_sigma = 0.5 * tf.layers.conv3d(
                h5,
                filters=self.dim_z,
                kernel_size=(1, 1, 1),
                strides=(1, 1, 1),
                padding='same',
                name='enc_x_log_sigma',
                reuse=tf.AUTO_REUSE)
            if self.is_train is True:
                z = tf.add(
                    z_mu,
                    tf.multiply(epsilon, tf.exp(z_log_sigma)),
                    name='enc_x_z')
            elif self.is_train is False:
                z = z_mu
        elif self.discriminative is False:
            z = h3
            z_mu = z
            z_log_sigma = z
        return z, z_mu, z_log_sigma, ssc

    def generate_full(self, Z, h3_, h4_, h5_):

        if self.discriminative is True:
            dims = Z.get_shape().as_list()
            vox_size_l1 = self.start_vox_size
            output_shape_l1 = [
                self.batch_size, vox_size_l1[0], vox_size_l1[1],
                vox_size_l1[2], self.dim_W1
            ]
            h1 = tf.nn.conv3d_transpose(
                Z,
                self.gen_y_W1,
                output_shape=output_shape_l1,
                strides=[1, 1, 1, 1, 1])

            vox_size_l2 = self.start_vox_size * 2
            output_shape_l2 = [
                self.batch_size, vox_size_l2[0], vox_size_l2[1],
                vox_size_l2[2], self.dim_W2
            ]
            h2 = tf.nn.relu(
                tf.nn.conv3d_transpose(
                    h1,
                    self.gen_y_W2,
                    output_shape=output_shape_l2,
                    strides=self.stride))
        elif self.discriminative is False:
            h2 = Z

        vox_size_l3 = self.start_vox_size * 4
        output_shape_l3 = [
            self.batch_size, vox_size_l3[0], vox_size_l3[1], vox_size_l3[2],
            self.dim_W3
        ]

        h3 = tf.nn.relu(
            tf.nn.conv3d_transpose(
                h2,
                self.gen_y_W3,
                output_shape=output_shape_l3,
                strides=self.stride))

        vox_size_l4 = self.start_vox_size * 8
        output_shape_l4 = [
            self.batch_size, vox_size_l4[0], vox_size_l4[1], vox_size_l4[2],
            self.dim_W4
        ]
        h4 = tf.nn.relu(
            tf.nn.conv3d_transpose(
                tf.concat([h3, h3_], -1),
                self.gen_y_W4,
                output_shape=output_shape_l4,
                strides=self.stride))

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

        #res1
        res1_1 = tf.nn.relu(
            tf.layers.conv3d(
                tf.concat([h5, h5_], -1),
                filters=16,
                kernel_size=3,
                padding='same',
                name='gen_y_res_1_1',
                reuse=tf.AUTO_REUSE))
        res1_2 = tf.nn.relu(
            tf.layers.conv3d(
                res1_1,
                filters=output_shape_l5[-1],
                kernel_size=3,
                padding='same',
                name='gen_y_res_1_2',
                reuse=tf.AUTO_REUSE))

        res1 = tf.nn.relu(tf.add(h5, res1_2))

        res_1_post = tf.layers.conv3d(
            res1,
            filters=self.vox_shape[-1],
            kernel_size=3,
            padding='same',
            name='gen_y_res_1_post',
            reuse=tf.AUTO_REUSE)

        stage1 = softmax(res_1_post, self.batch_size, self.vox_shape)

        # start to refine
        base = tf.nn.relu(
            tf.layers.conv3d(
                stage1,
                filters=16,
                kernel_size=3,
                padding='same',
                name='gen_y_ref_base',
                reuse=tf.AUTO_REUSE))

        #res2
        res2_1 = tf.nn.relu(
            tf.layers.conv3d(
                base,
                filters=16,
                kernel_size=3,
                padding='same',
                name='gen_y_ref_2_1',
                reuse=tf.AUTO_REUSE))
        res2_2 = tf.nn.relu(
            tf.layers.conv3d(
                res2_1,
                filters=16,
                kernel_size=3,
                padding='same',
                name='gen_y_ref_2_2',
                reuse=tf.AUTO_REUSE))

        res2 = tf.nn.relu(tf.add(base, res2_2))

        #res3
        res3_1 = tf.nn.relu(
            tf.layers.conv3d(
                res2,
                filters=16,
                kernel_size=3,
                padding='same',
                name='gen_y_ref_3_1',
                reuse=tf.AUTO_REUSE))
        res3_2 = tf.nn.relu(
            tf.layers.conv3d(
                res3_1,
                filters=16,
                kernel_size=3,
                padding='same',
                name='gen_y_ref_3_2',
                reuse=tf.AUTO_REUSE))

        res3 = tf.nn.relu(tf.add(res2, res3_2))

        #res4
        res4_1 = tf.nn.relu(
            tf.layers.conv3d(
                res3,
                filters=16,
                kernel_size=3,
                padding='same',
                name='gen_y_ref_4_1',
                reuse=tf.AUTO_REUSE))
        res4_2 = tf.nn.relu(
            tf.layers.conv3d(
                res4_1,
                filters=16,
                kernel_size=3,
                padding='same',
                name='gen_y_ref_4_2',
                reuse=tf.AUTO_REUSE))

        res4 = tf.nn.relu(tf.add(res3, res4_2))

        res_4_post = tf.layers.conv3d(
            res4,
            filters=self.vox_shape[-1],
            kernel_size=3,
            padding='same',
            name='gen_y_ref_4_post',
            reuse=tf.AUTO_REUSE)
        stage2 = softmax(res_4_post, self.batch_size, self.vox_shape)

        return stage1, stage2

    def generate_comp(self, Z):

        if self.discriminative is True:
            dims = Z.get_shape().as_list()
            vox_size_l1 = self.start_vox_size
            output_shape_l1 = [
                self.batch_size, vox_size_l1[0], vox_size_l1[1],
                vox_size_l1[2], self.dim_W1
            ]
            h1 = tf.nn.conv3d_transpose(
                Z,
                self.gen_x_W1,
                output_shape=output_shape_l1,
                strides=[1, 1, 1, 1, 1])

            vox_size_l2 = self.start_vox_size * 2
            output_shape_l2 = [
                self.batch_size, vox_size_l2[0], vox_size_l2[1],
                vox_size_l2[2], self.dim_W2
            ]
            h2 = tf.nn.relu(
                tf.nn.conv3d_transpose(
                    h1,
                    self.gen_x_W2,
                    output_shape=output_shape_l2,
                    strides=self.stride))
        elif self.discriminative is False:
            h2 = Z

        vox_size_l3 = self.start_vox_size * 4
        output_shape_l3 = [
            self.batch_size, vox_size_l3[0], vox_size_l3[1], vox_size_l3[2],
            self.dim_W3
        ]
        h3 = tf.nn.relu(
            tf.nn.conv3d_transpose(
                h2,
                self.gen_x_W3,
                output_shape=output_shape_l3,
                strides=self.stride))

        vox_size_l4 = self.start_vox_size * 8
        output_shape_l4 = [
            self.batch_size, vox_size_l4[0], vox_size_l4[1], vox_size_l4[2],
            self.dim_W4
        ]
        h4 = tf.nn.relu(
            tf.nn.conv3d_transpose(
                h3,
                self.gen_x_W4,
                output_shape=output_shape_l4,
                strides=self.stride))

        vox_size_l5 = self.start_vox_size * 16
        output_shape_l5 = [
            self.batch_size, vox_size_l5[0], vox_size_l5[1], vox_size_l5[2],
            self.complete_shape[-1]
        ]
        h5 = tf.nn.conv3d_transpose(
            h4,
            self.gen_x_W5,
            output_shape=output_shape_l5,
            strides=self.stride)

        # start to refine
        base = tf.nn.relu(
            tf.layers.conv3d(
                h5,
                filters=16,
                kernel_size=3,
                padding='same',
                name='gen_x_res_base',
                reuse=tf.AUTO_REUSE))

        #res1
        res1_1 = tf.nn.relu(
            tf.layers.conv3d(
                base,
                filters=16,
                kernel_size=3,
                padding='same',
                name='gen_x_res_1_1',
                reuse=tf.AUTO_REUSE))
        res1_2 = tf.nn.relu(
            tf.layers.conv3d(
                res1_1,
                filters=16,
                kernel_size=3,
                padding='same',
                name='gen_x_res_1_2',
                reuse=tf.AUTO_REUSE))

        res1 = tf.nn.relu(tf.add(base, res1_2))

        res_1_post = tf.layers.conv3d(
            res1,
            filters=self.complete_shape[-1],
            kernel_size=3,
            padding='same',
            name='gen_x_res_1_post',
            reuse=tf.AUTO_REUSE)

        stage1 = softmax(res_1_post, self.batch_size, self.complete_shape)
        return stage1, h3, h4, h5

    def generate_part(self, Z):

        if self.discriminative is True:
            dims = Z.get_shape().as_list()
            vox_size_l1 = self.start_vox_size
            output_shape_l1 = [
                self.batch_size, vox_size_l1[0], vox_size_l1[1],
                vox_size_l1[2], self.dim_W1
            ]
            h1 = tf.nn.conv3d_transpose(
                Z,
                self.gen_z_W1,
                output_shape=output_shape_l1,
                strides=[1, 1, 1, 1, 1])

            vox_size_l2 = self.start_vox_size * 2
            output_shape_l2 = [
                self.batch_size, vox_size_l2[0], vox_size_l2[1],
                vox_size_l2[2], self.dim_W2
            ]
            h2 = tf.nn.relu(
                tf.nn.conv3d_transpose(
                    h1,
                    self.gen_z_W2,
                    output_shape=output_shape_l2,
                    strides=self.stride))
        elif self.discriminative is False:
            h2 = Z

        vox_size_l3 = self.start_vox_size * 4
        output_shape_l3 = [
            self.batch_size, vox_size_l3[0], vox_size_l3[1], vox_size_l3[2],
            self.dim_W3
        ]
        h3 = tf.nn.relu(
            tf.nn.conv3d_transpose(
                h2,
                self.gen_z_W3,
                output_shape=output_shape_l3,
                strides=self.stride))

        vox_size_l4 = self.start_vox_size * 8
        output_shape_l4 = [
            self.batch_size, vox_size_l4[0], vox_size_l4[1], vox_size_l4[2],
            self.dim_W4
        ]
        h4 = tf.nn.relu(
            tf.nn.conv3d_transpose(
                h3,
                self.gen_z_W4,
                output_shape=output_shape_l4,
                strides=self.stride))

        vox_size_l5 = self.start_vox_size * 16
        output_shape_l5 = [
            self.batch_size, vox_size_l5[0], vox_size_l5[1], vox_size_l5[2], 1
        ]
        h5 = tf.nn.conv3d_transpose(
            h4,
            self.gen_z_W5,
            output_shape=output_shape_l5,
            strides=self.stride)

        return h5

    def discriminate_full(self, vox):

        h1 = lrelu(
            tf.layers.conv3d(
                vox,
                filters=16,
                kernel_size=(7, 7, 7),
                strides=(2, 2, 2),
                padding='same',
                name='dis_y_ssc_0',
                reuse=tf.AUTO_REUSE))
        h1_0 = lrelu(
            tf.layers.conv3d(
                h1,
                filters=32,
                kernel_size=(1, 1, 1),
                strides=(1, 1, 1),
                padding='same',
                name='dis_y_ssc_0_0',
                reuse=tf.AUTO_REUSE))
        h1_1 = lrelu(
            tf.layers.conv3d(
                h1,
                filters=32,
                kernel_size=(3, 3, 3),
                strides=(1, 1, 1),
                padding='same',
                name='dis_y_ssc_0_1',
                reuse=tf.AUTO_REUSE))
        h1_2 = lrelu(
            tf.layers.conv3d(
                h1_1,
                filters=32,
                kernel_size=(3, 3, 3),
                strides=(1, 1, 1),
                padding='same',
                name='dis_y_ssc_0_2',
                reuse=tf.AUTO_REUSE))

        base_4 = h1_0 + h1_2
        base_5 = tf.layers.conv3d(
            base_4,
            filters=64,
            kernel_size=(3, 3, 3),
            strides=(1, 1, 1),
            padding='same',
            dilation_rate=(2, 2, 2),
            name='dis_y_ssc_1',
            reuse=tf.AUTO_REUSE)

        base_6 = base_5 + tf.layers.conv3d(
            base_5,
            filters=64,
            kernel_size=(3, 3, 3),
            strides=(1, 1, 1),
            padding='same',
            dilation_rate=(2, 2, 2),
            name='dis_y_ssc_2',
            reuse=tf.AUTO_REUSE)

        base_7 = tf.layers.conv3d(
            base_6,
            filters=64,
            kernel_size=(3, 3, 3),
            strides=(1, 1, 1),
            padding='same',
            dilation_rate=(2, 2, 2),
            name='dis_y_ssc_3',
            reuse=tf.AUTO_REUSE)

        base_8 = base_7 + tf.layers.conv3d(
            base_7,
            filters=64,
            kernel_size=(3, 3, 3),
            strides=(1, 1, 1),
            padding='same',
            dilation_rate=(2, 2, 2),
            name='dis_y_ssc_4',
            reuse=tf.AUTO_REUSE)
        base_9 = tf.concat([base_4, base_6, base_8], -1)

        h1_1 = tf.layers.conv3d(
            base_9,
            filters=128,
            kernel_size=(1, 1, 1),
            strides=(1, 1, 1),
            padding='same',
            dilation_rate=(1, 1, 1),
            name='dis_y_ssc_5',
            reuse=tf.AUTO_REUSE)
        h1_2 = tf.layers.conv3d(
            h1_1,
            filters=32,
            kernel_size=(1, 1, 1),
            strides=(1, 1, 1),
            padding='same',
            dilation_rate=(1, 1, 1),
            name='dis_y_ssc_6',
            reuse=tf.AUTO_REUSE)

        h2 = lrelu(
            layernormalize(
                tf.nn.conv3d(
                    h1_2,
                    self.dis_y_W2,
                    strides=self.stride,
                    dilations=self.dilations,
                    padding='SAME'),
                g=self.dis_y_bn_g2,
                b=self.dis_y_bn_b2))
        h3 = lrelu(
            layernormalize(
                tf.nn.conv3d(
                    h2,
                    self.dis_y_W3,
                    strides=self.stride,
                    dilations=self.dilations,
                    padding='SAME'),
                g=self.dis_y_bn_g3,
                b=self.dis_y_bn_b3))
        h4 = lrelu(
            layernormalize(
                tf.nn.conv3d(
                    h3,
                    self.dis_y_W4,
                    strides=self.stride,
                    dilations=self.dilations,
                    padding='SAME'),
                g=self.dis_y_bn_g4,
                b=self.dis_y_bn_b4))

        h5 = tf.layers.conv3d(
            h4,
            filters=1,
            kernel_size=(1, 1, 1),
            strides=(1, 1, 1),
            padding='same',
            dilation_rate=(1, 1, 1),
            name='dis_y_final',
            reuse=tf.AUTO_REUSE)
        # h4 = tf.reshape(h4, [self.batch_size, -1])
        # h5 = tf.matmul(h4, self.dis_y_W5)
        y = tf.nn.sigmoid(h5)

        return h5

    def discriminate_comp(self, vox):

        h1 = lrelu(
            tf.layers.conv3d(
                vox,
                filters=16,
                kernel_size=(7, 7, 7),
                strides=(2, 2, 2),
                padding='same',
                name='dis_g_ssc_0',
                reuse=tf.AUTO_REUSE))
        h1_0 = lrelu(
            tf.layers.conv3d(
                h1,
                filters=32,
                kernel_size=(1, 1, 1),
                strides=(1, 1, 1),
                padding='same',
                name='dis_g_ssc_0_0',
                reuse=tf.AUTO_REUSE))
        h1_1 = lrelu(
            tf.layers.conv3d(
                h1,
                filters=32,
                kernel_size=(3, 3, 3),
                strides=(1, 1, 1),
                padding='same',
                name='dis_g_ssc_0_1',
                reuse=tf.AUTO_REUSE))
        h1_2 = lrelu(
            tf.layers.conv3d(
                h1_1,
                filters=32,
                kernel_size=(3, 3, 3),
                strides=(1, 1, 1),
                padding='same',
                name='dis_g_ssc_0_2',
                reuse=tf.AUTO_REUSE))

        base_4 = h1_0 + h1_2
        base_5 = tf.layers.conv3d(
            base_4,
            filters=64,
            kernel_size=(3, 3, 3),
            strides=(1, 1, 1),
            padding='same',
            dilation_rate=(2, 2, 2),
            name='dis_g_ssc_1',
            reuse=tf.AUTO_REUSE)

        base_6 = base_5 + tf.layers.conv3d(
            base_5,
            filters=64,
            kernel_size=(3, 3, 3),
            strides=(1, 1, 1),
            padding='same',
            dilation_rate=(2, 2, 2),
            name='dis_g_ssc_2',
            reuse=tf.AUTO_REUSE)

        base_7 = tf.layers.conv3d(
            base_6,
            filters=64,
            kernel_size=(3, 3, 3),
            strides=(1, 1, 1),
            padding='same',
            dilation_rate=(2, 2, 2),
            name='dis_g_ssc_3',
            reuse=tf.AUTO_REUSE)

        base_8 = base_7 + tf.layers.conv3d(
            base_7,
            filters=64,
            kernel_size=(3, 3, 3),
            strides=(1, 1, 1),
            padding='same',
            dilation_rate=(2, 2, 2),
            name='dis_g_ssc_4',
            reuse=tf.AUTO_REUSE)
        base_9 = tf.concat([base_4, base_6, base_8], -1)

        h1_1 = tf.layers.conv3d(
            base_9,
            filters=128,
            kernel_size=(1, 1, 1),
            strides=(1, 1, 1),
            padding='same',
            dilation_rate=(1, 1, 1),
            name='dis_g_ssc_5',
            reuse=tf.AUTO_REUSE)
        h1_2 = tf.layers.conv3d(
            h1_1,
            filters=32,
            kernel_size=(1, 1, 1),
            strides=(1, 1, 1),
            padding='same',
            dilation_rate=(1, 1, 1),
            name='dis_g_ssc_6',
            reuse=tf.AUTO_REUSE)

        h2 = lrelu(
            layernormalize(
                tf.nn.conv3d(
                    h1_2,
                    self.dis_g_W2,
                    strides=self.stride,
                    dilations=self.dilations,
                    padding='SAME'),
                g=self.dis_g_bn_g2,
                b=self.dis_g_bn_b2))
        h3 = lrelu(
            layernormalize(
                tf.nn.conv3d(
                    h2,
                    self.dis_g_W3,
                    strides=self.stride,
                    dilations=self.dilations,
                    padding='SAME'),
                g=self.dis_g_bn_g3,
                b=self.dis_g_bn_b3))
        h4 = lrelu(
            layernormalize(
                tf.nn.conv3d(
                    h3,
                    self.dis_g_W4,
                    strides=self.stride,
                    dilations=self.dilations,
                    padding='SAME'),
                g=self.dis_g_bn_g4,
                b=self.dis_g_bn_b4))

        h5 = tf.layers.conv3d(
            h4,
            filters=1,
            kernel_size=(1, 1, 1),
            strides=(1, 1, 1),
            padding='same',
            dilation_rate=(1, 1, 1),
            name='dis_g_final',
            reuse=tf.AUTO_REUSE)
        # h4 = tf.reshape(h4, [self.batch_size, -1])
        # h5 = tf.matmul(h4, self.dis_g_W5)
        y = tf.nn.sigmoid(h5)

        return h5

    def discriminate_part(self, vox):

        h1 = lrelu(
            tf.nn.conv3d(
                vox,
                self.dis_x_W1,
                strides=self.stride,
                dilations=self.dilations,
                padding='SAME'))
        h2 = lrelu(
            layernormalize(
                tf.nn.conv3d(
                    h1,
                    self.dis_x_W2,
                    strides=self.stride,
                    dilations=self.dilations,
                    padding='SAME'),
                g=self.dis_x_bn_g2,
                b=self.dis_x_bn_b2))
        h3 = lrelu(
            layernormalize(
                tf.nn.conv3d(
                    h2,
                    self.dis_x_W3,
                    strides=self.stride,
                    dilations=self.dilations,
                    padding='SAME'),
                g=self.dis_x_bn_g3,
                b=self.dis_x_bn_b3))
        h4 = lrelu(
            layernormalize(
                tf.nn.conv3d(
                    h3,
                    self.dis_x_W4,
                    strides=self.stride,
                    dilations=self.dilations,
                    padding='SAME'),
                g=self.dis_x_bn_g4,
                b=self.dis_x_bn_b4))
        h4 = tf.reshape(h4, [self.batch_size, -1])
        h5 = tf.matmul(h4, self.dis_x_W5)
        y = tf.nn.sigmoid(h5)

        return h5

    def samples_generator(self, visual_size):
        Z = tf.placeholder(tf.float32, [
            visual_size, self.start_vox_size[0], self.start_vox_size[1],
            self.start_vox_size[2], self.dim_z
        ])

        part = self.generate_part(Z)
        comp, h3_t, h4_t, h5_t = self.generate_comp(Z)
        surf, full = self.generate_full(Z, h3_t, h4_t, h5_t)
        scores = tf.concat([
            tf.math.sigmoid(self.discriminate_part(part)),
            tf.math.sigmoid(
                tf.reduce_mean(self.discriminate_comp(comp), [1, 2, 3])),
            tf.math.sigmoid(
                tf.reduce_mean(self.discriminate_full(full), [1, 2, 3]))
        ], 0)
        return Z, comp, surf, full, part, scores
