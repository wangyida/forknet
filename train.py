import numpy as np
import tensorflow as tf

from config import cfg
from util import DataProcess, scene_model_id_pair
from model import depvox_gan

from colorama import init
from termcolor import colored

init()


def learning_rate(rate, step):
    if step < rate[1]:
        lr = rate[0]
    else:
        lr = rate[2]
    return lr


def train(n_epochs, learning_rate_G, learning_rate_D, batch_size, mid_flag,
          check_num):
    beta_G = cfg.TRAIN.ADAM_BETA_G
    beta_D = cfg.TRAIN.ADAM_BETA_D
    n_vox = cfg.CONST.N_VOX
    dim = cfg.NET.DIM
    vox_shape = [n_vox[0], n_vox[1], n_vox[2], dim[-1]]
    part_shape = [n_vox[0], n_vox[1], n_vox[2], 2]
    dim_z = cfg.NET.DIM_Z
    start_vox_size = cfg.NET.START_VOX
    kernel = cfg.NET.KERNEL
    stride = cfg.NET.STRIDE
    dilations = cfg.NET.DILATIONS
    freq = cfg.CHECK_FREQ
    record_vox_num = cfg.RECORD_VOX_NUM
    discriminative = cfg.NET.DISCRIMINATIVE

    depvox_gan_model = depvox_gan(
        batch_size=batch_size,
        vox_shape=vox_shape,
        part_shape=part_shape,
        dim_z=dim_z,
        dim=dim,
        start_vox_size=start_vox_size,
        kernel=kernel,
        stride=stride,
        dilations=dilations,
        discriminative=discriminative,
        is_train=True)

    Z_tf, z_part_enc_tf, full_tf, full_gen_tf, full_gen_dec_tf, full_gen_dec_ref_tf,\
    gen_loss_tf, discrim_loss_tf, recons_loss_tf, encode_loss_tf, refine_loss_tf, summary_tf,\
    part_tf, complete_gen_tf, complete_gen_decode_tf = depvox_gan_model.build_model()
    global_step = tf.Variable(0, name='global_step', trainable=False)
    config_gpu = tf.ConfigProto()
    config_gpu.gpu_options.allow_growth = True
    sess = tf.Session(config=config_gpu)
    saver = tf.train.Saver(max_to_keep=cfg.SAVER_MAX)

    data_paths = scene_model_id_pair(dataset_portion=cfg.TRAIN.DATASET_PORTION)
    print '---amount of data:' + str(len(data_paths))
    data_process = DataProcess(data_paths, batch_size, repeat=True)

    encode_vars = filter(lambda x: x.name.startswith('enc'),
                         tf.trainable_variables())
    discrim_vars = filter(lambda x: x.name.startswith('discrim'),
                          tf.trainable_variables())
    gen_vars = filter(lambda x: x.name.startswith('gen'),
                      tf.trainable_variables())
    refine_vars = filter(lambda x: x.name.startswith('gen_y_ref'),
                         tf.trainable_variables())

    lr_VAE = tf.placeholder(tf.float32, shape=[])

    # main optimiser
    train_op_pred = tf.train.AdamOptimizer(
        learning_rate_G, beta1=beta_G, beta2=0.9).minimize(
            recons_loss_tf, var_list=encode_vars + gen_vars)

    # variational optimiser
    train_op_encode = tf.train.AdamOptimizer(
        lr_VAE, beta1=beta_D, beta2=0.9).minimize(
            encode_loss_tf, var_list=encode_vars + gen_vars)

    # refine optimiser
    train_op_refine = tf.train.AdamOptimizer(
        learning_rate_G, beta1=beta_G, beta2=0.9).minimize(
            refine_loss_tf, var_list=refine_vars)

    if discriminative is True:
        train_op_gen = tf.train.AdamOptimizer(
            learning_rate_G, beta1=beta_G, beta2=0.9).minimize(
                gen_loss_tf, var_list=gen_vars)
        train_op_discrim = tf.train.AdamOptimizer(
            learning_rate_D, beta1=beta_D, beta2=0.9).minimize(
                discrim_loss_tf,
                var_list=discrim_vars,
                global_step=global_step)

    if discriminative is True:
        Z_tf_sample, full_tf_sample, full_ref_tf_sample, part_tf_sample = depvox_gan_model.samples_generator(
            visual_size=batch_size)

    writer = tf.summary.FileWriter(cfg.DIR.LOG_PATH, sess.graph_def)
    tf.initialize_all_variables().run(session=sess)

    if mid_flag:
        chckpt_path = cfg.DIR.CHECK_PT_PATH + str(check_num)
        saver.restore(sess, chckpt_path)
        Z_var_np_sample = np.load(cfg.DIR.TRAIN_OBJ_PATH +
                                  '/sample_z.npy').astype(np.float32)
        """
        Z_var_np_sample = np.random.normal(
            size=(batch_size, start_vox_size[0], start_vox_size[1],
                  start_vox_size[2], dim_z)).astype(np.float32)
        """
        Z_var_np_sample = Z_var_np_sample[:batch_size]
        print '---weights restored'
    else:
        Z_var_np_sample = np.random.normal(
            size=(batch_size, start_vox_size[0], start_vox_size[1],
                  start_vox_size[2], dim_z)).astype(np.float32)
        np.save(cfg.DIR.TRAIN_OBJ_PATH + '/sample_z.npy', Z_var_np_sample)

    ite = check_num * freq + 1
    cur_epochs = int(ite / int(len(data_paths) / batch_size))

    #training
    for epoch in np.arange(cur_epochs, n_epochs):
        epoch_flag = True
        while epoch_flag:
            print colored('---Iteration:%d, epoch:%d', 'blue') % (ite, epoch)
            db_inds, epoch_flag = data_process.get_next_minibatch()
            batch_voxel = data_process.get_voxel(db_inds)
            batch_tsdf = data_process.get_tsdf(db_inds)

            if cfg.TYPE_TASK == 'scene':
                # Evaluation masks
                space_effective = np.clip(
                    np.where(batch_voxel > 0, 1, 0) + np.where(
                        batch_tsdf > -1.01, 1, 0), 0, 1)
                batch_voxel *= space_effective
                batch_tsdf *= space_effective
                # occluded region
                batch_tsdf[batch_tsdf < -1] = 0

            lr = learning_rate(cfg.LEARNING_RATE_V, ite)

            batch_z_var = np.random.normal(
                size=(batch_size, start_vox_size[0], start_vox_size[1],
                      start_vox_size[2], dim_z)).astype(np.float32)

            # updating for the main network
            """
            _, _, _ = sess.run(
                [train_op_encode, train_op_pred, train_op_refine],
                feed_dict={
                    Z_tf: batch_z_var,
                    full_tf: batch_voxel,
                    part_tf: batch_tsdf,
                    lr_VAE: lr
                },
            )
            """
            _, _ = sess.run(
                [train_op_pred, train_op_refine],
                feed_dict={
                    Z_tf: batch_z_var,
                    full_tf: batch_voxel,
                    part_tf: batch_tsdf,
                    lr_VAE: lr
                },
            )
            gen_vae_loss_val, z_part_enc_val = sess.run(
                [recons_loss_tf, z_part_enc_tf],
                feed_dict={
                    Z_tf: batch_z_var,
                    full_tf: batch_voxel,
                    part_tf: batch_tsdf,
                    lr_VAE: lr
                },
            )

            if discriminative:
                # for s in range(2):
                _, gen_loss_val = sess.run(
                    [train_op_gen, gen_loss_tf],
                    feed_dict={
                        Z_tf: batch_z_var,
                        full_tf: batch_voxel,
                        part_tf: batch_tsdf,
                        lr_VAE: lr
                    },
                )
                discrim_loss_val = sess.run(
                    discrim_loss_tf,
                    feed_dict={
                        Z_tf: batch_z_var,
                        full_tf: batch_voxel,
                        part_tf: batch_tsdf,
                    },
                )
                # if discrim_loss_val > 0.01:
                _ = sess.run(
                    train_op_discrim,
                    feed_dict={
                        Z_tf: batch_z_var,
                        full_tf: batch_voxel,
                        part_tf: batch_tsdf,
                    },
                )

            print(colored('gan', 'red'))
            print '    reconstruct loss:', gen_vae_loss_val if (
                'gen_vae_loss_val' in locals()) else 'None'

            print '            gen loss:', gen_loss_val if (
                'gen_loss_val' in locals()) else 'None'

            print '      output discrim:', discrim_loss_val if (
                'discrim_loss_val' in locals()) else 'None'

            print '     avarage of code:', np.mean(
                np.mean(z_part_enc_val,
                        4)) if ('z_part_enc_val' in locals()) else 'None'

            print '         std of code:', np.mean(
                np.std(z_part_enc_val,
                       4)) if ('z_part_enc_val' in locals()) else 'None'

            if np.mod(ite, freq) == 0:
                if discriminative is True:
                    full_models = sess.run(
                        full_tf_sample,
                        feed_dict={Z_tf_sample: Z_var_np_sample},
                    )
                    full_models_cat = np.argmax(full_models, axis=4)
                    record_vox = full_models_cat[:record_vox_num]
                    np.save(
                        cfg.DIR.TRAIN_OBJ_PATH + '/' + str(ite / freq) +
                        '.npy', record_vox)
                save_path = saver.save(
                    sess,
                    cfg.DIR.CHECK_PT_PATH + str(ite / freq),
                    global_step=None)

            ite += 1
