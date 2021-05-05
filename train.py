import numpy as np
import tensorflow as tf

from config import cfg
from util import DataProcess, id_models_train
from model import network

from colorama import init
from termcolor import colored

init()
tf.compat.v1.disable_eager_execution()


def learning_rate(rate, step):
    if step < rate[1]:
        lr = rate[0]
    else:
        lr = rate[2]
    return lr


def train(n_epochs, learning_rate_G, learning_rate_D, batch_size, mid_flag,
          check_num, discriminative, data_list):
    beta_G = cfg.TRAIN.ADAM_BETA_G
    beta_D = cfg.TRAIN.ADAM_BETA_D
    n_vox = cfg.CONST.N_VOX
    dim = cfg.NET.DIM
    vox_shape = [n_vox[0], n_vox[1], n_vox[2], dim[-1]]
    com_shape = [n_vox[0], n_vox[1], n_vox[2], 2]
    dim_z = cfg.NET.DIM_Z
    start_vox_size = cfg.NET.START_VOX
    kernel = cfg.NET.KERNEL
    stride = cfg.NET.STRIDE
    dilations = cfg.NET.DILATIONS
    freq = cfg.CHECK_FREQ
    record_vox_num = cfg.RECORD_VOX_NUM

    network_model = network(
        batch_size=batch_size,
        vox_shape=vox_shape,
        com_shape=com_shape,
        dim_z=dim_z,
        dim=dim,
        start_vox_size=start_vox_size,
        kernel=kernel,
        stride=stride,
        dilations=dilations,
        discriminative=discriminative,
        is_train=True)

    Z_tf, z_part_enc_tf, surf_tf, full_tf, full_gen_tf, surf_dec_tf, full_dec_tf,\
    gen_loss_tf, discrim_loss_tf, recons_ssc_loss_tf, recons_com_loss_tf, recons_sem_loss_tf, encode_loss_tf, refine_loss_tf, summary_tf,\
    space_effective_tf, part_tf, part_dec_tf, comp_gt_tf, comp_gen_tf, comp_dec_tf, sscnet_tf, scores_tf = network_model.build_model()
    global_step = tf.Variable(0, name='global_step', trainable=False)
    config_gpu = tf.compat.v1.ConfigProto()
    config_gpu.gpu_options.allow_growth = True
    sess = tf.compat.v1.Session(config=config_gpu)
    saver = tf.compat.v1.train.Saver(max_to_keep=cfg.SAVER_MAX)

    data_paths = id_models_train(
        dataset_portion=cfg.TRAIN.DATASET_PORTION, data_list=data_list)
    print(colored('The amount of data: %d' % len(data_paths), 'green'))
    data_process = DataProcess(data_paths, batch_size, repeat=True)

    enc_sscnet_vars = list(
        filter(lambda x: x.name.startswith('enc_ssc'),
               tf.compat.v1.trainable_variables()))
    enc_sdf_vars = list(
        filter(lambda x: x.name.startswith('enc_x'),
               tf.compat.v1.trainable_variables()))
    dis_sdf_vars = list(
        filter(lambda x: x.name.startswith('dis_x'),
               tf.compat.v1.trainable_variables()))
    dis_com_vars = list(
        filter(lambda x: x.name.startswith('dis_g'),
               tf.compat.v1.trainable_variables()))
    dis_sem_vars = list(
        filter(lambda x: x.name.startswith('dis_y'),
               tf.compat.v1.trainable_variables()))
    gen_com_vars = list(
        filter(lambda x: x.name.startswith('gen_x'),
               tf.compat.v1.trainable_variables()))
    gen_sem_vars = list(
        filter(lambda x: x.name.startswith('gen_y'),
               tf.compat.v1.trainable_variables()))
    gen_sdf_vars = list(
        filter(lambda x: x.name.startswith('gen_z'),
               tf.compat.v1.trainable_variables()))
    refine_vars = list(
        filter(lambda x: x.name.startswith('gen_y_ref'),
               tf.compat.v1.trainable_variables()))

    lr_VAE = tf.compat.v1.placeholder(tf.float32, shape=[])

    # main optimiser
    train_op_pd_sscnet = tf.compat.v1.train.AdamOptimizer(
        learning_rate_G, beta1=beta_G, beta2=0.9).minimize(
            recons_ssc_loss_tf, var_list=enc_sscnet_vars)
    train_op_pd_com = tf.compat.v1.train.AdamOptimizer(
        learning_rate_G, beta1=beta_G, beta2=0.9).minimize(
            recons_com_loss_tf,
            var_list=enc_sdf_vars + gen_com_vars + gen_sdf_vars)
    train_op_pd_sem = tf.compat.v1.train.AdamOptimizer(
        learning_rate_G, beta1=beta_G, beta2=0.9).minimize(
            recons_sem_loss_tf,
            var_list=enc_sdf_vars + gen_sem_vars + gen_sdf_vars)

    # refine optimiser
    train_op_refine = tf.compat.v1.train.AdamOptimizer(
        learning_rate_G, beta1=beta_G, beta2=0.9).minimize(
            refine_loss_tf, var_list=refine_vars)

    if discriminative is True:
        train_op_gen_sdf = tf.compat.v1.train.AdamOptimizer(
            learning_rate_G, beta1=beta_G, beta2=0.9).minimize(
                gen_loss_tf, var_list=gen_sdf_vars)
        train_op_gen_com = tf.compat.v1.train.AdamOptimizer(
            learning_rate_G, beta1=beta_G, beta2=0.9).minimize(
                gen_loss_tf, var_list=gen_com_vars)
        train_op_gen_sem = tf.compat.v1.train.AdamOptimizer(
            learning_rate_G, beta1=beta_G, beta2=0.9).minimize(
                gen_loss_tf, var_list=gen_sem_vars + gen_com_vars)
        train_op_dis_sdf = tf.compat.v1.train.AdamOptimizer(
            learning_rate_D, beta1=beta_D, beta2=0.9).minimize(
                discrim_loss_tf, var_list=dis_sdf_vars)
        train_op_dis_com = tf.compat.v1.train.AdamOptimizer(
            learning_rate_D, beta1=beta_D, beta2=0.9).minimize(
                discrim_loss_tf, var_list=dis_com_vars)
        train_op_dis_sem = tf.compat.v1.train.AdamOptimizer(
            learning_rate_D, beta1=beta_D, beta2=0.9).minimize(
                discrim_loss_tf,
                var_list=dis_sem_vars,
                global_step=global_step)

        Z_tf_samp, comp_tf_samp, full_tf_samp, full_ref_tf_samp, part_tf_samp, scores_tf_samp = network_model.samples_generator(
            visual_size=batch_size)

        model_path = cfg.DIR.CHECK_POINT_PATH + '-d'
    else:
        model_path = cfg.DIR.CHECK_POINT_PATH

    writer = tf.compat.v1.summary.FileWriter(cfg.DIR.LOG_PATH, sess.graph_def)
    tf.compat.v1.initialize_all_variables().run(session=sess)

    if mid_flag:
        chckpt_path = model_path + '/checkpoint' + str(check_num)
        saver.restore(sess, chckpt_path)
        print('---weights restored')

    Z_var_np_samp = np.random.normal(
        size=(batch_size, start_vox_size[0], start_vox_size[1],
              start_vox_size[2], dim_z)).astype(np.float32)
    np.save(cfg.DIR.TRAIN_OBJ_PATH + '/sample_z.npy', Z_var_np_samp)

    ite = check_num * freq + 1
    cur_epochs = int(ite / int(len(data_paths) / batch_size))

    #training
    for epoch in np.arange(cur_epochs, n_epochs):
        epoch_flag = True
        while epoch_flag:
            print(colored('---Iteration:%d, epoch:%d' % (ite, epoch), 'blue'))
            db_inds, epoch_flag = data_process.get_next_minibatch()
            bth_tsdf = data_process.get_tsdf(db_inds)
            bth_surf = data_process.get_surf(db_inds)
            bth_voxel = data_process.get_voxel(db_inds)

            # Evaluation masks
            # NOTICE that the target should never have negative values,
            # otherwise the one-hot coding never works for that region
            if cfg.dataset == 'scene' or 'fusion':
                space_effective = np.where(bth_tsdf > -1, 1, 0)
                """
                bth_voxel *= space_effective
                bth_tsdf *= space_effective
                # occluded region
                """
                bth_tsdf[bth_tsdf < -1] = 0
                bth_surf[bth_surf < 0] = 0
                bth_voxel[bth_voxel < 0] = 0

            lr = learning_rate(cfg.LEARNING_RATE_V, ite)

            bth_z_var = np.random.normal(
                size=(batch_size, start_vox_size[0], start_vox_size[1],
                      start_vox_size[2], dim_z)).astype(np.float32)

            # updating for the main network
            is_supervised = True
            if is_supervised is True:
                _, _, _, _ = sess.run(
                    [
                        train_op_pd_sscnet, train_op_pd_com, train_op_pd_sem,
                        train_op_refine
                    ],
                    feed_dict={
                        space_effective_tf: space_effective,
                        part_tf: bth_tsdf,
                        surf_tf: bth_surf,
                        full_tf: bth_voxel,
                        lr_VAE: lr
                    },
                )
            gen_com_loss_val, gen_sem_loss_val, z_part_enc_val = sess.run(
                [recons_com_loss_tf, recons_sem_loss_tf, z_part_enc_tf],
                feed_dict={
                    Z_tf: bth_z_var,
                    space_effective_tf: space_effective,
                    part_tf: bth_tsdf,
                    surf_tf: bth_surf,
                    full_tf: bth_voxel,
                    lr_VAE: lr
                },
            )

            if discriminative is True:
                discrim_loss_val, gen_loss_val, scores_discrim = sess.run(
                    [discrim_loss_tf, gen_loss_tf, scores_tf],
                    feed_dict={
                        Z_tf: bth_z_var,
                        space_effective_tf: space_effective,
                        part_tf: bth_tsdf,
                        surf_tf: bth_surf,
                        full_tf: bth_voxel,
                    },
                )
                if np.abs(scores_discrim[0] -
                          1) < 0.2 and np.abs(scores_discrim[1] - 0) < 0.2:
                    _ = sess.run(
                        train_op_gen_sdf,
                        feed_dict={
                            Z_tf: bth_z_var,
                            space_effective_tf: space_effective,
                            part_tf: bth_tsdf,
                            surf_tf: bth_surf,
                            full_tf: bth_voxel,
                            lr_VAE: lr
                        },
                    )
                # else:
                _ = sess.run(
                    train_op_dis_sdf,
                    feed_dict={
                        Z_tf: bth_z_var,
                        space_effective_tf: space_effective,
                        part_tf: bth_tsdf,
                        surf_tf: bth_surf,
                        full_tf: bth_voxel,
                    },
                )

                if np.abs(scores_discrim[2] -
                          1) < 0.2 and np.abs(scores_discrim[3] - 0) < 0.2:
                    _ = sess.run(
                        train_op_gen_com,
                        feed_dict={
                            Z_tf: bth_z_var,
                            space_effective_tf: space_effective,
                            part_tf: bth_tsdf,
                            surf_tf: bth_surf,
                            full_tf: bth_voxel,
                            lr_VAE: lr
                        },
                    )
                # else:
                _ = sess.run(
                    train_op_dis_com,
                    feed_dict={
                        Z_tf: bth_z_var,
                        space_effective_tf: space_effective,
                        part_tf: bth_tsdf,
                        surf_tf: bth_surf,
                        full_tf: bth_voxel,
                    },
                )

                if np.abs(scores_discrim[4] -
                          1) < 0.2 and np.abs(scores_discrim[5] - 0) < 0.2:
                    _ = sess.run(
                        train_op_gen_sem,
                        feed_dict={
                            Z_tf: bth_z_var,
                            space_effective_tf: space_effective,
                            part_tf: bth_tsdf,
                            surf_tf: bth_surf,
                            full_tf: bth_voxel,
                            lr_VAE: lr
                        },
                    )
                # else:
                _ = sess.run(
                    train_op_dis_sem,
                    feed_dict={
                        Z_tf: bth_z_var,
                        space_effective_tf: space_effective,
                        part_tf: bth_tsdf,
                        surf_tf: bth_surf,
                        full_tf: bth_voxel,
                    },
                )

            print('GAN')
            np.set_printoptions(precision=2)
            print('reconstruct-com loss: %.1f' % gen_com_loss_val)

            print('reconstruct-sem loss: %.1f' % gen_sem_loss_val)

            if discriminative is True:
                print('            gen loss: %.2f' % gen_loss_val if (
                    'gen_loss_val' in locals()) else 'None')

                print('      output discrim: %.2f' % discrim_loss_val if (
                    'discrim_loss_val' in locals()) else 'None')
                """
                print(
                    '      scores discrim:'
                    colored("%.2f" % scores_discrim[0], 'green'),
                    colored("%.2f" % scores_discrim[1], 'magenta'),
                    colored("%.2f" % scores_discrim[2], 'green'),
                    colored("%.2f" % scores_discrim[3], 'magenta'),
                    colored("%.2f" % scores_discrim[4], 'green'),
                    colored("%.2f" % scores_discrim[5], 'magenta') if
                    ('scores_discrim' in locals()) else 'None')
                """

            print('     avarage of code: %.2f' %
                  np.mean(np.mean(z_part_enc_val, 4)) if (
                      'z_part_enc_val' in locals()) else 'None')

            print('         std of code: %.2f' %
                  np.mean(np.std(z_part_enc_val, 4)) if (
                      'z_part_enc_val' in locals()) else 'None')

            if np.mod(ite, freq) == 0:
                if discriminative is True:
                    full_models = sess.run(
                        full_tf_samp,
                        feed_dict={Z_tf_samp: Z_var_np_samp},
                    )
                    full_models_cat = np.argmax(full_models, axis=4)
                    record_vox = full_models_cat[:record_vox_num]
                    np.save(
                        cfg.DIR.TRAIN_OBJ_PATH + '/' + str(ite // freq) +
                        '.npy', record_vox)
                save_path = saver.save(
                    sess,
                    model_path + '/checkpoint' + str(ite // freq),
                    global_step=None)

            ite += 1
