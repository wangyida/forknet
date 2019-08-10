import numpy as np
import tensorflow as tf

from config import cfg
from model import depvox_gan
from util import DataProcess, scene_model_id_pair, onehot, scene_model_id_pair_test
from sklearn.metrics import average_precision_score
import copy

from colorama import init
from termcolor import colored
from pca import pca

# use Colorama to make Termcolor work on Windows too
init()


def IoU(on_gt, on_pred, IoU_class, vox_shape):
    # calc_IoU
    if vox_shape[3] == 12:
        name_list = [
            'empty', 'ceili', 'floor', ' wall', 'windo', 'chair', '  bed',
            ' sofa', 'table', '  tvs', 'furni', 'objec'
        ]
    elif vox_shape[3] == 5:
        name_list = ['empty', 'bench', 'chair', 'couch', 'table']
    elif vox_shape[3] == 2:
        name_list = ['empty', 'objec']
    num = on_gt.shape[0]
    for class_n in np.arange(vox_shape[3]):
        on_pred_ = on_pred[:, :, :, :, class_n]
        on_gt_ = on_gt[:, :, :, :, class_n]
        mother = np.sum(np.clip(np.add(on_pred_, on_gt_), 0, 1), (0, 1, 2, 3))
        child = np.sum(np.multiply(on_pred_, on_gt_), (0, 1, 2, 3))

        IoU_calc = np.round(child / mother, 3)
        IoU_class[class_n] = IoU_calc
        print 'IoU of ' + name_list[class_n] + ':' + str(IoU_calc)
    if vox_shape[3] != 2:
        IoU_class[vox_shape[3]] = np.round(
            np.sum(IoU_class[1:vox_shape[3]]) / (vox_shape[3] - 1), 3)
    elif vox_shape[3] == 2:
        IoU_class[vox_shape[3]] = np.round(np.sum(IoU_class) / vox_shape[3], 3)
    print 'IoU average: ' + str(IoU_class[vox_shape[3]])
    print ''
    return IoU_class


def evaluate(batch_size, checknum, mode, discriminative):

    n_vox = cfg.CONST.N_VOX
    dim = cfg.NET.DIM
    vox_shape = [n_vox[0], n_vox[1], n_vox[2], dim[4]]
    complete_shape = [n_vox[0], n_vox[1], n_vox[2], 2]
    dim_z = cfg.NET.DIM_Z
    start_vox_size = cfg.NET.START_VOX
    kernel = cfg.NET.KERNEL
    stride = cfg.NET.STRIDE
    dilations = cfg.NET.DILATIONS
    freq = cfg.CHECK_FREQ

    save_path = cfg.DIR.EVAL_PATH
    if discriminative is True:
        model_path = cfg.DIR.CHECK_POINT_PATH + '-d'
    else:
        model_path = cfg.DIR.CHECK_POINT_PATH
    chckpt_path = model_path + '/checkpoint' + str(checknum)

    depvox_gan_model = depvox_gan(
        batch_size=batch_size,
        vox_shape=vox_shape,
        complete_shape=complete_shape,
        dim_z=dim_z,
        dim=dim,
        start_vox_size=start_vox_size,
        kernel=kernel,
        stride=stride,
        dilations=dilations,
        discriminative=discriminative,
        is_train=False)


    Z_tf, z_enc_tf, surf_tf, full_tf, full_gen_tf, surf_dec_tf, full_dec_tf,\
    gen_loss_tf, discrim_loss_tf, recons_ssc_loss_tf, recons_com_loss_tf, recons_sem_loss_tf, encode_loss_tf, refine_loss_tf, summary_tf,\
    part_tf, part_dec_tf, complete_gt_tf, complete_gen_tf, complete_dec_tf, ssc_tf, scores_tf = depvox_gan_model.build_model()
    if discriminative is True:
        Z_tf_sample, comp_tf_sample, surf_tf_sample, full_tf_sample, part_tf_sample, scores_tf_sample = depvox_gan_model.samples_generator(
            visual_size=batch_size)
    sess = tf.InteractiveSession()
    saver = tf.train.Saver()

    # Restore variables from disk.
    saver.restore(sess, chckpt_path)

    print("...Weights restored.")

    if mode == 'recons':
        # evaluation for reconstruction
        voxel_test, surf_test, part_test, num, data_paths = scene_model_id_pair_test(
            dataset_portion=cfg.TRAIN.DATASET_PORTION)

        # Evaluation masks
        if cfg.TYPE_TASK == 'scene':
            # occluded region
            """
            space_effective = np.where(voxel_test > -1, 1, 0) * np.where(
                part_test > -1, 1, 0)
            voxel_test *= space_effective
            part_test *= space_effective
            """
            # occluded region
            part_test[part_test < -1] = 0
            surf_test[surf_test < 0] = 0
            voxel_test[voxel_test < 0] = 0

        num = voxel_test.shape[0]
        print("test voxels loaded")
        for i in np.arange(int(num / batch_size)):
            batch_tsdf = part_test[i * batch_size:i * batch_size + batch_size]
            batch_surf = surf_test[i * batch_size:i * batch_size + batch_size]
            batch_voxel = voxel_test[i * batch_size:i * batch_size +
                                     batch_size]

            batch_pred_surf, batch_pred_full, batch_pred_part, batch_part_enc_Z, batch_complete_gt, batch_pred_complete, batch_ssc = sess.run(
                [
                    surf_dec_tf, full_dec_tf, part_dec_tf, z_enc_tf,
                    complete_gt_tf, complete_dec_tf, ssc_tf
                ],
                feed_dict={
                    part_tf: batch_tsdf,
                    surf_tf: batch_surf,
                    full_tf: batch_voxel
                })

            if i == 0:
                pred_part = batch_pred_part
                pred_surf = batch_pred_surf
                pred_full = batch_pred_full
                pred_ssc = batch_ssc
                part_enc_Z = batch_part_enc_Z
                complete_gt = batch_complete_gt
                pred_complete = batch_pred_complete
            else:
                pred_part = np.concatenate((pred_part, batch_pred_part),
                                           axis=0)
                pred_surf = np.concatenate((pred_surf, batch_pred_surf),
                                           axis=0)
                pred_full = np.concatenate((pred_full, batch_pred_full),
                                           axis=0)
                pred_ssc = np.concatenate((pred_ssc, batch_ssc), axis=0)
                part_enc_Z = np.concatenate((part_enc_Z, batch_part_enc_Z),
                                            axis=0)
                complete_gt = np.concatenate((complete_gt, batch_complete_gt),
                                             axis=0)
                pred_complete = np.concatenate(
                    (pred_complete, batch_pred_complete), axis=0)

        print("forwarded")

        # For visualization
        bin_file = np.uint8(voxel_test)
        bin_file.tofile(save_path + '/scene.bin')

        surface = np.array(part_test)
        if cfg.TYPE_TASK == 'scene':
            surface = np.abs(surface)
            surface *= 10
            pred_part = np.abs(pred_part)
            pred_part *= 10
        elif cfg.TYPE_TASK == 'object':
            surface = np.clip(surface, 0, 1)
            pred_part = np.clip(pred_part, 0, 1)
        surface.astype('uint8').tofile(save_path + '/surface.bin')
        pred_part.astype('uint8').tofile(save_path + '/dec_part.bin')

        depsem_gt = np.multiply(voxel_test, np.clip(surface, 0, 1))
        if cfg.TYPE_TASK == 'scene':
            depsem_gt[depsem_gt < 0] = 0
        depsem_gt.astype('uint8').tofile(save_path + '/depth_seg_scene.bin')

        # decoded
        np.argmax(
            pred_ssc,
            axis=4).astype('uint8').tofile(save_path + '/dec_ssc.bin')
        error = np.array(
            np.clip(np.argmax(pred_ssc, axis=4), 0, 1) +
            np.argmax(complete_gt, axis=4) * 2)
        error.astype('uint8').tofile(save_path + '/dec_ssc_error.bin')
        np.argmax(
            pred_surf,
            axis=4).astype('uint8').tofile(save_path + '/dec_surf.bin')
        error = np.array(
            np.clip(np.argmax(pred_surf, axis=4), 0, 1) +
            np.argmax(complete_gt, axis=4) * 2)
        error.astype('uint8').tofile(save_path + '/dec_surf_error.bin')
        np.argmax(
            pred_full,
            axis=4).astype('uint8').tofile(save_path + '/dec_full.bin')
        error = np.array(
            np.clip(np.argmax(pred_full, axis=4), 0, 1) +
            np.argmax(complete_gt, axis=4) * 2)
        error.astype('uint8').tofile(save_path + '/dec_full_error.bin')
        np.argmax(
            pred_complete,
            axis=4).astype('uint8').tofile(save_path + '/dec_complete.bin')
        np.argmax(
            complete_gt,
            axis=4).astype('uint8').tofile(save_path + '/complete_gt.bin')

        # reconstruction and generation from normal distribution evaluation
        # generator from random distribution
        if discriminative is True:
            np.save(save_path + '/decode_z.npy', part_enc_Z)
            sample_times = 10
            for j in np.arange(sample_times):
                Z_var_np_sample = np.random.normal(
                    size=(batch_size, start_vox_size[0], start_vox_size[1],
                          start_vox_size[2], dim_z)).astype(np.float32)

                z_comp_rand, z_surf_rand, z_full_rand, z_part_rand, scores_sample = sess.run(
                    [
                        comp_tf_sample, surf_tf_sample, full_tf_sample,
                        part_tf_sample, scores_tf_sample
                    ],
                    feed_dict={Z_tf_sample: Z_var_np_sample})
                if j == 0:
                    z_comp_rand_all = z_comp_rand
                    z_part_rand_all = z_part_rand
                    z_surf_rand_all = z_surf_rand
                    z_full_rand_all = z_full_rand
                else:
                    z_comp_rand_all = np.concatenate(
                        [z_comp_rand_all, z_comp_rand], axis=0)
                    z_part_rand_all = np.concatenate(
                        [z_part_rand_all, z_part_rand], axis=0)
                    z_surf_rand_all = np.concatenate(
                        [z_surf_rand_all, z_surf_rand], axis=0)
                    z_full_rand_all = np.concatenate(
                        [z_full_rand_all, z_full_rand], axis=0)
                    print(scores_sample)
            Z_var_np_sample.astype('float32').tofile(save_path +
                                                     '/sample_z.bin')
            np.argmax(
                z_comp_rand_all,
                axis=4).astype('uint8').tofile(save_path + '/gen_comp.bin')
            np.argmax(
                z_surf_rand_all,
                axis=4).astype('uint8').tofile(save_path + '/gen_surf.bin')
            np.argmax(
                z_full_rand_all,
                axis=4).astype('uint8').tofile(save_path + '/gen_full.bin')
            if cfg.TYPE_TASK == 'scene':
                z_part_rand_all = np.abs(z_part_rand_all)
                z_part_rand_all *= 10
            elif cfg.TYPE_TASK == 'object':
                z_part_rand_all[z_part_rand_all <= 0.4] = 0
                z_part_rand_all[z_part_rand_all > 0.4] = 1
                z_part_rand = np.squeeze(z_part_rand)
            z_part_rand_all.astype('uint8').tofile(save_path + '/gen_part.bin')

            eigen_shape = False
            if eigen_shape:
                z_U, z_V = pca(
                    np.reshape(part_enc_Z, [
                        200, start_vox_size[0] * start_vox_size[1] *
                        start_vox_size[2] * dim_z
                    ]),
                    dim_remain=200)
                z_V = np.reshape(
                    np.transpose(z_V[:, 0:8]), [
                        8, start_vox_size[0], start_vox_size[1],
                        start_vox_size[2], dim_z
                    ])
                z_surf_rand, z_full_rand, z_part_rand = sess.run(
                    [surf_tf_sample, full_tf_sample, part_tf_sample],
                    feed_dict={Z_tf_sample: z_V})
                np.argmax(
                    z_surf_rand,
                    axis=4).astype('uint8').tofile(save_path + '/gen_surf.bin')
                if cfg.TYPE_TASK == 'scene':
                    z_part_rand = np.abs(z_part_rand)
                    z_part_rand *= 10
                elif cfg.TYPE_TASK == 'object':
                    z_part_rand[z_part_rand <= 0.4] = 0
                    z_part_rand[z_part_rand > 0.4] = 1
                    z_part_rand = np.squeeze(z_part_rand)
                z_part_rand.astype('uint8').tofile(save_path + '/gen_sdf.bin')

        print("voxels saved")

        # numerical evalutation

        # calc_IoU
        # completion
        on_complete_gt = complete_gt
        complete_gen = np.argmax(pred_complete, axis=4)
        on_complete_gen = onehot(complete_gen, 2)
        IoU_comp = np.zeros([2 + 1])
        AP_comp = np.zeros([2 + 1])
        print(colored("Completion", 'cyan'))
        IoU_comp = IoU(on_complete_gt, on_complete_gen, IoU_comp,
                       [vox_shape[0], vox_shape[1], vox_shape[2], 2])

        # depth segmentation
        on_depsem_gt = onehot(depsem_gt, vox_shape[3])
        on_depsem_ssc = np.multiply(
            onehot(np.argmax(pred_ssc, axis=4), vox_shape[3]),
            np.expand_dims(np.clip(surface, 0, 1), -1))
        on_depsem_dec = np.multiply(
            onehot(np.argmax(pred_full, axis=4), vox_shape[3]),
            np.expand_dims(np.clip(surface, 0, 1), -1))
        print(colored("Geometric segmentation", 'cyan'))
        IoU_class = np.zeros([vox_shape[3] + 1])
        IoU_class = IoU(on_depsem_gt, on_depsem_ssc, IoU_class, vox_shape)
        IoU_all = np.expand_dims(IoU_class, axis=1)

        print(colored("Generative segmentation", 'cyan'))
        IoU_class = np.zeros([vox_shape[3] + 1])
        IoU_class = IoU(on_depsem_gt, on_depsem_dec, IoU_class, vox_shape)
        IoU_all = np.expand_dims(IoU_class, axis=1)

        # volume segmentation
        on_surf_gt = onehot(surf_test, vox_shape[3])
        on_full_gt = onehot(voxel_test, vox_shape[3])
        print(colored("Geometric semantic completion", 'cyan'))
        on_pred = onehot(np.argmax(pred_ssc, axis=4), vox_shape[3])
        IoU_class = IoU(on_full_gt, on_pred, IoU_class, vox_shape)
        IoU_all = np.concatenate((IoU_all, np.expand_dims(IoU_class, axis=1)),
                                 axis=1)
        print(colored("Generative semantic completion", 'cyan'))
        on_pred = onehot(np.argmax(pred_surf, axis=4), vox_shape[3])
        IoU_class = IoU(on_full_gt, on_pred, IoU_class, vox_shape)
        IoU_all = np.concatenate((IoU_all, np.expand_dims(IoU_class, axis=1)),
                                 axis=1)
        print(colored("Solid generative semantic completion", 'cyan'))
        on_pred = onehot(np.argmax(pred_full, axis=4), vox_shape[3])
        IoU_class = IoU(on_full_gt, on_pred, IoU_class, vox_shape)
        IoU_all = np.concatenate((IoU_all, np.expand_dims(IoU_class, axis=1)),
                                 axis=1)

        np.savetxt(
            save_path + '/IoU.csv',
            np.transpose(IoU_all[1:] * 100),
            delimiter=" & ",
            fmt='%2.1f')

    # interpolation evaluation
    if mode == 'interpolate':
        interpolate_num = 8
        #interpolatioin latent vectores
        decode_z = np.load(save_path + '/decode_z.npy')
        print(save_path)
        decode_z = decode_z[20:20 + batch_size]
        for l in np.arange(batch_size):
            for r in np.arange(batch_size):
                if l != r:
                    print l, r
                    base_num_left = l
                    base_num_right = r
                    left = np.reshape(decode_z[base_num_left], [
                        1, start_vox_size[0], start_vox_size[1],
                        start_vox_size[2], dim_z
                    ])
                    right = np.reshape(decode_z[base_num_right], [
                        1, start_vox_size[0], start_vox_size[1],
                        start_vox_size[2], dim_z
                    ])

                    duration = (right - left) / (interpolate_num - 1)
                    # left is the reference sample and Z_np_sample is the remaining samples
                    if base_num_left == 0:
                        Z_np_sample = decode_z[1:]
                    elif base_num_left == batch_size - 1:
                        Z_np_sample = decode_z[:batch_size - 1]
                    else:
                        Z_np_sample_before = np.reshape(
                            decode_z[:base_num_left], [
                                base_num_left, start_vox_size[0],
                                start_vox_size[1], start_vox_size[2], dim_z
                            ])
                        Z_np_sample_after = np.reshape(
                            decode_z[base_num_left + 1:], [
                                batch_size - base_num_left - 1,
                                start_vox_size[0], start_vox_size[1],
                                start_vox_size[2], dim_z
                            ])
                        Z_np_sample = np.concatenate(
                            [Z_np_sample_before, Z_np_sample_after], axis=0)
                    for i in np.arange(interpolate_num):
                        if i == 0:
                            Z = copy.copy(left)
                            interpolate_z = copy.copy(Z)
                        else:
                            Z = Z + duration
                            interpolate_z = np.concatenate([interpolate_z, Z],
                                                           axis=0)

                        # Z_np_sample is used to fill up the batch
                        Z_var_np_sample = np.concatenate([Z, Z_np_sample],
                                                         axis=0)
                        pred_full_rand, pred_part_rand = sess.run(
                            [full_tf_sample, part_tf_sample],
                            feed_dict={Z_tf_sample: Z_var_np_sample})
                        interpolate_vox = np.reshape(pred_full_rand[0], [
                            1, vox_shape[0], vox_shape[1], vox_shape[2],
                            vox_shape[3]
                        ])
                        interpolate_part = np.reshape(pred_part_rand[0], [
                            1, vox_shape[0], vox_shape[1], vox_shape[2],
                            complete_shape[3]
                        ])

                        if i == 0:
                            pred_full = interpolate_vox
                            pred_part = interpolate_part
                        else:
                            pred_full = np.concatenate(
                                [pred_full, interpolate_vox], axis=0)
                            pred_part = np.concatenate(
                                [pred_part, interpolate_part], axis=0)
                    interpolate_z.astype('uint8').tofile(
                        save_path + '/interpolate/interpolation_z' + str(l) +
                        '-' + str(r) + '.bin')

                    full_models_cat = np.argmax(pred_full, axis=4)
                    full_models_cat.astype('uint8').tofile(
                        save_path + '/interpolate/interpolation_f' + str(l) +
                        '-' + str(r) + '.bin')
                    if cfg.TYPE_TASK == 'scene':
                        pred_part = np.abs(pred_part)
                        pred_part[pred_part < 0.2] = 0
                        pred_part[pred_part >= 0.2] = 1
                    elif cfg.TYPE_TASK == 'object':
                        pred_part = np.argmax(pred_part, axis=4)
                    pred_part.astype('uint8').tofile(
                        save_path + '/interpolate/interpolation_p' + str(l) +
                        '-' + str(r) + '.bin')
        print("voxels saved")

    # add noise evaluation
    if mode == 'noise':
        decode_z = np.load(save_path + '/decode_z.npy')
        decode_z = decode_z[:batch_size]
        noise_num = 10
        for base_num in np.arange(batch_size):
            print base_num
            base = np.reshape(decode_z[base_num], [
                1, start_vox_size[0], start_vox_size[1], start_vox_size[2],
                dim_z
            ])
            eps = np.random.normal(size=(noise_num - 1,
                                         dim_z)).astype(np.float32)

            if base_num == 0:
                Z_np_sample = decode_z[1:]
            elif base_num == batch_size - 1:
                Z_np_sample = decode_z[:batch_size - 1]
            else:
                Z_np_sample_before = np.reshape(decode_z[:base_num], [
                    base_num, start_vox_size[0], start_vox_size[1],
                    start_vox_size[2], dim_z
                ])
                Z_np_sample_after = np.reshape(decode_z[base_num + 1:], [
                    batch_size - base_num - 1, start_vox_size[0],
                    start_vox_size[1], start_vox_size[2], dim_z
                ])
                Z_np_sample = np.concatenate(
                    [Z_np_sample_before, Z_np_sample_after], axis=0)

            for c in np.arange(start_vox_size[0]):
                for l in np.arange(start_vox_size[1]):
                    for d in np.arange(start_vox_size[2]):

                        for i in np.arange(noise_num):
                            if i == 0:
                                Z = copy.copy(base)
                                noise_z = copy.copy(Z)
                            else:
                                Z = copy.copy(base)
                                Z[0, c, l, d, :] += eps[i - 1]
                                noise_z = np.concatenate([noise_z, Z], axis=0)
                            Z_var_np_sample = np.concatenate([Z, Z_np_sample],
                                                             axis=0)
                            pred_full_rand = sess.run(
                                full_tf_sample,
                                feed_dict={Z_tf_sample: Z_var_np_sample})
                            """
                            refined_voxs_rand = sess.run(
                                sample_refine_full_tf,
                                feed_dict={
                                    sample_full_tf: pred_full_rand
                                })
                            """
                            noise_vox = np.reshape(pred_full_rand[0], [
                                1, vox_shape[0], vox_shape[1], vox_shape[2],
                                vox_shape[3]
                            ])
                            if i == 0:
                                pred_full = noise_vox
                            else:
                                pred_full = np.concatenate(
                                    [pred_full, noise_vox], axis=0)

                        np.save(
                            save_path + '/noise_z' + str(base_num) + '_' +
                            str(c) + str(l) + str(d) + '.npy', noise_z)

                        full_models_cat = np.argmax(pred_full, axis=4)
                        np.save(
                            save_path + '/noise' + str(base_num) + '_' + str(c)
                            + str(l) + str(d) + '.npy', full_models_cat)

        print("voxels saved")
