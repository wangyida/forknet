import numpy as np
import tensorflow as tf
import matplotlib
import os

from config import cfg
from model import depvox_gan
from util import DataProcess, onehot, id_models_test
from sklearn.metrics import average_precision_score
import copy

from colorama import init
from termcolor import colored
import colorama
from pca import pca
from io_util import read_pcd, save_pcd

# use Colorama to make Termcolor work on Windows too
init()
tf.compat.v1.disable_eager_execution()


def IoU(on_gt, on_pd, vox_shape, IoU_compared=None):
    # calc_IoU
    epsilon = 0.1
    if vox_shape[3] == 41:
        name_list = ['empty'
               , 'wall'
 , 'floor'
 , 'cabinet'
 , 'bed'
 , 'chair'
 , 'sofa'
 , 'table'
 , 'door'
 , 'window'
 , 'bookshelf'
 , 'picture'
 , 'counter'
 , 'blinds'
 , 'desk'
 , 'shelves'
 , 'curtain'
 , 'dresser'
 , 'pillow'
 , 'mirror'
 , 'floor mat'
 , 'clothes'
 , 'ceiling'
 , 'books'
 , 'refridgerator'
 , 'television'
 , 'paper'
 , 'towel'
 , 'shower curtain'
 , 'box'
 , 'whiteboard'
 , 'person'
 , 'night stand'
 , 'toilet'
 , 'sink'
 , 'lamp'
 , 'bathtub'
 , 'bag'
 , 'otherstructure'
 , 'otherfurniture'
 , 'otherprop']
    elif vox_shape[3] == 12:
        name_list = [
            'emp', 'ceil', 'floor', 'wall', 'wind', 'chair', 'bed', 'sofa',
            'table', 'tvs', 'furn', 'obj'
        ]
    elif vox_shape[3] == 5:
        name_list = ['empty', 'bench', 'chair', 'couch', 'table']
    elif vox_shape[3] == 2:
        name_list = ['empty', 'objec']
    IoUs = np.zeros([vox_shape[3] + 1])
    for class_n in np.arange(vox_shape[3]):
        IoU_calc = 0
        for sample_n in np.arange(on_gt.shape[0]):
            on_pd_ = on_pd[sample_n, :, :, :, class_n]
            on_gt_ = on_gt[sample_n, :, :, :, class_n]
            mother = np.sum(np.clip(np.add(on_pd_, on_gt_), 0, 1), (0, 1, 2))
            child = np.sum(np.multiply(on_pd_, on_gt_), (0, 1, 2))
            IoU_calc += (child + epsilon) / (mother + epsilon)

        IoUs[class_n] = np.round(IoU_calc * 100 / on_gt.shape[0], 1)
        if IoU_compared is not None:
            IoU_diff = np.round(IoUs[class_n] - IoU_compared[class_n], 3)
            text_diff = ' +' + colored(
                IoU_diff, 'green') if IoU_diff >= 0 else ' ' + colored(
                    IoU_diff, 'red')
            print('IoU of ' + name_list[class_n] + ': ' + str(IoUs[class_n]) +
                  text_diff)
        else:
            print('IoU of ' + name_list[class_n] + ': ' + str(IoUs[class_n]))
    if vox_shape[3] != 2:
        IoUs[vox_shape[3]] = np.round(
            np.sum(IoUs[1:vox_shape[3]]) / (vox_shape[3] - 1), 1)
    elif vox_shape[3] == 2:
        IoUs[vox_shape[3]] = np.round(np.sum(IoUs) / vox_shape[3], 1)
    print('IoU average: ', str(IoUs[vox_shape[3]]))
    print('')
    return IoUs


def evaluate(batch_size, checknum, mode, discriminative, data_list):

    n_vox = cfg.CONST.N_VOX
    dim = cfg.NET.DIM
    vox_shape = [n_vox[0], n_vox[1], n_vox[2], dim[4]]
    com_shape = [n_vox[0], n_vox[1], n_vox[2], 2]
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
        com_shape=com_shape,
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
    space_effective_tf, part_tf, part_dec_tf, comp_gt_tf, comp_gen_tf, comp_dec_tf, ssc_tf, scores_tf = depvox_gan_model.build_model()
    if discriminative is True:
        Z_tf_samp, comp_tf_samp, surf_tf_samp, full_tf_samp, part_tf_samp, scores_tf_samp = depvox_gan_model.samples_generator(
            visual_size=batch_size)
    config_gpu = tf.compat.v1.ConfigProto()
    config_gpu.gpu_options.allow_growth = True
    sess = tf.compat.v1.InteractiveSession(config=config_gpu)
    saver = tf.compat.v1.train.Saver()

    # Restore variables from disk.
    saver.restore(sess, chckpt_path)

    print("...Weights restored.")

    if mode == 'recons':
        # evaluation for reconstruction
        voxel_test, surf_test, part_test, num, data_paths = id_models_test(
            dataset_portion=cfg.TRAIN.DATASET_PORTION, data_list=data_list)

        # Evaluation masks
        if cfg.dataset == 'scene' or 'fusion' or '3rscan':
            # occluded region
            space_effective = np.where(part_test > -1, 1, 0)
            """
            voxel_test *= space_effective
            part_test *= space_effective
            # occluded region
            """
            part_test[part_test < -1] = 0
            surf_test[surf_test < 0] = 0
            voxel_test[voxel_test < 0] = 0

        num = voxel_test.shape[0]
        print("test voxels loaded")
        from progressbar import ProgressBar
        pbar = ProgressBar()
        for i in pbar(np.arange(int(num / batch_size))):
            bth_space = space_effective[i * batch_size:i * batch_size +
                                        batch_size]
            bth_tsdf = part_test[i * batch_size:i * batch_size + batch_size]
            bth_surf = surf_test[i * batch_size:i * batch_size + batch_size]
            bth_voxel = voxel_test[i * batch_size:i * batch_size + batch_size]

            bth_pd_surf, bth_pd_full, bth_pd_part, bth_part_enc_Z, bth_comp_gt, bth_pd_comp, bth_ssc = sess.run(
                [
                    surf_dec_tf, full_dec_tf, part_dec_tf, z_enc_tf,
                    comp_gt_tf, comp_dec_tf, ssc_tf
                ],
                feed_dict={
                    space_effective_tf: bth_space,
                    part_tf: bth_tsdf,
                    surf_tf: bth_surf,
                    full_tf: bth_voxel
                })

            if i == 0:
                pd_part = bth_pd_part
                pd_surf = bth_pd_surf
                pd_full = bth_pd_full
                pd_ssc = bth_ssc
                part_enc_Z = bth_part_enc_Z
                comp_gt = bth_comp_gt
                pd_comp = bth_pd_comp
            else:
                pd_part = np.concatenate((pd_part, bth_pd_part), axis=0)
                pd_surf = np.concatenate((pd_surf, bth_pd_surf), axis=0)
                pd_full = np.concatenate((pd_full, bth_pd_full), axis=0)
                pd_ssc = np.concatenate((pd_ssc, bth_ssc), axis=0)
                part_enc_Z = np.concatenate((part_enc_Z, bth_part_enc_Z),
                                            axis=0)
                comp_gt = np.concatenate((comp_gt, bth_comp_gt), axis=0)
                pd_comp = np.concatenate((pd_comp, bth_pd_comp), axis=0)

        print("forwarded")

        # For visualization
        bin_file = np.uint8(voxel_test)
        bin_file.tofile(save_path + '/scene.bin')

        sdf_volume = np.round(10 * np.abs(np.array(part_test)))
        observed = np.array(part_test)
        if cfg.dataset == 'scene':
            observed = np.abs(observed)
            observed *= 10
            observed -= 7
            observed = np.round(observed)
            pd_part = np.abs(pd_part)
            pd_part *= 10
            pd_part -= 7
        elif cfg.dataset == 'object':
            observed = np.clip(observed, 0, 1)
            pd_part = np.clip(pd_part, 0, 1)
        sdf_volume.astype('uint8').tofile(save_path + '/surface.bin')
        pd_part.astype('uint8').tofile(save_path + '/dec_part.bin')

        depsem_gt = np.multiply(voxel_test, np.clip(observed, 0, 1))
        if cfg.dataset == 'scene' or cfg.dataset == '3rscan':
            depsem_gt[depsem_gt < 0] = 0
        depsem_gt.astype('uint8').tofile(save_path + '/depth_seg_scene.bin')

        # decoded
        do_save_pcd = True
        if do_save_pcd is True:
            results_pcds = np.argmax(pd_comp, axis=4)
            for i in range(np.shape(results_pcds)[0]):
                pcd_idx = np.where(results_pcds[i] > 0)
                pts_coord = np.float32(np.transpose(pcd_idx)) / 64 - 0.5
                # pts_coord = np.float32(np.transpose(pcd_idx))
                pts_color = matplotlib.cm.rainbow(
                    np.float32(results_pcds[i][pcd_idx]) / dim[4])
                output_name = os.path.join('results_pcds',
                                           '%s.pcd' % data_paths[i][1][:-4])
                output_pcds = np.concatenate((pts_coord, pts_color[:, 0:3]),
                                             -1)
                if data_paths[i][1][:-4].find('/') > 0:
                    synset_id, _ = data_paths[i][1][:-4].split('/')
                    os.makedirs(
                        os.path.join('results_pcds', synset_id), exist_ok=True)
                save_pcd(output_name, output_pcds)

        np.argmax(
            pd_ssc, axis=4).astype('uint8').tofile(save_path + '/dec_ssc.bin')
        error = np.array(
            np.clip(np.argmax(pd_ssc, axis=4), 0, 1) +
            np.argmax(comp_gt, axis=4) * 2)
        error.astype('uint8').tofile(save_path + '/dec_ssc_error.bin')
        np.argmax(
            pd_surf,
            axis=4).astype('uint8').tofile(save_path + '/dec_surf.bin')
        error = np.array(
            np.clip(np.argmax(pd_surf, axis=4), 0, 1) +
            np.argmax(comp_gt, axis=4) * 2)
        error.astype('uint8').tofile(save_path + '/dec_surf_error.bin')
        np.argmax(
            pd_full,
            axis=4).astype('uint8').tofile(save_path + '/dec_full.bin')
        error = np.array(
            np.clip(np.argmax(pd_full, axis=4), 0, 1) +
            np.argmax(comp_gt, axis=4) * 2)
        error.astype('uint8').tofile(save_path + '/dec_full_error.bin')
        np.argmax(
            pd_comp,
            axis=4).astype('uint8').tofile(save_path + '/dec_complete.bin')
        np.argmax(
            comp_gt,
            axis=4).astype('uint8').tofile(save_path + '/complete_gt.bin')

        # reconstruction and generation from normal distribution evaluation
        # generator from random distribution
        if discriminative is True:
            np.save(save_path + '/decode_z.npy', part_enc_Z)
            sample_times = 10
            for j in np.arange(sample_times):
                gaussian_samp = np.random.normal(
                    size=(batch_size, start_vox_size[0], start_vox_size[1],
                          start_vox_size[2], dim_z)).astype(np.float32)

                z_comp_rnd, z_surf_rnd, z_full_rnd, z_part_rnd, scores_samp = sess.run(
                    [
                        comp_tf_samp, surf_tf_samp, full_tf_samp, part_tf_samp,
                        scores_tf_samp
                    ],
                    feed_dict={Z_tf_samp: gaussian_samp})
                if j == 0:
                    z_comp_rnd_all = z_comp_rnd
                    z_part_rnd_all = z_part_rnd
                    z_surf_rnd_all = z_surf_rnd
                    z_full_rnd_all = z_full_rnd
                else:
                    z_comp_rnd_all = np.concatenate(
                        [z_comp_rnd_all, z_comp_rnd], axis=0)
                    z_part_rnd_all = np.concatenate(
                        [z_part_rnd_all, z_part_rnd], axis=0)
                    z_surf_rnd_all = np.concatenate(
                        [z_surf_rnd_all, z_surf_rnd], axis=0)
                    z_full_rnd_all = np.concatenate(
                        [z_full_rnd_all, z_full_rnd], axis=0)
                    print('Discrim score: ' +
                          colored(np.mean(scores_samp), 'blue'))
            gaussian_samp.astype('float32').tofile(save_path + '/sample_z.bin')
            np.argmax(
                z_comp_rnd_all,
                axis=4).astype('uint8').tofile(save_path + '/gen_comp.bin')
            np.argmax(
                z_surf_rnd_all,
                axis=4).astype('uint8').tofile(save_path + '/gen_surf.bin')
            np.argmax(
                z_full_rnd_all,
                axis=4).astype('uint8').tofile(save_path + '/gen_full.bin')
            if cfg.dataset == 'scene':
                z_part_rnd_all = np.abs(z_part_rnd_all)
                z_part_rnd_all *= 10
                z_part_rnd_all -= 7
            elif cfg.dataset == 'object':
                z_part_rnd_all[z_part_rnd_all <= 0.4] = 0
                z_part_rnd_all[z_part_rnd_all > 0.4] = 1
                z_part_rnd = np.squeeze(z_part_rnd)
            z_part_rnd_all.astype('uint8').tofile(save_path + '/gen_part.bin')

        print("voxels saved")

        # numerical evalutation
        iou_eval = True
        if iou_eval is True:
            # completion
            print(colored("Completion:", 'red'))
            on_gt = comp_gt
            pd_max = np.argmax(pd_comp, axis=4)
            on_pd = onehot(pd_max, 2)
            IoU_comp = np.zeros([2 + 1])
            AP_comp = np.zeros([2 + 1])
            IoU_comp = IoU(on_gt, on_pd,
                           [vox_shape[0], vox_shape[1], vox_shape[2], 2])

            # depth segmentation
            print(colored("Segmentation:", 'red'))
            print(colored("encoded", 'cyan'))
            on_gt = onehot(depsem_gt, vox_shape[3])
            on_pd = np.multiply(
                onehot(np.argmax(pd_comp, axis=4), vox_shape[3]),
                np.expand_dims(np.clip(observed, 0, 1), -1))
            # IoUs = np.zeros([vox_shape[3] + 1])
            IoU_temp = IoU(on_gt, on_pd, vox_shape)
            IoU_all = np.expand_dims(IoU_temp, axis=1)

            print(colored("decoded", 'cyan'))
            on_pd = np.multiply(
                onehot(np.argmax(pd_surf, axis=4), vox_shape[3]),
                np.expand_dims(np.clip(observed, 0, 1), -1))
            IoU_temp = IoU(
                on_gt, on_pd, vox_shape, IoU_compared=IoU_all[:, -1])
            IoU_all = np.concatenate(
                (IoU_all, np.expand_dims(IoU_temp, axis=1)), axis=1)

            print(colored("solidly decoded", 'cyan'))
            on_pd = np.multiply(
                onehot(np.argmax(pd_full, axis=4), vox_shape[3]),
                np.expand_dims(np.clip(observed, 0, 1), -1))
            IoU_temp = IoU(
                on_gt, on_pd, vox_shape, IoU_compared=IoU_all[:, -1])
            IoU_all = np.concatenate(
                (IoU_all, np.expand_dims(IoU_temp, axis=1)), axis=1)

            # volume segmentation
            print(colored("Semantic Completion:", 'red'))
            on_surf_gt = onehot(surf_test, vox_shape[3])
            on_gt = onehot(voxel_test, vox_shape[3])
            print(colored("encoded", 'cyan'))
            on_pd = onehot(np.argmax(pd_ssc, axis=4), vox_shape[3])
            IoU_temp = IoU(on_gt, on_pd, vox_shape)
            IoU_all = np.concatenate(
                (IoU_all, np.expand_dims(IoU_temp, axis=1)), axis=1)

            print(colored("decoded", 'cyan'))
            on_pd = onehot(np.argmax(pd_surf, axis=4), vox_shape[3])
            IoU_temp = IoU(
                on_gt, on_pd, vox_shape, IoU_compared=IoU_all[:, -1])
            IoU_all = np.concatenate(
                (IoU_all, np.expand_dims(IoU_temp, axis=1)), axis=1)

            print(colored("solidly decoded", 'cyan'))
            on_pd = onehot(np.argmax(pd_full, axis=4), vox_shape[3])
            IoU_temp = IoU(
                on_gt, on_pd, vox_shape, IoU_compared=IoU_all[:, -1])
            IoU_all = np.concatenate(
                (IoU_all, np.expand_dims(IoU_temp, axis=1)), axis=1)

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
                    print(l, r)
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
                    # left is the reference sample and Z_np_samp is the remaining samples
                    if base_num_left == 0:
                        Z_np_samp = decode_z[1:]
                    elif base_num_left == batch_size - 1:
                        Z_np_samp = decode_z[:batch_size - 1]
                    else:
                        Z_np_samp_before = np.reshape(
                            decode_z[:base_num_left], [
                                base_num_left, start_vox_size[0],
                                start_vox_size[1], start_vox_size[2], dim_z
                            ])
                        Z_np_samp_after = np.reshape(
                            decode_z[base_num_left + 1:], [
                                batch_size - base_num_left - 1,
                                start_vox_size[0], start_vox_size[1],
                                start_vox_size[2], dim_z
                            ])
                        Z_np_samp = np.concatenate(
                            [Z_np_samp_before, Z_np_samp_after], axis=0)
                    for i in np.arange(interpolate_num):
                        if i == 0:
                            Z = copy.copy(left)
                            interpolate_z = copy.copy(Z)
                        else:
                            Z = Z + duration
                            interpolate_z = np.concatenate([interpolate_z, Z],
                                                           axis=0)

                        # Z_np_samp is used to fill up the batch
                        gaussian_samp = np.concatenate([Z, Z_np_samp], axis=0)
                        pd_full_rnd, pd_part_rnd = sess.run(
                            [full_tf_samp, part_tf_samp],
                            feed_dict={Z_tf_samp: gaussian_samp})
                        interpolate_vox = np.reshape(pd_full_rnd[0], [
                            1, vox_shape[0], vox_shape[1], vox_shape[2],
                            vox_shape[3]
                        ])
                        interpolate_part = np.reshape(pd_part_rnd[0], [
                            1, vox_shape[0], vox_shape[1], vox_shape[2],
                            com_shape[3]
                        ])

                        if i == 0:
                            pd_full = interpolate_vox
                            pd_part = interpolate_part
                        else:
                            pd_full = np.concatenate(
                                [pd_full, interpolate_vox], axis=0)
                            pd_part = np.concatenate(
                                [pd_part, interpolate_part], axis=0)
                    interpolate_z.astype('uint8').tofile(
                        save_path + '/interpolate/interpolation_z' + str(l) +
                        '-' + str(r) + '.bin')

                    full_models_cat = np.argmax(pd_comp, axis=4)
                    full_models_cat.astype('uint8').tofile(
                        save_path + '/interpolate/interpolation_f' + str(l) +
                        '-' + str(r) + '.bin')
                    if cfg.dataset == 'scene':
                        pd_part = np.abs(pd_part)
                        pd_part *= 10
                        pd_part -= 7
                    elif cfg.dataset == 'object':
                        pd_part = np.argmax(pd_part, axis=4)
                    pd_part.astype('uint8').tofile(
                        save_path + '/interpolate/interpolation_p' + str(l) +
                        '-' + str(r) + '.bin')
        print("voxels saved")

    # add noise evaluation
    if mode == 'noise':
        decode_z = np.load(save_path + '/decode_z.npy')
        decode_z = decode_z[:batch_size]
        noise_num = 10
        for base_num in np.arange(batch_size):
            print(base_num)
            base = np.reshape(decode_z[base_num], [
                1, start_vox_size[0], start_vox_size[1], start_vox_size[2],
                dim_z
            ])
            eps = np.random.normal(size=(noise_num - 1,
                                         dim_z)).astype(np.float32)

            if base_num == 0:
                Z_np_samp = decode_z[1:]
            elif base_num == batch_size - 1:
                Z_np_samp = decode_z[:batch_size - 1]
            else:
                Z_np_samp_before = np.reshape(decode_z[:base_num], [
                    base_num, start_vox_size[0], start_vox_size[1],
                    start_vox_size[2], dim_z
                ])
                Z_np_samp_after = np.reshape(decode_z[base_num + 1:], [
                    batch_size - base_num - 1, start_vox_size[0],
                    start_vox_size[1], start_vox_size[2], dim_z
                ])
                Z_np_samp = np.concatenate([Z_np_samp_before, Z_np_samp_after],
                                           axis=0)

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
                            gaussian_samp = np.concatenate([Z, Z_np_samp],
                                                           axis=0)
                            pd_full_rnd = sess.run(
                                full_tf_samp,
                                feed_dict={Z_tf_samp: gaussian_samp})
                            """
                            refined_voxs_rnd = sess.run(
                                sample_refine_full_tf,
                                feed_dict={
                                    sample_full_tf: pd_full_rnd
                                })
                            """
                            noise_vox = np.reshape(pd_full_rnd[0], [
                                1, vox_shape[0], vox_shape[1], vox_shape[2],
                                vox_shape[3]
                            ])
                            if i == 0:
                                pd_full = noise_vox
                            else:
                                pd_full = np.concatenate([pd_full, noise_vox],
                                                         axis=0)

                        np.save(
                            save_path + '/noise_z' + str(base_num) + '_' +
                            str(c) + str(l) + str(d) + '.npy', noise_z)

                        full_models_cat = np.argmax(pd_full, axis=4)
                        np.save(
                            save_path + '/noise' + str(base_num) + '_' + str(c)
                            + str(l) + str(d) + '.npy', full_models_cat)

        print("voxels saved")
