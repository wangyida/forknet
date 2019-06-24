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


def IoU_AP_calc(on_gt, on_pred, pred_voxs, IoU_class, AP_class, vox_shape):
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

    #calc_AP
    """
    for class_n in np.arange(vox_shape[3]):
        on_pred_ = pred_voxs[:, :, :, :, class_n]
        on_gt_ = on_gt[:, :, :, :, class_n]

        AP = 0.
        for i in np.arange(num):
            y_true = np.reshape(on_gt_[i], [-1])
            y_scores = np.reshape(on_pred_[i], [-1])
            if np.sum(y_true) > 0.:
                AP += average_precision_score(y_true, y_scores)
        AP = np.round(AP / num, 3)
        AP_class[class_n] = AP
        print 'AP class ' + str(class_n) + '=' + str(AP)
    AP_class[vox_shape[3]] = np.round(
        np.sum(AP_class[1:(vox_shape[3] - 1)]) / (vox_shape[3] - 1), 3)
    print 'AP category-wise = ' + str(AP_class[vox_shape[3]])
    """
    """
    on_pred_ = pred_voxs[:, :, :, :, 1:vox_shape[3]]
    on_gt_ = on_gt[:, :, :, :, 1:vox_shape[3]]
    AP = 0.
    for i in np.arange(num):
        y_true = np.reshape(on_gt_[i], [-1])
        y_scores = np.reshape(on_pred_[i], [-1])
        if np.sum(y_true) > 0.:
            AP += average_precision_score(y_true, y_scores)

    AP = np.round(AP / num, 3)
    AP_class[vox_shape[3]] = AP
    print 'AP space-wise =' + str(AP)
    """
    print ''
    return IoU_class, AP_class


def evaluate(batch_size, checknum, mode):

    n_vox = cfg.CONST.N_VOX
    dim = cfg.NET.DIM
    vox_shape = [n_vox[0], n_vox[1], n_vox[2], dim[4]]
    part_shape = [n_vox[0], n_vox[1], n_vox[2], 2]
    dim_z = cfg.NET.DIM_Z
    start_vox_size = cfg.NET.START_VOX
    kernel = cfg.NET.KERNEL
    stride = cfg.NET.STRIDE
    dilations = cfg.NET.DILATIONS
    freq = cfg.CHECK_FREQ
    discriminative = cfg.NET.DISCRIMINATIVE

    save_path = cfg.DIR.EVAL_PATH
    chckpt_path = cfg.DIR.CHECK_PT_PATH + str(checknum)

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
        is_train=False)


    Z_tf, z_part_enc_tf, full_tf, full_gen_tf, full_gen_dec_tf, full_gen_dec_ref_tf,\
    gen_loss_tf, discrim_loss_tf, recons_loss_tf, encode_loss_tf, refine_loss_tf, summary_tf,\
    part_tf, complete_gen_tf, complete_gen_decode_tf = depvox_gan_model.build_model()
    if discriminative is True:
        Z_tf_sample, full_tf_sample, full_ref_tf_sample, part_tf_sample = depvox_gan_model.samples_generator(
            visual_size=batch_size)
    sess = tf.InteractiveSession()
    saver = tf.train.Saver()

    # Restore variables from disk.
    saver.restore(sess, chckpt_path)

    print("...Weights restored.")

    if mode == 'recons':
        # evaluation for reconstruction
        voxel_test, part_test, num, data_paths = scene_model_id_pair_test(
            dataset_portion=cfg.TRAIN.DATASET_PORTION)

        # Evaluation masks
        if cfg.TYPE_TASK == 'scene':
            space_effective = np.clip(
                np.where(voxel_test > 0, 1, 0) + np.where(
                    part_test > -1.01, 1, 0), 0, 1)
            voxel_test *= space_effective
            part_test *= space_effective
            # occluded region
            part_test[part_test < -1] = 0

        num = voxel_test.shape[0]
        print("test voxels loaded")
        for i in np.arange(int(num / batch_size)):
            batch_voxel = voxel_test[i * batch_size:i * batch_size +
                                     batch_size]
            batch_tsdf = part_test[i * batch_size:i * batch_size + batch_size]
            if cfg.TYPE_TASK == 'scene':
                batch_effective = space_effective[i * batch_size:i *
                                                  batch_size + batch_size]

            batch_pred_voxs, batch_pred_ref_voxs, batch_part_enc_Z, batch_pred_complete = sess.run(
                [
                    full_gen_dec_tf, full_gen_dec_ref_tf, z_part_enc_tf,
                    complete_gen_decode_tf
                ],
                feed_dict={
                    part_tf: batch_tsdf,
                    full_tf: batch_voxel
                })
            add_noising = False
            if add_noising:
                batch_part_enc_Z += np.random.normal(
                    loc=0.0,
                    scale=1.0,
                    size=(batch_size, start_vox_size[0], start_vox_size[1],
                          start_vox_size[2], dim_z)).astype(np.float32)
                batch_pred_voxs, batch_pred_complete = sess.run(
                    [full_tf_sample, part_tf_sample],
                    feed_dict={Z_tf_sample: batch_part_enc_Z})

            # Masked
            if cfg.TYPE_TASK == 'scene':
                batch_pred_voxs *= np.expand_dims(batch_effective, -1)
                batch_pred_ref_voxs *= np.expand_dims(batch_effective, -1)
                batch_pred_complete *= np.expand_dims(batch_effective, -1)

            if i == 0:
                pred_voxs = batch_pred_voxs
                pred_ref_voxs = batch_pred_ref_voxs
                part_enc_Z = batch_part_enc_Z
                pred_complete = batch_pred_complete
            else:
                pred_voxs = np.concatenate((pred_voxs, batch_pred_voxs),
                                           axis=0)
                pred_ref_voxs = np.concatenate(
                    (pred_ref_voxs, batch_pred_ref_voxs), axis=0)
                part_enc_Z = np.concatenate((part_enc_Z, batch_part_enc_Z),
                                            axis=0)
                pred_complete = np.concatenate(
                    (pred_complete, batch_pred_complete), axis=0)

        print("forwarded")

        # For visualization
        voxel_test.astype('uint8').tofile(save_path + '/scene.bin')

        observe = np.array(part_test)
        observe[observe == -1] = 3
        observe.astype('uint8').tofile(save_path + '/observe.bin')

        surface = np.array(part_test)
        if cfg.TYPE_TASK == 'scene':
            surface = np.abs(surface)
            surface *= 10
            surface -= 4 
            surface[surface < 0] = 0
            """
            surface[surface < -1] = 0
            surface[surface > 1] = 0
            surface[np.abs(surface) > 0.9] = 1
            surface[abs(surface) < 1] = 0
            """
        elif cfg.TYPE_TASK == 'object':
            surface = np.clip(surface, 0, 1)
        surface.astype('uint8').tofile(save_path + '/surface.bin')

        depth_seg_gt = np.multiply(voxel_test, np.clip(surface, 0, 1))
        if cfg.TYPE_TASK == 'scene':
            depth_seg_gt[depth_seg_gt < 0] = 0
        depth_seg_gt.astype('uint8').tofile(save_path + '/depth_seg_scene.bin')

        complete_gt = np.clip(voxel_test, 0, 1)
        complete_gt.astype('uint8').tofile(save_path + '/complete_gt.bin')

        # decoded
        # check for some bad categories
        if cfg.TYPE_TASK == 'scene':
            """
            pred_voxs[:,:,:,:,4] -= 0.7
            pred_voxs[[0,6,45,71,139,143,193],:,:,:,4] = 0
            addtioanl_vox = np.array(surface)
            addtioanl_vox *= pred_voxs[:,:,:,:,0]
            addtioanl_vox *= onehot(voxel_test, vox_shape[3])[:,:,:,:,0]
            """
            # pred_voxs[[3,35,46],:,:,:,10] += addtioanl_vox[[3,35,46],:,:,:]
            # pred_complete[:,:,:,:,1][np.where(addtioanl_vox)] = 1
            """
            correct_vis = np.array(pred_voxs)
            correct_vis -= onehot(np.argmax(correct_vis, axis=4), vox_shape[3])*correct_vis 
            pred_voxs[np.argmax(correct_vis, axis=4) == 3] = 1
            """
        np.argmax(
            pred_voxs,
            axis=4).astype('uint8').tofile(save_path + '/gen_vox.bin')
        error = np.array(
            np.clip(np.argmax(pred_voxs, axis=4), 0, 1) + complete_gt*2)
        error.astype('uint8').tofile(save_path + '/gen_vox_error.bin')
        np.argmax(
            pred_ref_voxs,
            axis=4).astype('uint8').tofile(save_path + '/gen_ref_vox.bin')
        error = np.array(
            np.clip(np.argmax(pred_ref_voxs, axis=4), 0, 1) + complete_gt*2)
        error.astype('uint8').tofile(save_path + '/gen_ref_vox_error.bin')
        np.argmax(
            pred_complete,
            axis=4).astype('uint8').tofile(save_path + '/gen_complete.bin')
        """
        if cfg.TYPE_TASK == 'scene':
            vae_tsdf = np.abs(vae_tsdf)
            vae_tsdf[vae_tsdf < 0.2] = 0
            vae_tsdf[vae_tsdf >= 0.2] = 1
        elif cfg.TYPE_TASK == 'object':
            vae_tsdf[vae_tsdf <= 0.2] = 0
            vae_tsdf[vae_tsdf > 0.2] = 1
            vae_tsdf = np.squeeze(vae_tsdf)
        vae_tsdf.astype('uint8').tofile(save_path + '/vae_tsdf.bin')
        """

        np.save(save_path + '/decode_z.npy', part_enc_Z)

        # reconstruction and generation from normal distribution evaluation
        # generator from random distribution
        if discriminative is True:
            sample_times = 10
            for j in np.arange(sample_times):
                for i in np.arange(batch_size):
                    Z_np_sample = np.random.normal(
                        size=(1, start_vox_size[0], start_vox_size[1],
                              start_vox_size[2], dim_z)).astype(np.float32)
                    """
                    Z_np_sample += (part_enc_Z[j * batch_size + i] +
                                    part_enc_Z[j * batch_size + i + 2]) / 2
                    """
                    if i == 0:
                        Z_var_np_sample = Z_np_sample
                    else:
                        Z_var_np_sample = np.concatenate(
                            (Z_var_np_sample, Z_np_sample), axis=0)

                z_voxs_fromrand, z_voxs_ref_fromrand, z_part_fromrand = sess.run(
                    [full_tf_sample, full_ref_tf_sample, part_tf_sample],
                    feed_dict={Z_tf_sample: Z_var_np_sample})
                if j == 0:
                    z_voxs_fromrand_all = z_voxs_fromrand
                    z_voxs_ref_fromrand_all = z_voxs_ref_fromrand
                    z_part_fromrand_all = z_part_fromrand
                else:
                    z_voxs_fromrand_all = np.concatenate(
                        [z_voxs_fromrand_all, z_voxs_fromrand], axis=0)
                    z_voxs_ref_fromrand_all = np.concatenate(
                        [z_voxs_ref_fromrand_all, z_voxs_ref_fromrand], axis=0)
                    z_part_fromrand_all = np.concatenate(
                        [z_part_fromrand_all, z_part_fromrand], axis=0)
            Z_var_np_sample.astype('float32').tofile(save_path +
                                                     '/sample_z.bin')
            np.argmax(
                z_voxs_fromrand_all,
                axis=4).astype('uint8').tofile(save_path + '/generate.bin')
            np.argmax(
                z_voxs_ref_fromrand_all,
                axis=4).astype('uint8').tofile(save_path + '/generate_ref.bin')
            if cfg.TYPE_TASK == 'scene':
                z_part_fromrand_all = np.abs(z_part_fromrand_all)
                z_part_fromrand_all *= 10
                z_part_fromrand_all -= 4
                z_part_fromrand_all[z_part_fromrand_all < 0] = 0
            elif cfg.TYPE_TASK == 'object':
                z_part_fromrand_all[z_part_fromrand_all <= 0.4] = 0
                z_part_fromrand_all[z_part_fromrand_all > 0.4] = 1
                z_part_fromrand = np.squeeze(z_part_fromrand)
            z_part_fromrand_all.astype('uint8').tofile(save_path +
                                                       '/generate_sdf.bin')

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
                z_voxs_fromrand, z_voxs_ref_fromrand, z_part_fromrand = sess.run(
                    [full_tf_sample, full_ref_tf_sample, part_tf_sample],
                    feed_dict={Z_tf_sample: z_V})
                np.argmax(
                    z_voxs_fromrand,
                    axis=4).astype('uint8').tofile(save_path + '/generate.bin')
                if cfg.TYPE_TASK == 'scene':
                    z_part_fromrand = np.abs(z_part_fromrand)
                    z_part_fromrand *= 10
                    z_part_fromrand -= 4
                    z_part_fromrand[z_part_fromrand < 0] = 0
                elif cfg.TYPE_TASK == 'object':
                    z_part_fromrand[z_part_fromrand <= 0.4] = 0
                    z_part_fromrand[z_part_fromrand > 0.4] = 1
                    z_part_fromrand = np.squeeze(z_part_fromrand)
                z_part_fromrand.astype('uint8').tofile(save_path +
                                                       '/generate_sdf.bin')

        print("voxels saved")

        # numerical evalutation
        on_gt = onehot(voxel_test, vox_shape[3])
        on_depth_seg_gt = onehot(depth_seg_gt, vox_shape[3])
        on_depth_seg_pred = np.multiply(
            onehot(np.argmax(pred_voxs, axis=4), vox_shape[3]),
            np.expand_dims(surface, -1))
        on_complete_gt = onehot(complete_gt, 2)
        complete_gen = np.argmax(pred_complete, axis=4)
        on_complete_gen = onehot(complete_gen, 2)

        # calc_IoU
        # completion
        IoU_comp = np.zeros([2 + 1])
        AP_comp = np.zeros([2 + 1])
        print(colored("Completion", 'cyan'))
        IoU_comp, AP_comp = IoU_AP_calc(
            on_complete_gt, on_complete_gen, complete_gen, IoU_comp, AP_comp,
            [vox_shape[0], vox_shape[1], vox_shape[2], 2])

        # depth segmentation
        print(colored("Depth segmentation", 'cyan'))
        IoU_class = np.zeros([vox_shape[3] + 1])
        AP_class = np.zeros([vox_shape[3] + 1])
        IoU_class, AP_class = IoU_AP_calc(
            on_depth_seg_gt, on_depth_seg_pred,
            np.multiply(pred_voxs, np.expand_dims(np.clip(surface, 0, 1), -1)), IoU_class,
            AP_class, vox_shape)
        IoU_all = np.expand_dims(IoU_class, axis=1)
        AP_all = np.expand_dims(AP_class, axis=1)

        # volume segmentation
        print(colored("Decoded segmentation", 'cyan'))
        on_pred = onehot(np.argmax(pred_voxs, axis=4), vox_shape[3])
        IoU_class, AP_class = IoU_AP_calc(on_gt, on_pred, pred_voxs, IoU_class,
                                          AP_class, vox_shape)
        IoU_all = np.concatenate((IoU_all, np.expand_dims(IoU_class, axis=1)),
                                 axis=1)
        AP_all = np.concatenate((AP_all, np.expand_dims(AP_class, axis=1)),
                                axis=1)
        print(colored("Refined segmentation", 'cyan'))
        on_pred_ref = onehot(np.argmax(pred_ref_voxs, axis=4), vox_shape[3])
        IoU_class, AP_class = IoU_AP_calc(on_gt, on_pred_ref, pred_ref_voxs,
                                          IoU_class, AP_class, vox_shape)
        IoU_all = np.concatenate((IoU_all, np.expand_dims(IoU_class, axis=1)),
                                 axis=1)
        AP_all = np.concatenate((AP_all, np.expand_dims(AP_class, axis=1)),
                                axis=1)

        np.savetxt(
            save_path + '/IoU.csv',
            np.transpose(IoU_all[1:] * 100),
            delimiter=" & ",
            fmt='%2.1f')
        np.savetxt(
            save_path + '/AP.csv',
            np.transpose(AP_all[1:] * 100),
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
                        pred_voxs_fromrand, pred_part_fromrand = sess.run(
                            [full_tf_sample, part_tf_sample],
                            feed_dict={Z_tf_sample: Z_var_np_sample})
                        interpolate_vox = np.reshape(pred_voxs_fromrand[0], [
                            1, vox_shape[0], vox_shape[1], vox_shape[2],
                            vox_shape[3]
                        ])
                        interpolate_part = np.reshape(pred_part_fromrand[0], [
                            1, vox_shape[0], vox_shape[1], vox_shape[2],
                            part_shape[3]
                        ])

                        if i == 0:
                            pred_voxs = interpolate_vox
                            pred_part = interpolate_part
                        else:
                            pred_voxs = np.concatenate(
                                [pred_voxs, interpolate_vox], axis=0)
                            pred_part = np.concatenate(
                                [pred_part, interpolate_part], axis=0)
                    interpolate_z.astype('uint8').tofile(
                        save_path + '/interpolate/interpolation_z' + str(l) +
                        '-' + str(r) + '.bin')

                    full_models_cat = np.argmax(pred_voxs, axis=4)
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
                            pred_voxs_fromrand = sess.run(
                                full_tf_sample,
                                feed_dict={Z_tf_sample: Z_var_np_sample})
                            """
                            refined_voxs_fromrand = sess.run(
                                sample_refine_full_tf,
                                feed_dict={
                                    sample_full_tf: pred_voxs_fromrand
                                })
                            """
                            noise_vox = np.reshape(pred_voxs_fromrand[0], [
                                1, vox_shape[0], vox_shape[1], vox_shape[2],
                                vox_shape[3]
                            ])
                            if i == 0:
                                pred_voxs = noise_vox
                            else:
                                pred_voxs = np.concatenate(
                                    [pred_voxs, noise_vox], axis=0)

                        np.save(
                            save_path + '/noise_z' + str(base_num) + '_' +
                            str(c) + str(l) + str(d) + '.npy', noise_z)

                        full_models_cat = np.argmax(pred_voxs, axis=4)
                        np.save(
                            save_path + '/noise' + str(base_num) + '_' + str(c)
                            + str(l) + str(d) + '.npy', full_models_cat)

        print("voxels saved")
