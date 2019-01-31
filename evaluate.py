import numpy as np
import tensorflow as tf

from config import cfg
from model import depvox_gan
from util import DataProcess, scene_model_id_pair, onehot, scene_model_id_pair_test
from sklearn.metrics import average_precision_score
import copy

from colorama import init
from termcolor import colored

# use Colorama to make Termcolor work on Windows too
init()


def IoU_AP_calc(on_real, on_pred, pred_voxs, IoU_class, AP_class,
                vox_shape):
    # calc_IoU
    if vox_shape[3] == 13:
        name_list = [
            'empty', 'ceili', 'floor', ' wall', 'windo', ' door', 'chair',
            '  bed', ' sofa', 'table', '  tvs', 'furni', 'objec'
        ]
    elif vox_shape[3] == 5:
        name_list = ['empty', 'bench', 'chair', 'couch', 'table']
    elif vox_shape[3] == 2:
        name_list = ['empty', 'objec']
    num = on_real.shape[0]
    for class_n in np.arange(vox_shape[3]):
        on_pred_ = on_pred[:, :, :, :, class_n]
        on_real_ = on_real[:, :, :, :, class_n]
        mother = np.sum(np.add(on_pred_, on_real_), (1, 2, 3))
        # Here the child must be doubled because the mother is calculated twice
        child = np.sum(np.multiply(on_pred_, on_real_), (1, 2, 3)) * 2
        count = 0
        IoU_element = 0
        for i in np.arange(num):
            if mother[i] != 0:
                IoU_element += child[i] / mother[i]
                count += 1
        if count > 0:
            IoU_calc = np.round(IoU_element / count, 3)
            IoU_class[class_n] = IoU_calc
            print 'IoU of ' + name_list[class_n] + ':' + str(IoU_calc)
        else:
            IoU_class[class_n] = 1
            print 'IoU of ' + name_list[class_n] + ': nothing exists'
    if vox_shape[3] == 13:
        IoU_class[vox_shape[3]] = np.round(
            (np.sum(IoU_class[1:(vox_shape[3])]) - IoU_class[5]) /
            (vox_shape[3] - 2), 3)
    elif vox_shape[3] == 5:
        IoU_class[vox_shape[3]] = np.round(
            np.sum(IoU_class[1:(vox_shape[3])]) / (vox_shape[3] - 1), 3)
    elif vox_shape[3] == 2:
        IoU_class[vox_shape[3]] = np.round(np.sum(IoU_class) / vox_shape[3], 3)
    print 'IoU average: ' + str(IoU_class[vox_shape[3]])

    #calc_AP
    """
    for class_n in np.arange(vox_shape[3]):
        on_pred_ = pred_voxs[:, :, :, :, class_n]
        on_real_ = on_real[:, :, :, :, class_n]

        AP = 0.
        for i in np.arange(num):
            y_true = np.reshape(on_real_[i], [-1])
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
    on_real_ = on_real[:, :, :, :, 1:vox_shape[3]]
    AP = 0.
    for i in np.arange(num):
        y_true = np.reshape(on_real_[i], [-1])
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
    tsdf_shape = [n_vox[0], n_vox[1], n_vox[2], 3]
    dim_z = cfg.NET.DIM_Z
    start_vox_size = cfg.NET.START_VOX
    kernel = cfg.NET.KERNEL
    stride = cfg.NET.STRIDE
    dilations = cfg.NET.DILATIONS
    freq = cfg.CHECK_FREQ
    discriminative = cfg.NET.DISCRIMINATIVE
    generative = cfg.NET.GENERATIVE

    save_path = cfg.DIR.EVAL_PATH
    chckpt_path = cfg.DIR.CHECK_PT_PATH + str(checknum)

    depvox_gan_model = depvox_gan(
        batch_size=batch_size,
        vox_shape=vox_shape,
        tsdf_shape=tsdf_shape,
        dim_z=dim_z,
        dim=dim,
        start_vox_size=start_vox_size,
        kernel=kernel,
        stride=stride,
        dilations=dilations,
        generative=generative)


    Z_tf, z_tsdf_enc_tf, z_vox_enc_tf, vox_tf, vox_gen_tf, vox_gen_decode_tf, vox_vae_decode_tf, vox_cc_decode_tf,\
    recon_vae_loss_tf, recon_cc_loss_tf, recon_gen_loss_tf, code_encode_loss_tf, gen_loss_tf, discrim_loss_tf,\
    cost_enc_tf, cost_code_tf, cost_gen_tf, cost_discrim_tf, summary_tf,\
    tsdf_tf, tsdf_gen_tf, tsdf_gen_decode_tf, tsdf_vae_decode_tf, tsdf_cc_decode_tf = depvox_gan_model.build_model()
    Z_tf_sample, vox_tf_sample = depvox_gan_model.samples_generator(
        visual_size=batch_size)
    sess = tf.InteractiveSession()
    saver = tf.train.Saver()

    # Restore variables from disk.
    saver.restore(sess, chckpt_path)

    print("...Weights restored.")

    if mode == 'recons':
        # reconstruction and generation from normal distribution evaluation
        # generator from random distribution
        for i in np.arange(batch_size):
            Z_np_sample = np.random.normal(
                size=(1, start_vox_size[0], start_vox_size[1],
                      start_vox_size[2], dim_z)).astype(np.float32)
            if i == 0:
                Z_var_np_sample = Z_np_sample
            else:
                Z_var_np_sample = np.concatenate(
                    (Z_var_np_sample, Z_np_sample), axis=0)
        # np.save(save_path + '/sample_z.npy', Z_var_np_sample)
        Z_var_np_sample.astype('float32').tofile(save_path + '/sample_z.bin')

        generated_voxs_fromrand = sess.run(
            vox_tf_sample, feed_dict={Z_tf_sample: Z_var_np_sample})
        # np.save(save_path + '/generate.npy', np.argmax(generated_voxs_fromrand, axis=4))
        np.argmax(
            generated_voxs_fromrand,
            axis=4).astype('uint8').tofile(save_path + '/generate.bin')

        # evaluation for reconstruction
        voxel_test, tsdf_test, num, data_paths = scene_model_id_pair_test(
            dataset_portion=cfg.TRAIN.DATASET_PORTION)

        num = voxel_test.shape[0]
        print("test voxels loaded")
        for i in np.arange(int(num / batch_size)):
            batch_voxel = voxel_test[i * batch_size:i * batch_size +
                                     batch_size]
            batch_tsdf = tsdf_test[i * batch_size:i * batch_size + batch_size]

            # Evaluation masks
            if cfg.TYPE_TASK is 'scene':
                # Evaluation masks
                volume_effective = np.clip(
                    np.where(batch_voxel > 0, 1, 0) + np.where(
                        batch_tsdf > 0, 1, 0), 0, 1)
                batch_voxel *= volume_effective
                batch_tsdf *= volume_effective

                # batch_tsdf[batch_tsdf > 1] = 0
                # batch_tsdf[np.where(batch_voxel == 10)] = 1

            batch_pred_voxs, batch_vae_voxs, batch_cc_voxs,\
               batch_tsdf_enc_Z, batch_vox_enc_Z,\
               batch_pred_tsdf, batch_vae_tsdf, batch_cc_tsdf = sess.run(
                [
                    vox_gen_decode_tf, vox_vae_decode_tf, vox_cc_decode_tf,
                    z_tsdf_enc_tf, z_vox_enc_tf, 
                    tsdf_gen_decode_tf, tsdf_vae_decode_tf, tsdf_cc_decode_tf
                ],
                feed_dict={
                    tsdf_tf: batch_tsdf,
                    vox_tf: batch_voxel
                })

            # Masked
            if cfg.TYPE_TASK is 'scene':
                batch_pred_voxs *= np.expand_dims(volume_effective, -1)
                batch_vae_voxs *= np.expand_dims(volume_effective, -1)
                batch_cc_voxs *= np.expand_dims(volume_effective, -1)

            if i == 0:
                pred_voxs = batch_pred_voxs
                vae_voxs = batch_vae_voxs
                cc_voxs = batch_cc_voxs
                complete_gen = batch_vae_tsdf
                tsdf_enc_Z = batch_tsdf_enc_Z
                vox_enc_Z = batch_vox_enc_Z
                pred_tsdf = batch_pred_tsdf
                vae_tsdf = batch_vae_tsdf
                cc_tsdf = batch_cc_tsdf
            else:
                pred_voxs = np.concatenate(
                    (pred_voxs, batch_pred_voxs), axis=0)
                vae_voxs = np.concatenate((vae_voxs, batch_vae_voxs), axis=0)
                cc_voxs = np.concatenate((cc_voxs, batch_cc_voxs), axis=0)
                complete_gen = np.concatenate((complete_gen, batch_vae_tsdf),
                                              axis=0)
                tsdf_enc_Z = np.concatenate((tsdf_enc_Z, batch_tsdf_enc_Z),
                                            axis=0)
                vox_enc_Z = np.concatenate((vox_enc_Z, batch_vox_enc_Z),
                                           axis=0)
                pred_tsdf = np.concatenate(
                    (pred_tsdf, batch_pred_tsdf), axis=0)
                vae_tsdf = np.concatenate((vae_tsdf, batch_vae_tsdf), axis=0)
                cc_tsdf = np.concatenate((cc_tsdf, batch_cc_tsdf), axis=0)

        print("forwarded")

        # For visualization
        # np.save(save_path + '/scene.npy', voxel_test)
        voxel_test.astype('uint8').tofile(save_path + '/scene.bin')

        observe = np.array(tsdf_test)
        observe[observe == -1] = 3
        # np.save(save_path + '/observe.npy', observe)
        observe.astype('uint8').tofile(save_path + '/observe.bin')

        surface = np.array(tsdf_test)
        if cfg.TYPE_TASK is 'scene':
            surface[surface < 0] = 0
            surface[surface > 1] = 0
        elif cfg.TYPE_TASK is 'object':
            surface = np.clip(surface, 0, 1)
        # np.save(save_path + '/surface.npy', surface)
        surface.astype('uint8').tofile(save_path + '/surface.bin')

        depth_seg_real = np.multiply(voxel_test, surface)
        if cfg.TYPE_TASK is 'scene':
            depth_seg_real[depth_seg_real < 0] = 0
        # np.save(save_path + '/depth_seg_scene.npy', depth_seg_real)
        depth_seg_real.astype('uint8').tofile(save_path +
                                              '/depth_seg_scene.bin')

        complete_real = np.clip(voxel_test, 0, 1)
        # np.save(save_path + '/complete_scene.npy', complete_real)
        complete_real.astype('uint8').tofile(save_path + '/complete_real.bin')

        # decoded
        # np.save(save_path + '/gen_vox.npy', np.argmax( pred_voxs, axis=4))
        np.argmax(
            pred_voxs,
            axis=4).astype('uint8').tofile(save_path + '/gen_vox.bin')
        error = np.array(
            np.clip(np.argmax(pred_voxs, axis=4), 0, 1) + complete_real)
        # error[error == 2] = 0
        error.astype('uint8').tofile(save_path + '/gen_vox_error.bin')

        # np.save(save_path + '/vae_vox.npy', np.argmax(vae_voxs, axis=4))
        np.argmax(
            vae_voxs,
            axis=4).astype('uint8').tofile(save_path + '/vae_vox.bin')
        error = np.array(
            np.clip(np.argmax(vae_voxs, axis=4), 0, 1) + complete_real)
        # error[error == 2] = 0
        error.astype('uint8').tofile(save_path + '/vae_vox_error.bin')

        # np.save(save_path + '/cc_vox.npy', np.argmax(cc_voxs, axis=4))
        np.argmax(
            cc_voxs, axis=4).astype('uint8').tofile(save_path + '/cc_vox.bin')
        error = np.array(
            np.clip(np.argmax(cc_voxs, axis=4), 0, 1) + complete_real)
        # error[error == 2] = 0
        error.astype('uint8').tofile(save_path + '/cc_vox_error.bin')

        # np.save(save_path + '/gen_tsdf.npy', np.argmax(pred_tsdf, axis=4))
        if cfg.TYPE_TASK is 'scene':
            pred_tsdf = np.argmax(pred_tsdf, axis=4)
            # pred_tsdf[pred_tsdf < 0] = 0
            pred_tsdf[pred_tsdf > 1] = 0
            vae_tsdf = np.argmax(vae_tsdf, axis=4)
            # vae_tsdf[vae_tsdf < 0] = 0
            vae_tsdf[vae_tsdf > 1] = 0
            cc_tsdf = np.argmax(cc_tsdf, axis=4)
            # cc_tsdf[cc_tsdf < 0] = 0
            cc_tsdf[cc_tsdf > 1] = 0
        elif cfg.TYPE_TASK is 'object':
            pred_tsdf = np.argmax(pred_tsdf, axis=4)
            vae_tsdf = np.argmax(vae_tsdf, axis=4)
            cc_tsdf = np.argmax(cc_tsdf, axis=4)
        pred_tsdf.astype('uint8').tofile(save_path + '/gen_tsdf.bin')
        vae_tsdf.astype('uint8').tofile(save_path + '/vae_tsdf.bin')
        cc_tsdf.astype('uint8').tofile(save_path + '/cc_tsdf.bin')

        # np.save(save_path + '/complete_gen.npy', np.argmax( complete_gen, axis=4))
        np.argmax(
            complete_gen,
            axis=4).astype('uint8').tofile(save_path + '/complete_gen.bin')

        np.save(save_path + '/decode_z_tsdf.npy', tsdf_enc_Z)
        np.save(save_path + '/decode_z_vox.npy', vox_enc_Z)

        print("voxels saved")

        # numerical evalutation
        on_real = onehot(voxel_test, vox_shape[3])
        on_depth_seg_real = onehot(
            np.multiply(voxel_test, surface), vox_shape[3])
        on_complete_real = onehot(complete_real, 2)
        on_complete_gen = onehot(np.argmax(complete_gen, axis=4), 2)

        # calc_IoU
        # completion
        IoU_comp = np.zeros([2 + 1])
        AP_comp = np.zeros([2 + 1])
        print(colored("Completion", 'cyan'))
        IoU_comp, AP_comp = IoU_AP_calc(
            on_complete_real, on_complete_gen, complete_gen, IoU_comp, AP_comp,
            [vox_shape[0], vox_shape[1], vox_shape[2], 2])

        # depth segmentation
        print(colored("Depth segmentation", 'cyan'))
        IoU_class = np.zeros([vox_shape[3] + 1])
        AP_class = np.zeros([vox_shape[3] + 1])
        IoU_class, AP_class = IoU_AP_calc(
            on_depth_seg_real, on_depth_seg_real,
            np.multiply(pred_voxs, np.expand_dims(surface, -1)),
            IoU_class, AP_class, vox_shape)
        IoU_all = np.expand_dims(IoU_class, axis=1)
        AP_all = np.expand_dims(AP_class, axis=1)

        # volume segmentation
        print(colored("Decoded segmentation", 'cyan'))
        on_pred = onehot(np.argmax(pred_voxs, axis=4), vox_shape[3])
        IoU_class, AP_class = IoU_AP_calc(on_real, on_pred, pred_voxs,
                                          IoU_class, AP_class, vox_shape)
        IoU_all = np.concatenate((IoU_all, np.expand_dims(IoU_class, axis=1)),
                                 axis=1)
        AP_all = np.concatenate((AP_all, np.expand_dims(AP_class, axis=1)),
                                axis=1)

        print(colored("VAE segmentation", 'cyan'))
        on_pred = onehot(np.argmax(vae_voxs, axis=4), vox_shape[3])
        IoU_class, AP_class = IoU_AP_calc(on_real, on_pred, vae_voxs,
                                          IoU_class, AP_class, vox_shape)
        IoU_all = np.concatenate((IoU_all, np.expand_dims(IoU_class, axis=1)),
                                 axis=1)
        AP_all = np.concatenate((AP_all, np.expand_dims(AP_class, axis=1)),
                                axis=1)

        print(colored("Cycle consistency segmentation", 'cyan'))
        on_pred = onehot(np.argmax(cc_voxs, axis=4), vox_shape[3])
        IoU_class, AP_class = IoU_AP_calc(on_real, on_pred, cc_voxs,
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
        decode_z = np.load(save_path + '/decode_z_vox.npy')
        print(save_path)
        decode_z = decode_z[:batch_size]
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
                        Z_var_np_sample = np.concatenate([Z, Z_np_sample],
                                                         axis=0)
                        pred_voxs_fromrand = sess.run(
                            vox_tf_sample,
                            feed_dict={Z_tf_sample: Z_var_np_sample})
                        """
                        refined_voxs_fromrand = sess.run(
                            sample_refine_vox_tf,
                            feed_dict={sample_vox_tf: pred_voxs_fromrand})
                        interpolate_vox = np.reshape(
                            refined_voxs_fromrand[0], [
                                1, vox_shape[0], vox_shape[1], vox_shape[2],
                                vox_shape[3]
                            ])
                        interpolate_vox = np.reshape(
                            pred_voxs_fromrand[0], [
                                1, vox_shape[0], vox_shape[1], vox_shape[2],
                                vox_shape[3]
                            ])
                        """
                        if i == 0:
                            pred_voxs = interpolate_vox
                        else:
                            pred_voxs = np.concatenate(
                                [pred_voxs, interpolate_vox], axis=0)
                    """
                    np.save(
                        save_path + '/interpolate/interpolation_z' + str(l) + '-' + str(r)
                        + '.npy', interpolate_z)
                    """
                    interpolate_z.astype('uint8').tofile(
                        save_path + '/interpolate/interpolation_z' + str(l) +
                        '-' + str(r) + '.bin')

                    vox_models_cat = np.argmax(pred_voxs, axis=4)
                    """
                    np.save(
                        save_path + '/interpolate/interpolation' + str(l) + '-' + str(r) +
                        '.npy', vox_models_cat)
                    """
                    vox_models_cat.astype('uint8').tofile(
                        save_path + '/interpolate/interpolation' + str(l) +
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
                                vox_tf_sample,
                                feed_dict={Z_tf_sample: Z_var_np_sample})
                            """
                            refined_voxs_fromrand = sess.run(
                                sample_refine_vox_tf,
                                feed_dict={
                                    sample_vox_tf: pred_voxs_fromrand
                                })
                            noise_vox = np.reshape(refined_voxs_fromrand[0], [
                                1, vox_shape[0], vox_shape[1], vox_shape[2],
                                vox_shape[3]
                            ])
                            """
                            if i == 0:
                                pred_voxs = noise_vox
                            else:
                                pred_voxs = np.concatenate(
                                    [pred_voxs, noise_vox], axis=0)

                        np.save(
                            save_path + '/noise_z' + str(base_num) + '_' +
                            str(c) + str(l) + str(d) + '.npy', noise_z)

                        vox_models_cat = np.argmax(pred_voxs, axis=4)
                        np.save(
                            save_path + '/noise' + str(base_num) + '_' + str(c)
                            + str(l) + str(d) + '.npy', vox_models_cat)

        print("voxels saved")
