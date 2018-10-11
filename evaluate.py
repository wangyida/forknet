import numpy as np
import tensorflow as tf

from config import cfg
from model import FCR_aGAN
from util import DataProcess, scene_model_id_pair, onehot, scene_model_id_pair_test
from sklearn.metrics import average_precision_score
import copy

from colorama import init
from termcolor import colored

# use Colorama to make Termcolor work on Windows too
init()


def IoU_AP_calc(on_real, on_recons, generated_voxs, IoU_class, AP_class,
                vox_shape):
    # calc_IoU
    num = on_real.shape[0]
    for class_n in np.arange(vox_shape[3]):
        on_recons_ = on_recons[:, :, :, :, class_n]
        on_real_ = on_real[:, :, :, :, class_n]
        mother = np.sum(np.add(on_recons_, on_real_), (1, 2, 3))
        # Here the child must be doubled because the mother is calculated twice
        child = np.sum(np.multiply(on_recons_, on_real_), (1, 2, 3)) * 2
        count = 0
        IoU_element = 0
        for i in np.arange(num):
            if mother[i] != 0:
                IoU_element += child[i] / mother[i]
                count += 1
        IoU_calc = np.round(IoU_element / count, 3)
        IoU_class[class_n] = IoU_calc
        print 'IoU class ' + str(class_n) + '=' + str(IoU_calc)
    print 'IoU category-wise = ' + str(
        np.round(np.sum(IoU_class[:vox_shape[3] - 1]) / vox_shape[3]), 3)

    on_recons_ = on_recons[:, :, :, :, 1:vox_shape[3]]
    on_real_ = on_real[:, :, :, :, 1:vox_shape[3]]
    mother = np.sum(np.add(on_recons_, on_real_), (1, 2, 3, 4))
    child = np.sum(np.multiply(on_recons_, on_real_), (1, 2, 3, 4)) * 2
    count = 0
    IoU_element = 0
    for i in np.arange(num):
        if mother[i] != 0:
            IoU_element += child[i] / mother[i]
            count += 1
    IoU_calc = np.round(IoU_element / count, 3)
    IoU_class[vox_shape[3]] = IoU_calc
    print 'IoU space-wise =' + str(IoU_calc)

    #calc_AP
    for class_n in np.arange(vox_shape[3]):
        on_recons_ = generated_voxs[:, :, :, :, class_n]
        on_real_ = on_real[:, :, :, :, class_n]

        AP = 0.
        for i in np.arange(num):
            y_true = np.reshape(on_real_[i], [-1])
            y_scores = np.reshape(on_recons_[i], [-1])
            if np.sum(y_true) > 0.:
                AP += average_precision_score(y_true, y_scores)
        AP = np.round(AP / num, 3)
        AP_class[class_n] = AP
        print 'AP class ' + str(class_n) + '=' + str(AP)
    print 'AP category-wise = ' + str(
        np.round(np.sum(AP_class[:vox_shape[3] - 1]) / vox_shape[3]), 3)

    on_recons_ = generated_voxs[:, :, :, :, 1:vox_shape[3]]
    on_real_ = on_real[:, :, :, :, 1:vox_shape[3]]
    AP = 0.
    for i in np.arange(num):
        y_true = np.reshape(on_real_[i], [-1])
        y_scores = np.reshape(on_recons_[i], [-1])
        if np.sum(y_true) > 0.:
            AP += average_precision_score(y_true, y_scores)

    AP = np.round(AP / num, 3)
    AP_class[vox_shape[3]] = AP
    print 'AP space-wise =' + str(AP)
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
    refine_ch = cfg.NET.REFINE_CH
    refine_kernel = cfg.NET.REFINE_KERNEL
    refiner = cfg.NET.REFINER
    discriminative = cfg.NET.DISCRIMINATIVE
    generative = cfg.NET.GENERATIVE

    save_path = cfg.DIR.EVAL_PATH
    chckpt_path = cfg.DIR.CHECK_PT_PATH + str(checknum)

    fcr_agan_model = FCR_aGAN(
        batch_size=batch_size,
        vox_shape=vox_shape,
        dim_z=dim_z,
        dim=dim,
        start_vox_size=start_vox_size,
        kernel=kernel,
        stride=stride,
        dilations=dilations,
        refine_ch=refine_ch,
        refine_kernel=refine_kernel,
        refiner=refiner,
        generative=generative)


    Z_tf, z_tsdf_enc_tf, z_vox_enc_tf, vox_tf, vox_gen_tf, vox_gen_decode_tf, vox_vae_decode_tf, vox_cc_decode_tf, vox_gen_complete_tf, tsdf_seg_tf, vox_refine_dec_tf, vox_refine_gen_tf,\
    recons_vae_loss_tf, recons_cc_loss_tf, recons_gen_loss_tf, code_encode_loss_tf, gen_loss_tf, discrim_loss_tf, recons_loss_refine_tfs, gen_loss_refine_tf, discrim_loss_refine_tf,\
    cost_enc_tf, cost_code_tf, cost_gen_tf, cost_discrim_tf, cost_gen_ref_tf, cost_discrim_ref_tf, summary_tf,\
    tsdf_tf, tsdf_gen_tf, tsdf_gen_decode_tf, tsdf_vae_decode_tf, tsdf_cc_decode_tf = fcr_agan_model.build_model()
    Z_tf_sample, vox_tf_sample = fcr_agan_model.samples_generator(
        visual_size=batch_size)
    if refiner is 'sscnet':
        sample_vox_tf, sample_refine_vox_tf = fcr_agan_model.refine_generator_sscnet(
            visual_size=batch_size)
    else:
        sample_vox_tf, sample_refine_vox_tf = fcr_agan_model.refine_generator_resnet(
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
        np.save(save_path + '/sample_z.npy', Z_var_np_sample)

        generated_voxs_fromrand = sess.run(
            vox_tf_sample, feed_dict={Z_tf_sample: Z_var_np_sample})
        np.save(save_path + '/generate.npy',
                np.argmax(generated_voxs_fromrand, axis=4))

        refined_voxs_fromrand = sess.run(
            sample_refine_vox_tf,
            feed_dict={sample_vox_tf: generated_voxs_fromrand})
        np.save(save_path + '/generate_refine.npy',
                np.argmax(refined_voxs_fromrand, axis=4))

        # evaluation for reconstruction
        voxel_test, tsdf_test, num = scene_model_id_pair_test(
            dataset_portion=cfg.TRAIN.DATASET_PORTION)
        voxel_test = np.multiply(voxel_test, np.where(tsdf_test > 0, 1, 0))
        num = voxel_test.shape[0]
        print("test voxels loaded")
        for i in np.arange(int(num / batch_size)):
            batch_voxel_test = voxel_test[i * batch_size:i * batch_size +
                                          batch_size]
            batch_tsdf_test = tsdf_test[i * batch_size:i * batch_size +
                                        batch_size]

            batch_generated_voxs, batch_vae_voxs, batch_cc_voxs, batch_depth_seg_gen, batch_complete_gen, batch_tsdf_enc_Z, batch_vox_enc_Z, batch_generated_tsdf, batch_vae_tsdf, batch_cc_tsdf = sess.run(
                [
                    vox_gen_decode_tf, vox_vae_decode_tf, vox_cc_decode_tf,
                    tsdf_seg_tf, vox_gen_complete_tf, z_tsdf_enc_tf,
                    z_vox_enc_tf, tsdf_gen_decode_tf, tsdf_vae_decode_tf,
                    tsdf_cc_decode_tf
                ],
                # feed_dict={tsdf_tf: batch_tsdf_test})
                feed_dict={
                    tsdf_tf: batch_tsdf_test,
                    vox_tf: batch_voxel_test
                })

            # This can eliminate some false positive
            batch_generated_voxs = np.multiply(
                batch_generated_voxs,
                np.expand_dims(np.where(batch_voxel_test > 0, 1, 0), -1))
            batch_vae_voxs = np.multiply(
                batch_vae_voxs,
                np.expand_dims(np.where(batch_voxel_test > 0, 1, 0), -1))
            batch_cc_voxs = np.multiply(
                batch_cc_voxs,
                np.expand_dims(np.where(batch_voxel_test > 0, 1, 0), -1))

            batch_refined_vox = sess.run(
                sample_refine_vox_tf,
                feed_dict={sample_vox_tf: batch_generated_voxs})

            # This can eliminate some false positive
            batch_refined_vox = np.multiply(
                batch_refined_vox,
                np.expand_dims(np.where(batch_voxel_test > 0, 1, 0), -1))

            if i == 0:
                generated_voxs = batch_generated_voxs
                vae_voxs = batch_vae_voxs
                cc_voxs = batch_cc_voxs
                depth_seg_gen = batch_depth_seg_gen
                complete_gen = batch_complete_gen
                refined_voxs = batch_refined_vox
                tsdf_enc_Z = batch_tsdf_enc_Z
                vox_enc_Z = batch_vox_enc_Z
                generated_tsdf = batch_generated_tsdf
                vae_tsdf = batch_vae_tsdf
                cc_tsdf = batch_cc_tsdf
            else:
                generated_voxs = np.concatenate(
                    (generated_voxs, batch_generated_voxs), axis=0)
                vae_voxs = np.concatenate((vae_voxs, batch_vae_voxs), axis=0)
                cc_voxs = np.concatenate((cc_voxs, batch_cc_voxs), axis=0)
                depth_seg_gen = np.concatenate(
                    (depth_seg_gen, batch_depth_seg_gen), axis=0)
                complete_gen = np.concatenate(
                    (complete_gen, batch_complete_gen), axis=0)
                refined_voxs = np.concatenate(
                    (refined_voxs, batch_refined_vox), axis=0)
                tsdf_enc_Z = np.concatenate((tsdf_enc_Z, batch_tsdf_enc_Z),
                                            axis=0)
                vox_enc_Z = np.concatenate((vox_enc_Z, batch_vox_enc_Z),
                                           axis=0)
                generated_tsdf = np.concatenate(
                    (generated_tsdf, batch_generated_tsdf), axis=0)
                vae_tsdf = np.concatenate((vae_tsdf, batch_vae_tsdf), axis=0)
                cc_tsdf = np.concatenate((cc_tsdf, batch_cc_tsdf), axis=0)

        print("forwarded")

        # real
        np.save(save_path + '/real.npy', voxel_test)
        np.save(save_path + '/tsdf.npy', tsdf_test)
        depth_seg_real = np.multiply(voxel_test, np.where(
            tsdf_test == 1, 1, 0))
        np.save(save_path + '/depth_seg_real.npy', depth_seg_real)
        complete_real = np.clip(voxel_test, 0, 1)
        np.save(save_path + '/complete_real.npy', complete_real)

        # decoded
        np.save(save_path + '/recons_vox.npy', np.argmax(
            generated_voxs, axis=4))
        np.save(save_path + '/vae_vox.npy', np.argmax(vae_voxs, axis=4))
        np.save(save_path + '/cc_vox.npy', np.argmax(cc_voxs, axis=4))
        np.save(save_path + '/recons_tsdf.npy',
                np.argmax(generated_tsdf, axis=4))
        np.save(save_path + '/vae_tsdf.npy', np.argmax(vae_tsdf, axis=4))
        np.save(save_path + '/cc_tsdf.npy', np.argmax(cc_tsdf, axis=4))
        np.save(save_path + '/recons_refine_vox.npy',
                np.argmax(refined_voxs, axis=4))
        np.save(save_path + '/depth_seg_gen.npy',
                np.argmax(depth_seg_gen, axis=4))
        np.save(save_path + '/complete_gen.npy', np.argmax(
            complete_gen, axis=4))
        np.save(save_path + '/decode_z_tsdf.npy', tsdf_enc_Z)
        np.save(save_path + '/decode_z_vox.npy', vox_enc_Z)

        print("voxels saved")

        # numerical evalutation
        on_real = onehot(voxel_test, vox_shape[3])
        on_depth_seg_real = onehot(depth_seg_real, vox_shape[3])
        on_complete_real = onehot(complete_real, 2)
        on_recons = onehot(np.argmax(generated_voxs, axis=4), vox_shape[3])
        on_vae = onehot(np.argmax(vae_voxs, axis=4), vox_shape[3])
        on_cc = onehot(np.argmax(cc_voxs, axis=4), vox_shape[3])
        on_depth_seg_gen = onehot(
            np.multiply(
                np.argmax(depth_seg_gen, axis=4), np.where(
                    tsdf_test == 1, 1, 0)), vox_shape[3])
        on_complete_gen = onehot(np.argmax(complete_gen, axis=4), 2)

        # calc_IoU
        # completion
        IoU_class = np.zeros([2 + 1])
        AP_class = np.zeros([2 + 1])
        print(colored("Completion", 'cyan'))
        IoU_class, AP_class = IoU_AP_calc(
            on_complete_real, on_complete_gen, complete_gen, IoU_class,
            AP_class, [vox_shape[0], vox_shape[1], vox_shape[2], 2])

        # depth segmentation
        print(colored("Depth segmentation", 'cyan'))
        IoU_class = np.zeros([vox_shape[3] + 1])
        AP_class = np.zeros([vox_shape[3] + 1])
        IoU_class, AP_class = IoU_AP_calc(
            on_depth_seg_real, on_depth_seg_gen,
            np.multiply(depth_seg_gen,
                        np.expand_dims(np.where(tsdf_test == 1, 1, 0), -1)),
            IoU_class, AP_class, vox_shape)
        IoU_all = np.expand_dims(IoU_class, axis=1)
        AP_all = np.expand_dims(AP_class, axis=1)

        # volume segmentation
        print(colored("Decoded segmentation", 'cyan'))
        IoU_class, AP_class = IoU_AP_calc(on_real, on_recons, generated_voxs,
                                          IoU_class, AP_class, vox_shape)
        IoU_all = np.concatenate((IoU_all, np.expand_dims(IoU_class, axis=1)),
                                 axis=1)
        AP_all = np.concatenate((AP_all, np.expand_dims(AP_class, axis=1)),
                                axis=1)

        print(colored("VAE segmentation", 'cyan'))
        IoU_class, AP_class = IoU_AP_calc(on_real, on_vae, vae_voxs, IoU_class,
                                          AP_class, vox_shape)
        IoU_all = np.concatenate((IoU_all, np.expand_dims(IoU_class, axis=1)),
                                 axis=1)
        AP_all = np.concatenate((AP_all, np.expand_dims(AP_class, axis=1)),
                                axis=1)

        print(colored("Cycle consistency segmentation", 'cyan'))
        IoU_class, AP_class = IoU_AP_calc(on_real, on_cc, cc_voxs, IoU_class,
                                          AP_class, vox_shape)
        IoU_all = np.concatenate((IoU_all, np.expand_dims(IoU_class, axis=1)),
                                 axis=1)
        AP_all = np.concatenate((AP_all, np.expand_dims(AP_class, axis=1)),
                                axis=1)

        # refine for volume segmentation
        print(colored("refined volume segmentation", 'cyan'))
        on_recons = onehot(np.argmax(refined_voxs, axis=4), vox_shape[3])
        IoU_class, AP_class = IoU_AP_calc(on_real, on_recons, refined_voxs,
                                          IoU_class, AP_class, vox_shape)
        IoU_all = np.concatenate((IoU_all, np.expand_dims(IoU_class, axis=1)),
                                 axis=1)
        AP_all = np.concatenate((AP_all, np.expand_dims(AP_class, axis=1)),
                                axis=1)
        np.savetxt(
            save_path + '/IoU.csv', np.transpose(IoU_all * 100), delimiter="|")
        np.savetxt(
            save_path + '/AP.csv', np.transpose(AP_all * 100), delimiter="|")

    # interpolation evaluation
    if mode == 'interpolate':
        interpolate_num = 30
        #interpolatioin latent vectores
        decode_z = np.load(save_path + '/decode_z.npy')
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
                        generated_voxs_fromrand = sess.run(
                            vox_tf_sample,
                            feed_dict={Z_tf_sample: Z_var_np_sample})
                        refined_voxs_fromrand = sess.run(
                            sample_refine_vox_tf,
                            feed_dict={sample_vox_tf: generated_voxs_fromrand})
                        interpolate_vox = np.reshape(
                            refined_voxs_fromrand[0], [
                                1, vox_shape[0], vox_shape[1], vox_shape[2],
                                vox_shape[3]
                            ])
                        if i == 0:
                            generated_voxs = interpolate_vox
                        else:
                            generated_voxs = np.concatenate(
                                [generated_voxs, interpolate_vox], axis=0)

                    np.save(
                        save_path + '/interpolation_z' + str(l) + '-' + str(r)
                        + '.npy', interpolate_z)

                    vox_models_cat = np.argmax(generated_voxs, axis=4)
                    np.save(
                        save_path + '/interpolation' + str(l) + '-' + str(r) +
                        '.npy', vox_models_cat)
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
                            generated_voxs_fromrand = sess.run(
                                vox_tf_sample,
                                feed_dict={Z_tf_sample: Z_var_np_sample})
                            refined_voxs_fromrand = sess.run(
                                sample_refine_vox_tf,
                                feed_dict={
                                    sample_vox_tf: generated_voxs_fromrand
                                })
                            noise_vox = np.reshape(refined_voxs_fromrand[0], [
                                1, vox_shape[0], vox_shape[1], vox_shape[2],
                                vox_shape[3]
                            ])
                            if i == 0:
                                generated_voxs = noise_vox
                            else:
                                generated_voxs = np.concatenate(
                                    [generated_voxs, noise_vox], axis=0)

                        np.save(
                            save_path + '/noise_z' + str(base_num) + '_' +
                            str(c) + str(l) + str(d) + '.npy', noise_z)

                        vox_models_cat = np.argmax(generated_voxs, axis=4)
                        np.save(
                            save_path + '/noise' + str(base_num) + '_' + str(c)
                            + str(l) + str(d) + '.npy', vox_models_cat)

        print("voxels saved")
