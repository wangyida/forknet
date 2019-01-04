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


def byproduct(batch_size, checknum, mode):

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


    Z_tf, z_tsdf_enc_tf, z_vox_enc_tf, vox_tf, vox_gen_tf, vox_gen_decode_tf, vox_vae_decode_tf, vox_cc_decode_tf, tsdf_seg_tf,\
    gen_vae_loss_tf, gen_cc_loss_tf, gen_gen_loss_tf, code_encode_loss_tf, gen_loss_tf, discrim_loss_tf,\
    cost_enc_tf, cost_code_tf, cost_gen_tf, cost_discrim_tf, summary_tf,\
    tsdf_tf, tsdf_gen_tf, tsdf_gen_decode_tf, tsdf_vae_decode_tf, tsdf_cc_decode_tf = depvox_gan_model.build_model()
    sess = tf.InteractiveSession()
    saver = tf.train.Saver()

    # Restore variables from disk.
    saver.restore(sess, chckpt_path)

    print("...Weights restored.")

    # evaluation for reconstruction
    voxel_test, tsdf_test, num, data_paths = scene_model_id_pair_test(
        dataset_portion=cfg.TRAIN.DATASET_PORTION)

    num = voxel_test.shape[0]
    print("test voxels loaded")
    for i in np.arange(int(num / batch_size)):
        batch_voxel_test = voxel_test[i * batch_size:i * batch_size +
                                      batch_size]
        batch_tsdf_test = tsdf_test[i * batch_size:i * batch_size + batch_size]

        # Evaluation masks
        if cfg.TYPE_TASK is 'scene':
            # Evaluation masks
            volume_effective = np.clip(
                np.where(batch_voxel_test > 0, 1, 0) + np.where(
                    batch_tsdf_test > 0, 1, 0), 0, 1)
            batch_voxel_test *= volume_effective
            batch_tsdf_test *= volume_effective

            batch_tsdf_test[batch_tsdf_test > 1] = 0
            # batch_tsdf_test[np.where(batch_voxel_test == 10)] = 1

        batch_generated_voxs, batch_vae_voxs, batch_cc_voxs, batch_depth_seg_gen, batch_tsdf_enc_Z, batch_vox_enc_Z, batch_generated_tsdf, batch_vae_tsdf, batch_cc_tsdf = sess.run(
            [
                vox_gen_decode_tf, vox_vae_decode_tf, vox_cc_decode_tf,
                tsdf_seg_tf, z_tsdf_enc_tf, z_vox_enc_tf, tsdf_gen_decode_tf,
                tsdf_vae_decode_tf, tsdf_cc_decode_tf
            ],
            feed_dict={
                tsdf_tf: batch_tsdf_test,
                vox_tf: batch_voxel_test
            })

        # Masked
        if cfg.TYPE_TASK is 'scene':
            batch_generated_voxs *= np.expand_dims(volume_effective, -1)
            batch_vae_voxs *= np.expand_dims(volume_effective, -1)
            batch_cc_voxs *= np.expand_dims(volume_effective, -1)

        if i == 0:
            generated_voxs = batch_generated_voxs
            vae_voxs = batch_vae_voxs
            cc_voxs = batch_cc_voxs
            depth_seg_gen = batch_depth_seg_gen
            complete_gen = batch_vae_tsdf
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
            complete_gen = np.concatenate((complete_gen, batch_vae_tsdf),
                                          axis=0)
            tsdf_enc_Z = np.concatenate((tsdf_enc_Z, batch_tsdf_enc_Z), axis=0)
            vox_enc_Z = np.concatenate((vox_enc_Z, batch_vox_enc_Z), axis=0)
            generated_tsdf = np.concatenate(
                (generated_tsdf, batch_generated_tsdf), axis=0)
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
        generated_tsdf[generated_tsdf < 0] = 0
        generated_tsdf[generated_tsdf > 1] = 0
        vae_tsdf[vae_tsdf < 0] = 0
        vae_tsdf[vae_tsdf > 1] = 0
        cc_tsdf[cc_tsdf < 0] = 0
        cc_tsdf[cc_tsdf > 1] = 0
    elif cfg.TYPE_TASK is 'object':
        surface = np.clip(surface, 0, 1)
    # np.save(save_path + '/surface.npy', surface)
    surface.astype('uint8').tofile(save_path + '/surface.bin')

    depth_seg_real = np.multiply(voxel_test, surface)
    if cfg.TYPE_TASK is 'scene':
        depth_seg_real[depth_seg_real < 0] = 0
    # np.save(save_path + '/depth_seg_scene.npy', depth_seg_real)
    depth_seg_real.astype('uint8').tofile(save_path + '/depth_seg_scene.bin')

    complete_real = np.clip(voxel_test, 0, 1)
    # np.save(save_path + '/complete_scene.npy', complete_real)
    complete_real.astype('uint8').tofile(save_path + '/complete_real.bin')

    # decoded
    # np.save(save_path + '/gen_vox.npy', np.argmax( generated_voxs, axis=4))
    np.argmax(
        generated_voxs,
        axis=4).astype('uint8').tofile(save_path + '/gen_vox.bin')
    error = np.array(
        np.clip(np.argmax(generated_voxs, axis=4), 0, 1) + complete_real)
    # error[error == 2] = 0
    error.astype('uint8').tofile(save_path + '/gen_vox_error.bin')

    # np.save(save_path + '/vae_vox.npy', np.argmax(vae_voxs, axis=4))
    np.argmax(
        vae_voxs, axis=4).astype('uint8').tofile(save_path + '/vae_vox.bin')
    error = np.array(
        np.clip(np.argmax(vae_voxs, axis=4), 0, 1) + complete_real)
    # error[error == 2] = 0
    error.astype('uint8').tofile(save_path + '/vae_vox_error.bin')

    # np.save(save_path + '/cc_vox.npy', np.argmax(cc_voxs, axis=4))
    np.argmax(
        cc_voxs, axis=4).astype('uint8').tofile(save_path + '/cc_vox.bin')
    error = np.array(np.clip(np.argmax(cc_voxs, axis=4), 0, 1) + complete_real)
    # error[error == 2] = 0
    error.astype('uint8').tofile(save_path + '/cc_vox_error.bin')

    # np.save(save_path + '/gen_tsdf.npy', np.argmax(generated_tsdf, axis=4))
    if cfg.TYPE_TASK is 'scene':
        generated_tsdf = np.argmax(generated_tsdf, axis=4)
        generated_tsdf[generated_tsdf < 0] = 0
        generated_tsdf[generated_tsdf > 1] = 0
        vae_tsdf = np.argmax(vae_tsdf, axis=4)
        vae_tsdf[vae_tsdf < 0] = 0
        vae_tsdf[vae_tsdf > 1] = 0
        cc_tsdf = np.argmax(cc_tsdf, axis=4)
        cc_tsdf[cc_tsdf < 0] = 0
        cc_tsdf[cc_tsdf > 1] = 0
    generated_tsdf.astype('uint8').tofile(save_path + '/gen_tsdf.bin')
    vae_tsdf.astype('uint8').tofile(save_path + '/vae_tsdf.bin')
    cc_tsdf.astype('uint8').tofile(save_path + '/cc_tsdf.bin')

    # np.save(save_path + '/depth_seg_gen.npy', np.argmax(depth_seg_gen, axis=4))
    np.argmax(
        depth_seg_gen,
        axis=4).astype('uint8').tofile(save_path + '/depth_seg_gen.bin')

    # np.save(save_path + '/complete_gen.npy', np.argmax( complete_gen, axis=4))
    np.argmax(
        complete_gen,
        axis=4).astype('uint8').tofile(save_path + '/complete_gen.bin')

    np.save(save_path + '/decode_z_tsdf.npy', tsdf_enc_Z)
    np.save(save_path + '/decode_z_vox.npy', vox_enc_Z)

    print("voxels saved")
