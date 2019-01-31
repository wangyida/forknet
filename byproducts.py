import numpy as np
import tensorflow as tf
import os

from config import cfg
from model import depvox_gan
from util import DataProcess, scene_model_id_pair, onehot, scene_model_id_pair_test
from sklearn.metrics import average_precision_score
import copy

from colorama import init
from termcolor import colored

# use Colorama to make Termcolor work on Windows too
init()


def byproduct(batch_size, checknum):

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
    sess = tf.InteractiveSession()
    saver = tf.train.Saver()

    # Restore variables from disk.
    saver.restore(sess, chckpt_path)

    print("...Weights restored.")

    model_path = cfg.DIR.ROOT_PATH
    models = os.listdir(model_path)
    scene_name_pair = []  # full path of the objs files
    scene_name_pair.extend([(model_path, model_id) for model_id in models])
    num_models = len(scene_name_pair)

    batch_tsdf = np.zeros((batch_size, n_vox[0], n_vox[1], n_vox[2]),
                          dtype=np.float32)
    batch_voxel = np.zeros((batch_size, n_vox[0], n_vox[1], n_vox[2]),
                           dtype=np.float32)

    for i in np.arange(num_models):
        sceneId, model_id = scene_name_pair[i]

        voxel_fn = cfg.DIR.VOXEL_PATH % (model_id)
        voxel_data = np.load(voxel_fn)
        batch_voxel[0, :, :, :] = voxel_data

        tsdf_fn = cfg.DIR.TSDF_PATH % (model_id)
        tsdf_data = np.load(tsdf_fn)
        batch_tsdf[0, :, :, :] = tsdf_data

        # Evaluation masks
        if cfg.TYPE_TASK is 'scene':
            volume_effective = np.clip(
                np.where(batch_voxel > 0, 1, 0) + np.where(
                    batch_tsdf > 0, 1, 0), 0, 1)
            batch_voxel *= volume_effective
            batch_tsdf *= volume_effective

            # batch_tsdf[batch_tsdf > 1] = 0
            # batch_tsdf_test[np.where(batch_voxel_test == 10)] = 1

        batch_pred_voxs, batch_vae_voxs, batch_cc_voxs,\
        batch_pred_tsdf, batch_vae_tsdf, batch_cc_tsdf = sess.run(
            [
                vox_gen_decode_tf, vox_vae_decode_tf, vox_cc_decode_tf,
                tsdf_gen_decode_tf, tsdf_vae_decode_tf, tsdf_cc_decode_tf
            ],
            feed_dict={
                tsdf_tf: batch_tsdf,
                vox_tf: batch_voxel
            })

        batch_pred_tsdf = np.argmax(
            batch_pred_tsdf, axis=4).astype('float32')
        import ipdb
        ipdb.set_trace()
        # batch_pred_tsdf[batch_tsdf == -1] = -1
        np.save(
            '/media/wangyida/SSD2T/database/SUNCG_Yida/test/depth_tsdf_vae_npy/'
            + models[i], batch_pred_tsdf[0])


if __name__ == '__main__':
    byproduct(1, 464)
