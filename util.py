import numpy as np
import os
import random
import imageio
import matplotlib.image as mpimg

from config import cfg
from colorama import init
from termcolor import colored


class DataProcess():
    def __init__(self, data_paths, batch_size, repeat=True):
        self.data_paths = data_paths
        self.num_data = len(data_paths)
        self.repeat = repeat

        self.batch_size = batch_size
        self.shuffle_db_inds()
        self.n_vox = cfg.CONST.N_VOX
        # self.n_dep = cfg.CONST.N_DEP

    def shuffle_db_inds(self):
        # Randomly permute the training roidb
        if self.repeat:
            self.perm = np.random.permutation(np.arange(self.num_data))
        else:
            self.perm = np.arange(self.num_data)
        self.cur = 0

    def get_next_minibatch(self):
        flag = True
        if (self.cur + self.batch_size) >= self.num_data and self.repeat:
            self.shuffle_db_inds()
            flag = False

        db_inds = self.perm[self.cur:min(self.cur +
                                         self.batch_size, self.num_data)]
        self.cur += self.batch_size
        return db_inds, flag

    def get_tsdf(self, db_inds):
        batch_tsdf = np.zeros(
            (self.batch_size, self.n_vox[0], self.n_vox[1], self.n_vox[2]),
            dtype=np.float32)

        for batch_id, db_ind in enumerate(db_inds):
            sceneId, model_id = self.data_paths[db_ind]

            tsdf_fn = cfg.DIR.TSDF_PATH + model_id
            tsdf_data = np.load(tsdf_fn)

            batch_tsdf[batch_id, :, :, :] = tsdf_data
        return batch_tsdf

    def get_voxel(self, db_inds):
        batch_voxel = np.zeros(
            (self.batch_size, self.n_vox[0], self.n_vox[1], self.n_vox[2]),
            dtype=np.float32)

        for batch_id, db_ind in enumerate(db_inds):
            sceneId, model_id = self.data_paths[db_ind]

            voxel_fn = cfg.DIR.VOXEL_PATH + model_id
            voxel_data = np.load(voxel_fn)

            batch_voxel[batch_id, :, :, :] = voxel_data
        return batch_voxel

    def get_surf(self, db_inds):
        batch_surf = np.zeros(
            (self.batch_size, self.n_vox[0], self.n_vox[1], self.n_vox[2]),
            dtype=np.float32)

        for batch_id, db_ind in enumerate(db_inds):
            sceneId, model_id = self.data_paths[db_ind]

            surf_fn = cfg.DIR.SURF_PATH + model_id
            surf_data = np.load(surf_fn)

            batch_surf[batch_id, :, :, :] = surf_data
        return batch_surf


def id_models_train(dataset_portion=[], data_list='./train_3rscan.list'):
    '''
    Load sceneId, model names from a suncg dataset.
    '''

    scene_name_pair = []  # full path of the objs files

    model_path = cfg.DIR.TSDF_PATH
    """
    models = os.listdir(model_path)
    """
    with open(data_list) as file:
        models = file.read().splitlines()

    scene_name_pair.extend([(model_path, model_id) for model_id in models])

    num_models = len(scene_name_pair)
    portioned_scene_name_pair = scene_name_pair[int(num_models *
                                                    dataset_portion[0]):]

    return portioned_scene_name_pair


def id_models_test(dataset_portion=[],
                   data_list='./lists_infer/test_3rscan.list'):

    amount_of_test_sample = 50

    scene_name_pair = []  # full path of the objs files

    model_path = cfg.DIR.TSDF_PATH
    with open(data_list) as file:
        models = file.read().splitlines()

    scene_name_pair.extend([(model_path, model_id) for model_id in models])

    num_models = len(scene_name_pair)
    data_paths_test = scene_name_pair[int(num_models * dataset_portion[0]) +
                                      0:]
    random.seed(1)
    random.shuffle(data_paths_test)
    # print('The first sample is:', data_paths_test[0][1])

    data_paths = data_paths_test
    # data_paths = data_paths_test[:amount_of_test_sample]

    num_models = len(data_paths)
    print('The amount of test data: %d' % num_models)

    n_vox = cfg.CONST.N_VOX

    batch_tsdf = np.zeros((num_models, n_vox[0], n_vox[1], n_vox[2]),
                          dtype=np.float32)
    batch_surf = np.zeros((num_models, n_vox[0], n_vox[1], n_vox[2]),
                          dtype=np.float32)
    batch_voxel = np.zeros((num_models, n_vox[0], n_vox[1], n_vox[2]),
                           dtype=np.float32)

    for i in np.arange(num_models):
        sceneId, model_id = data_paths[i]

        # save depth images accordingly
        depth_fn = sceneId.replace("depth_tsdf_camera_npy",
                                   "depth_real_png") + "/" + model_id.replace(
                                       ".npy", ".png")
        if os.path.isfile(depth_fn):
            img = mpimg.imread(depth_fn)
            img_uint8 = img.astype(np.uint8)
            imageio.imwrite(
                'results_depth/' + data_paths_test[i][1][:-4] + '.png',
                img_uint8)

        tsdf_fn = cfg.DIR.TSDF_PATH + model_id
        tsdf_data = np.load(tsdf_fn)
        batch_tsdf[i, :, :, :] = tsdf_data

        surf_fn = cfg.DIR.SURF_PATH + model_id
        surf_data = np.load(surf_fn)
        batch_surf[i, :, :, :] = surf_data

        voxel_fn = cfg.DIR.VOXEL_PATH + model_id
        voxel_data = np.load(voxel_fn)
        batch_voxel[i, :, :, :] = voxel_data

    return batch_voxel, batch_surf, batch_tsdf, num_models, data_paths[:
                                                                       num_models]


def onehot(voxel, class_num):
    return np.eye(class_num)[voxel.astype(int)]
    """
    onehot_voxels = np.zeros((voxel.shape[0], voxel.shape[1], voxel.shape[2],
                              voxel.shape[3], class_num))
    for i in np.arange(class_num):
        onehot_voxel = np.zeros(voxel.shape)
        onehot_voxel[np.where(voxel == i)] = 1
        onehot_voxels[:, :, :, :, i] = onehot_voxel[:, :, :, :]
    return onehot_voxels
    """
