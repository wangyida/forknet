import numpy as np
import os
import random
import matplotlib.image as mpimg
from scipy.misc import imsave

from config import cfg


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

            tsdf_fn = cfg.DIR.TSDF_PATH % (model_id)
            tsdf_data = np.load(tsdf_fn)

            batch_tsdf[batch_id, :, :, :] = tsdf_data
        return batch_tsdf

    def get_voxel(self, db_inds):
        batch_voxel = np.zeros(
            (self.batch_size, self.n_vox[0], self.n_vox[1], self.n_vox[2]),
            dtype=np.float32)

        for batch_id, db_ind in enumerate(db_inds):
            sceneId, model_id = self.data_paths[db_ind]

            voxel_fn = cfg.DIR.VOXEL_PATH % (model_id)
            voxel_data = np.load(voxel_fn)

            batch_voxel[batch_id, :, :, :] = voxel_data
        return batch_voxel

    """
    def get_depth(self, db_inds):
        batch_depth = np.zeros(
                    (self.batch_size, self.n_dep[0], self.n_dep[1], self.n_dep[2]), dtype=np.float32)
    
        for batch_id, db_ind in enumerate(db_inds):
            sceneId, model_id = self.data_paths[db_ind]

            depth_fn = cfg.DIR.DEPTH_PATH % (model_id)
            depth_data = np.load(depth_fn)

            batch_depth[batch_id, :, :, :] = np.reshape(depth_data, [self.n_dep[0], self.n_dep[1], self.n_dep[2]])
        return batch_depth
    """


def scene_model_id_pair(dataset_portion=[]):
    '''
    Load sceneId, model names from a suncg dataset.
    '''

    scene_name_pair = []  # full path of the objs files

    model_path = cfg.DIR.ROOT_PATH
    models = os.listdir(model_path)

    scene_name_pair.extend([(model_path, model_id) for model_id in models])

    num_models = len(scene_name_pair)
    portioned_scene_name_pair = scene_name_pair[int(num_models *
                                                    dataset_portion[0]):]

    return portioned_scene_name_pair


def scene_model_id_pair_test(dataset_portion=[]):

    amount_of_test_sample = 76

    scene_name_pair = []  # full path of the objs files

    model_path = cfg.DIR.ROOT_PATH
    models = os.listdir(model_path)

    scene_name_pair.extend([(model_path, model_id) for model_id in models])

    num_models = len(scene_name_pair)
    data_paths_test = scene_name_pair[int(num_models * dataset_portion[0]) +
                                      1:]
    # random.shuffle(data_paths_test)
    #data_paths = scene_name_pair[int(num_models * dataset_portion[1])+1:int(num_models * dataset_portion[1])+amount_of_test_sample+1]
    data_paths = data_paths_test[:amount_of_test_sample]

    num_models = len(data_paths)
    print '---amount of test data:' + str(num_models)

    n_vox = cfg.CONST.N_VOX

    batch_voxel = np.zeros((num_models, n_vox[0], n_vox[1], n_vox[2]),
                           dtype=np.float32)
    batch_tsdf = np.zeros((num_models, n_vox[0], n_vox[1], n_vox[2]),
                          dtype=np.float32)

    for i in np.arange(num_models):
        sceneId, model_id = data_paths[i]

        # save depth images accordingly
        depth_fn = sceneId.replace("voxel_semantic_npy",
                                   "depth_real_png") + "/" + model_id.replace(
                                       ".npy", ".png")
        img = mpimg.imread(depth_fn)
        imsave('eval/vis_depth/' + str(i) + '.png', img)

        voxel_fn = cfg.DIR.VOXEL_PATH % (model_id)
        voxel_data = np.load(voxel_fn)

        batch_voxel[i, :, :, :] = voxel_data

        tsdf_fn = cfg.DIR.TSDF_PATH % (model_id)
        tsdf_data = np.load(tsdf_fn)

        batch_tsdf[i, :, :, :] = tsdf_data

    return batch_voxel, batch_tsdf, num_models


def onehot(voxel, class_num):
    onehot_voxels = np.zeros((voxel.shape[0], voxel.shape[1], voxel.shape[2],
                              voxel.shape[3], class_num))
    for i in np.arange(class_num):
        onehot_voxel = np.zeros(voxel.shape)
        onehot_voxel[np.where(voxel == i)] = 1
        onehot_voxels[:, :, :, :, i] = onehot_voxel[:, :, :, :]
    return onehot_voxels
