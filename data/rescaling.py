from struct import *
import numpy as np
# I considered using multiprocessing package, but I find this code version is fine.
# Welcome for your version with multiprocessing to make the reading faster.
# from joblib import Parallel, delayed
import multiprocessing
import time
from scipy import misc
import os
import argparse
from progressbar import ProgressBar
from skimage.measure import block_reduce
from skimage.transform import resize


def voxel_grid_padding(a):
    x_d = a.shape[0]
    y_d = a.shape[1]
    z_d = a.shape[2]
    ori_vox_res = 256
    size = [ori_vox_res, ori_vox_res, ori_vox_res]
    b = np.zeros(size, dtype=np.float32)

    bx_s = 0
    bx_e = size[0]
    by_s = 0
    by_e = size[1]
    bz_s = 0
    bz_e = size[2]
    ax_s = 0
    ax_e = x_d
    ay_s = 0
    ay_e = y_d
    az_s = 0
    az_e = z_d
    if x_d > size[0]:
        ax_s = int((x_d - size[0]) / 2)
        ax_e = int((x_d - size[0]) / 2) + size[0]
    else:
        bx_s = int((size[0] - x_d) / 2)
        bx_e = int((size[0] - x_d) / 2) + x_d

    if y_d > size[1]:
        ay_s = int((y_d - size[1]) / 2)
        ay_e = int((y_d - size[1]) / 2) + size[1]
    else:
        by_s = int((size[1] - y_d) / 2)
        by_e = int((size[1] - y_d) / 2) + y_d

    if z_d > size[2]:
        az_s = int((z_d - size[2]) / 2)
        az_e = int((z_d - size[2]) / 2) + size[2]
    else:
        bz_s = int((size[2] - z_d) / 2)
        bz_e = int((size[2] - z_d) / 2) + z_d
    b[bx_s:bx_e, by_s:by_e, bz_s:bz_e] = a[ax_s:ax_e, ay_s:ay_e, az_s:az_e]

    return b


def vox_down_single(vox, to_res):
    from_res = vox.shape[0]
    step = int(from_res / to_res)
    vox = np.reshape(vox, [from_res, from_res, from_res])
    new_vox = block_reduce(vox, (step, step, step), func=np.max)
    return new_vox


def rescale_npy(file_np, dir_tar_voxel, category):
    ans = np.load(file_np)
    """
    vox = resize(
        np.squeeze(ans['arr_0']), (64, 64, 64),
        preserve_range=True,
        mode='constant')
    """
    vox = np.squeeze(ans['arr_0'])
    vox = voxel_grid_padding(vox)
    vox = vox_down_single(vox, to_res=64)
    vox[vox > 0] = category
    name_start = int(file_np.rfind('/'))
    name_end = int(file_np.find('.', name_start))
    np.save(dir_tar_voxel + file_np[name_start:name_end] + '.npy', vox)


class ScanFile(object):
    def __init__(self, directory, prefix=None, postfix='.bin'):
        self.directory = directory
        self.prefix = prefix
        self.postfix = postfix

    def scan_files(self):
        files_list = []

        for dirpath, dirnames, filenames in os.walk(self.directory):
            for special_file in filenames:
                if self.postfix:
                    if special_file.endswith(self.postfix):
                        files_list.append(os.path.join(dirpath, special_file))
                elif self.prefix:
                    if special_file.startswith(self.prefix):
                        files_list.append(os.path.join(dirpath, special_file))
                else:
                    files_list.append(os.path.join(dirpath, special_file))

        return files_list

    def scan_subdir(self):
        subdir_list = []
        for dirpath, dirnames, files in os.walk(self.directory):
            subdir_list.append(dirpath)
        return subdir_list


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Parser added')
    parser.add_argument(
        '-s',
        action="store",
        dest="dir_src",
        default="/media/wangyida/D0-P1/database/SUNCGtrain_3001_5000",
        help='folder of paired depth and voxel')
    parser.add_argument(
        '-tv',
        action="store",
        dest="dir_tar_voxel",
        default="/media/wangyida/D0-P1/database/SUNCGtrain_3001_5000_depvox",
        help='for storing generated npy')
    parser.add_argument(
        '-c',
        action="store",
        type=int,
        dest="category",
        default="1",
        help='number of classes for rendering cubics')
    parser.print_help()
    results = parser.parse_args()

    # folder of paired depth and voxel
    dir_src = results.dir_src
    # for storing generated npy
    dir_tar_voxel = results.dir_tar_voxel
    category = results.category

    # scan for depth files
    scan_npz = ScanFile(directory=dir_src, postfix='.npz')
    files_npz = scan_npz.scan_files()

    # making directories
    try:
        os.stat(dir_tar_voxel)
    except:
        os.mkdir(dir_tar_voxel)

    # save voxel as npy files
    pbar = ProgressBar()
    # parallel processing for samples
    from joblib import Parallel, delayed
    import multiprocessing
    num_cores = multiprocessing.cpu_count()
    Parallel(n_jobs=num_cores)(
        delayed(rescale_npy)(file_np, dir_tar_voxel, category)
        for file_np in pbar(files_npz))
    # below is the normal procedure for processing
    """
    for file_np in pbar(files_npz):
        rescale_npy(file_np, dir_tar_voxel, category)
    """
