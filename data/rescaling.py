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
# from skimage.measure import block_reduce
from skimage.transform import resize


def rescale_npy(file_np, dir_tar_voxel, category):
    ans = np.load(file_np)
    vox = resize(
        np.squeeze(ans['arr_0']), (64, 64, 64),
        preserve_range=True,
        mode='constant')
    vox[vox >= 0.3] = category
    vox[vox < 0.3] = 0
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
