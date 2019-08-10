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
from skimage.util import view_as_blocks

# from astropy.nddata.utils import block_reduce

# parallel processing for samples
from joblib import Parallel, delayed
import multiprocessing


def label_assign(vox):
    vox = np.reshape(vox,
                     (np.shape(vox)[0], np.shape(vox)[1], np.shape(vox)[2],
                      np.shape(vox)[3] * np.shape(vox)[4] * np.shape(vox)[5]))
    u, indices = np.unique(vox, return_inverse=True)
    axis_there = 3
    label = u[np.argmax(
        np.apply_along_axis(np.bincount, axis_there,
                            indices.reshape(vox.shape), None,
                            np.max(indices) + 1),
        axis=axis_there)]
    return label


def bin2array(file_bin, dir_tar_voxel):
    with open(file_bin, 'r') as f:
        float_size = 4
        uint_size = 4
        total_count = 0
        cor = f.read(float_size * 3)
        cors = unpack('fff', cor)
        cam = f.read(float_size * 16)
        cams = unpack('ffffffffffffffff', cam)
        vox = f.read()
        numC = len(vox) / uint_size
        checkVoxValIter = unpack('I' * numC, vox)
        checkVoxVal = checkVoxValIter[0::2]
        checkVoxIter = checkVoxValIter[1::2]
        checkVox = [
            i for (val, repeat) in zip(checkVoxVal, checkVoxIter)
            for i in np.tile(val, repeat)
        ]

        # Down sampling according to maximum label
        vox_max = np.reshape(checkVox, (240, 144, 240))

        # convert 255 to -1
        vox_max[vox_max == 255] = -1
        vox_max[vox_max == 0] = 0
        vox_max[vox_max == 1] = 1
        vox_max[vox_max == 2] = 2
        vox_max[vox_max == 3] = 3
        vox_max[vox_max == 4] = 4
        vox_max[vox_max == 5] = 11
        vox_max[vox_max == 6] = 5
        vox_max[vox_max == 7] = 6
        vox_max[vox_max == 8] = 7
        vox_max[vox_max == 9] = 8
        vox_max[vox_max == 10] = 8
        vox_max[vox_max == 11] = 10
        vox_max[vox_max == 12] = 10
        vox_max[vox_max == 13] = 10
        vox_max[vox_max == 14] = 11
        vox_max[vox_max == 15] = 11
        vox_max[vox_max == 16] = 9
        vox_max[vox_max == 17] = 8
        vox_max[vox_max == 18] = 11
        vox_max[vox_max == 19] = 11
        vox_max[vox_max == 20] = 11
        vox_max[vox_max == 21] = 11
        vox_max[vox_max == 22] = 11
        vox_max[vox_max == 23] = 11
        vox_max[vox_max == 24] = 11
        vox_max[vox_max == 25] = 11
        vox_max[vox_max == 26] = 11
        vox_max[vox_max == 27] = 10
        vox_max[vox_max == 28] = 10
        vox_max[vox_max == 29] = 11
        vox_max[vox_max == 30] = 8
        vox_max[vox_max == 31] = 10
        vox_max[vox_max == 32] = 11
        vox_max[vox_max == 33] = 9
        vox_max[vox_max == 34] = 11
        vox_max[vox_max == 35] = 11
        vox_max[vox_max == 36] = 11

        vox_max = block_reduce(vox_max, block_size=(3, 3, 3), func=np.max)

        """
        # Down sampling according to common label
        vox_com = np.reshape(checkVox, (240, 144, 240))

        # mapping
        vox_com[vox_com == 255] = -1
        vox_com[vox_com == 0] = 0
        vox_com[vox_com == 1] = 1
        vox_com[vox_com == 2] = 2
        vox_com[vox_com == 3] = 3
        vox_com[vox_com == 4] = 4
        vox_com[vox_com == 5] = 11
        vox_com[vox_com == 6] = 5
        vox_com[vox_com == 7] = 6
        vox_com[vox_com == 8] = 7
        vox_com[vox_com == 9] = 8
        vox_com[vox_com == 10] = 8
        vox_com[vox_com == 11] = 10
        vox_com[vox_com == 12] = 10
        vox_com[vox_com == 13] = 10
        vox_com[vox_com == 14] = 11
        vox_com[vox_com == 15] = 11
        vox_com[vox_com == 16] = 9
        vox_com[vox_com == 17] = 8
        vox_com[vox_com == 18] = 11
        vox_com[vox_com == 19] = 11
        vox_com[vox_com == 20] = 11
        vox_com[vox_com == 21] = 11
        vox_com[vox_com == 22] = 11
        vox_com[vox_com == 23] = 11
        vox_com[vox_com == 24] = 11
        vox_com[vox_com == 25] = 11
        vox_com[vox_com == 26] = 11
        vox_com[vox_com == 27] = 10
        vox_com[vox_com == 28] = 10
        vox_com[vox_com == 29] = 11
        vox_com[vox_com == 30] = 8
        vox_com[vox_com == 31] = 10
        vox_com[vox_com == 32] = 11
        vox_com[vox_com == 33] = 9
        vox_com[vox_com == 34] = 11
        vox_com[vox_com == 35] = 11
        vox_com[vox_com == 36] = 11
        vox_com[vox_com == -1] = 255
        vox_com = label_assign(view_as_blocks(vox_com, (3, 3, 3)))
        vox_com[vox_com == 255] = -1

        # Merge 2 results
        locations = np.where((vox_max > 0) & (vox_com > 0))
        vox_max[locations] = vox_com[locations]
        """

        # Save
        name_start = int(file_bin.rfind('/'))
        name_end = int(file_bin.find('.', name_start))
        np.save(dir_tar_voxel + file_bin[name_start:name_end] + '.npy',
                vox_max)
    f.close()

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
    parser.print_help()
    results = parser.parse_args()

    # folder of paired depth and voxel
    dir_src = results.dir_src
    # for storing generated npy
    dir_tar_voxel = results.dir_tar_voxel

    # scan for depth files
    scan_png = ScanFile(directory=dir_src, postfix='.png')
    files_png = scan_png.scan_files()

    # scan for semantic voxel files
    scan_bin = ScanFile(directory=dir_src, postfix='.bin')
    files_bin = scan_bin.scan_files()

    # making directories
    try:
        os.stat(dir_tar_voxel)
    except:
        os.mkdir(dir_tar_voxel)

    # save voxel as npy files
    pbar = ProgressBar()
    """
    num_cores = multiprocessing.cpu_count()
    Parallel(n_jobs=num_cores)(delayed(bin2array)(file_bin, dir_tar_voxel)
                               for file_bin in pbar(files_bin))
    # below is the normal procedure for processing
    """
    for file_bin in pbar(files_bin):
        bin2array(file_bin=file_bin, dir_tar_voxel=dir_tar_voxel)
