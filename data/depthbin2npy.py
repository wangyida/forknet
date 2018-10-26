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
from astropy.nddata.utils import block_reduce


def bin2array(file_bin, dir_tar_voxel):
    with open(file_bin, 'r') as f:
        float_size = 4
        uint_size = 4
        total_count = 0
        cor = f.read(float_size * 3)
        cors = unpack('fff', cor)
        # print("cors is {}",cors)
        cam = f.read(float_size * 16)
        cams = unpack('ffffffffffffffff', cam)
        # print("cams %16f",cams)
        vox = f.read()
        numC = len(vox) / uint_size
        # print('numC is {}'.format(numC))
        checkVoxValIter = unpack('I' * numC, vox)
        checkVoxVal = checkVoxValIter[0::2]
        checkVoxIter = checkVoxValIter[1::2]
        checkVox = [
            i for (val, repeat) in zip(checkVoxVal, checkVoxIter)
            for i in np.tile(val, repeat)
        ]
        checkVox = np.reshape(checkVox, (240, 144, 240))
        # Firstly convert 255 to 0
        checkVox[checkVox == 255] = -1
        checkVox = block_reduce(checkVox, block_size=(3, 3, 3), func=np.max)
        checkVox[checkVox > 12] = 12
        name_start = int(file_bin.rfind('/'))
        name_end = int(file_bin.find('.', name_start))
        np.save(dir_tar_voxel + file_bin[name_start:name_end] + '.npy', checkVox)
    f.close()


def png2array(file):
    image = misc.imread(file)
    image = misc.imresize(image, 50)
    return image


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
        '-td',
        action="store",
        dest="dir_tar_depth",
        default="/media/wangyida/D0-P1/database/SUNCGtrain_3001_5000_depvox",
        help='for storing generated npy')
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
    dir_tar_depth = results.dir_tar_depth
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
    try:
        os.stat(dir_tar_depth)
    except:
        os.mkdir(dir_tar_depth)
    """
    pbar1 = ProgressBar()
    # save depth as npy files
    for file_png in pbar1(files_png):
        depth = png2array(file=file_png)
        name_start = int(file_png.rfind('/'))
        name_end = int(file_png.find('.', name_start))
        np.save(dir_tar_depth + file_png[name_start:name_end] + '.npy', depth)
    """

    # save voxel as npy files
    pbar = ProgressBar()
    # parallel processing for samples
    from joblib import Parallel, delayed
    import multiprocessing
    num_cores = multiprocessing.cpu_count()
    Parallel(n_jobs=num_cores)(
        delayed(bin2array)(file_bin, dir_tar_voxel)
        for file_bin in pbar(files_bin))
    # below is the normal procedure for processing
    """
    for file_bin in pbar(files_bin):
        bin2array(file_bin=file_bin, dir_tar_voxel=dir_tar_voxel)
    """
