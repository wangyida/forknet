import os.path
from struct import *
from subprocess import call
import numpy as np
# I considered using multiprocessing package, but I find this code version is fine.
# Welcome for your version with multiprocessing to make the reading faster.
# from joblib import Parallel, delayed
import time
from scipy import misc
import os
import argparse
from progressbar import ProgressBar
from skimage.measure import block_reduce
from numba import autojit, prange


def bin2array(file):
    with open(file, 'r') as f:
        float_size = 4
        uint_size = 4
        total_count = 0
        vox = f.read()
        numC = len(vox) / float_size
        # print('numC is {}'.format(numC))
        checkVox = unpack('f' * numC, vox)
        # print('checkVox shape is {}'.format(len(checkVox)))
        checkVox = np.reshape(checkVox, (48, 80, 80))
        checkVox = np.swapaxes(checkVox, 0, 1)
        checkVox = np.swapaxes(checkVox, 0, 2)
        # checkVox = np.flip(checkVox, 0)
        # checkVox = np.where(checkVox < 1.0, 1, 0)
        # checkVox = block_reduce(checkVox, block_size=(3, 3, 3), func=np.max)
    f.close()
    # print "reading voxel file takes {} mins".format((time.time()-start_time)/60)
    return checkVox


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


def process_data(file_depth, dir_voxel, dir_ply):

    img_path = file_depth
    camera_intrinsic = "./depth-tsdf/data/camera-intrinsics.txt"
    camera_extrinsic = img_path.replace("depth_real_png", "camera")
    camera_extrinsic = camera_extrinsic.replace(".png", ".txt")
    camera_origin = camera_extrinsic.replace("camera", "origin")
    tsdf_bin_path = img_path.replace("depth_real_png", "depth_tsdf_bin")
    tsdf_bin_path = tsdf_bin_path.replace(".png", ".bin")
    if os.path.isfile(camera_origin):
        call([
            "./depth-tsdf/demo", camera_intrinsic, camera_origin,
            camera_extrinsic, img_path, tsdf_bin_path
        ])
        voxel = bin2array(file=tsdf_bin_path)
        name_start = int(img_path.rfind('/'))
        name_end = int(img_path.find('.', name_start))

        # save numpy
        np.save(dir_voxel + img_path[name_start:name_end] + '.npy', voxel)

        # save ply
        call([
            "cp", "./tsdf.ply",
            dir_ply + img_path[name_start:name_end] + '.ply'
        ])
        # call(["rm -rf", tsdf_bin_path])
    else:
        print(camera_origin, 'does not exist.')


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
        dest="dir_tar",
        default="/media/wangyida/D0-P1/database/SUNCGtrain_3001_5000_depvox",
        help='for storing generated npy')
    parser.add_argument(
        '-tp',
        action="store",
        dest="dir_ply",
        default="/media/wangyida/D0-P1/database/SUNCGtrain_3001_5000_depvox",
        help='for storing generated ply')
    parser.print_help()
    results = parser.parse_args()

    # folder of paired depth and voxel
    dir_src = results.dir_src
    # for storing generated npy
    dir_voxel = results.dir_tar
    dir_ply = results.dir_ply

    # scan for depth files
    scan_png = ScanFile(directory=dir_src, postfix='.png')
    files_png = scan_png.scan_files()

    # make directories
    try:
        os.stat(dir_voxel)
    except:
        os.mkdir(dir_voxel)

    try:
        os.stat(dir_ply)
    except:
        os.mkdir(dir_ply)

    # save voxel as npy files
    pbar = ProgressBar()
    # parallel processing for samples
    from joblib import Parallel, delayed
    import multiprocessing
    num_cores = multiprocessing.cpu_count()
    Parallel(n_jobs=num_cores)(
        delayed(process_data)(file_depth, dir_voxel, dir_ply)
        for file_depth in pbar(files_png))
    # below is the normal procedure for processing
    """
    for file_depth in pbar(files_png):
        process_data(file_depth, dir_voxel, dir_ply)
    """
