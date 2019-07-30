from struct import *
import numpy as np
# I considered using multiprocessing package, but I find this code version is fine.
# Welcome for your version with multiprocessing to make the reading faster.
# from joblib import Parallel, delayed
import multiprocessing
import time
import os
import argparse
from open3d import *
from progressbar import ProgressBar

# from astropy.nddata.utils import block_reduce

# parallel processing for samples
from joblib import Parallel, delayed


def voxel2pcd(file_npy, dir_tar_pcd, type='partial'):
    voxels = np.load(file_npy)
    pcd = PointCloud()
    if type == 'partial':
        coordinate = np.transpose(np.where(voxels > 0.5))
        pcd.points = Vector3dVector(coordinate)
        # colors_cat = np.transpose(np.tile(voxels[voxels > 0.5], ( 3, 1)))
        colors_cat = np.ones_like(coordinate)
        pcd.colors = Vector3dVector(colors_cat)
    else:
        coordinate = np.transpose(np.where(voxels > 0))
        pcd.points = Vector3dVector(coordinate)
        colors_cat = np.float32(np.transpose(np.tile(voxels[voxels > 0], ( 3, 1))))
        pcd.colors = Vector3dVector(colors_cat)
    # Save
    name_start = int(file_npy.rfind('/'))
    name_end = int(file_npy.find('.', name_start))
    write_point_cloud(dir_tar_pcd + file_npy[name_start:name_end] + '.pcd', pcd)/11
    # write_point_cloud(dir_tar_pcd + file_npy[name_start:name_end] + '.ply', pcd)


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
        dest="dir_tar_pcd",
        default="/media/wangyida/D0-P1/database/SUNCGtrain_3001_5000_depvox",
        help='for storing generated npy')
    parser.add_argument(
        '-dt',
        action="store",
        dest="data_type",
        default='partial',
        help='for storing generated npy')
    parser.print_help()
    results = parser.parse_args()

    # folder of paired depth and voxel
    dir_src = results.dir_src
    # for storing generated npy
    dir_tar_pcd = results.dir_tar_pcd
    data_type = results.data_type

    # scan for voxel files
    scan_npy = ScanFile(directory=dir_src, postfix='.npy')
    files_npy = scan_npy.scan_files()

    # making directories
    try:
        os.stat(dir_tar_pcd)
    except:
        os.mkdir(dir_tar_pcd)

    # save voxel as npy files
    pbar = ProgressBar()
    num_cores = multiprocessing.cpu_count()
    Parallel(n_jobs=num_cores)(delayed(voxel2pcd)(file_npy, dir_tar_pcd, type=data_type)
                               for file_npy in pbar(files_npy))
    # below is the normal procedure for processing
    """
    for file_npy in pbar(files_npy):
        voxel2pcd(file_npy, dir_tar_pcd, type=data_type)
    """
