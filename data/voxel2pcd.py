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


def voxel2pcd(npy_sec, dir_pcd_dep, dir_pcd_sec):
    npy_dep = npy_sec.replace('surface_semantic_npy', 'depth_tsdf_camera_npy')
    voxel_sec = np.load(npy_sec)
    voxel_dep = np.load(npy_dep)
    voxel_dep[voxel_dep < -1] = 0
    # voxel_dep = np.round(np.abs(voxel_dep))
    voxel_dep = np.round(np.abs(voxel_dep)) * voxel_sec
    pcd = PointCloud()

    idx_sec = np.where(voxel_sec > 0)
    coor_sec = np.float32(np.transpose(idx_sec))/80 - 0.5
    pcd.points = Vector3dVector(coor_sec)
    colo_cat_sec = np.float32(np.transpose(np.tile(voxel_sec[idx_sec], (3, 1))))/11
    pcd.colors = Vector3dVector(colo_cat_sec)

    # Save
    name_start = int(npy_sec.rfind('/'))
    name_end = int(npy_sec.find('.', name_start))
    write_point_cloud(dir_pcd_sec + npy_sec[name_start:name_end] + '.pcd', pcd)

    idx_dep = np.where(voxel_dep > 0)
    coor_dep = np.float32(np.transpose(idx_dep))/80 - 0.5
    pcd.points = Vector3dVector(coor_dep)
    colo_cat_dep = np.float32(np.transpose(np.tile(voxel_dep[idx_dep], (3, 1))))
    pcd.colors = Vector3dVector(colo_cat_dep)

    # Save
    name_start = int(npy_dep.rfind('/'))
    name_end = int(npy_dep.find('.', name_start))
    write_point_cloud(dir_pcd_dep + npy_dep[name_start:name_end] + '.pcd', pcd)

    """
    if np.array(np.where(voxel_sec > 4)).size * np.array(np.where(voxel_dep > 4)).size > 0:
        idx_sec = np.where((voxel_sec == 1) | (voxel_sec > 4))
        coor_sec = np.float32(np.transpose(idx_sec))/80 - 0.5
        pcd.points = Vector3dVector(coor_sec)
        colo_cat_sec = np.float32(np.transpose(np.tile(voxel_sec[idx_sec], (3, 1))))/11
        pcd.colors = Vector3dVector(colo_cat_sec)

        # Save
        name_start = int(npy_sec.rfind('/'))
        name_end = int(npy_sec.find('.', name_start))
        write_point_cloud(dir_pcd_sec + npy_sec[name_start:name_end] + '.pcd', pcd)

        idx_dep = np.where((voxel_dep == 1) | (voxel_dep > 4))
        coor_dep = np.float32(np.transpose(idx_dep))/80 - 0.5
        pcd.points = Vector3dVector(coor_dep)
        colo_cat_dep = np.float32(np.transpose(np.tile(voxel_dep[idx_dep], (3, 1))))
        pcd.colors = Vector3dVector(colo_cat_dep)

        # Save
        name_start = int(npy_dep.rfind('/'))
        name_end = int(npy_dep.find('.', name_start))
        write_point_cloud(dir_pcd_dep + npy_dep[name_start:name_end] + '.pcd', pcd)
    """

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
        '-tv_partial',
        action="store",
        dest="dir_tar_pcd_partial",
        default="/media/wangyida/D0-P1/database/SUNCGtrain_3001_5000_depvox",
        help='for storing generated npy')
    parser.print_help()
    results = parser.parse_args()

    # folder of paired depth and voxel
    dir_src = results.dir_src
    # for storing generated npy
    dir_tar_pcd = results.dir_tar_pcd
    dir_tar_pcd_partial = results.dir_tar_pcd_partial

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
    Parallel(n_jobs=num_cores)(delayed(voxel2pcd)(file_npy, dir_tar_pcd_partial, dir_tar_pcd)
                               for file_npy in pbar(files_npy))
    # below is the normal procedure for processing
    """
    for file_npy in pbar(files_npy):
        voxel2pcd(file_npy, dir_tar_pcd_partial, dir_tar_pcd)
    """
