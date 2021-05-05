# Author: Wentao Yuan (wyuan1@cs.cmu.edu) 05/31/2018

import numpy as np
import open3d as o3d
import matplotlib


def read_pcd(filename):
    o3d.set_verbosity_level(VerbosityLevel.Error)
    pcd = o3d.read_point_cloud(filename)
    if pcd.colors:
        return np.concatenate([np.array(pcd.points), np.array(pcd.colors)], 1)
    else:
        colors = matplotlib.cm.cool(np.array(pcd.points)[:, 0])
        return np.concatenate([np.array(pcd.points), colors[:, 0:3]], 1)


def save_pcd(filename, points):
    # o3d.set_verbosity_level(VerbosityLevel.Error)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points[:, 0:3])
    pcd.colors = o3d.utility.Vector3dVector(points[:, 3:6])
    o3d.io.write_point_cloud(filename, pcd, write_ascii=True, compressed=True)
