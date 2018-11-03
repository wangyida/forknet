import os
import matplotlib as mpl
if os.environ.get('DISPLAY', '') == '':
    print('no display found. Using non-interactive Agg backend')
    mpl.use('Agg')
import matplotlib.pyplot as plt
import os
import numpy as np
from matplotlib import cm
from skimage.transform import resize
import argparse
from progressbar import ProgressBar


class ScanFile(object):
    def __init__(self, directory, prefix=None, postfix='.jpg'):
        self.directory = directory
        self.prefix = prefix
        self.postfix = postfix

    def scan_files(self):
        files_list = []

        for dirpath, dirnames, filenames in os.walk(self.directory):
            '''''
            dirpath is a string, the path to the directory.
            dirnames is a list of the names of the subdirectories in dirpath
            (excluding '.' and '..').
            filenames is a list of the names of the non-directory files
            in dirpath.
            '''
            for special_file in filenames:
                if self.postfix:
                    special_file.endswith(self.postfix)
                    files_list.append(os.path.join(dirpath, special_file))
                elif self.prefix:
                    special_file.startswith(self.prefix)
                    files_list.append(os.path.join(dirpath, special_file))
                else:
                    files_list.append(os.path.join(dirpath, special_file))

        return files_list

    def scan_subdir(self):
        subdir_list = []
        for dirpath, dirnames, files in os.walk(self.directory):
            subdir_list.append(dirpath)
        return subdir_list


class ScanFile(object):
    def __init__(self, directory, prefix=None, postfix='.jpg'):
        self.directory = directory
        self.prefix = prefix
        self.postfix = postfix

    def scan_files(self):
        files_list = []

        for dirpath, dirnames, filenames in os.walk(self.directory):
            '''''
            dirpath is a string, the path to the directory.
            dirnames is a list of the names of the subdirectories in dirpath
            (excluding '.' and '..').
            filenames is a list of the names of the non-directory files
            in dirpath.
            '''
            for special_file in filenames:
                if self.postfix:
                    special_file.endswith(self.postfix)
                    files_list.append(os.path.join(dirpath, special_file))
                elif self.prefix:
                    special_file.startswith(self.prefix)
                    files_list.append(os.path.join(dirpath, special_file))
                else:
                    files_list.append(os.path.join(dirpath, special_file))

        return files_list

    def scan_subdir(self):
        subdir_list = []
        for dirpath, dirnames, files in os.walk(self.directory):
            subdir_list.append(dirpath)
        return subdir_list


def normalize(arr):
    arr_min = np.min(arr)
    return (arr - arr_min) / (np.max(arr) - arr_min)


def show_histogram(values):
    n, bins, patches = plt.hist(values.reshape(-1), 50, normed=1)
    bin_centers = 0.5 * (bins[:-1] + bins[1:])

    for c, p in zip(normalize(bin_centers), patches):
        plt.setp(p, 'facecolor', cm.hsv(c))

    plt.show()


def explode(data):
    shape_arr = np.array(data.shape)
    size = shape_arr[:3] * 2 - 1
    exploded = np.zeros(
        np.concatenate([size, shape_arr[3:]]), dtype=data.dtype)
    exploded[::2, ::2, ::2] = data
    return exploded


def expand_coordinates(indices):
    x, y, z = indices
    x[1::2, :, :] += 0.4
    y[:, 1::2, :] += 0.4
    z[:, :, 1::2] += 0.4
    return x, y, z


def scale_by(arr, fac):
    mean = np.mean(arr)
    return (arr - mean) * fac + mean


def plot_image(arr, name='depth.png'):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    # ax.set_axis_off()
    arr = (arr - np.min(arr)) / (np.max(arr) - np.min(arr)) * 255
    arr = np.uint8(arr)
    ax.set_axis_off()
    # ax.set_aspect('equal')

    plt.imshow(arr, cmap="hot")
    plt.savefig(name, bbox_inches='tight', pad_inches=0, transparent=True)
    plt.close(fig)


def plot_cube(cube, name='voxel', angle=20, IMG_DIM=80, num_class=12):
    from mpl_toolkits.mplot3d import Axes3D

    # cube = normalize(cube)
    # Note that cm.Paired has 12 colors and Set2 has 8 colors
    # cube[np.where(cube > num_class)] = 10
    if num_class == 12:
        cube[cube == -1] = 0
        facecolors = cm.Paired((np.round(cube) / 13))
        facecolors[:, :, :, -1] = 0.2 * np.tanh(
            cube * 1000) + 0.1 * (cube > 5) + 0.1 * (cube == 2) 
    elif num_class == 4:
        facecolors = cm.Dark2((np.round(cube) / 9))
        facecolors[:, :, :, -1] = 0.4 * np.tanh(cube * 1000)
    elif num_class == 3:
        # cube[cube == -1] = 3
        cube[cube < 0] = 0
        facecolors = cm.Set2((np.round(cube) / 9))
        facecolors[:, :, :, -1] = 0.03 * np.tanh(
            cube * 1000) + 0.6 * (cube == 1)
    elif num_class == 1:
        cube[cube < 0] = 0
        facecolors = cm.Dark2((np.round(cube) / 1))
        facecolors[:, :, :, -1] =  0.5 * (cube == 1)

    # make the alpha channel more similar to each others while 0 is still 0
    facecolors = explode(facecolors)
    filled = facecolors[:, :, :, -1] != 0

    x, y, z = expand_coordinates(np.indices(np.array(filled.shape) + 1).astype(float))

    # Here is a loop for generating demo files
    for idx, val in enumerate(np.arange(160, 170, 10)):
        fig = plt.figure(figsize=(45 / 2.54, 45 / 2.54))  # , dpi=150)
        # plot
        ax1 = fig.add_subplot(111, projection='3d')
        # For samples in SUNCG, 20, -40 is a good choice for visualization
        # ax1.view_init(np.abs(90-val/2), val)
        ax1.view_init(angle, val)
        ax1.set_xlim(right=IMG_DIM * 2)
        ax1.set_ylim(top=IMG_DIM * 2)
        ax1.set_zlim(top=IMG_DIM * 2)
        ax1.set_axis_off()
        ax1.voxels(
            x,
            y,
            z,
            filled,
            facecolors=facecolors,
            edgecolors=np.clip(2 * facecolors - 0.5, 0, 1),
            linewidth=0.5)

        # plt.show()
        plt.savefig(
            name + '_' + format(idx, '03d') + '.png',
            bbox_inches='tight',
            pad_inches=0,
            transparent=True)
        plt.close(fig)

    """
    objects_name = ['empty', 'ceiling', 'floor', 'wall', 'window', 'door', 'chair', 'bed', 'sofa', 'table', 'tvs', 'furnture', 'object']
    for x in range(1, 11):
        fig = plt.figure(figsize=(30/2.54, 30/2.54))
        filled = explode(cube) == x
        ax1 = fig.add_subplot(111, projection='3d')
        ax1.view_init(20, angle)
        ax1.set_xlim(right=IMG_DIM*2)
        ax1.set_ylim(top=IMG_DIM*2)
        ax1.set_zlim(top=48*2)
        ax1.set_title(objects_name[x])
        ax1.set_axis_off()
        ax1.voxels(x, y, z, filled, facecolors=facecolors)
        # plt.show()
        plt.savefig(name.replace('.png', '_'+objects_name[x]+'.png'), bbox_inches='tight', pad_inches=0, transparent=True)
        plt.close(fig)
    """


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Parser added')
    parser.add_argument(
        '-d',
        action="store",
        dest="dir_dep",
        default="./SUNCGtrain_3001_5000",
        help='npy file for depth')
    parser.add_argument(
        '-v',
        action="store",
        dest="dir_vox",
        default="./SUNCGtrain_3001_5000",
        help='npy file for voxel')
    parser.add_argument(
        '-t',
        action="store",
        dest="target_folder",
        default="./target_folder",
        help='target folder for vis')
    parser.add_argument(
        '-n',
        action="store",
        type=int,
        dest="num_class",
        default="12",
        help='number of classes for rendering cubics')
    parser.print_help()
    results = parser.parse_args()

    dir_dep = results.dir_dep
    dir_vox = results.dir_vox
    target_folder = results.target_folder
    num_class = results.num_class
    scan = ScanFile(dir_dep)
    subdirs = scan.scan_subdir()
    files = scan.scan_files()
    try:
        os.stat(target_folder)
    except:
        os.mkdir(target_folder)

    # vis for 3D FGAN
    arr = np.load(results.dir_vox)
    # arr = np.expand_dims(arr, axis=0)
    pbar = ProgressBar()
    # parallel processing for samples
    from joblib import Parallel, delayed
    import multiprocessing
    num_cores = multiprocessing.cpu_count()
    sample_range = np.arange(0, arr.shape[0])
    """
    Parallel(n_jobs=num_cores)(delayed(plot_cube)(
        np.flip(np.rollaxis(np.squeeze(arr[idx, :, :, :]), 2, 0), 1),
        name=target_folder + '/' + format(idx, '03d'),
        num_class=num_class) for idx in pbar(sample_range))
    """
    # resized = resize(resized, (48, 80, 80), mode='constant')
    for idx in pbar(sample_range):
        plot_cube(
        np.flip(np.rollaxis(np.squeeze(arr[idx, :, :, :]), 2, 0), 1),
        name=target_folder + '/' + format(idx, '03d'),
        num_class=num_class)

