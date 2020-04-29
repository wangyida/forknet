from easydict import EasyDict as edict
__C = edict()
cfg = __C

#
# Common
#
__C.TYPE_TASK = 'scene'
__C.TYPE_EVAL = 'synthetic'
__C.TYPE_DATA = 'test'

__C.SUB_CONFIG_FILE = []

__C.CONST = edict()
__C.CONST.BATCH_SIZE = 8
__C.CONST.BATCH_SIZE_TEST = 5
__C.SAVER_MAX = 20000
__C.CHECK_FREQ = 10000
__C.RECORD_VOX_NUM = 10
__C.SWITCHING_ITE = 100001

# Network
__C.NET = edict()
__C.NET.DIM_Z = 64
# The last dimension of NET.DIM matters much for GPU consumption for loss function
__C.NET.KERNEL = [[4, 4, 4, 4, 4], [4, 4, 4, 4, 4], [4, 4, 4, 4, 4]]
__C.NET.STRIDE = [1, 2, 2, 2, 1]
__C.NET.DILATIONS = [1, 1, 1, 1, 1]

if __C.TYPE_TASK is 'scene':
    __C.CONST.N_VOX = [80, 48, 80]
    __C.NET.START_VOX = [5, 3, 5]
    __C.NET.DIM = [128, 64, 32, 16, 12]
elif __C.TYPE_TASK is 'object':
    __C.CONST.N_VOX = [64, 64, 64]
    __C.NET.START_VOX = [4, 4, 4]
    __C.NET.DIM = [512, 256, 128, 32, 5]

#
# Directories
#
__C.DIR = edict()
# Path where taxonomy.json is stored
path_ssd = '/media/wangyida/SSD2T/database/'
path_hdd = '/media/wangyida/HDD/database/'
if __C.TYPE_TASK is 'scene':
    if __C.TYPE_EVAL == 'real':
        __C.DIR.CHECK_POINT_PATH = '/media/wangyida/HDD/models/depvox-gan-scene-r'
        if __C.TYPE_DATA == 'test':
            __C.DIR.VOXEL_PATH = path_hdd + 'NYU_Yida/test/voxel_semantic_npy/'
            __C.DIR.SURF_PATH = path_hdd + 'NYU_Yida/test/surface_semantic_npy/'
            __C.DIR.TSDF_PATH = path_hdd + 'NYU_Yida/test/depth_tsdf_camera_npy/'
        elif __C.TYPE_DATA == 'train':
            __C.DIR.VOXEL_PATH = path_ssd + 'NYU_Yida/train/voxel_semantic_npy/'
            __C.DIR.SURF_PATH = path_ssd + 'NYU_Yida/train/surface_semantic_npy/'
            __C.DIR.TSDF_PATH = path_ssd + 'NYU_Yida/train/depth_tsdf_camera_npy/'
    elif __C.TYPE_EVAL == 'synthetic':
        __C.DIR.CHECK_POINT_PATH = '/media/wangyida/HDD/models/depvox-gan-scene-s'
        if __C.TYPE_DATA == 'test':
            __C.DIR.VOXEL_PATH = path_ssd + 'SUNCG_Yida/test/voxel_semantic_npy/'
            __C.DIR.SURF_PATH = path_ssd + 'SUNCG_Yida/test/surface_semantic_npy/'
            __C.DIR.TSDF_PATH = path_ssd + 'SUNCG_Yida/test/depth_tsdf_camera_npy/'
        elif __C.TYPE_DATA == 'train':
            __C.DIR.VOXEL_PATH = path_ssd + 'SUNCG_Yida/train/voxel_semantic_npy/'
            __C.DIR.SURF_PATH = path_ssd + 'SUNCG_Yida/train/surface_semantic_npy/'
            __C.DIR.TSDF_PATH = path_ssd + 'SUNCG_Yida/train/depth_tsdf_camera_npy/'
elif __C.TYPE_TASK is 'object':
    if __C.TYPE_EVAL == 'real':
        __C.DIR.CHECK_POINT_PATH = '/media/wangyida/HDD/models/depvox-gan-object-r'
        if __C.TYPE_DATA == 'test':
            __C.DIR.VOXEL_PATH = path_hdd + 'RecGAN_Yida/test/voxel_semantic_npy/'
            __C.DIR.TSDF_PATH = path_hdd + 'RecGAN_Yida/test/depth_tsdf_npy/'
        elif __C.TYPE_DATA == 'train':
            __C.DIR.VOXEL_PATH = path_hdd + 'RecGAN_Yida/train/voxel_semantic_npy/'
            __C.DIR.TSDF_PATH = path_hdd + 'RecGAN_Yida/train/depth_tsdf_npy/'
    elif __C.TYPE_EVAL == 'synthetic':
        __C.DIR.CHECK_POINT_PATH = '/media/wangyida/HDD/models/depvox-gan-object-s'
        if __C.TYPE_DATA == 'test':
            __C.DIR.VOXEL_PATH = path_hdd + 'Shapenet_Yida/test/voxel_semantic_npy/'
            __C.DIR.TSDF_PATH = path_hdd + 'Shapenet_Yida/test/depth_tsdf_npy/'
        elif __C.TYPE_DATA == 'train':
            __C.DIR.VOXEL_PATH = path_hdd + 'Shapenet_Yida/train/voxel_semantic_npy/'
            __C.DIR.TSDF_PATH = path_hdd + 'Shapenet_Yida/train/depth_tsdf_npy/'
    elif __C.TYPE_EVAL == 'register':
        __C.DIR.CHECK_POINT_PATH = '/media/wangyida/HDD/models/depvox-gan-object-s'
        __C.DIR.VOXEL_PATH = './data/object_registration/voxel_semantic_npy/'
        __C.DIR.TSDF_PATH = './data/object_registration/depth_tsdf_npy/'

__C.DIR.TRAIN_OBJ_PATH = './train_vox'
__C.DIR.EVAL_PATH = './eval'
__C.DIR.LOG_PATH = './log'

#
# Training
#
__C.TRAIN = edict()

__C.TRAIN.DATASET_PORTION = [0, 1.0]
__C.TRAIN.NUM_EPOCH = 50000  # maximum number of training epochs

# Learning
__C.LEARNING_RATE_G = 0.0001
__C.LEARNING_RATE_D = 0.0001
__C.LEARNING_RATE_V = [0.001, 1000, 0.0001]
__C.TRAIN.ADAM_BETA_G = 0.5
__C.TRAIN.ADAM_BETA_D = 0.5
__C.LAMDA_RECONS = 1
__C.LAMDA_GAMMA = 0.97


def cfg_from_file(filename):
    """Load a config file and merge it into the default options."""
    import yaml
    with open(filename, 'r') as f:
        yaml_cfg = edict(yaml.load(f))

    _merge_a_into_b(yaml_cfg, __C)


def cfg_from_list(cfg_list):
    """Set config keys via list (e.g., from command line)."""
    from ast import literal_eval
    assert len(cfg_list) % 2 == 0
    for k, v in zip(cfg_list[0::2], cfg_list[1::2]):
        key_list = k.split('.')
        d = __C
        for subkey in key_list[:-1]:
            assert subkey in d.keys()
            d = d[subkey]
        subkey = key_list[-1]
        assert subkey in d.keys()
        try:
            value = literal_eval(v)
        except:
            # handle the case when v is a string literal
            value = v
        assert type(value) == type(d[subkey]), \
            'type {} does not match original type {}'.format(
            type(value), type(d[subkey]))
        d[subkey] = value
