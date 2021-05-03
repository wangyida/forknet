from easydict import EasyDict as edict
__C = edict()
cfg = __C

#
# Common
#
__C.dataset = '3rscan'
__C.type = 'synthetic'
__C.tvt = 'test'

__C.SUB_CONFIG_FILE = []

__C.CONST = edict()
__C.CONST.BATCH_SIZE = 4
__C.CONST.BATCH_SIZE_TEST = 1
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

if __C.dataset is 'scene':
    __C.CONST.N_VOX = [80, 48, 80]
    __C.NET.START_VOX = [5, 3, 5]
    __C.NET.DIM = [128, 64, 32, 16, 12]
elif __C.dataset is 'object':
    __C.CONST.N_VOX = [64, 64, 64]
    __C.NET.START_VOX = [4, 4, 4]
    __C.NET.DIM = [512, 256, 128, 32, 5]
elif __C.dataset is 'fusion':
    __C.CONST.N_VOX = [64, 64, 64]
    __C.NET.START_VOX = [4, 4, 4]
    __C.NET.DIM = [128, 64, 32, 16, 12]
elif __C.dataset is '3rscan':
    __C.CONST.N_VOX = [64, 64, 64]
    __C.NET.START_VOX = [4, 4, 4]
    __C.NET.DIM = [128, 64, 32, 16, 41]

#
# Directories
#
__C.DIR = edict()
# Path where taxonomy.json is stored
path_ssd = '/media/wangyida/SSD2T/database/'
path_hdd = '/media/wangyida/HDD/database/'
if __C.dataset is 'scene':
    if __C.type == 'real':
        __C.DIR.CHECK_POINT_PATH = '/media/wangyida/HDD/models/depvox-gan-scene-r'

        __C.DIR.VOXEL_PATH = path_hdd + 'NYU_Yida/' + __C.tvt + '/voxel_semantic_npy/'
        __C.DIR.SURF_PATH = path_hdd + 'NYU_Yida/' + __C.tvt + '/surface_semantic_npy/'
        __C.DIR.TSDF_PATH = path_hdd + 'NYU_Yida/' + __C.tvt + '/depth_tsdf_camera_npy/'
    elif __C.type == 'synthetic':
        __C.DIR.CHECK_POINT_PATH = '/media/wangyida/HDD/models/depvox-gan-scene-s'

        __C.DIR.VOXEL_PATH = path_ssd + 'SUNCG_Yida/' + __C.tvt + '/voxel_semantic_npy/'
        __C.DIR.SURF_PATH = path_ssd + 'SUNCG_Yida/' + __C.tvt + '/surface_semantic_npy/'
        __C.DIR.TSDF_PATH = path_ssd + 'SUNCG_Yida/' + __C.tvt + '/depth_tsdf_camera_npy/'
elif __C.dataset is 'object':
    if __C.type == 'real':
        __C.DIR.CHECK_POINT_PATH = '/media/wangyida/HDD/models/depvox-gan-object-r'

        __C.DIR.VOXEL_PATH = path_hdd + 'RecGAN_Yida/' + __C.tvt + '/voxel_semantic_npy/'
        __C.DIR.TSDF_PATH = path_hdd + 'RecGAN_Yida/' + __C.tvt + '/depth_tsdf_npy/'
    elif __C.type == 'synthetic':
        __C.DIR.CHECK_POINT_PATH = '/media/wangyida/HDD/models/depvox-gan-object-s'

        __C.DIR.VOXEL_PATH = path_hdd + 'Shapenet_Yida/' + __C.tvt + '/voxel_semantic_npy/'
        __C.DIR.TSDF_PATH = path_hdd + 'Shapenet_Yida/' + __C.tvt + '/depth_tsdf_npy/'
elif __C.dataset is 'fusion':
    __C.DIR.CHECK_POINT_PATH = '/media/wangyida/HDD/models/depvox-gan-fusion'

    __C.DIR.VOXEL_PATH = path_hdd + '050_200/' + __C.tvt + '/gt/'
    __C.DIR.SURF_PATH = path_hdd + '050_200/' + __C.tvt + '/gt/'
    __C.DIR.TSDF_PATH = path_hdd + '050_200/' + __C.tvt + '/train/'
elif __C.dataset is '3rscan':
    __C.DIR.CHECK_POINT_PATH = '/media/wangyida/HDD/models/depvox-gan-3rscan'

    __C.DIR.VOXEL_PATH = path_hdd + '3RSCAN/voxelized/' + __C.tvt + '/complete/'
    __C.DIR.SURF_PATH = path_hdd + '3RSCAN/voxelized/' + __C.tvt + '/complete/'
    __C.DIR.TSDF_PATH = path_hdd + '3RSCAN/voxelized/' + __C.tvt + '/partial/'

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
# LAmbda_gamma says that this voxel is xxxx, (1-lambda_gamma says that this voxel is not xxxx,xxxx,xxxx...)
__C.LAMDA_GAMMA = 0.7


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
