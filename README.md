# ForkNet: Multi-branch Volumetric Semantic Completion from a Single Depth Image

The implementation of our paper accepted in **ICCV** *2019* (*International Conference on Computer Vision*, IEEE)
**[Yida Wang](https://wangyida.github.io/#about), David Tan, [Nassir Navab](http://campar.in.tum.de/Main/NassirNavab) and [Federico Tombari](http://campar.in.tum.de/Main/FedericoTombari)**

## ForkNet

![](iccv/teaser.png)

 <img src="iccv/PrasentationICCV.gif" alt="road condition" frameborder="0" style="border:0" >

### Architecture
![](iccv/architecture.png)
training.

### Generated synthetic samples
![](iccv/learning_dataset.png)

### More examples
![](iccv/qualitative.png)

## Data preprocessing

### Depth image to TSDF volumes
Firstly you need to go to depth-tsdf folder to compile the our depth converter. Then *camake* and *make* are suggested tools to compile our codes.

```shell
cmake . # configure
make # compiles demo executable
```
After the file named with **back-project** is compiled, depth images of NYU or SUNCG datasets could be converted into TSDF volumes parallelly.

```shell
CUDA_VISIBLE_DEVICES=0 python2 data/depth_backproject.py -s /media/wangyida/SSD2T/database/SUNCG_Yida/train/depth_real_png -tv /media/wangyida/HDD/database/SUNCG_Yida/train/depth_tsdf_camera_npy -tp /media/wangyida/HDD/database/SUNCG_Yida/train/depth_tsdf_pcd
```

### Semantic volumes used for training
We further convert the binary files from SUNCG and NYU datasets into numpy arrays in dimension of 80*48*80 with 12 semantic channels. Those voxel data are used as training ground truth.

```shell
python2 data/depthbin2npy.py -s /media/wangyida/HDD/database/SUNCGtrain_1001_2000  -tv /media/wangyida/HDD/database/SUNCG_Yida/train/voxel_semantic_npy &
python2 data/depthbin2npy.py -s /media/wangyida/HDD/database/SUNCGtrain_501_1000  -tv /media/wangyida/HDD/database/SUNCG_Yida/train/voxel_semantic_npy &
python2 data/depthbin2npy.py -s /media/wangyida/HDD/database/SUNCGtrain_1_1000  -tv /media/wangyida/HDD/database/SUNCG_Yida/train/voxel_semantic_npy &
python2 data/depthbin2npy.py -s /media/wangyida/HDD/database/SUNCGtrain_1001_3000  -tv /media/wangyida/HDD/database/SUNCG_Yida/train/voxel_semantic_npy &
python2 data/depthbin2npy.py -s /media/wangyida/HDD/database/SUNCGtrain_3001_5000  -tv /media/wangyida/HDD/database/SUNCG_Yida/train/voxel_semantic_npy &
python2 data/depthbin2npy.py -s /media/wangyida/HDD/database/SUNCGtrain_1_500  -tv /media/wangyida/HDD/database/SUNCG_Yida/train/voxel_semantic_npy &
python2 data/depthbin2npy.py -s /media/wangyida/HDD/database/SUNCGtrain_5001_7000  -tv /media/wangyida/HDD/database/SUNCG_Yida/train/voxel_semantic_npy &
python2 data/depthbin2npy.py -s /media/wangyida/HDD/database/depthbin_NYU_SUNCG/SUNCGtest_49700_49884 -tv /media/wangyida/HDD/database/SUNCG_Yida/test/voxel_semantic_npy &
python2 data/depthbin2npy.py -s /media/wangyida/HDD/database/depthbin_NYU_SUNCG/NYUtrain -tv /media/wangyida/HDD/database/NYU_Yida/train/voxel_semantic_npy &
python2 data/depthbin2npy.py -s /media/wangyida/HDD/database/depthbin_NYU_SUNCG/NYUtest -tv /media/wangyida/HDD/database/NYU_Yida/test/voxel_semantic_npy &
wait
```

## Train and Test
Then you can start to train with
```shell
CUDA_VISIBLE_DEVICES=0 python3 main.py --mode train --discriminative True
```
and test with
```shell
CUDA_VISIBLE_DEVICES=1 python main.py --mode evaluate_recons --conf_epoch 0
```
