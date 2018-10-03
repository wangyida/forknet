python2 data/depthbin2npy.py -s /media/wangyida/D0-P1/database/SUNCGtrain_1001_2000 -td /media/wangyida/SSD2T/database/SUNCG_Yida/train/depth_npy -tv /media/wangyida/SSD2T/database/SUNCG_Yida/train/voxel_semantic_npy &
python2 data/depthbin2npy.py -s /media/wangyida/D0-P1/database/SUNCGtrain_501_1000 -td /media/wangyida/SSD2T/database/SUNCG_Yida/train/depth_npy -tv /media/wangyida/SSD2T/database/SUNCG_Yida/train/voxel_semantic_npy &
python2 data/depthbin2npy.py -s /media/wangyida/D0-P1/database/SUNCGtrain_1_1000 -td /media/wangyida/SSD2T/database/SUNCG_Yida/train/depth_npy -tv /media/wangyida/SSD2T/database/SUNCG_Yida/train/voxel_semantic_npy &
python2 data/depthbin2npy.py -s /media/wangyida/D0-P1/database/SUNCGtrain_1001_3000 -td /media/wangyida/SSD2T/database/SUNCG_Yida/train/depth_npy -tv /media/wangyida/SSD2T/database/SUNCG_Yida/train/voxel_semantic_npy &
python2 data/depthbin2npy.py -s /media/wangyida/D0-P1/database/SUNCGtrain_3001_5000 -td /media/wangyida/SSD2T/database/SUNCG_Yida/train/depth_npy -tv /media/wangyida/SSD2T/database/SUNCG_Yida/train/voxel_semantic_npy &
python2 data/depthbin2npy.py -s /media/wangyida/D0-P1/database/SUNCGtrain_1_500 -td /media/wangyida/SSD2T/database/SUNCG_Yida/train/depth_npy -tv /media/wangyida/SSD2T/database/SUNCG_Yida/train/voxel_semantic_npy &
python2 data/depthbin2npy.py -s /media/wangyida/D0-P1/database/SUNCGtrain_5001_7000 -td /media/wangyida/SSD2T/database/SUNCG_Yida/train/depth_npy -tv /media/wangyida/SSD2T/database/SUNCG_Yida/train/voxel_semantic_npy &
wait

python2 data/depthbin2npy.py -s /media/wangyida/D0-P1/database/depthbin_NYU_SUNCG/SUNCGtest_49700_49884 -td /media/wangyida/SSD2T/database/SUNCG_Yida/test/depth_npy -tv /media/wangyida/SSD2T/database/SUNCG_Yida/test/voxel_semantic_npy &
python2 data/bin2camera.py -s /media/wangyida/D0-P1/database/depthbin_NYU_SUNCG/NYUtrain -td /media/wangyida/SSD2T/database/NYU_Yida/train/depth_npy -tv /media/wangyida/SSD2T/database/NYU_Yida/train/voxel_semantic_npy &
python2 data/bin2camera.py -s /media/wangyida/D0-P1/database/depthbin_NYU_SUNCG/NYUtest -td /media/wangyida/SSD2T/database/NYU_Yida/test/depth_npy -tv /media/wangyida/SSD2T/database/NYU_Yida/test/voxel_semantic_npy &
wait
