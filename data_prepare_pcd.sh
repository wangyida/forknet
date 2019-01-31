python2 data/voxel2pcd.py -s /media/wangyida/D0-P1/database/SUNCG_Yida/train/voxel_semantic_npy -tv /media/wangyida/D0-P1/database/SUNCG_Yida/train/pcd_complete -dt complete & 
python2 data/voxel2pcd.py -s /media/wangyida/D0-P1/database/SUNCG_Yida/test/voxel_semantic_npy -tv /media/wangyida/D0-P1/database/SUNCG_Yida/test/pcd_complete -dt complete &
python2 data/voxel2pcd.py -s /media/wangyida/D0-P1/database/NYU_Yida/train/voxel_semantic_npy -tv /media/wangyida/D0-P1/database/NYU_Yida/train/pcd_complete -dt complete &
python2 data/voxel2pcd.py -s /media/wangyida/D0-P1/database/NYU_Yida/test/voxel_semantic_npy -tv /media/wangyida/D0-P1/database/NYU_Yida/test/pcd_complete -dt complete &
python2 data/voxel2pcd.py -s /media/wangyida/D0-P1/database/SUNCG_Yida/train/depth_tsdf_camera_npy -tv /media/wangyida/D0-P1/database/SUNCG_Yida/train/pcd_partial -dt partial &
python2 data/voxel2pcd.py -s /media/wangyida/D0-P1/database/SUNCG_Yida/test/depth_tsdf_camera_npy -tv /media/wangyida/D0-P1/database/SUNCG_Yida/test/pcd_partial -dt partial &
python2 data/voxel2pcd.py -s /media/wangyida/D0-P1/database/NYU_Yida/train/depth_tsdf_camera_npy -tv /media/wangyida/D0-P1/database/NYU_Yida/train/pcd_partial -dt partial &
python2 data/voxel2pcd.py -s /media/wangyida/D0-P1/database/NYU_Yida/test/depth_tsdf_camera_npy -tv /media/wangyida/D0-P1/database/NYU_Yida/test/pcd_partial -dt partial &
wait
