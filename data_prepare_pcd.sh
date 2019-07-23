python2 data/voxel2pcd.py -s /media/wangyida/SSD2T/database/SUNCG_Yida/train/surface_semantic_npy -tv /media/wangyida/HDD/database/SUNCG_Yida/train/pcd_complete -dt complete & 
python2 data/voxel2pcd.py -s /media/wangyida/SSD2T/database/SUNCG_Yida/test/surface_semantic_npy -tv /media/wangyida/HDD/database/SUNCG_Yida/test/pcd_complete -dt complete &
python2 data/voxel2pcd.py -s /media/wangyida/SSD2T/database/NYU_Yida/train/surface_semantic_npy -tv /media/wangyida/HDD/database/NYU_Yida/train/pcd_complete -dt complete &
python2 data/voxel2pcd.py -s /media/wangyida/SSD2T/database/NYU_Yida/test/surface_semantic_npy -tv /media/wangyida/HDD/database/NYU_Yida/test/pcd_complete -dt complete &
python2 data/voxel2pcd.py -s /media/wangyida/SSD2T/database/SUNCG_Yida/train/depth_tsdf_camera_npy -tv /media/wangyida/HDD/database/SUNCG_Yida/train/pcd_partial -dt partial &
python2 data/voxel2pcd.py -s /media/wangyida/SSD2T/database/SUNCG_Yida/test/depth_tsdf_camera_npy -tv /media/wangyida/HDD/database/SUNCG_Yida/test/pcd_partial -dt partial &
python2 data/voxel2pcd.py -s /media/wangyida/SSD2T/database/NYU_Yida/train/depth_tsdf_camera_npy -tv /media/wangyida/HDD/database/NYU_Yida/train/pcd_partial -dt partial &
python2 data/voxel2pcd.py -s /media/wangyida/SSD2T/database/NYU_Yida/test/depth_tsdf_camera_npy -tv /media/wangyida/HDD/database/NYU_Yida/test/pcd_partial -dt partial &
wait
