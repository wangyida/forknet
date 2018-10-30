python2 data/rescaling.py -s /media/wangyida/D0-P1/database/data_3drecgan++/P1_02828884_bench/train_25d_vox256 -tv /media/wangyida/D0-P1/database/Shapenet_Yida/train/depth_tsdf_npy -c 1 &
python2 data/rescaling.py -s /media/wangyida/D0-P1/database/data_3drecgan++/P1_03001627_chair/train_25d_vox256  -tv /media/wangyida/D0-P1/database/Shapenet_Yida/train/depth_tsdf_npy -c 1 &
python2 data/rescaling.py -s /media/wangyida/D0-P1/database/data_3drecgan++/P1_04256520_coach/train_25d_vox256  -tv /media/wangyida/D0-P1/database/Shapenet_Yida/train/depth_tsdf_npy -c 1 &
python2 data/rescaling.py -s /media/wangyida/D0-P1/database/data_3drecgan++/P1_04379243_table/train_25d_vox256  -tv /media/wangyida/D0-P1/database/Shapenet_Yida/train/depth_tsdf_npy -c 1 &
# python2 data/rescaling.py -s /media/wangyida/D0-P1/database/data_3drecgan++/P2_02691156_airplane/train_25d_vox256 -tv /media/wangyida/D0-P1/database/Shapenet_Yida/train/depth_tsdf_npy -c 1 &
# python2 data/rescaling.py -s /media/wangyida/D0-P1/database/data_3drecgan++/P2_02958343_car/train_25d_vox256  -tv /media/wangyida/D0-P1/database/Shapenet_Yida/train/depth_tsdf_npy -c 1 &
# python2 data/rescaling.py -s /media/wangyida/D0-P1/database/data_3drecgan++/P2_03325088_faucet/train_25d_vox256  -tv /media/wangyida/D0-P1/database/Shapenet_Yida/train/depth_tsdf_npy -c 1 &
# python2 data/rescaling.py -s /media/wangyida/D0-P1/database/data_3drecgan++/P2_03467517_guitar/train_25d_vox256  -tv /media/wangyida/D0-P1/database/Shapenet_Yida/train/depth_tsdf_npy -c 1 &
# python2 data/rescaling.py -s /media/wangyida/D0-P1/database/data_3drecgan++/P2_04090263_gun/train_25d_vox256  -tv /media/wangyida/D0-P1/database/Shapenet_Yida/train/depth_tsdf_npy -c 1 &
# python2 data/rescaling.py -s /media/wangyida/D0-P1/database/data_3drecgan++/P2_03211117_monitor/train_25d_vox256  -tv /media/wangyida/D0-P1/database/Shapenet_Yida/train/depth_tsdf_npy -c 1 &
python2 data/rescaling.py -s /media/wangyida/D0-P1/database/data_3drecgan++/P1_02828884_bench/train_3d_vox256 -tv /media/wangyida/D0-P1/database/Shapenet_Yida/train/voxel_semantic_npy -c 1 &
python2 data/rescaling.py -s /media/wangyida/D0-P1/database/data_3drecgan++/P1_03001627_chair/train_3d_vox256  -tv /media/wangyida/D0-P1/database/Shapenet_Yida/train/voxel_semantic_npy -c 2 &
python2 data/rescaling.py -s /media/wangyida/D0-P1/database/data_3drecgan++/P1_04256520_coach/train_3d_vox256  -tv /media/wangyida/D0-P1/database/Shapenet_Yida/train/voxel_semantic_npy -c 3 &
python2 data/rescaling.py -s /media/wangyida/D0-P1/database/data_3drecgan++/P1_04379243_table/train_3d_vox256  -tv /media/wangyida/D0-P1/database/Shapenet_Yida/train/voxel_semantic_npy -c 4 &
# python2 data/rescaling.py -s /media/wangyida/D0-P1/database/data_3drecgan++/P2_02691156_airplane/train_3d_vox256 -tv /media/wangyida/D0-P1/database/Shapenet_Yida/train/voxel_semantic_npy -c 5 &
# python2 data/rescaling.py -s /media/wangyida/D0-P1/database/data_3drecgan++/P2_02958343_car/train_3d_vox256  -tv /media/wangyida/D0-P1/database/Shapenet_Yida/train/voxel_semantic_npy -c 6 &
# python2 data/rescaling.py -s /media/wangyida/D0-P1/database/data_3drecgan++/P2_03325088_faucet/train_3d_vox256  -tv /media/wangyida/D0-P1/database/Shapenet_Yida/train/voxel_semantic_npy -c 7 &
# python2 data/rescaling.py -s /media/wangyida/D0-P1/database/data_3drecgan++/P2_03467517_guitar/train_3d_vox256  -tv /media/wangyida/D0-P1/database/Shapenet_Yida/train/voxel_semantic_npy -c 8 &
# python2 data/rescaling.py -s /media/wangyida/D0-P1/database/data_3drecgan++/P2_04090263_gun/train_3d_vox256  -tv /media/wangyida/D0-P1/database/Shapenet_Yida/train/voxel_semantic_npy -c 9 &
# python2 data/rescaling.py -s /media/wangyida/D0-P1/database/data_3drecgan++/P2_03211117_monitor/train_3d_vox256  -tv /media/wangyida/D0-P1/database/Shapenet_Yida/train/voxel_semantic_npy -c 10 &
wait
python2 data/rescaling.py -s /media/wangyida/D0-P1/database/data_3drecgan++/P1_02828884_bench/test_25d_vox256 -tv /media/wangyida/D0-P1/database/Shapenet_Yida/test/depth_tsdf_npy -c 1 &
python2 data/rescaling.py -s /media/wangyida/D0-P1/database/data_3drecgan++/P1_03001627_chair/test_25d_vox256  -tv /media/wangyida/D0-P1/database/Shapenet_Yida/test/depth_tsdf_npy -c 1 &
python2 data/rescaling.py -s /media/wangyida/D0-P1/database/data_3drecgan++/P1_04256520_coach/test_25d_vox256  -tv /media/wangyida/D0-P1/database/Shapenet_Yida/test/depth_tsdf_npy -c 1 &
python2 data/rescaling.py -s /media/wangyida/D0-P1/database/data_3drecgan++/P1_04379243_table/test_25d_vox256  -tv /media/wangyida/D0-P1/database/Shapenet_Yida/test/depth_tsdf_npy -c 1 &
# python2 data/rescaling.py -s /media/wangyida/D0-P1/database/data_3drecgan++/P2_02691156_airplane/test_25d_vox256 -tv /media/wangyida/D0-P1/database/Shapenet_Yida/test/depth_tsdf_npy -c 1 &
# python2 data/rescaling.py -s /media/wangyida/D0-P1/database/data_3drecgan++/P2_02958343_car/test_25d_vox256  -tv /media/wangyida/D0-P1/database/Shapenet_Yida/test/depth_tsdf_npy -c 1 &
# python2 data/rescaling.py -s /media/wangyida/D0-P1/database/data_3drecgan++/P2_03325088_faucet/test_25d_vox256  -tv /media/wangyida/D0-P1/database/Shapenet_Yida/test/depth_tsdf_npy -c 1 &
# python2 data/rescaling.py -s /media/wangyida/D0-P1/database/data_3drecgan++/P2_03467517_guitar/test_25d_vox256  -tv /media/wangyida/D0-P1/database/Shapenet_Yida/test/depth_tsdf_npy -c 1 &
# python2 data/rescaling.py -s /media/wangyida/D0-P1/database/data_3drecgan++/P2_04090263_gun/test_25d_vox256  -tv /media/wangyida/D0-P1/database/Shapenet_Yida/test/depth_tsdf_npy -c 1 &
# python2 data/rescaling.py -s /media/wangyida/D0-P1/database/data_3drecgan++/P2_03211117_monitor/test_25d_vox256  -tv /media/wangyida/D0-P1/database/Shapenet_Yida/test/depth_tsdf_npy -c 1 &
python2 data/rescaling.py -s /media/wangyida/D0-P1/database/data_3drecgan++/P1_02828884_bench/test_3d_vox256 -tv /media/wangyida/D0-P1/database/Shapenet_Yida/test/voxel_semantic_npy -c 1 &
python2 data/rescaling.py -s /media/wangyida/D0-P1/database/data_3drecgan++/P1_03001627_chair/test_3d_vox256  -tv /media/wangyida/D0-P1/database/Shapenet_Yida/test/voxel_semantic_npy -c 2 &
python2 data/rescaling.py -s /media/wangyida/D0-P1/database/data_3drecgan++/P1_04256520_coach/test_3d_vox256  -tv /media/wangyida/D0-P1/database/Shapenet_Yida/test/voxel_semantic_npy -c 3 &
python2 data/rescaling.py -s /media/wangyida/D0-P1/database/data_3drecgan++/P1_04379243_table/test_3d_vox256  -tv /media/wangyida/D0-P1/database/Shapenet_Yida/test/voxel_semantic_npy -c 4 &
# python2 data/rescaling.py -s /media/wangyida/D0-P1/database/data_3drecgan++/P2_02691156_airplane/test_3d_vox256 -tv /media/wangyida/D0-P1/database/Shapenet_Yida/test/voxel_semantic_npy -c 5 &
# python2 data/rescaling.py -s /media/wangyida/D0-P1/database/data_3drecgan++/P2_02958343_car/test_3d_vox256  -tv /media/wangyida/D0-P1/database/Shapenet_Yida/test/voxel_semantic_npy -c 6 &
# python2 data/rescaling.py -s /media/wangyida/D0-P1/database/data_3drecgan++/P2_03325088_faucet/test_3d_vox256  -tv /media/wangyida/D0-P1/database/Shapenet_Yida/test/voxel_semantic_npy -c 7 &
# python2 data/rescaling.py -s /media/wangyida/D0-P1/database/data_3drecgan++/P2_03467517_guitar/test_3d_vox256  -tv /media/wangyida/D0-P1/database/Shapenet_Yida/test/voxel_semantic_npy -c 8 &
# python2 data/rescaling.py -s /media/wangyida/D0-P1/database/data_3drecgan++/P2_04090263_gun/test_3d_vox256  -tv /media/wangyida/D0-P1/database/Shapenet_Yida/test/voxel_semantic_npy -c 9 &
# python2 data/rescaling.py -s /media/wangyida/D0-P1/database/data_3drecgan++/P2_03211117_monitor/test_3d_vox256  -tv /media/wangyida/D0-P1/database/Shapenet_Yida/test/voxel_semantic_npy -c 10 &
wait
python2 data/rescaling.py -s /media/wangyida/D0-P1/database/data_3drecgan++/test/real_dataset/bench/real_50_25d_vox256  -tv /media/wangyida/D0-P1/database/RecGAN_Yida/test/depth_tsdf_npy -c 1 &
python2 data/rescaling.py -s /media/wangyida/D0-P1/database/data_3drecgan++/test/real_dataset/chair/real_50_25d_vox256  -tv /media/wangyida/D0-P1/database/RecGAN_Yida/test/depth_tsdf_npy -c 1 &
python2 data/rescaling.py -s /media/wangyida/D0-P1/database/data_3drecgan++/test/real_dataset/coach/real_50_25d_vox256  -tv /media/wangyida/D0-P1/database/RecGAN_Yida/test/depth_tsdf_npy -c 1 &
python2 data/rescaling.py -s /media/wangyida/D0-P1/database/data_3drecgan++/test/real_dataset/table/real_50_25d_vox256  -tv /media/wangyida/D0-P1/database/RecGAN_Yida/test/depth_tsdf_npy -c 1 &
python2 data/rescaling.py -s /media/wangyida/D0-P1/database/data_3drecgan++/test/real_dataset/bench/real_50_3d_vox256  -tv /media/wangyida/D0-P1/database/RecGAN_Yida/test/voxel_semantic_npy -c 1 &
python2 data/rescaling.py -s /media/wangyida/D0-P1/database/data_3drecgan++/test/real_dataset/chair/real_50_3d_vox256  -tv /media/wangyida/D0-P1/database/RecGAN_Yida/test/voxel_semantic_npy -c 2 &
python2 data/rescaling.py -s /media/wangyida/D0-P1/database/data_3drecgan++/test/real_dataset/coach/real_50_3d_vox256  -tv /media/wangyida/D0-P1/database/RecGAN_Yida/test/voxel_semantic_npy -c 3 &
python2 data/rescaling.py -s /media/wangyida/D0-P1/database/data_3drecgan++/test/real_dataset/table/real_50_3d_vox256  -tv /media/wangyida/D0-P1/database/RecGAN_Yida/test/voxel_semantic_npy -c 4 &
wait
