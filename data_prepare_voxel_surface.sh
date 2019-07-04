rm -rf /media/wangyida/HDD/database/SUNCG_Yida/train/surface_semantic_npy
rm -rf /media/wangyida/HDD/database/SUNCG_Yida/test/surface_semantic_npy
rm -rf /media/wangyida/HDD/database/NYU_Yida/train/surface_semantic_npy
rm -rf /media/wangyida/HDD/database/NYU_Yida/test/surface_semantic_npy
mkdir /media/wangyida/HDD/database/SUNCG_Yida/train/surface_semantic_npy
mkdir /media/wangyida/HDD/database/SUNCG_Yida/test/surface_semantic_npy
mkdir /media/wangyida/HDD/database/NYU_Yida/train/surface_semantic_npy
mkdir /media/wangyida/HDD/database/NYU_Yida/test/surface_semantic_npy

python2 data/depthbin2surface.py -s /media/gaoyafei/Yida/binary/SUNCGtrain_1_500_surface -tv /media/wangyida/HDD/database/SUNCG_Yida/train/surface_semantic_npy &
python2 data/depthbin2surface.py -s /media/gaoyafei/Yida/binary/SUNCGtrain_1_1000_surface -tv /media/wangyida/HDD/database/SUNCG_Yida/train/surface_semantic_npy &
python2 data/depthbin2surface.py -s /media/gaoyafei/Yida/binary/SUNCGtrain_501_1000_surface -tv /media/wangyida/HDD/database/SUNCG_Yida/train/surface_semantic_npy &
python2 data/depthbin2surface.py -s /media/gaoyafei/Yida/binary/SUNCGtrain_1001_2000_surface -tv /media/wangyida/HDD/database/SUNCG_Yida/train/surface_semantic_npy &
python2 data/depthbin2surface.py -s /media/gaoyafei/Yida/binary/SUNCGtrain_1001_3000_surface -tv /media/wangyida/HDD/database/SUNCG_Yida/train/surface_semantic_npy &
python2 data/depthbin2surface.py -s /media/gaoyafei/Yida/binary/SUNCGtrain_3001_5000_surface -tv /media/wangyida/HDD/database/SUNCG_Yida/train/surface_semantic_npy &
python2 data/depthbin2surface.py -s /media/gaoyafei/Yida/binary/SUNCGtrain_5001_7000_surface -tv /media/wangyida/HDD/database/SUNCG_Yida/train/surface_semantic_npy &
python2 data/depthbin2surface.py -s /media/gaoyafei/Yida/binary/SUNCGtest_49700_49884_surface -tv /media/wangyida/HDD/database/SUNCG_Yida/test/surface_semantic_npy &
python2 data/depthbin2surface.py -s /media/gaoyafei/Yida/binary/NYUtrain_surface -tv /media/wangyida/HDD/database/NYU_Yida/train/surface_semantic_npy &
python2 data/depthbin2surface.py -s /media/gaoyafei/Yida/binary/NYUtest_surface -tv /media/wangyida/HDD/database/NYU_Yida/test/surface_semantic_npy &
wait
