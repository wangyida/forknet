python data/bin2camera.py -s /media/wangyida/D0-P1/database/SUNCGtrain_1_500/ -t /media/wangyida/SSD2T/database/SUNCG_Yida/train & 
python data/bin2camera.py -s /media/wangyida/D0-P1/database/SUNCGtrain_1_1000/ -t /media/wangyida/SSD2T/database/SUNCG_Yida/train &
python data/bin2camera.py -s /media/wangyida/D0-P1/database/SUNCGtrain_501_1000/ -t /media/wangyida/SSD2T/database/SUNCG_Yida/train &
python data/bin2camera.py -s /media/wangyida/D0-P1/database/SUNCGtrain_1001_2000/ -t /media/wangyida/SSD2T/database/SUNCG_Yida/train &
python data/bin2camera.py -s /media/wangyida/D0-P1/database/SUNCGtrain_1001_3000/ -t /media/wangyida/SSD2T/database/SUNCG_Yida/train &
python data/bin2camera.py -s /media/wangyida/D0-P1/database/SUNCGtrain_3001_5000/ -t /media/wangyida/SSD2T/database/SUNCG_Yida/train &
python data/bin2camera.py -s /media/wangyida/D0-P1/database/SUNCGtrain_5001_7000/ -t /media/wangyida/SSD2T/database/SUNCG_Yida/train &
wait

python data/bin2camera.py -s /media/wangyida/D0-P1/database/depthbin_NYU_SUNCG/SUNCGtest_49700_49884 -t /media/wangyida/SSD2T/database/SUNCG_Yida/test &
python data/bin2camera.py -s /media/wangyida/D0-P1/database/depthbin_NYU_SUNCG/NYUtrain -t /media/wangyida/SSD2T/database/NYU_Yida/train &
python data/bin2camera.py -s /media/wangyida/D0-P1/database/depthbin_NYU_SUNCG/NYUtest -t /media/wangyida/SSD2T/database/NYU_Yida/test &
wait
