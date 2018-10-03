python3 visualization/voxviz.py -v ./eval/depth_seg_real.npy -t vis_depth_seg_real &
python3 visualization/voxviz.py -v ./eval/depth_seg_gen.npy -t vis_depth_seg_gen &
python3 visualization/voxviz.py -v ./eval/complete_real.npy -t vis_complete_real &
python3 visualization/voxviz.py -v ./eval/complete_gen.npy -t vis_complete_gen &
python3 visualization/voxviz.py -v ./eval/tsdf.npy -t vis_tsdf &
python3 visualization/voxviz.py -v ./eval/real.npy -t vis_real &
python3 visualization/voxviz.py -v ./eval/recons.npy -t vis_recons &
python3 visualization/voxviz.py -v ./eval/recons_refine.npy -t vis_refine &
wait
