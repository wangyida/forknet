python3 visualization/voxviz.py -v ./eval/surface.npy -t vis_samples_depth_seg_x &
python3 visualization/voxviz.py -v ./eval/depth_segment.npy -t vis_samples_depth_seg_y &
python3 visualization/voxviz.py -v ./eval/complete.npy -t vis_samples_complete_y &
python3 visualization/voxviz.py -v ./eval/real.npy -t vis_samples_real &
python3 visualization/voxviz.py -v ./eval/recons.npy -t vis_samples_recons &
python3 visualization/voxviz.py -v ./eval/recons_refine.npy -t vis_samples_refine &
wait
