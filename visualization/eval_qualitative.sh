python3 visualization/voxviz.py -v ./eval/depth_seg_real.npy -t vis_depth_seg_real &
python3 visualization/voxviz.py -v ./eval/depth_seg_gen.npy -t vis_depth_seg_gen &
python3 visualization/voxviz.py -v ./eval/complete_real.npy -t vis_complete_real -n 4 &
python3 visualization/voxviz.py -v ./eval/complete_gen.npy -t vis_complete_gen -n 4 &
python3 visualization/voxviz.py -v ./eval/tsdf.npy -t vis_real_tsdf -n 4 &
python3 visualization/voxviz.py -v ./eval/real.npy -t vis_real_vox &
python3 visualization/voxviz.py -v ./eval/recons_vox.npy -t vis_recons_vox &
python3 visualization/voxviz.py -v ./eval/recons_refine_vox.npy -t vis_refine_vox &
python3 visualization/voxviz.py -v ./eval/vae_vox.npy -t vis_vae_vox &
python3 visualization/voxviz.py -v ./eval/cc_vox.npy -t vis_cc_vox &
python3 visualization/voxviz.py -v ./eval/recons_tsdf.npy -t vis_recons_tsdf -n 4 &
python3 visualization/voxviz.py -v ./eval/vae_tsdf.npy -t vis_vae_tsdf -n -4 &
python3 visualization/voxviz.py -v ./eval/cc_tsdf.npy -t vis_cc_tsdf -n 4 &
wait
