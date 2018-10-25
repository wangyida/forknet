python3 visualization/voxviz.py -v ./eval_synthetic/depth_seg_real.npy -t vis_synthetic/vis_depth_seg_real &
python3 visualization/voxviz.py -v ./eval_synthetic/depth_seg_gen.npy -t vis_synthetic/vis_depth_seg_gen &
python3 visualization/voxviz.py -v ./eval_synthetic/complete_real.npy -t vis_synthetic/vis_complete_real -n 4 &
python3 visualization/voxviz.py -v ./eval_synthetic/complete_gen.npy -t vis_synthetic/vis_complete_gen -n 4 &
python3 visualization/voxviz.py -v ./eval_synthetic/tsdf.npy -t vis_synthetic/vis_synthetic_tsdf -n 4 &
python3 visualization/voxviz.py -v ./eval_synthetic/real.npy -t vis_synthetic/vis_synthetic_vox &
python3 visualization/voxviz.py -v ./eval_synthetic/recons_vox.npy -t vis_synthetic/vis_recons_vox &
python3 visualization/voxviz.py -v ./eval_synthetic/recons_tsdf.npy -t vis_synthetic/vis_recons_tsdf -n 4 &
python3 visualization/voxviz.py -v ./eval_synthetic/recons_refine_vox.npy -t vis_synthetic/vis_refine_vox &
python3 visualization/voxviz.py -v ./eval_synthetic/vae_vox.npy -t vis_synthetic/vis_vae_vox &
python3 visualization/voxviz.py -v ./eval_synthetic/cc_vox.npy -t vis_synthetic/vis_cc_vox &
python3 visualization/voxviz.py -v ./eval_synthetic/vae_tsdf.npy -t vis_synthetic/vis_vae_tsdf -n 4 &
python3 visualization/voxviz.py -v ./eval_synthetic/cc_tsdf.npy -t vis_synthetic/vis_cc_tsdf -n 4 &
wait
