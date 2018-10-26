python3 visualization/voxviz.py -v ./eval_synthetic/tsdf.npy -t vis_synthetic/tsdf_gt -n 4 &
python3 visualization/voxviz.py -v ./eval_synthetic/observe.npy -t vis_synthetic/observe_gt -n 4 &
python3 visualization/voxviz.py -v ./eval_synthetic/scene.npy -t vis_synthetic/scene_gt &
python3 visualization/voxviz.py -v ./eval_synthetic/layout.npy -t vis_synthetic/layout_gt &
python3 visualization/voxviz.py -v ./eval_synthetic/depth_seg_scene.npy -t vis_synthetic/depth_seg_gt &
python3 visualization/voxviz.py -v ./eval_synthetic/depth_seg_gen.npy -t vis_synthetic/depth_seg_gen &
python3 visualization/voxviz.py -v ./eval_synthetic/complete_scene.npy -t vis_synthetic/complete_gt -n 4 &
python3 visualization/voxviz.py -v ./eval_synthetic/complete_gen.npy -t vis_synthetic/complete_gen -n 4 &
python3 visualization/voxviz.py -v ./eval_synthetic/recons_vox.npy -t vis_synthetic/scene_recon &
python3 visualization/voxviz.py -v ./eval_synthetic/recons_tsdf.npy -t vis_synthetic/tsdf_recons -n 4 &
python3 visualization/voxviz.py -v ./eval_synthetic/recons_refine_vox.npy -t vis_synthetic/scene_refine &
python3 visualization/voxviz.py -v ./eval_synthetic/vae_vox.npy -t vis_synthetic/scene_vae &
python3 visualization/voxviz.py -v ./eval_synthetic/cc_vox.npy -t vis_synthetic/scene_cc &
python3 visualization/voxviz.py -v ./eval_synthetic/vae_tsdf.npy -t vis_synthetic/tsdf_vae -n 4 &
python3 visualization/voxviz.py -v ./eval_synthetic/cc_tsdf.npy -t vis_synthetic/tsdf_cc -n 4 &
python3 visualization/voxviz.py -v ./eval_synthetic/sscnet_vox.npy -t vis_synthetic/scene_sccnet &
wait
