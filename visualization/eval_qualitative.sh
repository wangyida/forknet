if [ "$1" = "suncg" ]
then
	python3 visualization/voxviz.py -v ./eval_suncg/tsdf.npy -t vis_suncg/tsdf_gt -n 3 &
	python3 visualization/voxviz.py -v ./eval_suncg/observe.npy -t vis_suncg/observe_gt -n 3 &
	python3 visualization/voxviz.py -v ./eval_suncg/scene.npy -t vis_suncg/scene_gt -n 12 &
	python3 visualization/voxviz.py -v ./eval_suncg/layout.npy -t vis_suncg/layout_gt -n 12 &
	python3 visualization/voxviz.py -v ./eval_suncg/depth_seg_scene.npy -t vis_suncg/depth_seg_gt -n 12 &
	python3 visualization/voxviz.py -v ./eval_suncg/depth_seg_gen.npy -t vis_suncg/depth_seg_gen -n 12 &
	python3 visualization/voxviz.py -v ./eval_suncg/complete_scene.npy -t vis_suncg/complete_gt -n 3 &
	python3 visualization/voxviz.py -v ./eval_suncg/complete_gen.npy -t vis_suncg/complete_gen -n 3 &
	python3 visualization/voxviz.py -v ./eval_suncg/recons_vox.npy -t vis_suncg/scene_recon -n 12 &
	python3 visualization/voxviz.py -v ./eval_suncg/recons_tsdf.npy -t vis_suncg/tsdf_recons -n 3 &
	python3 visualization/voxviz.py -v ./eval_suncg/recons_refine_vox.npy -t vis_suncg/scene_refine -n 12 &
	python3 visualization/voxviz.py -v ./eval_suncg/vae_vox.npy -t vis_suncg/scene_vae -n 12 &
	python3 visualization/voxviz.py -v ./eval_suncg/cc_vox.npy -t vis_suncg/scene_cc -n 12 &
	python3 visualization/voxviz.py -v ./eval_suncg/vae_tsdf.npy -t vis_suncg/tsdf_vae -n 3 &
	python3 visualization/voxviz.py -v ./eval_suncg/cc_tsdf.npy -t vis_suncg/tsdf_cc -n 3 &
	python3 visualization/voxviz.py -v ./eval_suncg/sscnet_vox.npy -t vis_suncg/scene_sccnet -n 12 &
	wait
elif [ "$1" = "nyu"]
	python3 visualization/voxviz.py -v ./eval_nyu/tsdf.npy -t vis_nyu/tsdf_gt -n 3 &
	python3 visualization/voxviz.py -v ./eval_nyu/observe.npy -t vis_nyu/observe_gt -n 3 &
	python3 visualization/voxviz.py -v ./eval_nyu/scene.npy -t vis_nyu/scene_gt -n 12 &
	python3 visualization/voxviz.py -v ./eval_nyu/layout.npy -t vis_nyu/layout_gt -n 12 &
	python3 visualization/voxviz.py -v ./eval_nyu/depth_seg_scene.npy -t vis_nyu/depth_seg_gt -n 12 &
	python3 visualization/voxviz.py -v ./eval_nyu/depth_seg_gen.npy -t vis_nyu/depth_seg_gen -n 12 &
	python3 visualization/voxviz.py -v ./eval_nyu/complete_scene.npy -t vis_nyu/complete_gt -n 3 &
	python3 visualization/voxviz.py -v ./eval_nyu/complete_gen.npy -t vis_nyu/complete_gen -n 3 &
	python3 visualization/voxviz.py -v ./eval_nyu/recons_vox.npy -t vis_nyu/scene_recon -n 12 &
	python3 visualization/voxviz.py -v ./eval_nyu/recons_tsdf.npy -t vis_nyu/tsdf_recons -n 3 &
	python3 visualization/voxviz.py -v ./eval_nyu/recons_refine_vox.npy -t vis_nyu/scene_refine -n 12 &
	python3 visualization/voxviz.py -v ./eval_nyu/vae_vox.npy -t vis_nyu/scene_vae -n 12 &
	python3 visualization/voxviz.py -v ./eval_nyu/cc_vox.npy -t vis_nyu/scene_cc -n 12 &
	python3 visualization/voxviz.py -v ./eval_nyu/vae_tsdf.npy -t vis_nyu/tsdf_vae -n 3 &
	python3 visualization/voxviz.py -v ./eval_nyu/cc_tsdf.npy -t vis_nyu/tsdf_cc -n 3 &
	python3 visualization/voxviz.py -v ./eval_nyu/sscnet_vox.npy -t vis_nyu/scene_sccnet -n 12 &
	wait
then
elif [ "$1" = "shapenet"]
	python3 visualization/voxviz.py -v ./eval_shapenet/tsdf.npy -t vis_shapenet/tsdf_gt -n 3 &
	python3 visualization/voxviz.py -v ./eval_shapenet/scene.npy -t vis_shapenet/scene_gt -n 4 &
	python3 visualization/voxviz.py -v ./eval_shapenet/complete_scene.npy -t vis_shapenet/complete_gt -n 3 &
	python3 visualization/voxviz.py -v ./eval_shapenet/complete_gen.npy -t vis_shapenet/complete_gen -n 3 &
	python3 visualization/voxviz.py -v ./eval_shapenet/recons_vox.npy -t vis_shapenet/scene_recon -n 4 &
	python3 visualization/voxviz.py -v ./eval_shapenet/recons_tsdf.npy -t vis_shapenet/tsdf_recons -n 3 &
	python3 visualization/voxviz.py -v ./eval_shapenet/recons_refine_vox.npy -t vis_shapenet/scene_refine -n 4 &
	python3 visualization/voxviz.py -v ./eval_shapenet/vae_vox.npy -t vis_shapenet/scene_vae -n 4 &
	python3 visualization/voxviz.py -v ./eval_shapenet/cc_vox.npy -t vis_shapenet/scene_cc -n 4 &
	python3 visualization/voxviz.py -v ./eval_shapenet/vae_tsdf.npy -t vis_shapenet/tsdf_vae -n 3 &
	python3 visualization/voxviz.py -v ./eval_shapenet/cc_tsdf.npy -t vis_shapenet/tsdf_cc -n 3 &
	python3 visualization/voxviz.py -v ./eval_shapenet/sscnet_vox.npy -t vis_shapenet/scene_sccnet -n 4 &
	wait
then
fi
