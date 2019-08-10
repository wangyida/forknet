if [ "$1" = "suncg" ]
then
	python3 visualization/voxviz.py -v ./eval_suncg/tsdf.npy -t vis_suncg/surface_gt -n 3 &
	python3 visualization/voxviz.py -v ./eval_suncg/observe.npy -t vis_suncg/observe_gt -n 3 &
	python3 visualization/voxviz.py -v ./eval_suncg/scene.npy -t vis_suncg/scene_gt -n 12 &
	python3 visualization/voxviz.py -v ./eval_suncg/depth_seg_scene.npy -t vis_suncg/depth_seg_gt -n 12 &
	python3 visualization/voxviz.py -v ./eval_suncg/depth_seg_gen.npy -t vis_suncg/depth_seg_gen -n 12 &
	# python3 visualization/voxviz.py -v ./eval_suncg/complete_scene.npy -t vis_suncg/complete_gt -n 1 &
	# python3 visualization/voxviz.py -v ./eval_suncg/complete_gen.npy -t vis_suncg/complete_gen -n 1 &
	python3 visualization/voxviz.py -v ./eval_suncg/recons_vox.npy -t vis_suncg/scene_gen -n 12 &
	# python3 visualization/voxviz.py -v ./eval_suncg/recons_tsdf.npy -t vis_suncg/tsdf_recons -n 3 &
	python3 visualization/voxviz.py -v ./eval_suncg/refined_voxs_vae.npy -t vis_suncg/scene_ours -n 12 &
	python3 visualization/voxviz.py -v ./eval_suncg/refined_voxs_gen.npy -t vis_suncg/scene_ours_wo-refiner -n 12 &
	python3 visualization/voxviz.py -v ./eval_suncg/refined_voxs_cc.npy -t vis_suncg/scene_ours_wo-styletrans -n 12 &
	python3 visualization/voxviz.py -v ./eval_suncg/vae_vox.npy -t vis_suncg/scene_vae -n 12 &
	# python3 visualization/voxviz.py -v ./eval_suncg/vae_tsdf.npy -t vis_suncg/tsdf_vae -n 3 &
	# python3 visualization/voxviz.py -v ./eval_suncg/cc_tsdf.npy -t vis_suncg/tsdf_cc -n 3 &
	wait
elif [ "$1" = "nyu" ]
then
	python3 visualization/voxviz.py -v ./eval_nyu/tsdf.npy -t vis_nyu/surface_gt -n 3 &
	python3 visualization/voxviz.py -v ./eval_nyu/observe.npy -t vis_nyu/observe_gt -n 3 &
	python3 visualization/voxviz.py -v ./eval_nyu/scene.npy -t vis_nyu/scene_gt -n 12 &
	python3 visualization/voxviz.py -v ./eval_nyu/depth_seg_scene.npy -t vis_nyu/depth_seg_gt -n 12 &
	python3 visualization/voxviz.py -v ./eval_nyu/depth_seg_gen.npy -t vis_nyu/depth_seg_gen -n 12 &
	# python3 visualization/voxviz.py -v ./eval_nyu/complete_scene.npy -t vis_nyu/complete_gt -n 1 &
	# python3 visualization/voxviz.py -v ./eval_nyu/complete_gen.npy -t vis_nyu/complete_gen -n 1 &
	python3 visualization/voxviz.py -v ./eval_nyu/recons_vox.npy -t vis_nyu/scene_gen -n 12 &
	# python3 visualization/voxviz.py -v ./eval_nyu/recons_tsdf.npy -t vis_nyu/tsdf_recons -n 3 &
	python3 visualization/voxviz.py -v ./eval_nyu/refined_voxs_vae.npy -t vis_nyu/scene_ours -n 12 &
	python3 visualization/voxviz.py -v ./eval_nyu/refined_voxs_gen.npy -t vis_nyu/scene_ours_wo-refiner -n 12 &
	python3 visualization/voxviz.py -v ./eval_nyu/refined_voxs_cc.npy -t vis_nyu/scene_ours_wo-styletrans -n 12 &
	python3 visualization/voxviz.py -v ./eval_nyu/vae_vox.npy -t vis_nyu/scene_vae -n 12 &
	# python3 visualization/voxviz.py -v ./eval_nyu/vae_tsdf.npy -t vis_nyu/tsdf_vae -n 3 &
	# python3 visualization/voxviz.py -v ./eval_nyu/cc_tsdf.npy -t vis_nyu/tsdf_cc -n 3 &
	wait
elif [ "$1" = "sscnet" ]
then
	# python3 visualization/voxviz.py -v ./eval_sscnet/tsdf.npy -t vis_sscnet/surface_gt -n 3 &
	# python3 visualization/voxviz.py -v ./eval_sscnet/observe.npy -t vis_sscnet/observe_gt -n 3 &
	# python3 visualization/voxviz.py -v ./eval_sscnet/scene.npy -t vis_sscnet/scene_gt -n 12 &
	python3 visualization/voxviz.py -v ./eval_sscnet/depth_seg_scene.npy -t vis_sscnet/depth_seg_gt -n 12 &
	python3 visualization/voxviz.py -v ./eval_sscnet/depth_seg_gen.npy -t vis_sscnet/depth_seg_gen -n 12 &
	# python3 visualization/voxviz.py -v ./eval_sscnet/complete_scene.npy -t vis_sscnet/complete_gt -n 1 &
	# python3 visualization/voxviz.py -v ./eval_sscnet/complete_gen.npy -t vis_sscnet/complete_gen -n 1 &
	python3 visualization/voxviz.py -v ./eval_sscnet/recons_vox.npy -t vis_sscnet/scene_gen -n 12 &
	# python3 visualization/voxviz.py -v ./eval_sscnet/recons_tsdf.npy -t vis_sscnet/tsdf_recons -n 3 &
	python3 visualization/voxviz.py -v ./eval_sscnet/refined_voxs.npy -t vis_sscnet/scene_ours -n 12 &
	python3 visualization/voxviz.py -v ./eval_sscnet/refined_voxs_gen.npy -t vis_sscnet/scene_ours_wo-refiner -n 12 &
	python3 visualization/voxviz.py -v ./eval_sscnet/refined_voxs_cc.npy -t vis_sscnet/scene_ours_wo-styletrans -n 12 &
	python3 visualization/voxviz.py -v ./eval_sscnet/vae_vox.npy -t vis_sscnet/scene_vae -n 12 &
	# python3 visualization/voxviz.py -v ./eval_sscnet/vae_tsdf.npy -t vis_sscnet/tsdf_vae -n 3 &
	# python3 visualization/voxviz.py -v ./eval_sscnet/cc_tsdf.npy -t vis_sscnet/tsdf_cc -n 3 &
	wait
elif [ "$1" = "shapenet" ]
then
	python3 visualization/voxviz.py -v ./eval_shapenet/tsdf.npy -t vis_shapenet/surface_gt -n 3 &
	python3 visualization/voxviz.py -v ./eval_shapenet/scene.npy -t vis_shapenet/scene_gt -n 4 &
	python3 visualization/voxviz.py -v ./eval_shapenet/complete_scene.npy -t vis_shapenet/complete_gt -n 1 &
	python3 visualization/voxviz.py -v ./eval_shapenet/complete_gen.npy -t vis_shapenet/complete_gen -n 1 &
	python3 visualization/voxviz.py -v ./eval_shapenet/recons_vox.npy -t vis_shapenet/scene_gen -n 4 &
	# python3 visualization/voxviz.py -v ./eval_shapenet/recons_tsdf.npy -t vis_shapenet/tsdf_recons -n 3 &
	python3 visualization/voxviz.py -v ./eval_shapenet/refined_voxs.npy -t vis_shapenet/scene_ours -n 4 &
	python3 visualization/voxviz.py -v ./eval_shapenet/vae_vox.npy -t vis_shapenet/scene_ours_wo-refiner -n 4 &
	python3 visualization/voxviz.py -v ./eval_shapenet/cc_vox.npy -t vis_shapenet/scene_ours_wo-styletrans -n 4 &
	# python3 visualization/voxviz.py -v ./eval_shapenet/vae_tsdf.npy -t vis_shapenet/tsdf_vae -n 3 &
	# python3 visualization/voxviz.py -v ./eval_shapenet/cc_tsdf.npy -t vis_shapenet/tsdf_cc -n 3 &
	wait
fi
