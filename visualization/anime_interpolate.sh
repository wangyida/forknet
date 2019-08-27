python3 visualization/voxviz.py -v ./eval/interpolation0-1.npy -t vis_samples_interpolate0-1 &
python3 visualization/voxviz.py -v ./eval/interpolation0-2.npy -t vis_samples_interpolate0-2 &
python3 visualization/voxviz.py -v ./eval/interpolation0-3.npy -t vis_samples_interpolate0-3 &
python3 visualization/voxviz.py -v ./eval/interpolation0-4.npy -t vis_samples_interpolate0-4 &
python3 visualization/voxviz.py -v ./eval/interpolation0-5.npy -t vis_samples_interpolate0-5 &
python3 visualization/voxviz.py -v ./eval/interpolation0-6.npy -t vis_samples_interpolate0-6 &
python3 visualization/voxviz.py -v ./eval/interpolation0-7.npy -t vis_samples_interpolate0-7 &
wait

ffmpeg -framerate 15 -pattern_type glob -i "vis_samples_interpolate0-1/*_0000.png" -s:v 720x720 -c:v libx264 -profile:v high -crf 20 -pix_fmt yuv420p inter_0_1.mp4 &
ffmpeg -framerate 15 -pattern_type glob -i "vis_samples_interpolate0-2/*_0000.png" -s:v 720x720 -c:v libx264 -profile:v high -crf 20 -pix_fmt yuv420p inter_0_2.mp4 &
ffmpeg -framerate 15 -pattern_type glob -i "vis_samples_interpolate0-3/*_0000.png" -s:v 720x720 -c:v libx264 -profile:v high -crf 20 -pix_fmt yuv420p inter_0_3.mp4 &
ffmpeg -framerate 15 -pattern_type glob -i "vis_samples_interpolate0-4/*_0000.png" -s:v 720x720 -c:v libx264 -profile:v high -crf 20 -pix_fmt yuv420p inter_0_4.mp4 &
ffmpeg -framerate 15 -pattern_type glob -i "vis_samples_interpolate0-5/*_0000.png" -s:v 720x720 -c:v libx264 -profile:v high -crf 20 -pix_fmt yuv420p inter_0_5.mp4 &
ffmpeg -framerate 15 -pattern_type glob -i "vis_samples_interpolate0-6/*_0000.png" -s:v 720x720 -c:v libx264 -profile:v high -crf 20 -pix_fmt yuv420p inter_0_6.mp4 &
ffmpeg -framerate 15 -pattern_type glob -i "vis_samples_interpolate0-7/*_0000.png" -s:v 720x720 -c:v libx264 -profile:v high -crf 20 -pix_fmt yuv420p inter_0_7.mp4 &
wait

convert vis_samples_interpolate0-1/*_0000.png +append -fuzz 2% -transparent white inter_0_1.png
convert vis_samples_interpolate0-2/*_0000.png +append -fuzz 2% -transparent white inter_0_2.png
convert vis_samples_interpolate0-3/*_0000.png +append -fuzz 2% -transparent white inter_0_3.png
convert vis_samples_interpolate0-4/*_0000.png +append -fuzz 2% -transparent white inter_0_4.png
convert vis_samples_interpolate0-5/*_0000.png +append -fuzz 2% -transparent white inter_0_5.png
convert vis_samples_interpolate0-6/*_0000.png +append -fuzz 2% -transparent white inter_0_6.png
convert vis_samples_interpolate0-7/*_0000.png +append -fuzz 2% -transparent white inter_0_7.png
