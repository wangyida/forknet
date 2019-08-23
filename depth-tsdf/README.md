# Volumetric TSDF Fusion of Multiple Depth Maps

## Requirements
 * NVIDA GPU with [CUDA](https://developer.nvidia.com/cuda-downloads) support
 * [OpenCV](http://opencv.org/) (tested with OpenCV 4.0.1)

## Compile

There are 2 options to compile the tool

1. compile directly with *nvcc* from terminal by using our shell script

```shell
./compile.sh # compiles demo executable
```

2. using *cmake* to configure compiling environment and *make* to compile

```shell
cmake . # configure
make # compiles demo executable
```

