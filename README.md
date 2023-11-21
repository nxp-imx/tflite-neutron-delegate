# TfLite-neutron-delegate
tflite-neutron-delegate is one external delegate for tensorflow lite, it is constructed with neutron-software, it can run neutron converted model on imx Neutron NPU.

# Use tflite-neutron-delegate

## Prepare source code
```sh
git clone {this repo}
cd tflite-neutron-delegate
git submodule update --init --recursive
```
# Build from source with cmake

```sh
# set the toolchain env
source /PATH_TO_TOOLCHAIN/environment-setup-cortexa53-crypto-poky-linux

# build the delegate
mkdir build && cd build
cmake ..
make -j 8

# benchmark_model
make benchmark_model -j8
# label_image
make lable_image -j8
```

If you would like to build using local version of tensorflow, you can use `FETCHCONTENT_SOURCE_DIR_TENSORFLOW` cmake variable. Point this variable to your tensorflow tree. For additional details on this variable please see the [official cmake documentation](https://cmake.org/cmake/help/latest/module/FetchContent.html#command:fetchcontent_populate)

``` sh
cmake -DFETCHCONTENT_SOURCE_DIR_TENSORFLOW=/my/copy/of/tensorflow \
    -DOTHER_CMAKE_DEFINES...\
    ..
```
After cmake execution completes, build and run as usual.

## Enable external delegate support in benchmark_model/label_image

If tensorflow source code downloaded by cmake, you can find it in <build_output_dir>/_deps/tensorflow-src

## Run
```sh
./benchmark_model --external_delegate_path=<patch_to_libneutron_delegate.so> --graph=<tflite_neutron_model.tflite>
```
