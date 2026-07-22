#!/bin/bash
# Compile the standalone MILUV mono driver, linking the already-built ORB-SLAM3.
set -e
source /src/gs25058/miniconda3/etc/profile.d/conda.sh
conda activate slam_env

ORB=/src/gs25058/cr_RNE/ORB_SLAM3
OUT=/src/gs25058/cr_RNE/covor_slam/orbslam_driver
P=$CONDA_PREFIX

g++ -std=c++14 -O3 -w \
  "$OUT/mono_miluv.cc" -o "$OUT/mono_miluv" \
  -I"$ORB" -I"$ORB/include" -I"$ORB/include/CameraModels" \
  -I"$ORB/Thirdparty/Sophus" \
  -I"$P/include/eigen3" -I"$P/include" -I"$P/include/opencv4" \
  -L"$ORB/lib" -lORB_SLAM3 \
  "$ORB/Thirdparty/DBoW2/lib/libDBoW2.so" \
  "$ORB/Thirdparty/g2o/lib/libg2o.so" \
  -L"$P/lib" \
  -lopencv_core -lopencv_imgproc -lopencv_imgcodecs -lopencv_highgui -lopencv_features2d -lopencv_calib3d \
  -lpangolin -lboost_serialization -lGL -lGLEW \
  -Wl,-rpath,"$ORB/lib" -Wl,-rpath,"$P/lib"

echo "Built: $OUT/mono_miluv"
