#!/bin/bash
# Compile the standalone MILUV mono driver, linking the already-built ORB-SLAM3.
set -e
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate slam_env

PROJ="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ORB="${ORB_SLAM3_DIR:-$PROJ/../ORB_SLAM3}"   # path to a built ORB-SLAM3 checkout
OUT="$PROJ/orbslam_driver"
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
