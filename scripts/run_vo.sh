#!/bin/bash
# Run full ORB-SLAM3 monocular VO for one MILUV robot.
# Usage: run_vo.sh <sequence> <robot>
set -e
source /src/gs25058/miniconda3/etc/profile.d/conda.sh
conda activate slam_env
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}

SEQ=$1; ROBOT=$2
PROJ=/src/gs25058/cr_RNE/covor_slam
ORB=/src/gs25058/cr_RNE/ORB_SLAM3
cd $PROJ/vo_output

$PROJ/orbslam_driver/mono_miluv \
  $ORB/Vocabulary/ORBvoc.txt \
  $PROJ/configs/miluv_${ROBOT}.yaml \
  ${SEQ}_${ROBOT}_list.txt \
  ${SEQ}_${ROBOT}_kf.txt
