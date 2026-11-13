#!/bin/bash
# Run full ORB-SLAM3 monocular VO for one MILUV robot.
# Usage: run_vo.sh <sequence> <robot>
set -e
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate slam_env
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}

SEQ=$1; ROBOT=$2
PROJ="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ORB="${ORB_SLAM3_DIR:-$PROJ/../ORB_SLAM3}"   # path to a built ORB-SLAM3 checkout
cd $PROJ/vo_output

$PROJ/orbslam_driver/mono_miluv \
  $ORB/Vocabulary/ORBvoc.txt \
  $PROJ/configs/miluv_${ROBOT}.yaml \
  ${SEQ}_${ROBOT}_list.txt \
  ${SEQ}_${ROBOT}_kf.txt
