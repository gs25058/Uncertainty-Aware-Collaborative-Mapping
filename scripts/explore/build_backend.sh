#!/usr/bin/env bash
set -eu
# Source dependencies stay outside the working tree. No ROS sensor/odom shortcut.
task_external="${1:-/src/gs25058/cr_RNE/external_observation_20261105}"
task_script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
mkdir -p "$task_external/standalone"
if [ ! -d "$task_external/FUEL/.git" ]; then
  git clone https://github.com/HKUST-Aerial-Robotics/FUEL "$task_external/FUEL"
  git -C "$task_external/FUEL" checkout 662dd23c7b52b258d3c4a0155ff6632118e8984f
fi
test "$(git -C "$task_external/FUEL" rev-parse HEAD)" = 662dd23c7b52b258d3c4a0155ff6632118e8984f
if [ ! -d "$task_external/eigen_pkg/usr/include/eigen3" ]; then
  (cd "$task_external" && apt-get download libeigen3-dev=3.4.0-2ubuntu2)
  dpkg-deb -x "$task_external/libeigen3-dev_3.4.0-2ubuntu2_all.deb" "$task_external/eigen_pkg"
fi
cp "$task_script_dir/fuel_bridge.cpp" "$task_external/standalone/fuel_bridge.cpp"
g++ -O3 -std=c++14 -shared -fPIC -I "$task_external/eigen_pkg/usr/include/eigen3" -I "$task_external/FUEL/fuel_planner/poly_traj/include" "$task_external/standalone/fuel_bridge.cpp" -o "$task_external/standalone/libfuel_bridge.so"
