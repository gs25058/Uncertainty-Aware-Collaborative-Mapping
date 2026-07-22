#!/bin/bash
# Run ORB-SLAM3 monocular VO for all 3 robots of a sequence, with up to 3 retries
# per robot (the ifo001-style Sophus SO3::exp NaN crash is timing-dependent, so a
# retry usually succeeds). Reports keyframe counts; skips a robot after 3 failures.
# Usage: run_vo_all.sh <sequence>
SEQ=${1:-default_3_random_0}
PROJ=/src/gs25058/cr_RNE/covor_slam
MIN_KF=20

for r in ifo001 ifo002 ifo003; do
  KF=$PROJ/vo_output/${SEQ}_${r}_kf.txt
  ok=0
  for attempt in 1 2 3; do
    echo ">>> $r attempt $attempt ($(date +%H:%M:%S))"
    rm -f "$KF"
    bash $PROJ/scripts/run_vo.sh "$SEQ" "$r" > /tmp/vo_${SEQ}_${r}.log 2>&1
    rc=$?
    n=$(wc -l < "$KF" 2>/dev/null || echo 0)
    echo "    exit=$rc keyframes=$n"
    if [ "$n" -ge "$MIN_KF" ]; then ok=1; break; fi
  done
  if [ "$ok" = 1 ]; then
    echo "=== $r OK: $(wc -l < "$KF") keyframes ==="
  else
    echo "=== $r FAILED after 3 attempts (last log tail): ==="
    tail -5 /tmp/vo_${SEQ}_${r}.log
  fi
done
echo "ALL_VO_DONE"
