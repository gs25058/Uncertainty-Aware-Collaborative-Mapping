# 중단된 보완 장면 실험의 정확한 재개

2026-11-08 약00:45에 이전 계산 프로세스가 종료돼 3층20개 실행은 96–128초의 관측/지도 기록만 남았다. 원본은 `results/fire_recon_20261107`에 보존했다. 2층20개 완료 결과와 기하 검증도 보존한다. 이번 실행은 `results/fire_recon_20261108_resume`, 보고서는 이 폴더다.

기존 스냅샷은 분류 지도와 표면 요약이어서 전체 점유 log-odds와 누적 표면 모멘트를 그대로 복구할 수 없었다. 따라서 기록된 현재 프레임만 같은 시각/카메라/동료 포즈에서 재생해 정확한 내부 상태를 재구성한다. 미래 후보의 깊이를 선택에 사용하지 않는다. 포즈가 다르면 캐시를 재사용하지 않고 실패를 보고하며, 저장된 원본 지도가 있는 시점에는 모든 지도 필드가 원본과 같은지 검사한다. 중복 적분으로 증거 수를 늘리지 않는다.

이후에는4초마다 원자적 체크포인트를 저장한다. OctoMap의 전체 확률을 보존하는 `.ot`와 표면 누적 가중치/높이 모멘트/투표/음성 증거/실제XY, 점유 endpoint 최대높이, 기체 상태·고장·예약 궤적·작업 시각·카운터, 전체 출력 기록과 로그 위치를 저장한다. `.bt`의 최대우도 변환/가지치기를 사용하지 않는다. 중단 뒤 체크포인트 이후의 미완료 로그는 별도로 보존하고 최신 완료 상태부터 이어간다. 고장 드론을 되살리거나 가상 임무 시간을 늘리지 않는다.

공통 runner에는 기본값이None인 선택형 실행 I/O만 추가했다. 관측 가치·후보/배정·이동·센서 수치·지도 적분·사람 판정 알고리즘은 유지한다. 동결된 중단 전 코드의 연속 실행과 현재 코드의 캐시 재생/2초 중단/복구를 실제 비교해27관측의 지도12개 배열, 궤적·이벤트·후보/배정·목표 경로·지표가 모두 정확히 같았다. `validation/execution_v1/verification.json`과 원본 로그를 보존한다.

```bash
cd /src/gs25058/cr_RNE/covor_slam
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export COVOR_PY=/src/gs25058/miniconda3/envs/covor/bin/python
# 이미 작업이 살아 있으면 새로 중복 실행하지 않는다. 파일 잠금도 적용한다.
nohup setsid $COVOR_PY scripts/explore/run_recon_resumable_matrix.py \
  --root results/fire_recon_20261108_resume \
  --previous results/fire_recon_20261107 --workers 20 \
  >> results/fire_recon_20261108_resume/audit_matrix.log 2>&1 < /dev/null &
# 완료를 감시하고 집계·정확 관측 재생 감사·보고·층별 표를 작성
nohup setsid bash scripts/explore/finish_resumable_recon.sh \
  >> results/fire_recon_20261108_resume/audit_post.log 2>&1 < /dev/null &
```

프로세스가 없을 때 같은 명령으로 재개한다. 코드/설정 해시가 바뀌면 저장 상태를 조용히 혼합하지 않고 거절한다. `main/scene*/.execution/latest.json`에서 실제 완료한 가상 시각과 캐시/새 관측 수를 확인한다. 원본 중단 실행의 마지막 스냅샷 시각과 현재 재구성 시각은 다르다.

실제 본 실행 소스106개 사본/해시는 `main/source_snapshot`, `main/source_manifest.json`, 설정 해시는 `configuration_manifest.json`에 있다. 원래98개 소스는 이전 폴더에 그대로 남는다. 캐시 재생·상태 저장의 CPU 비용과 새 카메라 생성 비용을 분리해 해석해야 한다. 최종 비교 시간은 양쪽 모두 동일한180초·4대·4Hz이며, 재생은 추가 물리 촬영/추가 시간/새 독립 반복으로 집계하지 않는다.
