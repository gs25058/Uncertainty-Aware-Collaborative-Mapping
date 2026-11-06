# 외부 이동 코드 점검과 실행 백엔드

이번 비교의 명칭은 **공통 다중 드론 탐색 파이프라인에서 관측 가치 함수 비교**다. ROS 없는 동기식 시뮬레이션이며 RACER 전체 재현은 아니다.

| 공식 저장소 | 고정 커밋 | 점검 결과 |
|---|---|---|
| [RACER](https://github.com/Robotics-STAR-Lab/RACER) | `049c332e3634ef72d8beb155b4c13dc91ca52916` | `fast_exploration_manager.cpp::planTrajToView`의 초기 `optimistic` 분기는 사용하지 않음. 구역 분배를 사용하지 않아 재관측 후보도 동일 배정 대상으로 남음. |
| [FUEL](https://github.com/HKUST-Aerial-Robotics/FUEL) | `662dd23c7b52b258d3c4a0155ff6632118e8984f` | 공통 백엔드. 실제 빌드 대상 `path_searching/src/astar2.cpp`의 UNKNOWN 제외·팽창 점유 검사·선분 검사 구조를 Python 어댑터로 연결. 원본 `poly_traj/include/poly_traj/polynomial_traj.h::Polynomial`을 C++ 공유 라이브러리로 컴파일하고 매 궤적 위치·속도·가속도 계산에 직접 호출. |
| [EGO-Swarm](https://github.com/ZJU-FAST-Lab/ego-planner-swarm) | `92fe9f7227b2da819133eb8e0e8c7fc000f6ae20` | README의 CPU local_sensing 무 raycasting, fake_drone 명령→odom 직접 대입을 확인. 해당 센서는 사용하지 않음. |

외부 원본: `/src/gs25058/cr_RNE/external_observation_20261105/`. 외부 저장소는 수정하지 않았다. 어댑터는 `standalone/fuel_bridge.cpp`, 작업 트리 복사본 `scripts/explore/fuel_bridge.cpp`다. FUEL은 GPL-3.0이며 그 Polynomial 헤더와 라이선스를 유지한다.

현재 Ubuntu 22.04 서버에 ROS/catkin/colcon, PCL/Eigen 시스템 패키지가 없다. FUEL path_searching CMake 실행은 PCL 누락으로 실패했다(`results/closed_loop_20261105/audit/fuel_ros_build.log`). Eigen 개발 .deb만 외부 경로에 추출해 원본 Polynomial을 단독 컴파일했다. ROS 노드나 원래 B-spline 최적화를 실행한 결과로 표현하지 않는다.

공통 Python A*는 관측 FREE만 기체 반경 0.14 m+여유 0.04 m로 팽창 검사한다. 시야 밖 UNKNOWN으로 비행하지 않는다. 직선 단축 뒤 각 선분을 원본 FUEL Polynomial의 정지→정지 5차식으로 시간화한다. 속도 1.0 m/s, 가속도 1.5 m/s², yaw 90°/s, 짐벌 pitch 60°/s 제한이다. 선분마다 정지하므로 원본 연속 고속 최적화보다 느릴 수 있다. 상대 궤적의 0.1초 샘플 사이 최소 거리와 가속도 보수 여유를 검사하며, 정지한 동료와 예약 종점은 A*에서도 피한다. 최대 8초 기다림 후에도 안 되면 현재 공간에서 대기한다.

센서는 Open3D 메시 raycasting에 의한 좌우 영상(320×192 mono8, fx=210, B=0.08 m)이다. 다른 드론은 반경 0.14 m 구로 가림을 만들고, 원래 파이프라인 기본값과 같은 0.35 m 동료 제외 반경을 쓴다. 기존 SGBM 설정을 모든 정책/연기 조건에 동일 적용한다. 완전 깊이는 연기 합성 입력에만 사용하며 지도에는 연기 영상에서 다시 계산한 깊이를 적분한다. 잡음 강도 0, 영상 텍스처는 고정 장면 시드로 생성한다.

정확한 위치가 주어지는 `perfect_odometry`, 운동학적 정확 추종, 즉시 공유 통신을 가정한다. 자세 제어·다운워시·통신 지연·실시간 처리 성능은 검증 대상이 아니다. CPU 실행의 외부 계산 시간을 별도로 저장한다. GPU는 사용하지 않는다.
