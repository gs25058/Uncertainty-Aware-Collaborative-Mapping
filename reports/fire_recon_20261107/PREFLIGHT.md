# 보완 장면의 작은 실행 검증

현재 실제 완료한 항목과 실행 중인 항목을 구분한다. 이 문서는 신규 전체 비교의 성능 결과가 아니다.

## 완료한 검사

1. 기존 확대 화재 건물 10개의 생성 형상을 실제 보관 PLY·collision_geometry.json과 대조했다. 고체 꼭짓점(float32 PLY)/삼각형이 모두 일치했다. 보완본에서도 원래 부품·화재·목표는 유지하며, 3층 건물 5개에 하부 층계참 10개만 추가했다. `results/fire_recon_20261107/intake/asset_geometry_checks.{json,csv}`.
2. 선택형 보완, 원본 형상/화재 보존, 빠진 위치의 실제 지지 복구를 검사하는 테스트 3개가 통과했다. `intake/repair_tests.log`.
3. 보완 학교(scene1)의 독립 메시 정답 경로와 완전 관측의 공통 사람 판정기를 검사했다. **유효 경로 20/20, 올바른 차단 4/4, 무효 경로 0/20, 미확인 0/24, 잘못된 차단 0**. 3층 목표의 유효 경로가 **6개**다. 기준 양성 경로 20개도 별도로 모두 메시 유효였다.

3번은 완전한 기준 형상에서의 **입력·출력 및 판정기 검증**이다. 드론이 제한된 시간에 이 성능을 얻었다는 뜻이 아니다. 원본 학교의 도달 가능 14/24(3층 0)와 보완본 20/24(3층 6)를 섞어 집계하지 않는다.

학교 형상 SHA-256: `eda397340387143b2bd69a0ca7b2a429887f61bc96b4208d9c6cf55cee53a934`. 사람 크기·걸음 높이·지지 증거 규칙은 현재 공통 구현과 같다. 관측 가치 함수나 목표를 결과에 맞춰 수정하지 않았다.

실제 산출물: `results/fire_recon_20261107/preflight/scene1_complete/{summary.json,targets.csv,positive_controls.csv,reference_targets.json,prior.json,geometry.json,config.json}`. 코드/설정은 `preflight/source_manifest.json`과 source_snapshot에 동결했다.

## 완료한 작은 폐루프 연결 검사

`recon_preflight_driver.py`가 위 gate 합격을 확인한 뒤 **4대·16초·clean·unknown/entry**를 각각 실행한다. 원래 깊이 생성·지도 적분·사람 판정·가치·배정·이동·충돌·경로 평가를 그대로 재사용한다. 현재 코드에서는 world factory만 보완 어댑터로 바꾼다.

각 정책을 `preflight/pilot/scene1_seed0_clean_{unknown,entry}/`에 저장했다. 두 실행 모두 실제 촬영 212장, 팀 거리 19.2035m, 유효 1/20, 올바른 차단 0/4, 무효 0/1, 미확인 23/24, 실패 기체 0이었다. entry는 16/16초 공통 탐색으로 대체했다. 저장된 212장 깊이만 다시 적분한 labels/support/height/votes/evidence/std/실제XY/점유높이 배열이 모두 원본과 비트 동일했다. `preflight/cached_replay/verification.json`과 `preflight_data/summary_by_method.csv`에 있다. 새 입력은 매 0.25초의 실제 카메라 포즈에서 생성하며 동일 깊이를 복제하지 않는다. 본 비교 전체 완료로 표시하려면 아직 전체 실행과 집계 로그가 필요하다.

## 보존 자료와 수신 확인

원본 장면의 40회 비교는 `results/fire_closed_loop_20261107`에 보존·계속한다. 해당 실험에는 계단 결함이 있으므로 3층 유효 진입 능력의 검증으로 사용하지 않는다. 별도 요청 파일 `fire_buildings_3d(1).htm`의 서버 위치/기존 viewer와의 동일성, 그리고 이번 메시지 §2 이후의 추가 조건은 아직 확인되지 않았다. 현재 작은 검사는 서버에서 실제 확인한 기존 확대 화재 10개와 그 보완 복사본에 한정한다.
