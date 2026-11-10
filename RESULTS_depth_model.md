# 낙관적 깊이 관측 모델 실험

2026-11-06, `paper/conference-2026`, 기준 커밋 `5788821`.

`depth_error_model`/`ideal_depth`를 기존 지도·발판·사람 판정·다중 드론 폐루프에 연결했다. 기존 SGBM 기본 동작을 보존했다. **32회 주 비교와 2회 1Hz 대조를 완료**했다.

무연기 두 정책 모두 유효 경로 **16/16**, 올바른 차단 **3/12**다. entry는 약 저하에서 차단 하나를 더 확인했지만, 중에서 유효 경로 하나·강에서 차단 하나를 덜 확인했다. **일관된 entry 우위는 확인되지 않았다.** 주 실행의 무효 경로·충돌은 0이며 미확인과 계획 실패를 포함해 보고했다.

이는 **RealSense D455급 공개 사양을 참고한 낙관적 깊이 관측 모델과 통제된 관측 저하 조건**의 관측 배정 실험이다. 실물 센서·실화재 검증이나 SGBM 제거만의 독립 효과가 아니다.

- [전체 결과·분모·비용·한계](reports/depth_model_20261106/RESULTS.md)
- [센서 수식·공식 사양과 가정](reports/depth_model_20261106/SENSOR_MODEL.md)
- [재실행·깊이 캐시 재생](reports/depth_model_20261106/REPRODUCE.md)
- [방식별 집계](reports/depth_model_20261106/data/summary_by_method.csv)
- [목표별 판정·제시 경로](reports/depth_model_20261106/data/targets_final.csv)

새 원자료: `results/depth_model_20261106/`. 이전 v4는 [RESULTS_closed_loop.md](RESULTS_closed_loop.md)에 유지했다.
