# foodcourt-cross-eval

[foodcourt-adaptive-sim](https://github.com/phj9187/foodcourt-adaptive-sim)에서 이미 학습·시험이 끝난 AI를 **다시 학습하지 않고** 모든 점수표로 다시 채점합니다.

## 왜 필요한가
각 AI는 자기 점수표(λ, q, a, b, c)로만 채점됐습니다. 그래서 "가중치가 방향(선호)을 바꾼 것"과 "AI가 더 잘 배운 것"이 섞여 있습니다.
같은 기준으로 모두 다시 채점하면 둘을 가를 수 있습니다.

- 평가 점수표 k에서 k로 학습한 AI가 1등이면 → 차이는 **선호 방향**입니다.
- 어떤 점수표로 학습한 AI가 **여러 평가 기준에서 두루** 앞서면 → 그 가중치는 **더 잘 배우는 보상**입니다.

## 어떻게 가능한가
`paired_test.json`에 본시험의 하루 기록마다 5개 지표(맛, 가격 부담, 대기, 마진, 폐기)가 "추천 없음"과 "AI" 모두 저장돼 있습니다.
점수표는 이 5개의 가중합이라서(`kiosk_adaptive.design.Objective.score`) 어떤 점수표로든 정확히 다시 계산됩니다.

## 실행 (Colab)
`notebooks/cross_eval_colab.ipynb`를 Colab에서 열고 위에서부터 실행합니다. **CPU 고용량 RAM** 런타임을 쓰세요.
GPU는 필요 없습니다. 시간 대부분이 Drive에서 큰 JSON을 읽는 일입니다.

1. `python -m cross_eval.extract <실험 폴더> <캐시 폴더>`: AI마다 필요한 값만 작은 `.npz`로 뽑습니다(병렬, 캐시).
2. `python -m cross_eval.analyze <캐시 폴더> <결과 폴더>`: 재채점하고 요약합니다.

## 검산
- 추출할 때 AI마다 자기 점수표로 다시 계산한 ΔF가 저장된 `delta`와 같은지 확인합니다(`recheck_max_abs_err`).
- 분석할 때 자기 점수표의 푸드코트 평균 ΔF가 `summary.json`의 `mean_delta`와 같은지 확인합니다.

## 결과 파일
| 파일 | 내용 |
| --- | --- |
| `cross_matrix_<split>.csv` | 학습 점수표 × 평가 점수표, 반복 평균 ΔF |
| `cross_se_<split>.csv` | 반복 간 표준오차 |
| `self_vs_best_<split>.csv` | 평가 점수표마다 자기 점수표로 학습한 AI의 순위 |
| `generalist_<split>.csv` | 학습 점수표마다 여러 기준에서의 강도 |
| `common_metrics_<split>.csv` | 공통 지표 변화(5개 지표, 원 단위 이익)와 파레토 여부 |
| `heatmap_<split>.png` | 열마다 최대값으로 나눈 교차 행렬 그림 |
| `checks.json` | 검산 결과 |

`split`: `stable`은 메뉴가 그대로인 시험, `changed`는 중간에 메뉴가 바뀌는 시험입니다.

## 한계
- 단순 규칙은 하루 기록을 저장하지 않아서 교차 평가에서 빠집니다.
- 불확실성 단위는 독립 학습된 AI(반복)입니다. 점수표마다 반복은 5개뿐입니다.
- 서로 다른 실험(예: extended_main과 extended_taste)은 환경이 달라 섞지 않습니다.
