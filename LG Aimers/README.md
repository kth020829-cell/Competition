# 🍽️ 메뉴별 수요 예측 모델 개발
**LG Aimers 7기 AI 해커톤**
곤지암 리조트 식음업장의 실제 판매 데이터를 기반으로, 메뉴별 향후 7일치 매출 수량을 예측하는 AI 모델 개발

<br>

## 결과

| 항목 | 내용 |
|------|------|
| **최종 순위** | 예선 상위 21% |

<br>

## 프로젝트 개요

최대 28일치 메뉴별 매출 시계열 데이터를 입력으로 받아, XGBoost 기반 다중 출력 회귀 모델로 향후 7일치 매출 수량을 동시에 예측합니다. Optuna를 활용한 하이퍼파라미터 자동 최적화와 정교한 피처 엔지니어링을 결합하여 예측 정확도를 극대화했습니다.

**평가 지표**: SMAPE · NMAE · NRMSE · Pearson R² 복합 산식 (실제 매출 0인 행 제외)

<br>

## 기술 스택

- **모델**: XGBoost (`XGBRegressor` + `MultiOutputRegressor`)
- **튜닝**: Optuna (100 trials)
- **주요 라이브러리**: pandas, numpy, statsmodels, scikit-learn

<br>

## 구현 내용

### 피처 엔지니어링

| 카테고리 | 피처 |
|----------|------|
| 주기성 인코딩 | 월/일/요일/연중일/주차 → 사인·코사인 변환 |
| 롤링 통계 | 3·7·14·28일 이동평균, 표준편차, 합산 |
| 래그 피처 | 1·3·7·14일 전 매출 수량 |
| STL 시계열 분해 | 7일 주기 trend / seasonal / residual 분리 |
| 강건 통계 | 7일 rolling Q10, Q90, IQR, z-score |
| 추세 피처 | EWM(span=7), 7일 rolling log-slope |
| 모멘텀 | 직전값 대비 7일 평균의 하락 비율 |
| 집계 통계 | 주간/월간 평균·표준편차·합산·최소·최대, 누적 매출 |

### 카테고리 인코딩

학습·테스트 데이터 전체의 값 합집합 기준으로 Label Encoding 처리하여 테스트 시 미등장 카테고리를 `-1`로 안전하게 처리합니다.

### Optuna 하이퍼파라미터 튜닝

점포별 가중 sMAPE를 목적함수로, 100회 trial에 걸쳐 12개 XGBoost 파라미터와 점포별 가중치(`w_담하`, `w_미라시아`)를 동시에 최적화합니다.

| 파라미터 | 탐색 범위 |
|----------|-----------|
| `n_estimators` | 300 ~ 1600 |
| `learning_rate` | 0.008 ~ 0.25 (log scale) |
| `max_depth` | 3 ~ 12 |
| `subsample` / `colsample_*` | 0.55 ~ 1.0 |
| `reg_alpha` / `reg_lambda` | 1e-4 ~ 20.0 (log scale) |
| `gamma`, `min_child_weight`, `max_bin` | 각 범위 탐색 |

<br>

## 실행 방법

```bash
pip install xgboost optuna scikit-learn pandas numpy statsmodels tqdm holidays
jupyter notebook project_xgb.ipynb
```

실행 순서: 데이터 로드 → 피처 엔지니어링 → 카테고리 인코딩 → Optuna 튜닝 → 최종 모델 학습 → 예측 및 제출 파일 생성
