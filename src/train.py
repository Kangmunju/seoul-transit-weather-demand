import json
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

import features as F
from clean import build_clean_data


# 프로젝트 최상위 폴더
BASE_DIR = Path(__file__).resolve().parents[1]

# SQLite 데이터베이스 파일
DB_PATH = BASE_DIR / "data" / "seoul_transit.db"

# 모델 결과 저장 폴더
MODEL_DIR = BASE_DIR / "models"

# 랜덤포레스트 재현성을 위한 시드
SEED = 42


def load_data():
    # SQLite DB 연결
    con = sqlite3.connect(DB_PATH)

    try:
        # 지하철 데이터 불러오기
        subway = pd.read_sql_query(
            """
            SELECT
                date,
                line,
                station,
                io_type,
                total
            FROM subway_daily
            """,
            con,
        )

        # 날씨 데이터 불러오기
        weather = pd.read_sql_query(
            """
            SELECT
                date,
                temp_mean,
                temp_max,
                temp_min,
                precip_mm,
                rain_mm,
                snow_cm,
                wind_max,
                humidity
            FROM weather_daily
            """,
            con,
        )

    finally:
        # 중간에 오류가 나도 DB 연결은 닫음
        con.close()

    return subway, weather


def mape(y, pred):
    y = np.asarray(y, dtype=float)  # 실제 이옹객 수
    pred = np.asarray(pred, dtype=float)  # 모델이 예측한 이용객 수

    # 실제값이 0인 경우 0으로 나누는 문제 방지
    valid = y != 0

    return float(np.mean(np.abs((y[valid] - pred[valid]) / y[valid])) * 100)


def evaluate(y, pred):
    # 여러 평가 지표를 한 번에 계산
    return {
        "MAE": round(float(mean_absolute_error(y, pred)), 1),
        "RMSE": round(float(np.sqrt(mean_squared_error(y, pred))), 1),
        "MAPE_%": round(mape(y, pred), 2),
        "R2": round(float(r2_score(y, pred)), 4),
    }


def main():
    # 모델 결과 저장 폴더가 없으면 생성
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    # DB에서 지하철과 날씨 데이터 불러오기
    subway, weather = load_data()

    # 기존 정제 파이프라인 적용
    merged, _, _ = build_clean_data(subway, weather)

    # lag, rolling 등 시계열 피처 생성
    d = F.build(merged)

    # 여러 연도의 데이터인지 확인
    n_years = int(d["date"].dt.year.nunique())
    multiyear = n_years >= 2

    # 모델에서 사용할 피처 목록 생성
    feat = F.feature_columns(include_pandemic=multiyear)

    # lag와 rolling 생성 과정에서 생긴 결측 제거
    d = d.dropna(subset=feat + [F.TARGET]).reset_index(drop=True)

    # 현재 lag_7 등의 이유로 인해 초반 7일 정도 제외
    # 2,185일이 실제 학습 가능한 데이터가 됨

    # 모델 입력값과 정답 분리
    X = d[feat].values  # 모델이 예측할 때 참고하는 입력값
    y = d[F.TARGET].values.astype(float)  # 모델이 맞혀야 하는 정답

    # 시계열 데이터이므로 랜덤하게 섞지 않고
    # 앞 75%를 학습, 뒤 25%를 테스트로 사용
    cut = int(len(d) * 0.75)

    Xtr = X[:cut]
    Xte = X[cut:]
    ytr = y[:cut]
    yte = y[cut:]

    # 지난주 같은 요일의 이용객 수를 기본 예측값으로 사용
    # 본격적인 머신러닝 모델을 학습하기 전에 베이스라인 생성
    lag7_idx = feat.index("lag_7")
    baseline_pred = Xte[:, lag7_idx]
    baseline_result = evaluate(yte, baseline_pred)

    # 선형회귀 모델
    linear = LinearRegression()
    linear.fit(Xtr, ytr)

    linear_pred = linear.predict(Xte)
    linear_result = evaluate(yte, linear_pred)

    # 랜덤포레스트 모델
    # 이번 프로젝트에서 구하고자하는 ride는 연속적인 숫자값
    # 분류가 아닌 회귀(RandomForestRegressor)로 접근
    rf = RandomForestRegressor(
        n_estimators=400,
        random_state=SEED,
        min_samples_leaf=2,
        # 너무 사소한 한 건짜리 패턴까지는 학습하지 않도록 방지함
        n_jobs=-1,
    )
    rf.fit(Xtr, ytr)

    rf_pred = rf.predict(Xte)
    rf_result = evaluate(yte, rf_pred)

    # 모델별 성능 정리
    metrics = {
        "baseline_lag7": baseline_result,
        "linear_regression": linear_result,
        "random_forest": rf_result,
    }

    # 성능 결과 저장
    with open(MODEL_DIR / "metrics.json", "w", encoding="utf-8") as file:
        json.dump(metrics, file, ensure_ascii=False, indent=2)

    # 랜덤포레스트 변수 중요도 정리
    importance = (
        pd.DataFrame(
            {
                "feature": feat,
                "importance": rf.feature_importances_,
            }
        )
        .sort_values(
            "importance",
            ascending=False,
        )
        .reset_index(drop=True)
    )

    # 변수 중요도 저장
    importance.to_csv(
        MODEL_DIR / "feature_importance.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # 앞부분 metrics.json에는 모델의 전체적인 성능을 요약한 숫자만 저장하였음
    # 특정 날짜에 실제로 몇 명이었고 모델이 몇 명으로 예측했는지 파악 불가
    # 테스트 기간 실제값과 모델별 예측값 정리해 DataFrame으로 다시 저장
    # 실제값 / 단순 기준 예측 / 선형회귀 예측 / 랜덤포레스트 예측을 모두 직접 비교 가능하도록 구현함
    predictions = pd.DataFrame(
        {
            "date": d.loc[cut:, "date"].reset_index(drop=True),
            "actual": yte,
            "baseline_lag7": baseline_pred,
            "linear_regression": linear_pred,
            "random_forest": rf_pred,
        }
    )

    # 예측 결과 저장
    predictions.to_csv(
        MODEL_DIR / "test_predictions.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # 실행 결과 확인
    print(f"전체 학습 가능 데이터: {len(d):,}일")
    print(f"학습 기간: {d.loc[0, 'date'].date()} ~ {d.loc[cut - 1, 'date'].date()}")
    print(
        f"테스트 기간: {d.loc[cut, 'date'].date()} ~ {d.loc[len(d) - 1, 'date'].date()}"
    )

    print("\n[모델 성능]")
    for model_name, result in metrics.items():
        print(model_name, result)

    print("\n[랜덤포레스트 변수 중요도 상위 10개]")
    print(importance.head(10).to_string(index=False))

    print("\n모델 결과 저장 완료")
    print(f"- {MODEL_DIR / 'metrics.json'}")
    print(f"- {MODEL_DIR / 'feature_importance.csv'}")
    print(f"- {MODEL_DIR / 'test_predictions.csv'}")


if __name__ == "__main__":
    main()


# 전체 학습 가능 데이터: 2,185일
# 학습 기간: 2019-01-08 ~ 2023-07-03
# 테스트 기간: 2023-07-04 ~ 2024-12-31

# [모델 성능]
# baseline_lag7 {'MAE': 1042988.8, 'RMSE': 2243096.7, 'MAPE_%': 9.42, 'R2': 0.4075}
# linear_regression {'MAE': 1191487.5, 'RMSE': 2010003.3, 'MAPE_%': 10.49, 'R2': 0.5243}
# random_forest {'MAE': 347269.3, 'RMSE': 545907.3, 'MAPE_%': 2.99, 'R2': 0.9649}

# [랜덤포레스트 변수 중요도 상위 10개]
#           feature  importance
#        is_workday    0.598021
#             lag_1    0.164595
#    pandemic_acute    0.098174
#        roll7_mean    0.071761
#       roll28_mean    0.031430
#             lag_7    0.015822
#         roll7_std    0.002588
# pandemic_recovery    0.002078
#        is_holiday    0.001946
#          wind_max    0.001477

# 모델 결과 저장 완료
# - C:\Users\swrkd\Desktop\subway-demand\models\metrics.json
# - C:\Users\swrkd\Desktop\subway-demand\models\feature_importance.csv
# - C:\Users\swrkd\Desktop\subway-demand\models\test_predictions.csv


# ============================================================

# SQLite DB에서 지하철과 날씨 데이터를 불러옴
# clean.py의 정제 파이프라인과 features.py의 시계열 피처 생성 로직을 재사용
# 이를 이용하여 모델 학습용 데이터를 구성하였음

# 다년간의 데이터인지를 확인
# 다년 데이터인 경우 pandemic_acute, pandemic_recovery 피처를 포함하도록 구성함
# lag_1, lag_7, rolling 피처 생성 과정에서 결측값이 생기는 것을 확인함
# 이러한 문제를 실제 모델 입력 피처와 정답 ride에 필요한 행만 기준으로 제거하여 해결함

# 시계열 데이터 미래 정보가 학습에 섞이지 않도록 주의하며 구현
# 랜덤 셔플하지 않고 앞 75% 학습, 뒤 25%를 테스트 데이터로 분리
# 실제 학습 가능 데이터 = 총 2,185일
# 학습 기간 = 2019-01-08 ~ 2023-07-03
# 테스트 기간 = 2023-07-04 ~ 2024-12-31

# 본격적인 머신러닝 모델과 비교할 기준(Baseline)이 필요하다고 판단하였음
# 지난주 같은 요일의 이용객 수인 lag_7을 베이스라인 예측값으로 사용
# 이후 LinearRegression과 RandomForestRegressor를 같은 테스트 구간에서 비교함

# LinearRegression : 여러 피처와 이용객 수 사이의 기본적인 선형 관계를 확인하기 위해 사용
# RandomForestRegressor : 평일 여부, 과거 수요, 날씨 등 여러 피처 사이의 비선형 관계까지 파악하기 위해 사용
# random_state를 고정해 같은 데이터에서 결과를 재현할 수 있도록 구현함
# min_sample_leaf=2로 너무 작은 패턴까지 과도하게 학습하는 것을 일부 방지함

# 모델 성능은 MAE, RMSE, MAPE, R@를 함께 계산하여 비교
# 실제 실행 결과 RandomForest가 가장 좋은 성능을 보이는 것을 확인함
# 그 외 MAPE = 2.99%, R2 = 0.9649의 수치를 확인
# lag_7 베이스라인과 LinearRegression보다 테스트 구간 예측 성능이 크게 개선됨

# 랜덤 포레스트 변수 중요도에서는 is_workday가 가장 높음을 확인
# lag_1, pandemic_acute, roll7_mean, roll28_mean 순으로 높은 중요도를 보이는 것을 확인
# 현재 결과에서는 날씨 변수보다 평일 여부, 최근 수요 흐름이 지하철 일별 수요 예측에 더 크게 활용된 것으로 나타남
# cf) feature_importances_는 예측 과정에서의 중요도를 나타내는 값
#     해당 변수가 실제 수요 원인이라는 인과관계로 해석하지 않았음

# metrics.json : 모델이 얼마나 잘 맞췄는지 확인
# feature_importance.csv : 랜덤포레스트가 어떤 피처를 중요하게 봤는지 확인
# test_predictions.csv : 날짜별 실제값과 각 모델의 예측값이 어떻게 달랐는지 확인

# 새 결과 파일을 따로 저장함
# 전체 성능 비교, 변수 중요도 분석, 날짜별 실제값/예측값 비교를 각각 독립적으로 확인할 수 있도록 구성하였음

# if __name__ == "__main__":
# train.py를 직접 실행할 때만 모델 학습이 수행되도록 함
# 다른 파일에서 함수만 import할 때는 자동 실행되지 않도록 구현하였음
