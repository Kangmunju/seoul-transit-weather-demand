import argparse
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
from metrics_utils import mape


# 프로젝트 최상위 폴더
BASE_DIR = Path(__file__).resolve().parents[1]

# SQLite 데이터베이스 파일
DB_PATH = BASE_DIR / "data" / "seoul_transit.db"

# 모델 결과 저장 폴더
MODEL_DIR = BASE_DIR / "models"

# 랜덤포레스트 재현성을 위한 시드
SEED = 42

# 피드백 8번: 휴일 피처 비교 실험 결과 파일
HOLIDAY_COMPARISON_PATH = MODEL_DIR / "holiday_feature_comparison.json"
HOLIDAY_PREDICTIONS_PATH = MODEL_DIR / "holiday_feature_predictions.csv"
HOLIDAY_IMPORTANCE_PATH = MODEL_DIR / "holiday_feature_importance.csv"


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


def evaluate(y, pred):
    # 여러 평가 지표를 한 번에 계산
    return {
        "MAE": round(float(mean_absolute_error(y, pred)), 1),
        "RMSE": round(float(np.sqrt(mean_squared_error(y, pred))), 1),
        "MAPE_%": round(mape(y, pred), 2),
        "R2": round(float(r2_score(y, pred)), 4),
    }


def prepare_data():
    # DB에서 지하철과 날씨 데이터 불러오기
    subway, weather = load_data()

    # 기존 정제 파이프라인 적용
    merged, _, _ = build_clean_data(subway, weather)

    # lag, rolling 등 시계열 피처 생성
    d = F.build(merged)

    # 여러 연도의 데이터인지 확인
    n_years = int(d["date"].dt.year.nunique())
    multiyear = n_years >= 2

    return d, multiyear


def create_random_forest():
    # 이번 프로젝트에서 구하고자 하는 ride는 연속적인 숫자값
    # 분류가 아닌 회귀(RandomForestRegressor)로 접근
    return RandomForestRegressor(
        n_estimators=400,
        random_state=SEED,
        min_samples_leaf=2,
        # 너무 사소한 한 건짜리 패턴까지는 학습하지 않도록 방지함
        n_jobs=-1,
    )


def create_importance(model, features):
    # 랜덤포레스트 변수 중요도 정리
    return (
        pd.DataFrame(
            {
                "feature": features,
                "importance": model.feature_importances_,
            }
        )
        .sort_values(
            "importance",
            ascending=False,
        )
        .reset_index(drop=True)
    )


def main():
    # 기존 모델 학습 기능
    # 신규 휴일 피처는 포함하지 않아 기존 결과와 비교 가능하도록 유지함

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    d, multiyear = prepare_data()

    # 기존 모델과 동일한 피처 목록 사용
    feat = F.feature_columns(
        include_pandemic=multiyear,
        include_holiday_features=False,
    )

    # lag와 rolling 생성 과정에서 생긴 결측 제거
    d = d.dropna(subset=feat + [F.TARGET]).reset_index(drop=True)

    # 현재 lag_7 등의 이유로 인해 초반 7일 정도 제외
    # 2,185일이 실제 학습 가능한 데이터가 됨

    # 모델 입력값과 정답 분리
    X = d[feat].values
    y = d[F.TARGET].values.astype(float)

    # 시계열 데이터이므로 랜덤하게 섞지 않고
    # 앞 75%를 학습, 뒤 25%를 테스트로 사용
    cut = int(len(d) * 0.75)

    Xtr = X[:cut]
    Xte = X[cut:]
    ytr = y[:cut]
    yte = y[cut:]

    # 지난주 같은 요일의 이용량을 기본 예측값으로 사용
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
    rf = create_random_forest()
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
    importance = create_importance(rf, feat)

    # 변수 중요도 저장
    importance.to_csv(
        MODEL_DIR / "feature_importance.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # 전체 성능만으로는 날짜별 예측 오차를 확인하기 어려움
    # 테스트 기간 실제값과 모델별 예측값을 별도로 저장
    predictions = pd.DataFrame(
        {
            "date": d.loc[cut:, "date"].reset_index(drop=True),
            "actual": yte,
            "baseline_lag7": baseline_pred,
            "linear_regression": linear_pred,
            "random_forest": rf_pred,
        }
    )

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


def evaluate_day_types(test_df, actual, pred):
    # 날짜 유형별 MAPE 계산
    # 공휴일이 주말과 겹치는 경우 중복 집계하지 않도록
    # 공휴일 → 주말 → 평일 순서로 분류함

    result = {}

    masks = {
        "workday": ((test_df["is_holiday"] == 0) & (test_df["is_weekend"] == 0)),
        "weekend": ((test_df["is_holiday"] == 0) & (test_df["is_weekend"] == 1)),
        "holiday": test_df["is_holiday"] == 1,
        "labor_day": test_df["is_labor_day"] == 1,
        "day_after_holiday": test_df["is_day_after_holiday"] == 1,
    }

    for name, mask in masks.items():
        mask_array = mask.to_numpy(dtype=bool)
        count = int(mask_array.sum())

        result[name] = {
            "days": count,
            "MAPE_%": (
                round(mape(actual[mask_array], pred[mask_array]), 2)
                if count > 0
                else None
            ),
        }

    return result


def compare_holiday_features():
    # 피드백 8번: 휴일 피처 추가 전후의 예측 성능 비교
    # 기존 metrics.json 및 test_predictions.csv는 수정하지 않음

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    d, multiyear = prepare_data()

    # 기존 모델 피처
    old_features = F.feature_columns(
        include_pandemic=multiyear,
        include_holiday_features=False,
    )

    # 휴일 보강 모델 피처
    new_features = F.feature_columns(
        include_pandemic=multiyear,
        include_holiday_features=True,
    )

    # 두 모델이 완전히 동일한 날짜를 사용하도록
    # 모든 피처를 기준으로 결측값을 한 번만 제거
    all_features = list(dict.fromkeys(old_features + new_features))

    d = d.dropna(subset=all_features + [F.TARGET]).reset_index(drop=True)

    # 기존 학습과 동일한 75:25 시계열 분할
    cut = int(len(d) * 0.75)

    train_df = d.iloc[:cut]
    test_df = d.iloc[cut:].reset_index(drop=True)

    ytr = train_df[F.TARGET].to_numpy(dtype=float)
    yte = test_df[F.TARGET].to_numpy(dtype=float)

    # 기존 모델 학습
    old_model = create_random_forest()

    old_model.fit(
        train_df[old_features].to_numpy(),
        ytr,
    )

    old_pred = old_model.predict(test_df[old_features].to_numpy())

    # 휴일 보강 모델 학습
    new_model = create_random_forest()

    new_model.fit(
        train_df[new_features].to_numpy(),
        ytr,
    )

    new_pred = new_model.predict(test_df[new_features].to_numpy())

    # 전체 성능 비교
    old_result = evaluate(yte, old_pred)
    new_result = evaluate(yte, new_pred)

    # 날짜 유형별 성능 비교
    old_day_types = evaluate_day_types(test_df, yte, old_pred)
    new_day_types = evaluate_day_types(test_df, yte, new_pred)

    # MAPE는 낮을수록 좋음
    # 양수이면 신규 모델의 오차가 감소한 것
    mape_improvement = round(
        old_result["MAPE_%"] - new_result["MAPE_%"],
        2,
    )

    # 2024-05-01 근로자의 날 예측 오차 확인
    target_date = pd.Timestamp("2024-05-01")

    target_mask = test_df["date"] == target_date

    labor_day_2024 = None

    if target_mask.any():
        idx = int(np.flatnonzero(target_mask.to_numpy())[0])

        actual = float(yte[idx])
        old_value = float(old_pred[idx])
        new_value = float(new_pred[idx])

        labor_day_2024 = {
            "date": "2024-05-01",
            "actual": round(actual, 1),
            "old_prediction": round(old_value, 1),
            "new_prediction": round(new_value, 1),
            "old_APE_%": round(abs(actual - old_value) / actual * 100, 2)
            if actual != 0
            else None,
            "new_APE_%": round(abs(actual - new_value) / actual * 100, 2)
            if actual != 0
            else None,
        }

    # 결과 JSON 구성
    comparison = {
        "experiment": "holiday_feature_comparison",
        "seed": SEED,
        "train_ratio": 0.75,
        "test_ratio": 0.25,
        "train_start": str(train_df["date"].iloc[0].date()),
        "train_end": str(train_df["date"].iloc[-1].date()),
        "test_start": str(test_df["date"].iloc[0].date()),
        "test_end": str(test_df["date"].iloc[-1].date()),
        "train_days": len(train_df),
        "test_days": len(test_df),
        "old_feature_count": len(old_features),
        "new_feature_count": len(new_features),
        "added_features": F.HOLIDAY_FEATURES,
        "old_model": {
            "overall": old_result,
            "by_day_type": old_day_types,
        },
        "new_model": {
            "overall": new_result,
            "by_day_type": new_day_types,
        },
        "MAPE_improvement_pp": mape_improvement,
        "labor_day_2024": labor_day_2024,
    }

    # 기존 모델 결과와 구분하여 별도 JSON 저장
    with open(
        HOLIDAY_COMPARISON_PATH,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(comparison, file, ensure_ascii=False, indent=2)

    # 날짜별 실제값과 두 모델의 예측값 비교
    predictions = pd.DataFrame(
        {
            "date": test_df["date"],
            "actual": yte,
            "old_prediction": old_pred,
            "new_prediction": new_pred,
            "is_weekend": test_df["is_weekend"],
            "is_holiday": test_df["is_holiday"],
            "is_labor_day": test_df["is_labor_day"],
            "is_day_after_holiday": test_df["is_day_after_holiday"],
            "holiday_streak_length": test_df["holiday_streak_length"],
        }
    )

    # 날짜별 절대백분율오차(APE)
    # 실제값이 0이면 0으로 나누지 않도록 NaN 처리
    predictions["old_APE_%"] = np.where(
        predictions["actual"] != 0,
        abs(predictions["actual"] - predictions["old_prediction"])
        / predictions["actual"].replace(0, np.nan)
        * 100,
        np.nan,
    )

    predictions["new_APE_%"] = np.where(
        predictions["actual"] != 0,
        abs(predictions["actual"] - predictions["new_prediction"])
        / predictions["actual"].replace(0, np.nan)
        * 100,
        np.nan,
    )

    predictions.to_csv(
        HOLIDAY_PREDICTIONS_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    # 휴일 보강 모델 변수 중요도
    importance = create_importance(new_model, new_features)

    importance.to_csv(
        HOLIDAY_IMPORTANCE_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    # 실행 결과 출력
    print("\n[피드백 8번: 휴일 피처 비교 실험]")
    print(f"전체 학습 가능 데이터: {len(d):,}일")
    print(
        f"학습 기간: {train_df['date'].iloc[0].date()} "
        f"~ {train_df['date'].iloc[-1].date()}"
    )
    print(
        f"테스트 기간: {test_df['date'].iloc[0].date()} "
        f"~ {test_df['date'].iloc[-1].date()}"
    )

    print("\n[피처 구성]")
    print(f"기존 모델: {len(old_features)}개")
    print(f"보강 모델: {len(new_features)}개")
    print(f"추가 피처: {F.HOLIDAY_FEATURES}")

    print("\n[전체 성능 비교]")
    print("기존 모델:", old_result)
    print("보강 모델:", new_result)
    print(f"MAPE 개선 폭: {mape_improvement:+.2f}%p")

    print("\n[날짜 유형별 MAPE]")
    for day_type in old_day_types:
        old_info = old_day_types[day_type]
        new_info = new_day_types[day_type]

        print(
            f"{day_type}: "
            f"{old_info['days']}일 / "
            f"기존 {old_info['MAPE_%']}% / "
            f"보강 {new_info['MAPE_%']}%"
        )

    print("\n[2024-05-01 근로자의 날]")
    if labor_day_2024 is not None:
        print(labor_day_2024)
    else:
        print("테스트 기간에 해당 날짜가 없습니다.")

    print("\n[보강 모델 변수 중요도 상위 10개]")
    print(importance.head(10).to_string(index=False))

    print("\n비교 실험 결과 저장 완료")
    print(f"- {HOLIDAY_COMPARISON_PATH}")
    print(f"- {HOLIDAY_PREDICTIONS_PATH}")
    print(f"- {HOLIDAY_IMPORTANCE_PATH}")


if __name__ == "__main__":
    # --compare-holidays 옵션을 지정하면 휴일 피처 비교 실험만 수행
    # 옵션이 없으면 기존 모델 학습 기능을 실행함
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--compare-holidays",
        action="store_true",
        help="기존 모델과 휴일 보강 모델의 성능 비교",
    )

    args = parser.parse_args()

    if args.compare_holidays:
        compare_holiday_features()
    else:
        main()


# 기존 모델 실행 결과(휴일 피처 보강 전)
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
# [구현 핵심]
# ============================================================

# SQLite DB에서 지하철과 날씨 데이터를 불러와 기존 정제 파이프라인을 재사용함
# 시계열 피처 생성 후 앞 75% 학습, 뒤 25% 테스트로 분리함
# lag_7 베이스라인, 선형회귀, 랜덤포레스트 모델의 성능을 비교함
# MAE, RMSE, MAPE, R2를 계산하고 모델별 결과를 저장함
# 랜덤포레스트 변수 중요도와 날짜별 실제값·예측값을 별도 파일로 저장함
# 휴일 피처 보강 전후의 랜덤포레스트 성능을 동일한 조건에서 비교하도록 구현함
# 전체 성능과 날짜 유형별 MAPE, 2024-05-01 예측 오차를 함께 확인하도록 구성함
#
# [발견한 문제 및 수정]
# 1. 기존 모델은 근로자의 날과 연속 휴일 특성을 입력 피처로 사용하지 않았음
#    → 휴일 보강 피처 3개를 포함한 랜덤포레스트 모델을 추가로 학습함
#
# 2. 신규 모델만 학습하면 기존 모델과 공정하게 성능을 비교하기 어려움
#    → 동일한 날짜, 학습·테스트 분할, 모델 설정, 랜덤 시드를 사용함
#
# 3. 전체 MAPE만으로는 휴일에 발생하는 큰 예측 오차를 확인하기 어려움
#    → 평일·주말·공휴일·근로자의 날·휴일 다음 날 MAPE를 별도로 계산함
#    → 2024-05-01의 실제값과 두 모델의 예측값 및 APE를 따로 저장함
#
# 4. 비교 실험 과정에서 기존 모델 결과가 덮어써질 수 있었음
#    → --compare-holidays 옵션을 추가하고 실험 결과를 별도 파일에 저장함
#
# [추후 보완 사항]
# 1. 휴일 보강 모델의 실제 성능 개선 여부를 확인하고 최종 반영 결정
# 2. 날짜 유형별 오차와 2024-05-01 예측 결과를 분석 보고서에 반영
# 3. 명절 연휴 전용 피처 및 연휴 종료 다음 날 효과 추가 검토
# 4. 휴일 피처의 중요도와 연도별 성능 안정성 검증
