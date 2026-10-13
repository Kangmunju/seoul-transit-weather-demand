import json
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

from _common import load_merged

import features as F


BASE_DIR = Path(__file__).resolve().parents[1]
OUTPUT_PATH = BASE_DIR / "models" / "fixed_test_split_results.json"

SEED = 42

TEST_START = "2022-01-01"
TEST_END = "2022-12-31"


def create_model():
    # 기존 train.py와 동일한 Random Forest 설정
    return RandomForestRegressor(
        n_estimators=400,
        random_state=SEED,
        min_samples_leaf=2,
        n_jobs=-1,
    )


def evaluate(y, pred):
    # 실제값이 0인 날짜는 MAPE 계산에서 제외
    y = np.asarray(y, dtype=float)
    pred = np.asarray(pred, dtype=float)

    valid = y != 0

    return {
        "MAE": round(float(mean_absolute_error(y, pred)), 1),
        "MAPE_%": round(
            float(np.mean(np.abs((y[valid] - pred[valid]) / y[valid])) * 100),
            2,
        ),
        "R2": round(float(r2_score(y, pred)), 4),
    }


def fit_evaluate(df, train_idx, test_idx, features):
    X_train = df.iloc[train_idx][features]
    y_train = df.iloc[train_idx][F.TARGET]

    X_test = df.iloc[test_idx][features]
    y_test = df.iloc[test_idx][F.TARGET]

    model = create_model()
    model.fit(X_train, y_train)

    pred = model.predict(X_test)

    return evaluate(y_test, pred)


def main():
    merged, _, _ = load_merged()

    # 기존 프로젝트의 피처 생성 로직 재사용
    df = F.build(merged)

    features = F.feature_columns(include_pandemic=True)

    df = (
        df.dropna(subset=features + [F.TARGET])
        .sort_values("date")
        .reset_index(drop=True)
    )

    dates = pd.to_datetime(df["date"])

    test_start = pd.Timestamp(TEST_START)
    test_end = pd.Timestamp(TEST_END)

    # 2022년 전체를 동일한 테스트 기간으로 고정
    test_idx = np.flatnonzero(((dates >= test_start) & (dates <= test_end)).to_numpy())

    # 실제 미래 예측 방식: 테스트 시작일 이전만 학습
    past_idx = np.flatnonzero((dates < test_start).to_numpy())

    # 진단용 혼합 방식: 테스트 기간을 제외한 모든 날짜
    candidate_idx = np.flatnonzero(
        ((dates < test_start) | (dates > test_end)).to_numpy()
    )

    if len(test_idx) == 0 or len(past_idx) == 0:
        raise ValueError("학습 또는 테스트 데이터가 없습니다.")

    if len(candidate_idx) < len(past_idx):
        raise ValueError("랜덤 혼합 학습 후보가 부족합니다.")

    # 두 학습셋의 크기를 동일하게 유지
    rng = np.random.RandomState(SEED)

    mixed_idx = np.sort(
        rng.choice(
            candidate_idx,
            size=len(past_idx),
            replace=False,
        )
    )

    future_count = int((dates.iloc[mixed_idx] > test_end).sum())

    print("=" * 70)
    print("1. 실험 데이터 구성")
    print("=" * 70)

    print(f"전체 데이터: {len(df):,}일")
    print(f"피처: {len(features)}개")
    print(f"고정 테스트: {TEST_START} ~ {TEST_END}")
    print(f"테스트 날짜: {len(test_idx)}일")

    print("\n[시간순 학습]")
    print(f"학습 날짜: {len(past_idx)}일")
    print(
        f"학습 기간: "
        f"{dates.iloc[past_idx[0]].date()} ~ "
        f"{dates.iloc[past_idx[-1]].date()}"
    )

    print("\n[랜덤 혼합 학습]")
    print(f"학습 날짜: {len(mixed_idx)}일")
    print(f"테스트 이후 미래 날짜 포함: {future_count}일")

    print("\n" + "=" * 70)
    print("2. 동일 테스트 기간에서 모델 비교")
    print("=" * 70)

    time_result = fit_evaluate(df, past_idx, test_idx, features)

    mixed_result = fit_evaluate(df, mixed_idx, test_idx, features)

    print("\n[시간순 학습 → 2022년 테스트]")
    print(time_result)

    print("\n[과거·미래 혼합 학습 → 2022년 테스트]")
    print(mixed_result)

    mape_difference = round(
        time_result["MAPE_%"] - mixed_result["MAPE_%"],
        2,
    )

    print(f"\n혼합 학습의 MAPE 개선: {mape_difference:+.2f}%p")

    print("\n" + "=" * 70)
    print("3. 결과 해석 시 주의사항")
    print("=" * 70)

    print("1. 두 모델의 테스트 날짜와 학습 데이터 개수는 동일함.")
    print("2. 랜덤 혼합 모델에는 테스트 기간 이후의 미래 데이터가 포함됨.")
    print("3. 따라서 혼합 모델의 성능은 실제 미래 예측 성능으로 사용할 수 없음.")
    print(
        "4. lag/rolling 피처는 과거 실측 이용량을 사용하므로 "
        "일별 실측값이 순차적으로 확보되는 평가를 가정함."
    )
    print(
        "5. 이 실험은 학습 데이터의 시점 구성이 달라지는 효과를 "
        "보여주며, 누수의 크기를 독립적으로 추정하지는 못함."
    )

    output = {
        "experiment": "fixed_test_split",
        "seed": SEED,
        "test_start": TEST_START,
        "test_end": TEST_END,
        "test_days": len(test_idx),
        "features": features,
        "train_days_each": len(past_idx),
        "mixed_future_days": future_count,
        "time_split": time_result,
        "mixed_split": mixed_result,
        "mixed_mape_improvement_pp": mape_difference,
        "interpretation": (
            "동일한 테스트 기간과 학습 데이터 개수에서 "
            "과거 학습과 과거·미래 혼합 학습을 비교한 진단 실험. "
            "혼합 학습은 미래 정보를 허용하므로 "
            "실제 미래 예측 성능으로 해석할 수 없음."
        ),
    }

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        OUTPUT_PATH,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            output,
            file,
            ensure_ascii=False,
            indent=2,
        )

    print("\n" + "=" * 70)
    print("4. 분석 결과 저장")
    print("=" * 70)
    print(f"[ok ] {OUTPUT_PATH}")


if __name__ == "__main__":
    main()


# ============================================================
# [구현 핵심]
# ============================================================

# 2022년 전체를 고정 테스트 기간으로 설정함
# 두 모델의 테스트 날짜와 학습 데이터 개수를 동일하게 유지함
# 시간순 학습과 과거·미래 혼합 학습의 성능을 비교함
#
# [발견한 문제 및 수정]
# 1. 기존 Random Split과 Time Split은 테스트 날짜가 달랐음
#    → 2022년을 공통 테스트 기간으로 고정함
#
# 2. 학습 데이터 크기가 다르면 성능 차이의 원인을 구분하기 어려움
#    → 두 학습셋의 날짜 개수를 동일하게 맞춤
#
# 3. 랜덤 학습에 미래 데이터가 포함되는 문제가 있었음
#    → 미래 데이터 포함 개수를 출력하고 진단 실험임을 명시함
#
# 4. 시계열 피처가 실측 이용량을 사용한다는 전제가 불명확했음
#    → 과거 실측값이 순차적으로 확보되는 평가 조건을 명시함
#
# [추후 보완 사항]
# 1. 동일 테스트 기간 비교 결과를 분석 보고서에 반영
# 2. 실측 lag 피처를 사용한 일별 예측과 다일 예측을 구분
# 3. 시계열 누수 방지 원칙을 README에 명시
