import json
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import TimeSeriesSplit

from _common import load_merged

import features as F


# 프로젝트 최상위 폴더
BASE_DIR = Path(__file__).resolve().parents[1]

# 분석 결과 저장 폴더
MODEL_DIR = BASE_DIR / "models"

# 기존 train.py와 동일한 랜덤 시드
SEED = 42

# 호우일 기준: 일 강우량 30mm 이상
HEAVY_RAIN_MM = 30.0


def mape(y, pred):
    # 실제값과 예측값을 numpy 배열로 변환
    y = np.asarray(y, dtype=float)
    pred = np.asarray(pred, dtype=float)

    # 실제값이 0인 경우 0으로 나누지 않도록 처리
    valid = y != 0

    if not valid.any():
        return float("nan")

    return float(np.mean(np.abs((y[valid] - pred[valid]) / y[valid])) * 100)


def evaluate(y, pred):
    # 기존 모델과 동일한 성능 지표 사용
    return {
        "MAE": round(float(mean_absolute_error(y, pred)), 1),
        "MAPE_%": round(mape(y, pred), 2),
        "R2": round(float(r2_score(y, pred)), 4),
    }


def create_model():
    # train.py와 동일한 Random Forest 설정
    return RandomForestRegressor(
        n_estimators=400,
        random_state=SEED,
        min_samples_leaf=2,
        n_jobs=-1,
    )


def fit_predict(Xtr, ytr, Xte):
    # 모델 생성 및 학습
    model = create_model()
    model.fit(Xtr, ytr)

    # 테스트 데이터 예측
    pred = model.predict(Xte)

    return model, pred


def compare_models(d, train_idx, test_idx, full_features, no_weather_features):
    # 같은 학습·테스트 날짜를 사용하고 피처 목록만 변경
    ytr = d.iloc[train_idx][F.TARGET].to_numpy(dtype=float)
    yte = d.iloc[test_idx][F.TARGET].to_numpy(dtype=float)

    # 날씨 포함 모델
    Xtr_full = d.iloc[train_idx][full_features]
    Xte_full = d.iloc[test_idx][full_features]

    full_model, full_pred = fit_predict(Xtr_full, ytr, Xte_full)

    # 날씨 제외 모델
    Xtr_no_weather = d.iloc[train_idx][no_weather_features]
    Xte_no_weather = d.iloc[test_idx][no_weather_features]

    _, no_weather_pred = fit_predict(Xtr_no_weather, ytr, Xte_no_weather)

    full_result = evaluate(yte, full_pred)
    no_weather_result = evaluate(yte, no_weather_pred)

    # 양수면 날씨를 포함한 모델의 MAPE가 더 낮다는 의미
    improvement = no_weather_result["MAPE_%"] - full_result["MAPE_%"]

    return {
        "full_model": full_model,
        "full_pred": full_pred,
        "no_weather_pred": no_weather_pred,
        "full_result": full_result,
        "no_weather_result": no_weather_result,
        "improvement": round(improvement, 2),
    }


def main():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    # 1. 기존 전처리 및 피처 생성 로직 재사용
    merged, _, _ = load_merged()

    d = F.build(merged)

    # 기존 train.py와 동일한 팬데믹 피처 사용 조건
    multiyear = d["date"].dt.year.nunique() >= 2

    # 전체 피처
    full_features = F.feature_columns(include_pandemic=multiyear)

    # 날씨 관련 피처만 제외
    no_weather_features = [
        feature for feature in full_features if feature not in F.WEATHER_FEATURES
    ]

    # 두 모델에 동일한 데이터 행을 사용
    d = (
        d.dropna(subset=full_features + [F.TARGET])
        .sort_values("date")
        .reset_index(drop=True)
    )

    print("=" * 70)
    print("1. 실험 데이터 및 피처 확인")
    print("=" * 70)

    print(f"전체 데이터: {len(d):,}일")
    print(f"전체 피처: {len(full_features)}개")
    print(f"날씨 피처: {len(F.WEATHER_FEATURES)}개")
    print(f"날씨 제외 피처: {len(no_weather_features)}개")

    print("\n[제외한 날씨 피처]")
    print(F.WEATHER_FEATURES)

    # 2. 기존 train.py와 동일한 시간순 75:25 분할
    cut = int(len(d) * 0.75)

    train_idx = np.arange(cut)
    test_idx = np.arange(cut, len(d))

    print("\n" + "=" * 70)
    print("2. 시간순 75:25 분할 비교")
    print("=" * 70)

    print(
        f"학습 기간: "
        f"{d.loc[train_idx[0], 'date'].date()} ~ "
        f"{d.loc[train_idx[-1], 'date'].date()}"
    )

    print(
        f"테스트 기간: "
        f"{d.loc[test_idx[0], 'date'].date()} ~ "
        f"{d.loc[test_idx[-1], 'date'].date()}"
    )

    result = compare_models(
        d,
        train_idx,
        test_idx,
        full_features,
        no_weather_features,
    )

    print("\n[날씨 포함]")
    print(result["full_result"])

    print("\n[날씨 제외]")
    print(result["no_weather_result"])

    print(f"\n날씨 포함에 따른 MAPE 개선: {result['improvement']:+.2f}%p")

    # 3. 호우일만 별도로 평가
    print("\n" + "=" * 70)
    print("3. 호우일 예측 성능 비교")
    print("=" * 70)

    test_data = d.iloc[test_idx].copy()

    # 총강수가 아닌 실제 강우량 사용
    heavy_mask = test_data["rain_mm"].to_numpy() >= HEAVY_RAIN_MM

    heavy_count = int(heavy_mask.sum())

    heavy_result = {
        "threshold_mm": HEAVY_RAIN_MM,
        "days": heavy_count,
    }

    print(f"호우 기준: 일 강우량 {HEAVY_RAIN_MM:g}mm 이상")
    print(f"호우일 수: {heavy_count}일")

    if heavy_count > 0:
        y_heavy = test_data.loc[heavy_mask, F.TARGET].to_numpy(dtype=float)

        full_heavy_pred = result["full_pred"][heavy_mask]
        no_weather_heavy_pred = result["no_weather_pred"][heavy_mask]

        full_heavy_mape = round(mape(y_heavy, full_heavy_pred), 2)

        no_weather_heavy_mape = round(mape(y_heavy, no_weather_heavy_pred), 2)

        heavy_improvement = round(no_weather_heavy_mape - full_heavy_mape, 2)

        heavy_result.update(
            {
                "with_weather_mape": full_heavy_mape,
                "without_weather_mape": no_weather_heavy_mape,
                "improvement_pp": heavy_improvement,
            }
        )

        print(f"날씨 포함 MAPE: {full_heavy_mape:.2f}%")
        print(f"날씨 제외 MAPE: {no_weather_heavy_mape:.2f}%")
        print(f"MAPE 개선: {heavy_improvement:+.2f}%p")

    else:
        print("해당 기준을 만족하는 테스트 날짜가 없음")

    # 4. Walk-forward 5구간 비교
    print("\n" + "=" * 70)
    print("4. Walk-forward 5-Fold 비교")
    print("=" * 70)

    tscv = TimeSeriesSplit(n_splits=5)

    fold_results = []

    for fold, (tr_idx, te_idx) in enumerate(tscv.split(d), start=1):
        fold_result = compare_models(
            d,
            tr_idx,
            te_idx,
            full_features,
            no_weather_features,
        )

        row = {
            "fold": fold,
            "train_days": len(tr_idx),
            "test_days": len(te_idx),
            "test_start": str(d.loc[te_idx[0], "date"].date()),
            "test_end": str(d.loc[te_idx[-1], "date"].date()),
            "with_weather_mape": fold_result["full_result"]["MAPE_%"],
            "without_weather_mape": fold_result["no_weather_result"]["MAPE_%"],
            "improvement_pp": fold_result["improvement"],
        }

        fold_results.append(row)

        print(
            f"fold{fold}: "
            f"날씨 포함 {row['with_weather_mape']:.2f}% / "
            f"날씨 제외 {row['without_weather_mape']:.2f}% / "
            f"개선 {row['improvement_pp']:+.2f}%p"
        )

    fold_df = pd.DataFrame(fold_results)

    helpful_folds = int((fold_df["improvement_pp"] > 0).sum())

    print(f"\n날씨가 성능 개선에 기여한 Fold: {helpful_folds}/5개")

    print(f"Fold별 MAPE 개선 평균: {fold_df['improvement_pp'].mean():+.2f}%p")

    # 5. 테스트셋 순열 중요도
    print("\n" + "=" * 70)
    print("5. 테스트셋 순열 중요도")
    print("=" * 70)

    # 학습에 사용하지 않은 테스트 데이터에서
    # 피처 값을 섞었을 때 MAE가 얼마나 증가하는지 확인
    #
    # 계산 시간을 줄이기 위해:
    # 반복 횟수 3회, 테스트 표본의 50% 사용
    #
    # 원본 feature_importances_와 달리
    # 실제 테스트셋 예측 성능의 변화를 기준으로 평가
    Xte_full = d.iloc[test_idx][full_features]
    yte = d.iloc[test_idx][F.TARGET].to_numpy(dtype=float)

    perm = permutation_importance(
        result["full_model"],
        Xte_full,
        yte,
        scoring="neg_mean_absolute_error",
        n_repeats=3,
        random_state=SEED,
        n_jobs=-1,
        max_samples=0.5,
    )

    perm_df = pd.DataFrame(
        {
            "feature": full_features,
            "mae_increase": perm.importances_mean,
            "std": perm.importances_std,
        }
    )

    # 날씨 피처만 별도로 정리
    weather_perm = (
        perm_df[perm_df["feature"].isin(F.WEATHER_FEATURES)]
        .sort_values("mae_increase", ascending=False)
        .reset_index(drop=True)
    )

    print("\n[날씨 피처 순열 중요도]")
    print(weather_perm.to_string(index=False))

    print(
        "\n※ 양수: 해당 피처를 섞었을 때 MAE 증가"
        "\n※ 음수: 해당 피처를 섞었을 때 MAE 감소"
        "\n※ 상관된 피처가 많으면 개별 중요도가 낮게 나올 수 있음"
    )

    # 6. 결과 저장
    print("\n" + "=" * 70)
    print("6. 분석 결과 저장")
    print("=" * 70)

    output = {
        "experiment": "weather_ablation",
        "seed": SEED,
        "model": "RandomForestRegressor",
        "n_estimators": 400,
        "min_samples_leaf": 2,
        "total_days": len(d),
        "weather_features": F.WEATHER_FEATURES,
        "full_features": full_features,
        "no_weather_features": no_weather_features,
        "split_comparison": {
            "train_start": str(d.loc[train_idx[0], "date"].date()),
            "train_end": str(d.loc[train_idx[-1], "date"].date()),
            "test_start": str(d.loc[test_idx[0], "date"].date()),
            "test_end": str(d.loc[test_idx[-1], "date"].date()),
            "with_weather": result["full_result"],
            "without_weather": result["no_weather_result"],
            "improvement_pp": result["improvement"],
        },
        "heavy_rain": heavy_result,
        "walk_forward": fold_results,
        "helpful_folds": helpful_folds,
        "mean_fold_improvement_pp": round(float(fold_df["improvement_pp"].mean()), 2),
    }

    json_path = MODEL_DIR / "weather_ablation_results.json"
    fold_path = MODEL_DIR / "weather_ablation_folds.csv"
    perm_path = MODEL_DIR / "weather_permutation_importance.csv"

    with open(json_path, "w", encoding="utf-8") as file:
        json.dump(
            output,
            file,
            ensure_ascii=False,
            indent=2,
        )

    fold_df.to_csv(
        fold_path,
        index=False,
        encoding="utf-8-sig",
    )

    weather_perm.to_csv(
        perm_path,
        index=False,
        encoding="utf-8-sig",
    )

    print(f"[ok ] {json_path}")
    print(f"[ok ] {fold_path}")
    print(f"[ok ] {perm_path}")


if __name__ == "__main__":
    main()


# ============================================================
# [구현 핵심]
# ============================================================

# 동일한 Random Forest 설정과 학습·테스트 기간에서
# 날씨 포함 모델과 날씨 제외 모델의 예측 성능을 비교함
# MAPE 차이를 계산하여 날씨 변수의 실제 기여도를 확인하도록 구현함
#
# Walk-forward 5-Fold 검증을 통해 시간 구간별 성능 차이를 비교하고,
# 30mm 이상 호우일의 예측 성능을 별도로 평가하도록 구현함
# 테스트셋 순열 중요도로 날씨 피처의 예측 기여도도 확인함
#
# [발견한 문제 및 수정]
# 1. 기존 변수 중요도만으로 날씨의 실제 예측 기여도를 판단하기 어려웠음
#    → 날씨 변수만 제외한 모델을 추가 학습하여 성능을 직접 비교함
#
# 2. 단일 테스트 구간의 결과만으로 효과를 일반화하기 어려웠음
#    → Walk-forward 5구간에서 날씨 포함·제외 성능을 반복 비교함
#
# 3. 전체 평균 성능만으로 호우일의 예측 차이를 확인하기 어려웠음
#    → 30mm 이상 호우일을 별도로 분리하여 MAPE를 계산함
#
# 4. 기존 feature_importances_는 학습 과정의 중요도만 나타냈음
#    → 테스트셋 순열 중요도를 추가하여 예측 성능 변화로 평가함
#
# [추후 보완 사항]
# 1. 실제 실험 결과를 README와 분석 보고서에 반영
# 2. 날씨 포함·제외 모델의 MAPE 차이를 대시보드에 시각화
# 3. 실측 날씨 대신 예보 날씨를 사용하는 실제 미래 예측 환경 검토
