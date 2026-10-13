import json
from pathlib import Path

import numpy as np
import pandas as pd

from _common import load_merged


# 프로젝트 최상위 폴더
BASE_DIR = Path(__file__).resolve().parents[1]

# 분석 결과 저장 경로
OUTPUT_PATH = BASE_DIR / "models" / "rain_adjusted_results.json"


def compare_means(df):
    # 비 오는 날과 비가 오지 않는 날의 평균 이용량 비교
    rain = df[df["is_rain"] == 1]
    no_rain = df[df["is_rain"] == 0]

    rain_mean = float(rain["ride"].mean())
    no_rain_mean = float(no_rain["ride"].mean())

    return {
        "rain_days": len(rain),
        "no_rain_days": len(no_rain),
        "rain_mean": round(rain_mean, 1),
        "no_rain_mean": round(no_rain_mean, 1),
        "difference": round(rain_mean - no_rain_mean, 1),
        "difference_pct": round((rain_mean / no_rain_mean - 1) * 100, 2),
    }


def matched_comparison(df):
    # 같은 연도·월·요일에 속하는 날짜끼리 비교
    # 각 그룹에 비 오는 날과 비가 오지 않는 날이 모두 있어야 비교 가능
    grouped = df.groupby(
        ["year", "month", "dow"],
        observed=True,
    )

    rows = []

    for (year, month, dow), group in grouped:
        rain = group[group["is_rain"] == 1]
        no_rain = group[group["is_rain"] == 0]

        if rain.empty or no_rain.empty:
            continue

        rain_mean = float(rain["ride"].mean())
        no_rain_mean = float(no_rain["ride"].mean())

        rows.append(
            {
                "year": int(year),
                "month": int(month),
                "dow": int(dow),
                "rain_days": len(rain),
                "no_rain_days": len(no_rain),
                "rain_mean": rain_mean,
                "no_rain_mean": no_rain_mean,
                "difference": rain_mean - no_rain_mean,
                "difference_pct": ((rain_mean / no_rain_mean - 1) * 100),
            }
        )

    result = pd.DataFrame(rows)

    if result.empty:
        raise ValueError("같은 연도·월·요일에서 비교 가능한 그룹이 없습니다.")

    # 비 오는 날의 수를 가중치로 사용
    # 비교 가능한 각 비 오는 날에 동일한 가중치를 부여하는 방식
    weights = result["rain_days"].to_numpy(dtype=float)

    weighted_difference = float(np.average(result["difference"], weights=weights))

    weighted_pct = float(np.average(result["difference_pct"], weights=weights))

    return result, {
        "matched_groups": len(result),
        "matched_rain_days": int(result["rain_days"].sum()),
        "weighted_difference": round(weighted_difference, 1),
        "weighted_difference_pct": round(weighted_pct, 2),
    }


def adjusted_regression(df):
    # 연도·월 조합별 고정효과를 생성
    # 2020년 1월과 2021년 1월을 서로 다른 시기로 취급
    year_month = df["year"].astype(str) + "-" + df["month"].astype(str).str.zfill(2)

    # 요일과 공휴일 효과를 함께 통제
    controls = pd.concat(
        [
            pd.get_dummies(
                year_month,
                prefix="ym",
                drop_first=True,
                dtype=float,
            ),
            pd.get_dummies(
                df["dow"],
                prefix="dow",
                drop_first=True,
                dtype=float,
            ),
            df[["is_holiday"]].astype(float),
        ],
        axis=1,
    )

    # 첫 번째 열은 절편, 두 번째 열은 비 여부
    X = np.column_stack(
        [
            np.ones(len(df)),
            df["is_rain"].to_numpy(dtype=float),
            controls.to_numpy(dtype=float),
        ]
    )

    y = df["ride"].to_numpy(dtype=float)

    # 최소제곱법으로 회귀계수 계산
    coefficients, _, rank, _ = np.linalg.lstsq(X, y, rcond=None)

    if rank < X.shape[1]:
        raise ValueError("회귀 설계행렬에 완전한 선형 종속성이 있습니다.")

    rain_effect = float(coefficients[1])

    # HC1 이분산 강건 표준오차
    residual = y - X @ coefficients

    n, p = X.shape

    inverse = np.linalg.pinv(X.T @ X)

    meat = X.T @ (X * (residual**2)[:, np.newaxis])

    covariance = (inverse @ meat @ inverse) * (n / (n - p))

    standard_error = float(np.sqrt(max(covariance[1, 1], 0.0)))

    # 정규근사 95% 신뢰구간
    ci_low = rain_effect - 1.96 * standard_error
    ci_high = rain_effect + 1.96 * standard_error

    return {
        "observations": n,
        "control_variables": p - 2,
        "rain_effect": round(rain_effect, 1),
        "robust_se": round(standard_error, 1),
        "ci_95_low": round(ci_low, 1),
        "ci_95_high": round(ci_high, 1),
        "reference_mean": round(float(y.mean()), 1),
        "relative_to_mean_pct": round(rain_effect / float(y.mean()) * 100, 2),
    }


def main():
    # 기존 프로젝트의 정제 결과 재사용
    merged, _, _ = load_merged()

    df = (
        merged[
            [
                "date",
                "ride",
                "rain_mm",
                "is_rain",
                "year",
                "month",
                "dow",
                "is_holiday",
            ]
        ]
        .dropna()
        .sort_values("date")
        .reset_index(drop=True)
    )

    print("=" * 70)
    print("1. 분석 데이터 확인")
    print("=" * 70)

    print(f"전체 데이터: {len(df):,}일")
    print(f"기간: {df['date'].min().date()} ~ {df['date'].max().date()}")
    print("강우 기준: 일 강우량 1mm 이상")

    # 1. 단순 평균 비교
    simple = compare_means(df)

    print("\n" + "=" * 70)
    print("2. 단순 평균 비교")
    print("=" * 70)

    print(f"비 오는 날: {simple['rain_days']}일 / 평균 {simple['rain_mean']:,.0f}건")

    print(
        f"비 없는 날: {simple['no_rain_days']}일 / 평균 {simple['no_rain_mean']:,.0f}건"
    )

    print(
        f"평균 차이: {simple['difference']:+,.0f}건 ({simple['difference_pct']:+.2f}%)"
    )

    # 2. 동일 연도·월·요일 비교
    matched_df, matched = matched_comparison(df)

    print("\n" + "=" * 70)
    print("3. 동일 연도·월·요일 조건 비교")
    print("=" * 70)

    print(f"비교 가능한 그룹: {matched['matched_groups']}개")
    print(f"비교에 포함된 비 오는 날: {matched['matched_rain_days']}일")

    print(
        f"가중 평균 차이: "
        f"{matched['weighted_difference']:+,.0f}건 "
        f"({matched['weighted_difference_pct']:+.2f}%)"
    )

    # 3. 통제 회귀분석
    adjusted = adjusted_regression(df)

    print("\n" + "=" * 70)
    print("4. 연도·월·요일·공휴일 통제 회귀분석")
    print("=" * 70)

    print(f"비 오는 날의 추정 차이: {adjusted['rain_effect']:+,.0f}건")

    print(f"전체 일평균 대비: {adjusted['relative_to_mean_pct']:+.2f}%")

    print(f"HC1 강건 표준오차: {adjusted['robust_se']:,.0f}건")

    print(
        f"95% 신뢰구간: "
        f"[{adjusted['ci_95_low']:+,.0f}, "
        f"{adjusted['ci_95_high']:+,.0f}]건"
    )

    if adjusted["ci_95_low"] > 0 or adjusted["ci_95_high"] < 0:
        print("→ 95% 신뢰구간에 0이 포함되지 않음")
    else:
        print("→ 95% 신뢰구간에 0이 포함됨")

    print(
        "\n※ 회귀계수는 통제 변수들을 고려한 조건부 연관성임"
        "\n※ 비가 이용량 변화를 직접 유발했다는 인과적 증거는 아님"
        "\n※ 시계열 자기상관을 고려한 신뢰구간은 추가 검토 필요"
    )

    # 4. 분석 결과 저장
    output = {
        "experiment": "rain_adjusted",
        "rain_definition": "rain_mm >= 1.0",
        "simple_comparison": simple,
        "matched_comparison": matched,
        "adjusted_regression": adjusted,
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
    print("5. 분석 결과 저장")
    print("=" * 70)

    print(f"[ok ] {OUTPUT_PATH}")


if __name__ == "__main__":
    main()


# ============================================================
# [구현 핵심]
# ============================================================

# 비 오는 날과 비가 오지 않는 날의 평균 이용량을 비교하고,
# 연도·월·요일에 따른 수요 차이를 통제한 분석을 추가함
# 동일 연도·월·요일 그룹 비교와 회귀분석을 통해
# 단순 평균 차이가 시기적 요인에 영향을 받는지 확인하도록 구현함
#
# [발견한 문제 및 수정]
# 1. 기존 분석은 비 유무에 따른 단순 평균 차이만 비교했음
#    → 같은 연도·월·요일에 속하는 날짜끼리 추가 비교하도록 수정함
#
# 2. 팬데믹과 계절에 따른 수요 변화가 강우 효과에 섞일 수 있었음
#    → 연도·월 조합, 요일, 공휴일 여부를 통제하는 회귀분석을 추가함
#
# 3. 평균 차이만으로 결과의 불확실성을 확인하기 어려웠음
#    → HC1 강건 표준오차와 95% 신뢰구간을 함께 계산하도록 구현함
#
# [추후 보완 사항]
# 1. 실제 분석 결과를 README와 분석 보고서에 반영
# 2. 시계열 자기상관을 고려한 HAC 표준오차로 추가 검증
# 3. 강우량 구간별 차이 및 팬데믹 전후 강우 효과 비교
