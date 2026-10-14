import json
from pathlib import Path

import numpy as np
import pandas as pd

from _common import load_merged


BASE_DIR = Path(__file__).resolve().parents[1]
OUTPUT_PATH = BASE_DIR / "models" / "rain_workday_adjusted_results.json"

SEED = 42
BOOTSTRAP_ITERATIONS = 2000

# 강우 강도 구간
RAIN_LABELS = ["0mm", "0~5mm", "5~20mm", "20mm 초과"]


def prepare_data():
    # 기존 정제 파이프라인으로 지하철·날씨 데이터를 불러옴
    merged, _, _ = load_merged()

    required = [
        "date",
        "ride",
        "year",
        "month",
        "is_workday",
        "rain_mm",
        "is_rain",
    ]

    # 분석에 필요한 열만 선택
    df = merged[required].copy()

    df["date"] = pd.to_datetime(df["date"])

    # 결측값 제거
    df = df.dropna(subset=required)

    # 평일이면서 공휴일이 아닌 날짜만 사용
    df = df[df["is_workday"] == 1].copy()

    if df.empty:
        raise ValueError("분석 가능한 평일 데이터가 없습니다.")

    # 같은 연도·월의 평일 평균 이용량
    monthly_mean = df.groupby(["year", "month"])["ride"].transform("mean")

    if (monthly_mean <= 0).any():
        raise ValueError("연도·월 평균 이용량이 0 이하입니다.")

    # 해당 연도·월의 평일 평균 대비 상대적 이용량
    # 예: rel = -0.03이면 해당 월 평일 평균보다 3% 적음
    df["rel"] = df["ride"] / monthly_mean - 1

    # 부트스트랩에서 사용할 연도·월 식별자
    df["year_month"] = (
        df["year"].astype(str) + "-" + df["month"].astype(str).str.zfill(2)
    )

    # 강우량 구간 분류
    # 0mm, (0, 5], (5, 20], (20, +inf)
    df["rain_group"] = pd.cut(
        df["rain_mm"],
        bins=[-np.inf, 0, 5, 20, np.inf],
        labels=RAIN_LABELS,
    )

    return df.sort_values("date").reset_index(drop=True)


def bootstrap_difference(df, group_col, reference, target):
    # 연도·월 단위로 묶어 재표본추출
    # 같은 달 안의 날짜들을 개별 독립 관측치로
    # 가정하는 문제를 일부 완화하기 위한 방식
    groups = sorted(df["year_month"].unique())

    summaries = []

    for year_month in groups:
        part = df[df["year_month"] == year_month]

        ref_values = part.loc[part[group_col] == reference, "rel"]

        target_values = part.loc[part[group_col] == target, "rel"]

        summaries.append(
            [
                float(ref_values.sum()),
                len(ref_values),
                float(target_values.sum()),
                len(target_values),
            ]
        )

    summaries = np.asarray(summaries, dtype=float)

    # 연도·월 그룹을 복원추출
    rng = np.random.default_rng(SEED)

    sampled_indices = rng.integers(
        low=0,
        high=len(groups),
        size=(BOOTSTRAP_ITERATIONS, len(groups)),
    )

    # 각 반복에서 참조군·비교군의 합계와 개수를 계산
    sampled = summaries[sampled_indices].sum(axis=1)

    ref_sum = sampled[:, 0]
    ref_count = sampled[:, 1]

    target_sum = sampled[:, 2]
    target_count = sampled[:, 3]

    # 어느 한쪽 집단이 없는 반복은 제외
    valid = (ref_count > 0) & (target_count > 0)

    differences = (
        target_sum[valid] / target_count[valid] - ref_sum[valid] / ref_count[valid]
    ) * 100

    if len(differences) < BOOTSTRAP_ITERATIONS * 0.9:
        raise ValueError(f"{target}: 유효한 부트스트랩 반복이 부족합니다.")

    # 백분위수 기반 95% 신뢰구간
    low, high = np.percentile(differences, [2.5, 97.5])

    return round(float(low), 2), round(float(high), 2)


def compare_groups(df, group_col, reference, target):
    ref_values = df.loc[df[group_col] == reference, "rel"]

    target_values = df.loc[df[group_col] == target, "rel"]

    if ref_values.empty or target_values.empty:
        raise ValueError(f"비교할 데이터가 부족합니다: {reference}, {target}")

    # 같은 연도·월 평일 평균 대비 상대적 이용량 차이
    difference_pp = (target_values.mean() - ref_values.mean()) * 100

    ci_low, ci_high = bootstrap_difference(df, group_col, reference, target)

    return {
        "reference": str(reference),
        "target": str(target),
        "reference_days": int(len(ref_values)),
        "target_days": int(len(target_values)),
        "reference_rel_pct": round(float(ref_values.mean() * 100), 2),
        "target_rel_pct": round(float(target_values.mean() * 100), 2),
        "difference_pp": round(float(difference_pp), 2),
        "ci_95_low": ci_low,
        "ci_95_high": ci_high,
    }


def print_comparison(result):
    print(f"{result['reference']} → {result['target']}")

    print(f"  날짜: {result['reference_days']}일 / {result['target_days']}일")

    print(f"  보정 후 차이: {result['difference_pp']:+.2f}%p")

    print(
        f"  95% 신뢰구간: [{result['ci_95_low']:+.2f}, {result['ci_95_high']:+.2f}]%p"
    )


def main():
    df = prepare_data()

    print("=" * 70)
    print("1. 평일 분석 데이터 확인")
    print("=" * 70)

    print(f"평일 데이터: {len(df):,}일")

    print(f"분석 기간: {df['date'].min().date()} ~ {df['date'].max().date()}")

    print(f"연도·월 그룹: {df['year_month'].nunique()}개")

    print("보정 방법: 같은 연도·월 평일 평균 대비 상대 이용량")

    # 1. 평일 강우 여부 비교
    print("\n" + "=" * 70)
    print("2. 연도·월 보정 후 평일 강우 효과")
    print("=" * 70)

    rain_result = compare_groups(
        df,
        group_col="is_rain",
        reference=0,
        target=1,
    )

    print_comparison(rain_result)

    # 2. 강우 강도별 비교
    print("\n" + "=" * 70)
    print("3. 강우 강도별 이용량 비교")
    print("=" * 70)

    print("강우 구간: 0mm / 0~5mm / 5~20mm / 20mm 초과")

    print("※ 0~5mm 구간은 0mm를 제외한 강우일을 의미함")

    intensity_results = []

    for label in RAIN_LABELS[1:]:
        result = compare_groups(
            df,
            group_col="rain_group",
            reference=RAIN_LABELS[0],
            target=label,
        )

        intensity_results.append(result)

        print()
        print_comparison(result)

    # 3. 결과 해석 안내
    print("\n" + "=" * 70)
    print("4. 해석 시 주의사항")
    print("=" * 70)

    print("1. 평일·비공휴일만 사용하여 주말과 공휴일 효과를 줄임.")

    print("2. 연도·월 평균 대비 상대 이용량을 사용하여 시기별 수요 수준 차이를 보정함.")

    print("3. 95% 신뢰구간은 연도·월 단위 클러스터 부트스트랩으로 계산함.")

    print("4. 관측된 차이는 강우와 수요의 연관성이며 인과 효과로 해석할 수 없음.")

    print(
        "5. 강우 여부는 rain_mm >= 1mm 기준이지만, "
        "강우 강도 구간은 rain_mm > 0mm부터 시작함."
    )

    # 4. JSON 저장
    output = {
        "experiment": "rain_workday_adjusted",
        "analysis_period": {
            "start": str(df["date"].min().date()),
            "end": str(df["date"].max().date()),
        },
        "workday_days": int(len(df)),
        "year_month_groups": int(df["year_month"].nunique()),
        "adjustment": ("ride / same_year_month_workday_mean - 1"),
        "rain_definition": "rain_mm >= 1.0",
        "rain_intensity_bins": [
            "0mm",
            "0 < rain_mm <= 5",
            "5 < rain_mm <= 20",
            "rain_mm > 20",
        ],
        "bootstrap": {
            "method": "year_month_cluster",
            "iterations": BOOTSTRAP_ITERATIONS,
            "seed": SEED,
            "confidence_level": 0.95,
        },
        "rain_comparison": rain_result,
        "intensity_comparisons": intensity_results,
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

# 기존 전체 날짜 통제 회귀분석(a7)은 유지함
# 평일·비공휴일 데이터만 사용하여 원본 피드백의 조건을 반영함
# 같은 연도·월 평일 평균 대비 상대 이용량을 계산함
# 강우 여부 및 강우 강도별 이용량 차이를 비교함
# 연도·월 단위 부트스트랩으로 95% 신뢰구간을 계산함
#
# [발견한 문제 및 수정]
# 1. 기존 a7 분석은 전체 날짜를 대상으로 수행했음
#    → 평일·비공휴일만 대상으로 하는 보완 분석을 추가함
#
# 2. 기존 분석에는 강우 강도별 비교가 없었음
#    → 0mm, 0~5mm, 5~20mm, 20mm 초과로 구분함
#
# 3. 연도·월 보정 후 차이의 불확실성 확인이 필요했음
#    → 연도·월 클러스터 부트스트랩 신뢰구간을 추가함
#
# 4. 강우 여부와 강우 강도 구간의 기준이 서로 다를 수 있음
#    → 강우 여부는 1mm 이상, 강도 구간은 0mm 초과로 명시함
#
# [추후 보완 사항]
# 1. 실제 실행 결과와 원본 피드백의 참고 수치 비교
# 2. a7 통제 회귀분석과 이번 평일 보정 결과를 보고서에 함께 정리
# 3. 강우 강도별 결과를 대시보드에 시각화
