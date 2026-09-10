from _common import load_merged


def main():
    merged, sw, rep = load_merged()

    # 날짜 순서대로 정렬
    df = merged.sort_values("date").reset_index(drop=True).copy()

    print("[비 오는 날 vs 비 안 오는 날]")

    rain = df[df["is_rain"] == 1]
    no_rain = df[df["is_rain"] == 0]

    # 일단 아무런 조건 없이 전체 데이터를 기준으로 비와 수요 관계를 1차로 비교함
    print(f"비 오는 날 {len(rain)}일 평균 이용 {rain['ride'].mean():,.0f}")
    print(f"비 안 오는 날 {len(no_rain)}일 평균 이용 {no_rain['ride'].mean():,.0f}")

    rain_ratio = rain["ride"].mean() / no_rain["ride"].mean() * 100

    print(f"→ 비 오는 날은 비 안 오는 날의 {rain_ratio:.1f}% 수준")

    print("\n[평일만 비교 - 요일 효과 분리]")

    # 평균 이용객이 낮아진 원인(비, 단순 주말 포함)을 정확히 구분하는 것이 필요
    # 평일 데이터만 골라내 평일과 주말의 큰 수요 차이가 비 효과에 섞이는 것을 줄여줌
    workday = df[df["is_workday"] == 1]

    # 평일/주말 요인을 어느 정도 통제한 뒤에도 비 오는 날의 수요 차이가 남았는지 확인
    # 평일을 다시 두 그룹으로 분류
    workday_rain = workday[workday["is_rain"] == 1]
    workday_no_rain = workday[workday["is_rain"] == 0]

    # 평일 조건을 고정하고 비 유무를 비교
    print(
        f"평일 + 비 {len(workday_rain)}일 평균 이용 {workday_rain['ride'].mean():,.0f}"
    )
    print(
        f"평일 + 비 없음 {len(workday_no_rain)}일 "
        f"평균 이용 {workday_no_rain['ride'].mean():,.0f}"
    )

    workday_rain_ratio = (
        workday_rain["ride"].mean() / workday_no_rain["ride"].mean() * 100
    )

    print(
        f"→ 평일끼리 비교하면 비 오는 날은 "
        f"비 안 오는 날의 {workday_rain_ratio:.1f}% 수준"
    )

    print("\n[강수량과 이용객 수의 상관관계]")

    # 강수량과 지하철 이용객 수의 관계 확인
    precip_corr = df["ride"].corr(df["precip_mm"])
    rain_corr = df["ride"].corr(df["rain_mm"])

    print(f"강수량(precip_mm)과 이용객 수 상관계수: {precip_corr:.3f}")
    print(f"강우량(rain_mm)과 이용객 수 상관계수: {rain_corr:.3f}")


if __name__ == "__main__":
    main()


# [비 오는 날 vs 비 안 오는 날]
# 비 오는 날 604일 평균 이용 12,383,645
# 비 안 오는 날 1588일 평균 이용 12,779,473
# → 비 오는 날은 비 안 오는 날의 96.9% 수준

# [평일만 비교 - 요일 효과 분리]
# 평일 + 비 409일 평균 이용 14,157,646
# 평일 + 비 없음 1078일 평균 이용 14,584,040
# → 평일끼리 비교하면 비 오는 날은 비 안 오는 날의 97.1% 수준

# [강수량과 이용객 수의 상관관계]
# 강수량(precip_mm)과 이용객 수 상관계수: -0.062
# 강우량(rain_mm)과 이용객 수 상관계수: -0.062


# ============================================================

# 비 오는 날과 비 안 오는 날의 평균 이용객을 비교
# 비 오는 날은 비 안 오는 날의 96.9% 수준인 것으로 1차적 확인함

# 평일/주말 차이가 비의 효과로 섞이는 것을 줄일 필요가 있다고 판단함
# 평일만 다시 비교한 경우, 이 때에도 비 오는 날은 97.1% 수준으로 나타남
# 평일로 조건을맞춘 뒤에도작은 수요 차이가 남아 있음을 확인함

# precip_mm, rain_mm와 ride의 상관관계는 모두 -0.062
# 강수량과 지하철 수요 사이의 선형 관계가 매우 약하게 나타남
# 앞서 확인한 요일 효과에 비해 비의 영향은 상대적으로 작다고 볼 수 있음

# 단, 평균차이와 상관관계만으로 비가 수요 변화의 직접적인 원인이라고 해석하지는 않았음
