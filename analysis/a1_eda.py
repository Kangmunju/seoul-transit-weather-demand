from _common import load_merged


def main():
    merged, sw, rep = load_merged()
    # merged : 날짜별 지하철 수요 + 날씨 + 달력 정보가 합쳐진 최종 데이터
    # sw : 정제된 지하철 데이터
    # rep : 정제 과정에서 만들어진 보고용 정보

    # 날짜 순서대로 정렬
    df = merged.sort_values("date").reset_index(drop=True).copy()

    print("[데이터 개요]")
    print(
        f"기간: {df['date'].min().date()} ~ {df['date'].max().date()} ({len(df):,}일)"
    )
    # 데이터 기간, 빠진 날짜, 연도 범위를 일단 기초적으로 확인함

    # clean.py에서 ride = board(승차인원) + alight(하차인원)로 정의함
    # 수요의 규모와 변동 폭을 파악하기 위해 작성함
    print(
        f"일 총이용(승+하) 평균: {df['ride'].mean():,.0f} "
        f"최소 {df['ride'].min():,.0f} "
        f"최대 {df['ride'].max():,.0f}"
    )

    # 원본 지하철 정제 데이터에서 날짜별 수집 역 수 확인
    station_count = sw.groupby("date")["station"].nunique()

    # 실제 수요 감소가 아니라 데이터 누락으로 인해 변동 폭이 클 수 있으므로 수집 범위별로 역 개수 파악함
    print(f"수집 역 수(일별): {station_count.min():,}~{station_count.max():,}")

    print("\n[무엇이 수요를 지배하나 - 평일 vs 주말/공휴일]")

    workday = df[df["is_workday"] == 1]
    weekend = df[df["is_weekend"] == 1]
    # 단순하게 주말에서 공휴일을 빼버리면 집계 방식이 맞지 않는 것을 확인
    # 주말과 공휴일을 서로 겹칠 수 있게 보는 방식으로 맞춤
    holiday = df[df["is_holiday"] == 1]

    print(f"평일 {len(workday)}일 평균 이용 {workday['ride'].mean():,.0f}")
    print(f"주말 {len(weekend)}일 평균 이용 {weekend['ride'].mean():,.0f}")
    print(f"공휴일 {len(holiday)}일 평균 이용 {holiday['ride'].mean():,.0f}")

    weekend_ratio = weekend["ride"].mean() / workday["ride"].mean() * 100

    print(f"→ 주말은 평일의 {weekend_ratio:.0f}% 수준")

    print("\n[요일별 평균 이용객]")

    dow_names = {
        0: "월",
        1: "화",
        2: "수",
        3: "목",
        4: "금",
        5: "토",
        6: "일",
    }

    dow_mean = df.groupby("dow")["ride"].mean().reindex(range(7))

    for dow, mean_ride in dow_mean.items():
        print(f"{dow_names[dow]}요일: {mean_ride:,.0f}")

    print("\n[자기상관 - 어제/지난주가 오늘을 설명한다]")
    # 오늘의 지하철 이용량이 어제 이용량 또는 지난주 같은 요일 이용량과 얼마나 비슷하게 움직이는지 파악
    lag1_corr = df["ride"].corr(df["ride"].shift(1))
    # shift(1)을 사용해 어제의 이용량으로 값을 밀어주며 비교
    lag7_corr = df["ride"].corr(df["ride"].shift(7))

    print(f"lag-1 자기상관: {lag1_corr:.3f}")
    print(f"lag-7 자기상관: {lag7_corr:.3f}")

    if lag7_corr > lag1_corr:
        print("→ lag-7의 상관이 더 강해 주간 주기가 뚜렷합니다.")
    else:
        print("→ lag-1의 상관이 더 강합니다.")


if __name__ == "__main__":
    main()


# 실행 후 확인해야 할 결과
# - 전체 분헉 기간과 실제 분석 일수
# - 일평균 이용객 및 최소/최대 이용객
# - 날짜별 수집 역 수 범위
# - 평일/주말/공휴일의 평균 이용객 차이
# - 주말 수요가 평일 수요의 몇 %인지
# - 월~일 요일별 평균 이용객 패턴
# - lag-1과 lag-7 자기상관 중 어느 쪽이 더 강한지

# 결과를 확인한 뒤 실제 수치와 이 외 더 확인할 사항을 계속 추가할 예정

# [데이터 개요]
# 기간: 2019-01-01 ~ 2024-12-31 (2,192일)
# 일 총이용(승+하) 평균: 12,670,404 최소 3,149,873 최대 18,425,509
# 수집 역 수(일별): 509~529

# [무엇이 수요를 지배하나 - 평일 vs 주말/공휴일]
# 평일 1487일 평균 이용 14,466,760
# 주말 626일 평균 이용 9,065,391
# 공휴일 108일 평균 이용 7,309,681
# → 주말은 평일의 63% 수준

# [요일별 평균 이용객]
# 월요일: 13,552,892
# 화요일: 14,087,212
# 수요일: 14,091,305
# 목요일: 14,221,677
# 금요일: 14,604,432
# 토요일: 10,466,778
# 일요일: 7,664,005

# [자기상관 - 어제/지난주가 오늘을 설명한다]
# lag-1 자기상관: 0.492
# lag-7 자기상관: 0.808
# → lag-7의 상관이 더 강해 주간 주기가 뚜렷합니다.


# ============================================================

# _common.py의 load_merged()를 재사용해 정제된 지하철 및 날씨 데이터를 불러옴
# 전체 수요 규모, 날짜별 수집 역 수, 평일/주말/공휴일 및 요일별 수요 차이를 확인함

# 분석 결과 주말 수요는 평일의 약 63% 수준임을 확인
# 금요일 평균 이용객이 가장 높고 일요일이 가장 낮음을 확인
# 지하철 수요에 뚜렷한 요일 패턴이 있음을 파악함
# -> train.py에서 is_workday가 높은 변수 중요도를 보인 결과와 연결

# shift(1), shift(7)을 이용해 현재 수요와 과거 수요이 자기 상관을 비교
# lag-1 = 0.492, lag-7 = 0.808
# 지난주 같은 요일의 수요가 오늘 수요와 더 강하게 연결되어 있음을 확인함
# -> features.py의 lag-7 피처와 train.py의 lag_7 베이스라인 사용 근거

# 자기상관은 함께 움직이는 정도를 나타내므로 인과관계로 해석하지 않음
