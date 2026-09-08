import re

import numpy as np
import pandas as pd

KR_HOLIDAYS = {
    # 2019
    "2019-01-01",
    "2019-02-04",
    "2019-02-05",
    "2019-02-06",
    "2019-03-01",
    "2019-05-05",
    "2019-05-06",
    "2019-05-12",
    "2019-06-06",
    "2019-08-15",
    "2019-09-12",
    "2019-09-13",
    "2019-09-14",
    "2019-10-03",
    "2019-10-09",
    "2019-12-25",
    # 2020
    "2020-01-01",
    "2020-01-24",
    "2020-01-25",
    "2020-01-26",
    "2020-01-27",
    "2020-03-01",
    "2020-04-15",
    "2020-04-30",
    "2020-05-05",
    "2020-06-06",
    "2020-08-15",
    "2020-08-17",
    "2020-09-30",
    "2020-10-01",
    "2020-10-02",
    "2020-10-03",
    "2020-10-09",
    "2020-12-25",
    # 2021
    "2021-01-01",
    "2021-02-11",
    "2021-02-12",
    "2021-02-13",
    "2021-03-01",
    "2021-05-05",
    "2021-05-19",
    "2021-06-06",
    "2021-08-15",
    "2021-08-16",
    "2021-09-20",
    "2021-09-21",
    "2021-09-22",
    "2021-10-03",
    "2021-10-04",
    "2021-10-09",
    "2021-10-11",
    "2021-12-25",
    # 2022
    "2022-01-01",
    "2022-01-31",
    "2022-02-01",
    "2022-02-02",
    "2022-03-01",
    "2022-03-09",
    "2022-05-05",
    "2022-05-08",
    "2022-06-01",
    "2022-06-06",
    "2022-08-15",
    "2022-09-09",
    "2022-09-10",
    "2022-09-11",
    "2022-09-12",
    "2022-10-03",
    "2022-10-09",
    "2022-10-10",
    "2022-12-25",
    # 2023
    "2023-01-01",
    "2023-01-21",
    "2023-01-22",
    "2023-01-23",
    "2023-01-24",
    "2023-03-01",
    "2023-05-05",
    "2023-05-27",
    "2023-05-29",
    "2023-06-06",
    "2023-08-15",
    "2023-09-28",
    "2023-09-29",
    "2023-09-30",
    "2023-10-02",
    "2023-10-03",
    "2023-10-09",
    "2023-12-25",
    # 2024
    "2024-01-01",
    "2024-02-09",
    "2024-02-10",
    "2024-02-11",
    "2024-02-12",
    "2024-03-01",
    "2024-04-10",
    "2024-05-05",
    "2024-05-06",
    "2024-05-15",
    "2024-06-06",
    "2024-08-15",
    "2024-09-16",
    "2024-09-17",
    "2024-09-18",
    "2024-10-01",
    "2024-10-03",
    "2024-10-09",
    "2024-12-25",
}
# 근로자의 날(5월 1일) 제외함
# 이후 작업을 진행하면서 공휴일 수를 실제와 대조할 예정임
# 2019~2024 공휴일, 대체 공휴일, 선거일, 임시 공휴일 기준으로 구성
# 공휴일: 108 주말: 626 평일: 1487

PANDEMIC_START = "2020-02-01"  # 국내 확산·수요 급감 시작
PANDEMIC_ACUTE_END = "2022-04-17"  # 사회적 거리두기 전면 해제 직전
PANDEMIC_END = "2023-05-11"  # WHO 비상사태 해제·엔데믹 전환


class StepLog:
    """단계마다 행 수를 기록합니다."""

    def __init__(self):
        self.rows = []

    def __call__(self, name: str, df: pd.DataFrame) -> pd.DataFrame:
        prev = self.rows[-1][1] if self.rows else len(df)
        self.rows.append((name, len(df), len(df) - prev))
        return df

    def frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.rows, columns=["단계", "행수", "증감"])


def normalize_station(name: str) -> str:
    """역명 부기(괄호) 제거. '청량리(서울시립대입구)' → '청량리'."""

    if not isinstance(name, str):
        return name

    s = re.sub(r"\(.*?\)", "", name)  # 괄호와 그 안 내용 제거

    return s.replace(" ", "").strip()


def clean_subway(sub, log):
    df = log("0. 지하철 원본", sub.copy())

    # 날짜와 이용객 수를 알맞은 타입으로 바꿈
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["total"] = pd.to_numeric(df["total"], errors="coerce")

    # 날짜, 역명, 승하차 구분이 없는 행 제거
    df = log("1. 타입 강제", df.dropna(subset=["date", "station", "io_type"]))

    # 역명에 붙어 있는 괄호 설명 제거
    df["station"] = df["station"].map(normalize_station)
    df = log("2. 역명 표준화", df)

    # total이 없거나 0 이하이면 이상치로 표시
    # 실제 데이터에서 이상치 6,764건 확인
    # 행 자체를 지우지는 않고 total 값만 NaN으로 바꿈
    bad = df["total"].isna() | (df["total"] <= 0)

    df["total_flag_bad"] = bad
    df.loc[bad, "total"] = np.nan

    df = log("3. total<=0 → NaN(플래그)", df)

    # 승차/하차 2행을 날짜·노선·역 기준 1행으로 합침
    # total_flag_bad는 정제 확인용이기 때문에 여기서 최종 데이터에는 포함하지 않았음
    wide = df.pivot_table(
        index=["date", "line", "station"],
        columns="io_type",
        values="total",
        aggfunc="sum",
    ).reset_index()

    # pivot_table을 사용하고 남은 컬럼 이름 제거
    wide.columns.name = None

    # 승차와 하차 컬럼 이름 변경
    wide = wide.rename(columns={"승차": "board", "하차": "alight"})

    # 승차 + 하차 = 총 이용객
    wide["ride"] = wide[["board", "alight"]].sum(axis=1, min_count=1)

    df = log("4. 승하차 wide + 총이용", wide)

    return df


def clean_weather(w, log):
    df = w.copy()

    # 날짜를 날짜 타입으로 바꿈
    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    # 총강수에서 강우를 빼서 눈으로 인한 강수량을 따로 구함
    df["snow_from_precip"] = (df["precip_mm"] - df["rain_mm"]).clip(lower=0)

    # 비가 1mm 이상 온 날
    df["is_rain"] = (df["rain_mm"] >= 1.0).astype(int)

    # 총강수가 1mm 이상인 날
    df["is_precip"] = (df["precip_mm"] >= 1.0).astype(int)

    # 비가 30mm 이상 온 날
    df["is_heavy_rain"] = (df["rain_mm"] >= 30.0).astype(int)

    df = log("W3. 강수 파생변수", df)

    return df


def aggregate_daily(df, log):
    # 역별 이용객을 날짜별로 모두 합쳐서 서울 전체 하루 이용객 수를 구함
    daily = df.groupby("date", as_index=False)["ride"].sum(min_count=1)

    daily = log("5. 도시 일별 집계", daily)

    return daily


def merge_subway_weather(subway_daily, weather_daily, log):
    # 날짜를 기준으로 지하철과 날씨 데이터를 합침
    merged = pd.merge(subway_daily, weather_daily, on="date", how="inner")

    merged = log("6. 지하철⨝날씨(inner)", merged)

    return merged


def add_calendar(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # 요일 만들기: 월요일=0 ~ 일요일=6
    df["dow"] = df["date"].dt.dayofweek

    # 토요일이나 일요일이면 1
    df["is_weekend"] = (df["dow"] >= 5).astype(int)

    # 날짜를 문자열로 바꿔서 공휴일 목록과 비교
    ds = df["date"].dt.strftime("%Y-%m-%d")
    df["is_holiday"] = ds.isin(KR_HOLIDAYS).astype(int)

    # 주말도 아니고 공휴일도 아니면 평일
    df["is_workday"] = ((df["is_weekend"] == 0) & (df["is_holiday"] == 0)).astype(int)

    # 월과 연도
    df["month"] = df["date"].dt.month
    df["year"] = df["date"].dt.year

    # 팬데믹 구조 변화 구간
    d = df["date"]

    df["pandemic_acute"] = ((d >= PANDEMIC_START) & (d < PANDEMIC_ACUTE_END)).astype(
        int
    )

    df["pandemic_recovery"] = ((d >= PANDEMIC_ACUTE_END) & (d < PANDEMIC_END)).astype(
        int
    )
    # 팬데믹을 하나로 묶지 않고 acute와 recovery로 나눔
    # 급격히 이용량이 떨어진 시기와 회복하는 시기를 모델이 구분하도록 구현함

    df["is_pandemic"] = ((d >= PANDEMIC_START) & (d < PANDEMIC_END)).astype(int)

    return df


def build_clean_data(subway, weather):
    # 전체 정제 단계의 행 수를 기록할 로그
    log = StepLog()

    # 0~4. 지하철 정제
    sw = clean_subway(subway, log)

    # 5. 역별 데이터를 서울 전체 일별 데이터로 집계
    subway_daily = aggregate_daily(sw, log)

    # W3. 날씨 정제 및 강수 파생변수 생성
    weather_daily = clean_weather(weather, log)

    # 6. 날짜 기준으로 지하철과 날씨 결합
    merged = merge_subway_weather(subway_daily, weather_daily, log)

    # 7. 요일, 공휴일, 평일, 팬데믹 구간 추가
    merged = add_calendar(merged)
    merged = log("7. 달력 파생", merged)

    # 지금까지의 처리 이력을 표로 만듦
    rep = log.frame()

    return merged, sw, rep


# ============================================================

# 지하철 데이터의 날짜와 이용객 수 타입을 맞추고 역명 정리
# total이 없거나 0 이하인 값은 삭제하지 않고 NaN으로 바꿔 처리함
# 실제 데이터에서 이상치 6,764건을 확인함

# 승차와 하차로 나뉜 데이터를 날짜/노선/역 기준 1행으로 합침
# 승차 + 하차를 계산해 총 이용객 수 ride를 만듦
# 이후 역별 데이터를 날짜별로 합쳐 서울 전체의 일별 이용개 수로 만들었음

# 날씨 데이터에서는 강수량과 강우량을 이용해 눈으로 인한 강수량을 구함
# 비 온 날, 강수 있는 날, 호우 온 날을 구분하는 변수를 만들었음

# 지하철과 날씨 데이터는 date를 기준으로 inner join함
# 결합 후 2,192일이 그대로 유지되는 것을 확인함

# 요일, 주말, 공휴일, 평일 여부와 연도, 월을 추가
# 팬데믹 급성기와 회복기를 구분해 이후 모델에서 따로 사용할 수 있게 작업함

# StepLog로 각 전처리 단계의 행 수 변화를 기록함
# 실제 실행 결과
# 2,638,670행 -> 1,319,335행 -> 2,192행
# 최종 데이터는 2019~2024년 총 2,192일로 확인함
