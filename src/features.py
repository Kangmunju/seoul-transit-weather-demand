import numpy as np
import pandas as pd


# 예측할 값
TARGET = "ride"


# 날씨 관련 피처
WEATHER_FEATURES = [
    "temp_mean",
    "temp_max",
    "temp_min",
    "precip_mm",
    "rain_mm",
    "snow_from_precip",
    "wind_max",
    "humidity",
    "is_rain",
    "is_precip",
    "is_heavy_rain",
]


# 달력 관련 피처
CALENDAR_FEATURES = [
    "dow",
    "is_weekend",
    "is_holiday",
    "is_workday",
    "month",
]


# 과거 이용객 수를 이용해서 만든 시계열 피처
TIME_FEATURES = [
    "lag_1",
    "lag_7",
    "roll7_mean",
    "roll7_std",
    "roll28_mean",
    "dow_sin",
    "dow_cos",
]


# 팬데믹 구조 변화 피처
PANDEMIC_FEATURES = [
    "pandemic_acute",
    "pandemic_recovery",
]


def build(df: pd.DataFrame) -> pd.DataFrame:
    """시계열 순서를 보장한 뒤 lag/rolling 피처를 만듭니다."""

    # 날짜순으로 먼저 정렬
    # shift는 행 순서를 기준으로 하기 때문에 정렬이 먼저 되어야 함
    d = df.sort_values("date").reset_index(drop=True).copy()

    # 어제 이용객
    d["lag_1"] = d[TARGET].shift(1)

    # 지난주 같은 요일 이용객
    d["lag_7"] = d[TARGET].shift(7)

    # 오늘 값은 제외하고 어제까지의 값만 사용
    # 오늘 값을 포함하면 오늘을 예측할 때 오늘 값을 미리 보는 데이터 누수가 생김
    past = d[TARGET].shift(1)

    # 어제까지 최근 7일 평균
    d["roll7_mean"] = past.rolling(7, min_periods=3).mean()

    # 어제까지 최근 7일 표준편차
    d["roll7_std"] = past.rolling(7, min_periods=3).std()

    # 어제까지 최근 28일 평균
    d["roll28_mean"] = past.rolling(28, min_periods=7).mean()

    # 요일을 순환 형태로 표현
    # 일요일 다음이 다시 월요일이라는 관계를 선형모델에서도 표현하기 위해 사용함
    d["dow_sin"] = np.sin(2 * np.pi * d["dow"] / 7)

    d["dow_cos"] = np.cos(2 * np.pi * d["dow"] / 7)

    return d


def feature_columns(include_pandemic=True):
    # 기본으로 날씨 + 달력 + 시계열 피처를 사용
    features = WEATHER_FEATURES + CALENDAR_FEATURES + TIME_FEATURES

    # 여러 연도의 데이터를 사용할 때만 팬데믹 피처 추가
    if include_pandemic:
        features = features + PANDEMIC_FEATURES

    return features


# 실제 확인 결과
# 피처 생성 전: 2192
# 피처 생성 후: 2192
# lag_7 결측: 7
# 학습 가능 데이터: 2185
# 기간: 2019-01-08 00:00:00 ~ 2024-12-31 00:00:00


# ============================================================

# 지하철 수요 데이터는 날짜순으로 먼저 정렬한 뒤 시계열 피처를 만듦

# lag_1 = 어제 이용객 수
# lag_6 = 지난주 같은 요일의 이용객 수

# 오늘 값을 포함하면 예측할 값을 미리 보는 데이터 누수가 생길 가능성이 있다고 판단하였음
# rolling 피처의 오늘 값을 바로 사용하지 않고 shift(1)를 먼저 적용해 어제까지의 데이터만 이용하는 것으로 해결함

# roll7_mean = 최근 7일의 평균
# roll7_std = 최근 7일의 변동 정도
# roll28_mean = 최근 한달 정도의 수요 흐름

# 0~6의 단순 숫자로만 요일을 보면 선형 모델이 크기 차이로 오해할 가능성이 있다고 판단하였음
# dow_sin, dow_cos를 추가해 요일의 순환 구조를 함께 표현해 해결함

# feature_columns()에 날씨, 달력, 시계열 피처를 한꺼번에 모아 두었음
# 다년 데이터일 경우 팬데믹 급성기와 회복기 피처도 추가하도록 함
