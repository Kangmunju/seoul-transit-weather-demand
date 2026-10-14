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


# 피드백 8번: 추가 휴일 관련 피처
# 기존 모델과 비교할 수 있도록 별도 목록으로 관리
HOLIDAY_FEATURES = [
    "is_labor_day",
    "is_day_after_holiday",
    "holiday_streak_length",
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


def feature_columns(include_pandemic=True, include_holiday_features=False):
    # 기본으로 날씨 + 달력 + 시계열 피처를 사용
    features = WEATHER_FEATURES + CALENDAR_FEATURES + TIME_FEATURES

    # 피드백 8번: 휴일 보강 피처를 선택적으로 추가
    # False이면 기존 모델과 동일한 피처 목록을 사용함
    if include_holiday_features:
        features = features + HOLIDAY_FEATURES

    # 여러 연도의 데이터를 사용할 때만 팬데믹 피처 추가
    if include_pandemic:
        features = features + PANDEMIC_FEATURES

    return features


# 실제 확인 결과(기존 모델 기준)
# 피처 생성 전: 2192
# 피처 생성 후: 2192
# lag_7 결측: 7
# 학습 가능 데이터: 2185
# 기간: 2019-01-08 00:00:00 ~ 2024-12-31 00:00:00


# ============================================================
# [구현 핵심]
# ============================================================

# 지하철 수요 데이터는 날짜순으로 정렬한 뒤 시계열 피처를 생성함
# lag_1과 lag_7로 전날 및 지난주 같은 요일의 이용량을 반영함
# rolling 피처는 shift(1)를 적용해 예측 당일 이용량이 포함되지 않도록 함
# roll7_mean, roll7_std, roll28_mean으로 최근 수요 흐름과 변동성을 표현함
# dow_sin, dow_cos를 추가하여 요일의 순환 구조를 표현함
# 날씨, 달력, 시계열, 팬데믹 피처를 그룹별로 관리함
# 휴일 보강 피처 3개를 별도 목록으로 관리하고 선택적으로 사용하도록 구성함
#
# [발견한 문제 및 수정]
# 1. 기존 달력 피처에는 근로자의 날과 연속 휴일 정보가 없었음
#    → HOLIDAY_FEATURES에 신규 휴일 피처 3개를 추가함
#
# 2. 신규 휴일 피처를 기본 목록에 바로 추가하면 기존 모델과 비교하기 어려움
#    → feature_columns()에 include_holiday_features 옵션을 추가함
#    → False는 기존 피처, True는 휴일 보강 피처까지 포함하도록 구현함
#
# 3. 시계열 피처 생성 시 예측 당일 이용량이 포함될 가능성이 있었음
#    → 기존 shift(1) 적용 방식을 유지하여 데이터 누수를 방지함
#
# [추후 보완 사항]
# 1. train.py에서 기존 피처와 휴일 보강 피처의 모델 성능 비교
# 2. 평일·주말·공휴일별 MAPE와 근로자의 날 예측 오차 확인
# 3. 휴일 피처별 중요도와 실제 성능 개선 여부 검증
