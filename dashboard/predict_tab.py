"""'직접 예측해보기' 탭. 기존 Streamlit 대시보드(dashboard/app.py)에 붙이는 모듈.

[파일 목적]
- 사용자가 과거 날짜와 가정한 날씨 조건을 입력해 지하철 수요를 예측한다.
- 같은 날짜의 강우량을 0mm로 바꾼 결과와 비교한다.
- 과거 실제 이용량으로 만든 lag/rolling 피처를 사용자에게 공개한다.

[기존 초안 대비 변경점]
- 미래 날짜 예측을 제외하고 실제 이용량 기록이 있는 날짜만 허용한다.
- 학습 기간과 테스트 기간을 구분한다.
- 날짜별 날씨 기본값과 입력 범위를 안정적으로 처리한다.
- 모델 예측 차이를 강우의 인과 효과라고 표현하지 않는다.
- 재학습 모델의 성능을 기존 보고서의 MAPE와 동일하다고 단정하지 않는다.

[연결 방법]
1. 이 파일을 dashboard/predict_tab.py 로 저장한다.
2. 단독 실행으로 기능을 확인한다.
3. 이후 dashboard/app.py 에서 show_predict_tab() 을 호출한다.

[주의]
- 이 기능은 2019~2024년의 기록을 활용한 과거 날짜 시나리오 예측이다.
- 선택 날짜 이전의 실제 이용량을 사용하므로 임의의 미래 날짜를
  예측하는 서비스와는 다르다.
"""

import sys
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestRegressor


# 프로젝트 최상위 경로
BASE_DIR = Path(__file__).resolve().parents[1]

DATA_PATH = BASE_DIR / "dashboard" / "dashboard_data.csv"
MODEL_PATH = BASE_DIR / "models" / "rf_model.joblib"

# 기존 src 모듈 재사용
if str(BASE_DIR / "src") not in sys.path:
    sys.path.insert(0, str(BASE_DIR / "src"))

import features as F  # noqa: E402
from clean import StepLog, add_calendar, clean_weather  # noqa: E402


SEED = 42
DAY_NAMES = ["월", "화", "수", "목", "금", "토", "일"]


@st.cache_data
def load_history():
    """대시보드 데이터를 읽고 모델 입력에 필요한 날씨 피처를 준비한다."""

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"대시보드 데이터가 없습니다: {DATA_PATH}\n"
            "먼저 python3 dashboard/make_dashboard_data.py 를 실행하세요."
        )

    data = pd.read_csv(DATA_PATH, parse_dates=["date"])

    required = {
        "date",
        "ride",
        "temp_mean",
        "temp_max",
        "temp_min",
        "precip_mm",
        "rain_mm",
        "snow_cm",
        "wind_max",
        "humidity",
    }

    missing = sorted(required - set(data.columns))

    if missing:
        raise ValueError(
            "대시보드 데이터에 필요한 컬럼이 없습니다: " + ", ".join(missing)
        )

    data = data.sort_values("date").reset_index(drop=True)

    # 동일 날짜가 여러 번 존재하면 lag/rolling 계산이 왜곡될 수 있음
    if data["date"].duplicated().any():
        raise ValueError("대시보드 데이터에 중복 날짜가 있습니다.")

    # 기존 정제 함수로 강수 관련 파생 피처 생성
    data = clean_weather(data, StepLog())

    # 달력·팬데믹 피처도 기존 함수를 재사용
    data = add_calendar(data)

    if data["ride"].isna().any():
        raise ValueError("대시보드 데이터에 이용량 결측값이 있습니다.")

    return data


@st.cache_resource
def load_model():
    """저장 모델을 사용하거나 기존 학습 설정으로 모델을 준비한다."""

    history = load_history()

    # 기존 프로젝트와 동일한 피처 목록 사용
    feat = F.feature_columns(include_pandemic=True)

    data = F.build(history)

    data = data.dropna(subset=feat + [F.TARGET]).reset_index(drop=True)

    if len(data) < 100:
        raise ValueError("모델 학습에 필요한 데이터가 부족합니다.")

    # 기존 train.py 의 시간순 75:25 분할
    cut = int(len(data) * 0.75)

    train = data.iloc[:cut]
    test = data.iloc[cut:]

    if MODEL_PATH.exists():
        import joblib

        model = joblib.load(MODEL_PATH)

        # 저장된 모델의 피처 수가 현재 코드와 일치하는지 검사
        model_features = getattr(model, "n_features_in_", None)

        if model_features is not None and model_features != len(feat):
            raise ValueError("저장된 모델과 현재 피처 수가 일치하지 않습니다.")

        source = "저장된 Random Forest 모델"

    else:
        # 기존 train.py 와 동일한 주요 하이퍼파라미터
        model = RandomForestRegressor(
            n_estimators=400,
            random_state=SEED,
            min_samples_leaf=2,
            n_jobs=-1,
        )

        model.fit(
            train[feat].values,
            train[F.TARGET].values.astype(float),
        )

        source = "대시보드에서 시간순 학습 구간으로 재학습한 모델"

    return {
        "model": model,
        "features": feat,
        "source": source,
        "train_start": train.iloc[0]["date"].date(),
        "train_end": train.iloc[-1]["date"].date(),
        "test_start": test.iloc[0]["date"].date(),
        "test_end": test.iloc[-1]["date"].date(),
    }


def snow_mm_per_cm(history):
    """관측 데이터에서 적설(cm) 대비 강수(mm) 환산 비율을 추정한다."""

    snow = history[(history["snow_cm"] > 0) & history["snow_cm"].notna()].copy()

    if snow.empty:
        return 1.0

    ratio = (snow["precip_mm"] - snow["rain_mm"]) / snow["snow_cm"]

    ratio = ratio.replace([np.inf, -np.inf], np.nan)
    ratio = ratio[ratio > 0].dropna()

    if ratio.empty:
        return 1.0

    return float(ratio.median())


def weather_defaults(history, target):
    """선택 날짜의 실제 날씨를 슬라이더 초기값으로 사용한다."""

    same_day = history[history["date"] == pd.Timestamp(target)]

    if same_day.empty:
        raise ValueError("선택 날짜의 기록이 없습니다.")

    row = same_day.iloc[0]

    return (
        float(row["temp_mean"]),
        float(row["rain_mm"]),
        float(row["snow_cm"]),
    )


def make_row(
    history,
    target,
    temp_mean,
    rain_mm,
    snow_cm,
    mm_per_cm,
):
    """선택 날짜의 입력 피처를 기존 함수로 생성한다."""

    target_ts = pd.Timestamp(target)

    same_day = history[history["date"] == target_ts]

    if same_day.empty:
        raise ValueError("선택 날짜의 데이터가 없습니다.")

    base = same_day.iloc[0]

    # 실제 날짜의 최고·최저 기온과 평균 기온 사이의 차이를 유지
    spread_hi = max(
        float(base["temp_max"] - base["temp_mean"]),
        0.0,
    )

    spread_lo = max(
        float(base["temp_mean"] - base["temp_min"]),
        0.0,
    )

    # 가정한 적설량을 강수량으로 환산
    snow_mm = snow_cm * mm_per_cm

    row = pd.DataFrame(
        {
            "date": [target_ts],
            "ride": [np.nan],
            "temp_mean": [temp_mean],
            "temp_max": [temp_mean + spread_hi],
            "temp_min": [temp_mean - spread_lo],
            "precip_mm": [rain_mm + snow_mm],
            "rain_mm": [rain_mm],
            "snow_cm": [snow_cm],
            "wind_max": [base["wind_max"]],
            "humidity": [base["humidity"]],
        }
    )

    # 강수 파생 피처와 달력 피처는 기존 코드 사용
    row = clean_weather(row, StepLog())
    row = add_calendar(row)

    # 선택 날짜 이후 데이터는 사용하지 않음
    past = history[history["date"] < target_ts].copy()

    # 선택 날짜의 ride는 NaN으로 두고 피처를 생성
    # features.build() 내부에서 shift(1)을 사용하므로
    # 선택 날짜의 실제 이용량은 입력에 포함되지 않음
    combined = pd.concat(
        [past, row],
        ignore_index=True,
    )

    built = F.build(combined)

    return built.iloc[[-1]], past


def predict(model_info, row):
    """피처 누락·비정상값을 검사한 뒤 이용량을 예측한다."""

    feat = model_info["features"]

    values = row[feat]

    if values.isna().any().any():
        missing = values.columns[values.isna().any()].tolist()

        raise ValueError("예측 피처에 결측값이 있습니다: " + ", ".join(missing))

    array = values.to_numpy(dtype=float)

    if not np.isfinite(array).all():
        raise ValueError("예측 피처에 비정상적인 숫자가 있습니다.")

    result = model_info["model"].predict(array)

    return float(result[0])


def _reset_weather():
    """날짜를 변경하면 해당 날짜의 실제 날씨로 입력을 초기화한다."""

    target = st.session_state["pred_date"]

    temp, rain, snow = weather_defaults(
        load_history(),
        target,
    )

    st.session_state["pred_temp"] = float(np.clip(temp, -25.0, 40.0))

    st.session_state["pred_rain"] = float(np.clip(rain, 0.0, 300.0))

    st.session_state["pred_snow"] = float(np.clip(snow, 0.0, 50.0))


def show_predict_tab():
    """직접 예측 화면을 렌더링한다."""

    st.header("직접 예측해보기")

    st.caption(
        "과거 날짜와 날씨 조건을 바꿔 랜덤포레스트의 "
        "서울 지하철 일별 승하차 수요 예측을 비교합니다."
    )

    st.info(
        "이 기능은 과거 실제 이용량을 사용하는 시나리오 실험입니다. "
        "임의의 미래 날짜를 예측하는 서비스가 아닙니다."
    )

    try:
        history = load_history()

        with st.spinner("모델을 준비하는 중입니다..."):
            model_info = load_model()

    except (FileNotFoundError, ValueError, OSError) as exc:
        st.error(str(exc))
        st.stop()

    # 충분한 과거 기록이 존재하는 날짜부터 허용
    first = (history["date"].min() + pd.Timedelta(days=28)).date()

    # 실제 기록이 존재하는 날짜까지만 허용
    last = history["date"].max().date()

    available = history[
        (history["date"].dt.date >= first) & (history["date"].dt.date <= last)
    ]

    if available.empty:
        st.error("선택 가능한 날짜가 없습니다.")
        st.stop()

    default_date = pd.Timestamp("2024-07-10").date()

    if default_date < first or default_date > last:
        default_date = last

    col1, col2 = st.columns(2)

    with col1:
        target = st.date_input(
            "예측 날짜",
            value=default_date,
            min_value=first,
            max_value=last,
            key="pred_date",
            on_change=_reset_weather,
        )

    calendar = add_calendar(pd.DataFrame({"date": [pd.Timestamp(target)]})).iloc[0]

    dow = target.weekday()

    if calendar["is_holiday"]:
        day_type = "공휴일"
    elif calendar["is_weekend"]:
        day_type = "주말"
    else:
        day_type = "평일"

    with col2:
        st.metric(
            "요일 · 구분",
            f"{DAY_NAMES[dow]}요일 · {day_type}",
        )

    if target <= model_info["train_end"]:
        period_type = "학습 기간"
    else:
        period_type = "테스트 기간"

    st.caption(
        f"선택 날짜: {period_type} | "
        f"학습 종료: {model_info['train_end']} | "
        f"테스트 시작: {model_info['test_start']}"
    )

    # 최초 화면 진입 시 해당 날짜의 실제 날씨 사용
    if "pred_temp" not in st.session_state:
        _reset_weather()

    col1, col2, col3 = st.columns(3)

    temp_mean = col1.slider(
        "평균 기온 (°C)",
        min_value=-25.0,
        max_value=40.0,
        step=0.5,
        key="pred_temp",
    )

    rain_mm = col2.slider(
        "강우량 (mm)",
        min_value=0.0,
        max_value=300.0,
        step=0.5,
        key="pred_rain",
    )

    snow_cm = col3.slider(
        "적설 (cm)",
        min_value=0.0,
        max_value=50.0,
        step=0.5,
        key="pred_snow",
    )

    try:
        mm_per_cm = snow_mm_per_cm(history)

        row, past = make_row(
            history,
            target,
            temp_mean,
            rain_mm,
            snow_cm,
            mm_per_cm,
        )

        # 날짜·기온·적설은 동일하게 두고 강우량만 0mm로 변경
        row_dry, _ = make_row(
            history,
            target,
            temp_mean,
            0.0,
            snow_cm,
            mm_per_cm,
        )

        pred = predict(model_info, row)
        pred_dry = predict(model_info, row_dry)

    except (ValueError, KeyError, TypeError) as exc:
        st.error(f"예측 입력을 처리하지 못했습니다: {exc}")
        st.stop()

    # 평소 이용량 = 직전 4주 같은 요일 실제 이용량 평균
    recent = past[past["date"] >= (pd.Timestamp(target) - pd.Timedelta(days=28))]

    same_weekday = recent[recent["date"].dt.dayofweek == dow]

    usual = float(same_weekday["ride"].mean())

    same_day = history[history["date"] == pd.Timestamp(target)]

    actual = float(same_day.iloc[0]["ride"])

    st.divider()

    col1, col2, col3 = st.columns(3)

    if np.isfinite(usual) and usual > 0:
        usual_delta = (pred / usual - 1) * 100

        usual_text = f"{usual_delta:+.1f}%"
    else:
        usual_text = None

    col1.metric(
        "예상 이용량",
        f"{pred / 10000:,.1f}만",
        usual_text,
        delta_color="off",
    )

    col2.metric(
        "평소 이용량",
        (f"{usual / 10000:,.1f}만" if np.isfinite(usual) else "계산 불가"),
        help="선택 날짜 직전 4주의 같은 요일 실제 이용량 평균",
    )

    if pred_dry != 0:
        dry_delta = (pred / pred_dry - 1) * 100

        dry_text = f"{dry_delta:+.1f}%"
    else:
        dry_text = None

    col3.metric(
        "강우 0mm 가정 예측",
        f"{pred_dry / 10000:,.1f}만",
        dry_text,
        delta_color="off",
    )

    # 모델의 조건부 예측 결과를 비교하는 차트
    bars = pd.DataFrame(
        {
            "구분": [
                "평소",
                "현재 입력 예측",
                "강우 0mm 예측",
                "실제 기록",
            ],
            "이용량(만)": [
                usual / 10000,
                pred / 10000,
                pred_dry / 10000,
                actual / 10000,
            ],
        }
    )

    chart = (
        alt.Chart(bars)
        .mark_bar()
        .encode(
            x=alt.X(
                "구분:N",
                sort=list(bars["구분"]),
                title=None,
                axis=alt.Axis(labelAngle=0),
            ),
            y=alt.Y(
                "이용량(만):Q",
                title="이용량 (만)",
            ),
            color=alt.Color(
                "구분:N",
                legend=None,
            ),
            tooltip=[
                "구분:N",
                alt.Tooltip(
                    "이용량(만):Q",
                    format=",.1f",
                ),
            ],
        )
        .properties(height=320)
    )

    st.altair_chart(
        chart,
        use_container_width=True,
    )

    st.caption(
        "강우 0mm 비교는 같은 날짜의 다른 입력 조건에 대한 "
        "모델 예측값 차이입니다. 실제 강우의 인과적 효과를 "
        "뜻하지 않습니다."
    )

    if period_type == "학습 기간":
        st.warning(
            "선택 날짜는 모델의 학습 기간에 포함됩니다. "
            "따라서 실제값과 예측값의 차이를 "
            "일반화 성능으로 해석하면 안 됩니다."
        )

    # 예측에 사용된 과거 수요 피처 공개
    lag = row.iloc[0]

    st.info(
        "과거 이용량 피처는 선택 날짜 이전의 실제 기록으로 "
        "계산했습니다.\n\n"
        f"- 전날 이용량: {lag['lag_1'] / 10000:,.1f}만\n"
        f"- 7일 전 이용량: {lag['lag_7'] / 10000:,.1f}만\n"
        f"- 최근 7일 평균: {lag['roll7_mean'] / 10000:,.1f}만\n"
        f"- 최근 28일 평균: {lag['roll28_mean'] / 10000:,.1f}만"
    )

    with st.expander("모델과 입력값 자세히 보기"):
        st.write(f"**모델:** {model_info['source']}")

        st.write(
            f"**학습 기간:** {model_info['train_start']} ~ {model_info['train_end']}"
        )

        st.write(
            f"**테스트 기간:** {model_info['test_start']} ~ {model_info['test_end']}"
        )

        st.write(
            "**최고·최저 기온:** 선택 날짜의 실제 기온 차이를 "
            "유지하면서 입력한 평균 기온에 맞춰 조정"
        )

        st.write("**풍속·습도:** 선택 날짜의 관측값 사용")

        st.write(f"**적설 환산:** {mm_per_cm:.2f} mm/cm")

        st.write("**요일·공휴일·팬데믹:** 기존 clean.add_calendar() 사용")

        st.write(f"**사용 피처 수:** {len(model_info['features'])}개")

        st.caption(
            "기존 보고서의 MAPE 2.99%는 별도 검증된 "
            "학습 파이프라인의 결과입니다. "
            "이 화면에서 재학습한 모델의 성능이 "
            "자동으로 동일하다고 보장하지 않습니다."
        )


# 단독 실행 확인용
# python3 -m streamlit run dashboard/predict_tab.py
if __name__ == "__main__":
    st.set_page_config(
        page_title="직접 예측해보기",
        layout="wide",
    )

    show_predict_tab()


# ============================================================
# [구현 핵심]
# ============================================================

# 기존 clean.py와 features.py의 전처리·피처 생성 함수를 재사용함
# 사용자가 날짜, 평균 기온, 강우량, 적설량을 변경하여 지하철 수요를 예측하도록 구현함
# 선택 날짜 이전의 실제 이용량으로 lag 및 rolling 피처를 생성함
# 기존 모델의 시간순 75:25 분할과 Random Forest 학습 설정을 유지함
# 학습 기간과 테스트 기간을 구분하여 예측 결과를 해석하도록 구현함
# 동일 날짜에서 강우량을 0mm로 변경한 예측 결과와 비교하도록 구현함
# 실제 이용량, 평소 이용량, 모델 예측량을 차트로 비교하도록 구현함
#
# [발견한 문제 및 수정]
# 1. 기존 초안에서는 마지막 데이터 날짜의 다음 날까지 예측할 수 있었음
#    → 실제 이용량 기록이 존재하는 날짜로 예측 범위를 제한함
#    → 과거 기록 기반 예측과 미래 수요 예측을 구분함
# 2. 기존 초안에서는 강우량 변경에 따른 예측 차이를 '비의 영향'으로 표현했음
#    → 모델의 조건부 예측 차이로 표현을 수정함
#    → 예측 결과만으로 강우의 인과적 효과를 단정하지 않도록 안내함
# 3. 입력 데이터에 결측값이나 중복 날짜가 있으면 예측 결과가 왜곡될 수 있었음
#    → 날짜 중복과 필수 컬럼 누락 여부를 검사하도록 구현함
#    → 예측 피처의 결측값과 비정상적인 숫자를 확인하도록 보완함
# 4. 저장된 모델과 현재 피처 구성이 달라질 가능성이 있었음
#    → 저장 모델의 입력 피처 수와 현재 피처 수를 비교하도록 구현함
#    → 대시보드에서 재학습한 모델의 성능을 기존 보고서 성능과 동일하다고 단정하지 않음
#
# [추후 보완 사항]
# 1. dashboard/app.py에 직접 예측 메뉴를 연결하고 기존 화면과 통합 검증
# 2. src/train.py의 실제 학습 결과와 대시보드 재학습 결과 비교
# 3. 저장 모델의 피처 이름과 순서까지 검증하는 기능 추가
# 4. Streamlit Cloud에서 모델 로딩 시간과 메모리 사용량 확인
# 5. 웹페이지에 예측 가능 범위와 시나리오 분석의 한계를 명확하게 안내
