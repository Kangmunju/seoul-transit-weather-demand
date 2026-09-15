from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
import json

import pandas as pd
import streamlit as st
import altair as alt


# 프로젝트 최상위 경로
BASE_DIR = Path(__file__).resolve().parents[1]

# 파일 경로
MODEL_DIR = BASE_DIR / "models"
DATA_PATH = BASE_DIR / "dashboard" / "dashboard_data.csv"


# Streamlit 페이지 기본 설정
st.set_page_config(
    page_title="서울 지하철 수요 분석",
    page_icon="🚇",
    layout="wide",
)


@st.cache_data
def load_data():
    # 대시보드용 일별 데이터
    data = pd.read_csv(DATA_PATH)
    data["date"] = pd.to_datetime(data["date"])

    # 모델 성능
    with open(MODEL_DIR / "metrics.json", "r", encoding="utf-8") as file:
        metrics = json.load(file)

    # 변수 중요도
    importance = pd.read_csv(MODEL_DIR / "feature_importance.csv")

    # 테스트 기간 실제값 / 예측값
    predictions = pd.read_csv(MODEL_DIR / "test_predictions.csv")

    return data, metrics, importance, predictions


def show_overview(data, metrics):
    st.header("프로젝트 개요")

    # 핵심 지표 계산
    start_year = data["date"].min().year
    end_year = data["date"].max().year

    avg_ride = data["ride"].mean()

    rf_r2 = metrics["random_forest"]["R2"]
    rf_mape = metrics["random_forest"]["MAPE_%"]

    # 핵심 지표 표시
    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "분석 기간",
        f"{start_year} ~ {end_year}",
    )

    col2.metric(
        "분석 일수",
        f"{len(data):,}일",
    )

    col3.metric(
        "일평균 이용량",
        f"{avg_ride / 10000:,.0f}만 명",
    )

    col4.metric(
        "Random Forest R²",
        f"{rf_r2:.4f}",
    )

    st.caption(f"Random Forest 테스트 MAPE: {rf_mape:.2f}%")

    st.divider()

    # 연도별 일평균 이용량
    st.subheader("연도별 일평균 이용량")

    yearly = data.groupby("year")["ride"].mean().reset_index()

    yearly["year"] = yearly["year"].astype(str)

    st.line_chart(
        yearly,
        x="year",
        y="ride",
    )

    st.caption(
        "2020년 팬데믹 시기에 이용량이 크게 감소한 뒤 "
        "2021~2024년에 점진적으로 회복되는 흐름을 확인할 수 있습니다."
    )

    st.divider()

    # 프로젝트 핵심 결과
    st.subheader("핵심 분석 결과")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("#### 📅 주간 수요 패턴")
        st.write(
            "지난주 같은 요일의 수요와 현재 수요의 관계가 강하게 나타났으며, "
            "평일과 주말 사이에도 뚜렷한 이용량 차이가 확인되었습니다."
        )

    with col2:
        st.markdown("#### 🌧️ 날씨 영향")
        st.write(
            "비가 오는 날의 이용량은 감소하는 경향을 보였지만, "
            "요일 효과에 비하면 상대적으로 작은 영향을 보였습니다."
        )

    with col3:
        st.markdown("#### 🤖 수요 예측")
        st.write(
            f"Random Forest가 테스트 데이터에서 "
            f"R² {rf_r2:.4f}, MAPE {rf_mape:.2f}%의 성능을 기록했습니다."
        )


def show_usage_pattern(data):
    st.header("이용 패턴")

    # 평일 / 주말 평균 이용량
    weekday_avg = data.loc[
        data["is_workday"] == 1,
        "ride",
    ].mean()

    weekend_avg = data.loc[
        data["is_weekend"] == 1,
        "ride",
    ].mean()

    weekend_ratio = weekend_avg / weekday_avg * 100

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "평일 일평균 이용량",
        f"{weekday_avg / 10000:,.0f}만 명",
    )

    col2.metric(
        "주말 일평균 이용량",
        f"{weekend_avg / 10000:,.0f}만 명",
    )

    col3.metric(
        "주말 / 평일",
        f"{weekend_ratio:.1f}%",
    )

    st.divider()

    # 요일별 평균 이용량
    st.subheader("요일별 평균 이용량")

    day_names = {
        0: "월",
        1: "화",
        2: "수",
        3: "목",
        4: "금",
        5: "토",
        6: "일",
    }

    daily_pattern = data.groupby("dow")["ride"].mean().reset_index()

    daily_pattern["요일"] = daily_pattern["dow"].map(day_names)

    st.bar_chart(
        daily_pattern,
        x="요일",
        y="ride",
    )

    st.caption(
        "평일에는 높은 이용량이 유지되며 금요일이 가장 높고, "
        "주말에는 이용량이 크게 감소하는 주간 패턴이 나타납니다."
    )

    st.divider()

    # 시계열 자기상관
    st.subheader("주간 반복성")

    lag1_corr = data["ride"].corr(data["ride"].shift(1))

    lag7_corr = data["ride"].corr(data["ride"].shift(7))

    col1, col2 = st.columns(2)

    col1.metric(
        "전날과의 상관관계",
        f"{lag1_corr:.3f}",
    )

    col2.metric(
        "지난주 같은 요일과의 상관관계",
        f"{lag7_corr:.3f}",
    )

    st.info(
        "지난주 같은 요일(lag-7)의 상관관계가 전날(lag-1)보다 높습니다. "
        "지하철 이용량에 강한 주간 반복 패턴이 존재한다는 것을 보여줍니다."
    )


def show_weather_effect(data):
    st.header("날씨 영향")

    # 비 오는 날 / 비 안 오는 날
    rainy = data[data["is_rain"] == 1]
    dry = data[data["is_rain"] == 0]

    rainy_avg = rainy["ride"].mean()
    dry_avg = dry["ride"].mean()
    rainy_ratio = rainy_avg / dry_avg * 100

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "비 오는 날 일평균",
        f"{rainy_avg / 10000:,.0f}만 명",
    )

    col2.metric(
        "비 안 오는 날 일평균",
        f"{dry_avg / 10000:,.0f}만 명",
    )

    col3.metric(
        "비 오는 날 / 비 안 오는 날",
        f"{rainy_ratio:.1f}%",
    )

    st.caption(f"비 오는 날 {len(rainy):,}일 / 비 안 오는 날 {len(dry):,}일")

    st.divider()

    # 평일만 비교
    st.subheader("요일 효과를 통제하면?")

    workday = data[data["is_workday"] == 1]

    workday_rainy = workday[workday["is_rain"] == 1]
    workday_dry = workday[workday["is_rain"] == 0]

    workday_rainy_avg = workday_rainy["ride"].mean()
    workday_dry_avg = workday_dry["ride"].mean()

    workday_ratio = workday_rainy_avg / workday_dry_avg * 100

    comparison = pd.DataFrame(
        {
            "구분": ["평일 + 비", "평일 + 비 없음"],
            "일평균 이용량": [
                workday_rainy_avg,
                workday_dry_avg,
            ],
        }
    )

    st.bar_chart(
        comparison,
        x="구분",
        y="일평균 이용량",
    )

    st.info(
        f"평일끼리 비교해도 비 오는 날의 이용량은 "
        f"비가 오지 않는 날의 {workday_ratio:.1f}% 수준입니다. "
        "즉 비의 영향은 존재하지만 평일·주말 같은 달력 효과보다 "
        "상대적으로 작게 나타납니다."
    )

    st.divider()

    # 강우량과 이용량의 상관관계
    st.subheader("강우량과 이용량의 관계")

    rain_corr = data["rain_mm"].corr(data["ride"])

    col1, col2 = st.columns(2)

    col1.metric(
        "강우량 ↔ 이용량 상관계수",
        f"{rain_corr:.3f}",
    )

    col2.metric(
        "평일 비 오는 날 이용 수준",
        f"{workday_ratio:.1f}%",
    )

    st.caption(
        "상관계수가 0에 가까울수록 두 변수 사이의 단순한 "
        "선형 관계가 약하다는 의미입니다."
    )

    st.divider()

    # 강우 강도별 이용량
    st.subheader("강우 강도별 평균 이용량")

    rain_level = pd.cut(
        data["rain_mm"],
        bins=[-0.01, 0, 5, 20, float("inf")],
        labels=[
            "비 없음",
            "5mm 이하",
            "5~20mm",
            "20mm 초과",
        ],
    )

    rain_group = (
        data.assign(rain_level=rain_level)
        .groupby("rain_level", observed=True)["ride"]
        .mean()
        .reset_index()
    )

    st.bar_chart(
        rain_group,
        x="rain_level",
        y="ride",
    )

    st.caption(
        "단순히 비가 왔는지만 비교하는 것에서 더 나아가 "
        "강우량 수준에 따라 평균 이용량이 어떻게 달라지는지 확인합니다."
    )


def show_pandemic(data):
    st.header("팬데믹 변화")

    # 연도별 일평균 이용량
    yearly = data.groupby("year")["ride"].mean().reset_index()

    base_2019 = yearly.loc[
        yearly["year"] == 2019,
        "ride",
    ].iloc[0]

    yearly["2019 대비 변화율"] = (yearly["ride"] / base_2019 - 1) * 100

    # 핵심 수치
    ride_2019 = yearly.loc[
        yearly["year"] == 2019,
        "ride",
    ].iloc[0]

    ride_2020 = yearly.loc[
        yearly["year"] == 2020,
        "ride",
    ].iloc[0]

    ride_2024 = yearly.loc[
        yearly["year"] == 2024,
        "ride",
    ].iloc[0]

    drop_2020 = (ride_2020 / ride_2019 - 1) * 100
    diff_2024 = (ride_2024 / ride_2019 - 1) * 100

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "2019 일평균",
        f"{ride_2019 / 10000:,.0f}만 명",
    )

    col2.metric(
        "2020 변화",
        f"{drop_2020:.1f}%",
    )

    col3.metric(
        "2024 vs 2019",
        f"{diff_2024:.1f}%",
    )

    st.divider()

    # 연도별 변화
    st.subheader("급감과 회복")

    chart_data = yearly.copy()
    chart_data["year"] = chart_data["year"].astype(str)

    st.line_chart(
        chart_data,
        x="year",
        y="ride",
    )

    st.info(
        "2019년과 비교해 2020년 이용량이 크게 감소했고, "
        "이후 지속적으로 회복했습니다. 다만 2024년에도 "
        "2019년 수준을 완전히 회복하지는 못했습니다."
    )

    st.divider()

    # 팬데믹 국면별 평일 평균
    st.subheader("팬데믹 국면별 평일 이용량")

    workday = data[data["is_workday"] == 1].copy()

    def pandemic_period(row):
        if row["date"] < pd.Timestamp("2020-01-01"):
            return "팬데믹 이전"

        elif row["pandemic_acute"] == 1:
            return "급성기"

        elif row["pandemic_recovery"] == 1:
            return "회복기"

        else:
            return "엔데믹 이후"

    workday["국면"] = workday.apply(
        pandemic_period,
        axis=1,
    )

    period_order = [
        "팬데믹 이전",
        "급성기",
        "회복기",
        "엔데믹 이후",
    ]

    period_avg = (
        workday.groupby("국면")["ride"].mean().reindex(period_order).reset_index()
    )

    st.bar_chart(
        period_avg,
        x="국면",
        y="ride",
    )

    before_avg = period_avg.loc[
        period_avg["국면"] == "팬데믹 이전",
        "ride",
    ].iloc[0]

    acute_avg = period_avg.loc[
        period_avg["국면"] == "급성기",
        "ride",
    ].iloc[0]

    acute_change = (acute_avg / before_avg - 1) * 100

    st.caption(
        f"요일 효과를 줄이기 위해 평일만 비교했습니다. "
        f"급성기 평일 이용량은 팬데믹 이전보다 "
        f"{abs(acute_change):.1f}% 감소했습니다."
    )

    st.divider()

    # 주말 / 평일 구조 변화
    st.subheader("생활 패턴 자체도 변했을까?")

    def weekend_workday_ratio(df):
        weekday_avg = df.loc[
            df["is_workday"] == 1,
            "ride",
        ].mean()

        weekend_avg = df.loc[
            df["is_weekend"] == 1,
            "ride",
        ].mean()

        return weekend_avg / weekday_avg * 100

    ratio_2019 = weekend_workday_ratio(data[data["year"] == 2019])

    ratio_acute = weekend_workday_ratio(data[data["year"].isin([2020, 2021])])

    ratio_2024 = weekend_workday_ratio(data[data["year"] == 2024])

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "2019 주말 / 평일",
        f"{ratio_2019:.1f}%",
    )

    col2.metric(
        "2020~2021 주말 / 평일",
        f"{ratio_acute:.1f}%",
    )

    col3.metric(
        "2024 주말 / 평일",
        f"{ratio_2024:.1f}%",
    )

    st.info(
        "팬데믹 급성기에는 전체 이용량만 감소한 것이 아니라 "
        "주말 이용량이 평일보다 더 크게 위축되었습니다. "
        "2024년에는 주간 이용 패턴도 팬데믹 이전 수준에 가까워졌습니다."
    )


def show_prediction(metrics):
    st.header("수요 예측")

    # 테스트 예측 결과 불러오기
    pred_path = ROOT / "models" / "test_predictions.csv"
    pred = pd.read_csv(pred_path)

    pred["date"] = pd.to_datetime(pred["date"])

    # Random Forest 성능
    rf_metrics = metrics["random_forest"]

    mae = rf_metrics["MAE"]
    mape = rf_metrics["MAPE_%"]
    r2 = rf_metrics["R2"]

    # ---------------------------------------------------------
    # 핵심 수치
    # ---------------------------------------------------------

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "테스트 기간",
        f"{len(pred):,}일",
    )

    col2.metric(
        "MAE",
        f"{mae:,.0f}명",
    )

    col3.metric(
        "MAPE",
        f"{mape:.2f}%",
    )

    col4.metric(
        "R²",
        f"{r2:.4f}",
    )

    st.caption(
        f"테스트 기간: {pred['date'].min().date()} ~ {pred['date'].max().date()}"
    )

    st.divider()

    # ---------------------------------------------------------
    # 실제 이용량과 Random Forest 예측 비교
    # 전체 기간은 주간 평균으로 표시
    # ---------------------------------------------------------

    st.subheader("실제 이용량 vs Random Forest 예측")

    # 전체 547일을 그대로 표시하면 선이 너무 빽빽하므로
    # 7일 단위 평균으로 묶어서 전체적인 흐름을 비교
    weekly_pred = (
        pred.set_index("date")[["actual", "random_forest"]]
        .resample("7D")
        .mean()
        .reset_index()
    )

    # 실제 이용량: 파란색 실선
    actual_line = (
        alt.Chart(weekly_pred)
        .mark_line(
            color="#2563EB",
            strokeWidth=2.5,
        )
        .encode(
            x=alt.X(
                "date:T",
                title="날짜",
            ),
            y=alt.Y(
                "actual:Q",
                title="주간 평균 이용량",
                scale=alt.Scale(zero=False),
            ),
            tooltip=[
                alt.Tooltip(
                    "date:T",
                    title="주 시작일",
                    format="%Y-%m-%d",
                ),
                alt.Tooltip(
                    "actual:Q",
                    title="실제 이용량",
                    format=",.0f",
                ),
            ],
        )
    )

    # Random Forest 예측: 빨간색 점선
    prediction_line = (
        alt.Chart(weekly_pred)
        .mark_line(
            color="#DC2626",
            strokeWidth=2,
            strokeDash=[7, 5],
        )
        .encode(
            x=alt.X(
                "date:T",
                title="날짜",
            ),
            y=alt.Y(
                "random_forest:Q",
                title="주간 평균 이용량",
                scale=alt.Scale(zero=False),
            ),
            tooltip=[
                alt.Tooltip(
                    "date:T",
                    title="주 시작일",
                    format="%Y-%m-%d",
                ),
                alt.Tooltip(
                    "random_forest:Q",
                    title="Random Forest 예측",
                    format=",.0f",
                ),
            ],
        )
    )

    prediction_chart = alt.layer(
        actual_line,
        prediction_line,
    ).properties(
        height=420,
    )

    st.altair_chart(
        prediction_chart,
        use_container_width=True,
    )

    # 범례 직접 표시
    st.markdown(
        """
        <div style="display:flex; gap:28px; margin-top:-8px; margin-bottom:18px;">
            <span>
                <span style="color:#2563EB; font-weight:bold;">━━━━</span>
                실제 이용량
            </span>
            <span>
                <span style="color:#DC2626; font-weight:bold;">━ ━ ━</span>
                Random Forest 예측
            </span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.caption(
        "전체 테스트 기간의 흐름을 보기 쉽도록 "
        "실제 이용량과 예측 이용량을 7일 단위 평균으로 표시했습니다."
    )

    st.info(
        "Random Forest는 테스트 데이터에서 "
        f"MAPE {mape:.2f}%, R² {r2:.4f}를 기록했습니다. "
        "전체적인 이용량 변화와 주간 반복 패턴을 비교적 잘 따라갑니다."
    )

    st.divider()

    # ---------------------------------------------------------
    # 세 모델 비교
    # ---------------------------------------------------------

    st.subheader("모델별 예측 성능 비교")

    model_names = {
        "baseline_lag7": "지난주 동일 요일",
        "linear_regression": "Linear Regression",
        "random_forest": "Random Forest",
    }

    model_rows = []

    for key, name in model_names.items():
        result = metrics[key]

        model_rows.append(
            {
                "모델": name,
                "MAE": result["MAE"],
                "MAPE (%)": result["MAPE_%"],
                "R²": result["R2"],
            }
        )

    model_df = pd.DataFrame(model_rows)

    st.dataframe(
        model_df.style.format(
            {
                "MAE": "{:,.0f}",
                "MAPE (%)": "{:.2f}",
                "R²": "{:.4f}",
            }
        ),
        hide_index=True,
        use_container_width=True,
    )

    st.success(
        "Random Forest가 베이스라인과 선형회귀보다 "
        "가장 낮은 오차와 가장 높은 R²를 기록했습니다."
    )

    st.divider()

    # ---------------------------------------------------------
    # 날짜 범위 선택
    # ---------------------------------------------------------

    st.subheader("기간을 선택해 예측 결과 확인")

    min_date = pred["date"].min().date()
    max_date = pred["date"].max().date()

    # 처음 화면에서는 최근 30일 표시
    default_end = max_date
    default_start = max_date - pd.Timedelta(days=29)

    selected = st.date_input(
        "날짜 범위",
        value=(default_start, default_end),
        min_value=min_date,
        max_value=max_date,
    )

    if isinstance(selected, (tuple, list)) and len(selected) == 2:
        start_date, end_date = selected

        filtered = pred[
            (pred["date"].dt.date >= start_date) & (pred["date"].dt.date <= end_date)
        ].copy()

        # 선택 기간 실제 이용량: 파란색 실선
        filtered_actual_line = (
            alt.Chart(filtered)
            .mark_line(
                color="#2563EB",
                strokeWidth=2.5,
            )
            .encode(
                x=alt.X(
                    "date:T",
                    title="날짜",
                ),
                y=alt.Y(
                    "actual:Q",
                    title="이용량",
                    scale=alt.Scale(zero=False),
                ),
                tooltip=[
                    alt.Tooltip(
                        "date:T",
                        title="날짜",
                        format="%Y-%m-%d",
                    ),
                    alt.Tooltip(
                        "actual:Q",
                        title="실제 이용량",
                        format=",.0f",
                    ),
                ],
            )
        )

        # 선택 기간 Random Forest 예측: 빨간색 점선
        filtered_prediction_line = (
            alt.Chart(filtered)
            .mark_line(
                color="#DC2626",
                strokeWidth=2,
                strokeDash=[7, 5],
            )
            .encode(
                x=alt.X(
                    "date:T",
                    title="날짜",
                ),
                y=alt.Y(
                    "random_forest:Q",
                    title="이용량",
                    scale=alt.Scale(zero=False),
                ),
                tooltip=[
                    alt.Tooltip(
                        "date:T",
                        title="날짜",
                        format="%Y-%m-%d",
                    ),
                    alt.Tooltip(
                        "random_forest:Q",
                        title="Random Forest 예측",
                        format=",.0f",
                    ),
                ],
            )
        )

        filtered_prediction_chart = alt.layer(
            filtered_actual_line,
            filtered_prediction_line,
        ).properties(
            height=420,
        )

        st.altair_chart(
            filtered_prediction_chart,
            use_container_width=True,
        )

        # 선택 기간 그래프 범례
        st.markdown(
            """
            <div style="display:flex; gap:28px; margin-top:-8px; margin-bottom:18px;">
                <span>
                    <span style="color:#2563EB; font-weight:bold;">━━━━</span>
                    실제 이용량
                </span>
                <span>
                    <span style="color:#DC2626; font-weight:bold;">━ ━ ━</span>
                    Random Forest 예측
                </span>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.divider()

    # ---------------------------------------------------------
    # 예측 오차가 컸던 날짜
    # ---------------------------------------------------------

    st.subheader("예측 오차가 컸던 날짜")

    pred["absolute_error"] = (pred["actual"] - pred["random_forest"]).abs()

    pred["error_rate"] = pred["absolute_error"] / pred["actual"] * 100

    worst = pred.nlargest(
        10,
        "absolute_error",
    )[
        [
            "date",
            "actual",
            "random_forest",
            "absolute_error",
            "error_rate",
        ]
    ].copy()

    worst.columns = [
        "날짜",
        "실제 이용량",
        "예측 이용량",
        "절대 오차",
        "오차율 (%)",
    ]

    st.dataframe(
        worst.style.format(
            {
                "날짜": lambda x: x.strftime("%Y-%m-%d"),
                "실제 이용량": "{:,.0f}",
                "예측 이용량": "{:,.0f}",
                "절대 오차": "{:,.0f}",
                "오차율 (%)": "{:.2f}",
            }
        ),
        hide_index=True,
        use_container_width=True,
    )

    st.caption(
        "절대 오차가 큰 날짜를 별도로 확인하면 "
        "일반적인 주간 패턴으로 설명하기 어려운 특수한 날짜를 "
        "찾아볼 수 있습니다."
    )


def show_model_analysis():
    st.header("모델 분석")

    st.caption(
        "Random Forest가 어떤 정보를 중요하게 사용했는지와 "
        "시계열 검증에서 나타난 모델의 한계를 확인합니다."
    )

    # ---------------------------------------------------------
    # 변수 중요도
    # ---------------------------------------------------------

    st.subheader("Random Forest 변수 중요도 TOP 10")

    importance_path = ROOT / "models" / "feature_importance.csv"
    importance = pd.read_csv(importance_path)

    top10 = importance.sort_values("importance", ascending=False).head(10).copy()

    # 그래프에서 보기 쉽도록 변수명 한글 표시
    feature_names = {
        "is_workday": "평일 여부",
        "lag_1": "전날 이용량",
        "pandemic_acute": "팬데믹 급성기",
        "roll7_mean": "최근 7일 평균",
        "roll28_mean": "최근 28일 평균",
        "lag_7": "7일 전 이용량",
        "roll7_std": "최근 7일 변동성",
        "pandemic_recovery": "팬데믹 회복기",
        "is_holiday": "공휴일 여부",
        "wind_max": "최대 풍속",
    }

    top10["변수"] = top10["feature"].map(feature_names).fillna(top10["feature"])

    top10["중요도 (%)"] = top10["importance"] * 100

    # 중요도가 높은 변수가 위에 오도록 정렬
    importance_chart = (
        alt.Chart(top10)
        .mark_bar()
        .encode(
            x=alt.X(
                "중요도 (%):Q",
                title="변수 중요도 (%)",
            ),
            y=alt.Y(
                "변수:N",
                sort="-x",
                title=None,
            ),
            tooltip=[
                alt.Tooltip(
                    "변수:N",
                    title="변수",
                ),
                alt.Tooltip(
                    "feature:N",
                    title="원본 변수명",
                ),
                alt.Tooltip(
                    "중요도 (%):Q",
                    title="중요도",
                    format=".2f",
                ),
            ],
        )
        .properties(
            height=420,
        )
    )

    st.altair_chart(
        importance_chart,
        use_container_width=True,
    )

    st.info(
        "가장 중요한 변수는 '평일 여부(is_workday)'였습니다. "
        "그다음으로 전날 이용량(lag_1), 팬데믹 급성기 여부, "
        "최근 7일 평균 이용량 등이 높은 중요도를 보였습니다."
    )

    st.divider()

    # ---------------------------------------------------------
    # 주요 변수 해석
    # ---------------------------------------------------------

    st.subheader("모델은 무엇을 중요하게 봤을까?")

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "평일 여부",
        f"{top10.loc[top10['feature'] == 'is_workday', '중요도 (%)'].iloc[0]:.1f}%",
    )

    col2.metric(
        "전날 이용량",
        f"{top10.loc[top10['feature'] == 'lag_1', '중요도 (%)'].iloc[0]:.1f}%",
    )

    col3.metric(
        "팬데믹 급성기",
        f"{top10.loc[top10['feature'] == 'pandemic_acute', '중요도 (%)'].iloc[0]:.1f}%",
    )

    st.markdown(
        """
        **평일 여부 (is_workday)**  
        평일과 주말·공휴일의 이용량 차이가 크기 때문에
        수요를 예측하는 데 가장 중요한 변수로 나타났습니다.

        **전날 이용량 (lag_1)**  
        바로 전날의 이용량은 현재 수요 수준을 파악하는 데
        중요한 정보를 제공합니다.

        **팬데믹 급성기 (pandemic_acute)**  
        코로나19 시기의 급격한 이용량 감소는 일반적인
        요일·날씨 패턴만으로 설명하기 어려운 구조 변화였습니다.

        **최근 7일·28일 평균 (roll7_mean, roll28_mean)**  
        단일 날짜의 이용량뿐 아니라 최근 수요 수준과
        장기적인 흐름도 모델이 함께 활용했습니다.
        """
    )

    st.divider()

    # ---------------------------------------------------------
    # 랜덤 split vs 시간순 split
    # ---------------------------------------------------------

    st.subheader("왜 시계열 데이터는 시간순으로 검증해야 할까?")

    split_result = pd.DataFrame(
        {
            "검증 방식": [
                "랜덤 split",
                "시간순 split",
            ],
            "MAE": [
                349963,
                347269,
            ],
            "MAPE (%)": [
                3.61,
                2.99,
            ],
            "R²": [
                0.9723,
                0.9649,
            ],
            "가장 가까운 학습 데이터": [
                "중앙값 1일",
                "중앙값 274일",
            ],
        }
    )

    st.dataframe(
        split_result.style.format(
            {
                "MAE": "{:,.0f}",
                "MAPE (%)": "{:.2f}",
                "R²": "{:.4f}",
            }
        ),
        hide_index=True,
        use_container_width=True,
    )

    st.warning(
        "랜덤 split은 미래와 과거 데이터를 섞기 때문에 "
        "테스트 날짜와 매우 가까운 날짜가 학습 데이터에 포함될 수 있습니다. "
        "따라서 이 프로젝트의 최종 평가는 시간 순서를 유지한 "
        "시간순 split을 기준으로 사용했습니다."
    )

    st.divider()

    # ---------------------------------------------------------
    # Walk-forward 검증
    # ---------------------------------------------------------

    st.subheader("Walk-forward 시계열 검증")

    walk_forward = pd.DataFrame(
        {
            "fold": [
                "fold1",
                "fold2",
                "fold3",
                "fold4",
                "fold5",
            ],
            "학습일수": [
                365,
                729,
                1093,
                1457,
                1821,
            ],
            "테스트일수": [
                364,
                364,
                364,
                364,
                364,
            ],
            "MAE": [
                3092521,
                363771,
                659678,
                384803,
                336583,
            ],
            "MAPE (%)": [
                34.01,
                4.08,
                5.49,
                3.40,
                2.84,
            ],
            "R²": [
                -0.178,
                0.969,
                0.914,
                0.956,
                0.965,
            ],
        }
    )

    st.dataframe(
        walk_forward.style.format(
            {
                "MAE": "{:,.0f}",
                "MAPE (%)": "{:.2f}",
                "R²": "{:.3f}",
            }
        ),
        hide_index=True,
        use_container_width=True,
    )

    # fold별 R² 그래프
    fold_chart = (
        alt.Chart(walk_forward)
        .mark_bar()
        .encode(
            x=alt.X(
                "fold:N",
                title="검증 구간",
            ),
            y=alt.Y(
                "R²:Q",
                title="R²",
            ),
            tooltip=[
                alt.Tooltip(
                    "fold:N",
                    title="Fold",
                ),
                alt.Tooltip(
                    "학습일수:Q",
                    title="학습 일수",
                ),
                alt.Tooltip(
                    "MAPE (%):Q",
                    title="MAPE",
                    format=".2f",
                ),
                alt.Tooltip(
                    "R²:Q",
                    title="R²",
                    format=".3f",
                ),
            ],
        )
        .properties(
            height=350,
        )
    )

    st.altair_chart(
        fold_chart,
        use_container_width=True,
    )

    st.error(
        "fold1은 2019년 데이터로 학습한 뒤 2020년을 예측하는 구간으로, "
        "팬데믹에 따른 갑작스러운 수요 급락 때문에 R²가 -0.178까지 "
        "하락했습니다."
    )

    st.info(
        "이후 fold에서는 R²가 0.914~0.969 수준으로 회복되었습니다. "
        "이는 과거 패턴이 반복되는 일반적인 시기에는 높은 예측력을 보이지만, "
        "팬데믹처럼 과거 데이터에 없던 급격한 구조 변화에서는 "
        "성능이 크게 떨어질 수 있음을 보여줍니다."
    )

    st.divider()

    # ---------------------------------------------------------
    # 최종 해석
    # ---------------------------------------------------------

    st.subheader("모델 분석 요약")

    st.success(
        "Random Forest의 변수 중요도를 확인한 결과, 날씨 변수보다 "
        "평일 여부와 과거 이용량 같은 시간·달력 관련 변수가 예측에 더 중요하게 "
        "사용되었습니다. Random Forest는 일반적인 수요 패턴에서는 높은 예측력을 "
        "보였지만, 팬데믹과 같은 갑작스러운 구조 변화에는 취약할 수 있다는 한계도 "
        "시계열 검증을 통해 확인했습니다."
    )


def main():
    data, metrics, importance, predictions = load_data()

    # 제목
    st.title("🚇 서울 지하철 × 날씨 수요 분석")

    st.write(
        "2019~2024년 서울·수도권 철도 이용량과 날씨 데이터를 결합하여 "
        "수요 패턴을 분석하고 머신러닝으로 일별 이용량을 예측한 프로젝트입니다."
    )

    # 사이드바
    st.sidebar.title("분석 메뉴")

    menu = st.sidebar.radio(
        "페이지 선택",
        [
            "프로젝트 개요",
            "이용 패턴",
            "날씨 영향",
            "팬데믹 변화",
            "수요 예측",
            "모델 분석",
        ],
    )

    if menu == "프로젝트 개요":
        show_overview(data, metrics)

    elif menu == "이용 패턴":
        show_usage_pattern(data)

    elif menu == "날씨 영향":
        show_weather_effect(data)

    elif menu == "팬데믹 변화":
        show_pandemic(data)

    elif menu == "수요 예측":
        show_prediction(metrics)

    elif menu == "모델 분석":
        show_model_analysis()

    else:
        st.info(f"'{menu}' 화면은 다음 단계에서 추가합니다.")


if __name__ == "__main__":
    main()
