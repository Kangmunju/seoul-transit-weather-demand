# 추가 분석 결과 시각화: a6, a7, a9, a10
# 기존 대시보드의 기능을 유지하면서 분석 결과를 확장함

import json
from pathlib import Path

# a6의 모델 성능 비교, a7의 강우 영향 비교, a9의 신뢰구간, a10의 연도별 회복률 시각화 필요
# 막대그래프나 선그래프 등으로 시각화하기 위해 altair를 import함
import altair as alt
import pandas as pd
import streamlit as st


MODEL_DIR = Path(__file__).resolve().parents[1] / "models"


@st.cache_data
def load_result(filename):
    # 분석 결과 JSON을 읽음
    # 파일이 없는 경우 전체 대시보드를 중단하지 않음
    path = MODEL_DIR / filename

    if not path.is_file():
        return None

    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def show_adjusted_rain():
    # ------------------------------------------------------------
    # a7. 단순 비교, 동일 조건 매칭, 회귀 보정
    # ------------------------------------------------------------
    result = load_result("rain_adjusted_results.json")

    st.divider()
    st.subheader("강우 영향: 단순 비교와 통계적 보정 (a7)")

    if result is None:
        st.warning(
            "models/rain_adjusted_results.json 파일이 없어 "
            "a7 분석을 표시할 수 없습니다."
        )
        return

    simple = result["simple_comparison"]
    matched = result["matched_comparison"]
    regression = result["adjusted_regression"]

    st.caption(
        f"강우일 기준: {result['rain_definition']} · 단위: 일별 승하차 합산 건수"
    )

    values = pd.DataFrame(
        [
            {
                "분석 방법": "단순 평균",
                "차이 (%)": simple["difference_pct"],
            },
            {
                "분석 방법": "동일 조건 매칭",
                "차이 (%)": matched["weighted_difference_pct"],
            },
            {
                "분석 방법": "다변량 회귀 보정",
                "차이 (%)": regression["relative_to_mean_pct"],
            },
        ]
    )

    chart = (
        alt.Chart(values)
        .mark_bar()
        .encode(
            x=alt.X(
                "분석 방법:N",
                title=None,
                sort=None,
            ),
            y=alt.Y(
                "차이 (%):Q",
                title="비가 없는 날 대비 차이 (%)",
            ),
            tooltip=[
                "분석 방법:N",
                alt.Tooltip(
                    "차이 (%):Q",
                    format=".2f",
                ),
            ],
        )
        .properties(height=260)
    )

    st.altair_chart(chart, use_container_width=True)

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "단순 평균 차이",
        f"{simple['difference_pct']:+.2f}%",
    )

    col2.metric(
        "매칭 후 차이",
        f"{matched['weighted_difference_pct']:+.2f}%",
    )

    col3.metric(
        "회귀 보정 후 차이",
        f"{regression['relative_to_mean_pct']:+.2f}%",
    )

    st.info(
        f"회귀분석에서 강우일은 하루 평균 약 "
        f"{regression['rain_effect']:,.0f}건 낮은 이용량과 "
        f"연관되었습니다. "
        f"95% 신뢰구간은 "
        f"{regression['ci_95_low']:,.0f}~"
        f"{regression['ci_95_high']:,.0f}건입니다. "
        "관측자료에 대한 보정 분석이므로 비가 이용량 감소를 "
        "직접 유발했다고 단정할 수는 없습니다."
    )


def show_workday_rain():
    # ------------------------------------------------------------
    # a9. 같은 연도·월의 평일 평균을 기준으로 강우 영향 보정
    # ------------------------------------------------------------
    result = load_result("rain_workday_adjusted_results.json")

    st.divider()
    st.subheader("평일 강우 강도별 보정 효과 (a9)")

    if result is None:
        st.warning(
            "models/rain_workday_adjusted_results.json 파일이 없어 "
            "a9 분석을 표시할 수 없습니다."
        )
        return

    comparison = result["rain_comparison"]

    st.caption(
        f"분석 기간: {result['analysis_period']['start']}~"
        f"{result['analysis_period']['end']} · "
        f"평일 {result['workday_days']:,}일 · "
        f"보정 기준: {result['adjustment']} · "
        f"강우일: {result['rain_definition']}"
    )

    col1, col2 = st.columns(2)

    col1.metric(
        "평일 강우일 차이",
        f"{comparison['difference_pp']:+.2f}%p",
    )

    col2.metric(
        "95% 신뢰구간",
        f"{comparison['ci_95_low']:+.2f}~{comparison['ci_95_high']:+.2f}%p",
    )

    intensity = pd.DataFrame(result["intensity_comparisons"])
    intensity["구간"] = intensity["target"]

    order = intensity["구간"].tolist()

    # 강우 강도별 이용량 차이 막대그래프
    bars = (
        alt.Chart(intensity)
        .mark_bar()
        .encode(
            x=alt.X(
                "구간:N",
                sort=order,
                title="강우 강도",
            ),
            y=alt.Y(
                "difference_pp:Q",
                title="비가 없는 날 대비 차이 (%p)",
            ),
            tooltip=[
                alt.Tooltip("구간:N"),
                alt.Tooltip(
                    "difference_pp:Q",
                    title="차이 (%p)",
                    format=".2f",
                ),
                alt.Tooltip(
                    "ci_95_low:Q",
                    title="95% CI 하한",
                    format=".2f",
                ),
                alt.Tooltip(
                    "ci_95_high:Q",
                    title="95% CI 상한",
                    format=".2f",
                ),
            ],
        )
    )

    # 부트스트랩 신뢰구간 표시
    errors = (
        alt.Chart(intensity)
        .mark_rule(strokeWidth=3)
        .encode(
            x=alt.X("구간:N", sort=order),
            y="ci_95_low:Q",
            y2="ci_95_high:Q",
        )
    )

    st.altair_chart(
        (bars + errors).properties(height=300),
        use_container_width=True,
    )

    table = intensity[
        [
            "구간",
            "reference_days",
            "target_days",
            "difference_pp",
            "ci_95_low",
            "ci_95_high",
        ]
    ].rename(
        columns={
            "reference_days": "무강우 비교일",
            "target_days": "강우일",
            "difference_pp": "차이 (%p)",
            "ci_95_low": "95% CI 하한",
            "ci_95_high": "95% CI 상한",
        }
    )

    st.dataframe(
        table.style.format(
            {
                "차이 (%p)": "{:+.2f}",
                "95% CI 하한": "{:+.2f}",
                "95% CI 상한": "{:+.2f}",
            }
        ),
        hide_index=True,
        use_container_width=True,
    )

    st.caption(
        f"95% 신뢰구간은 {result['bootstrap']['method']} 방식의 "
        f"부트스트랩 {result['bootstrap']['iterations']:,}회로 "
        "계산했습니다. 약한 비(0~5mm)는 신뢰구간에 0이 포함됩니다. "
        "강우 강도별 비교는 정확히 0mm를 기준으로 하며, "
        "전체 강우일 비교는 1mm 이상을 기준으로 합니다."
    )


def show_weather_ablation():
    # ------------------------------------------------------------
    # a6. 날씨 변수 포함/제외 실험
    # ------------------------------------------------------------
    result = load_result("weather_ablation_results.json")

    st.divider()
    st.subheader("날씨 변수가 예측 성능에 기여했을까? (a6)")

    if result is None:
        st.warning(
            "models/weather_ablation_results.json 파일이 없어 "
            "a6 분석을 표시할 수 없습니다."
        )
        return

    split = result["split_comparison"]
    with_weather = split["with_weather"]
    without_weather = split["without_weather"]

    st.caption(
        f"테스트: {split['test_start']}~{split['test_end']} · "
        "동일한 모델 설정에서 날씨 변수 포함 여부 비교"
    )

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "날씨 포함 MAPE",
        f"{with_weather['MAPE_%']:.2f}%",
    )

    col2.metric(
        "날씨 제외 MAPE",
        f"{without_weather['MAPE_%']:.2f}%",
    )

    col3.metric(
        "날씨 포함 개선 폭",
        f"{split['improvement_pp']:+.2f}%p",
    )

    heavy = result["heavy_rain"]

    st.info(
        f"{heavy['threshold_mm']:.0f}mm 이상 강우일 "
        f"{heavy['days']}일에서는 날씨 포함 모델 MAPE "
        f"{heavy['with_weather_mape']:.2f}%, "
        f"제외 모델 {heavy['without_weather_mape']:.2f}%로 "
        f"{heavy['improvement_pp']:.2f}%p 차이가 나타났습니다. "
        "표본 수가 적으므로 일반화에 주의해야 합니다."
    )

    # Walk-forward 구간별 성능 비교
    walk_forward = pd.DataFrame(result["walk_forward"])

    walk_forward["fold_label"] = "Fold " + walk_forward["fold"].astype(str)

    walk_forward_long = walk_forward.melt(
        id_vars=["fold_label"],
        value_vars=[
            "with_weather_mape",
            "without_weather_mape",
        ],
        var_name="모델",
        value_name="MAPE (%)",
    )

    walk_forward_long["모델"] = walk_forward_long["모델"].map(
        {
            "with_weather_mape": "날씨 포함",
            "without_weather_mape": "날씨 제외",
        }
    )

    chart = (
        alt.Chart(walk_forward_long)
        .mark_bar()
        .encode(
            x=alt.X(
                "fold_label:N",
                sort=walk_forward["fold_label"].tolist(),
                title="검증 구간",
            ),
            xOffset="모델:N",
            y=alt.Y("MAPE (%):Q"),
            color="모델:N",
            tooltip=[
                "fold_label:N",
                "모델:N",
                alt.Tooltip(
                    "MAPE (%):Q",
                    format=".2f",
                ),
            ],
        )
        .properties(height=300)
    )

    st.altair_chart(chart, use_container_width=True)

    helpful_folds = int((walk_forward["improvement_pp"] > 0).sum())

    mean_improvement = walk_forward["improvement_pp"].mean()

    st.warning(
        f"Walk-forward {len(walk_forward)}개 구간 중 "
        f"날씨 변수가 도움이 된 구간은 {helpful_folds}개이며, "
        f"평균 개선 폭은 {mean_improvement:+.2f}%p였습니다. "
        "따라서 단일 테스트 구간의 성능 개선이 모든 시기에서 "
        "재현된다고 볼 수는 없습니다."
    )


def show_fixed_station_pandemic():
    # ------------------------------------------------------------
    # a10. 전체 역과 고정 역의 팬데믹 회복률 비교
    # ------------------------------------------------------------
    result = load_result("pandemic_station_fixed_results.json")

    st.divider()
    st.subheader("역 구성 변화의 영향을 제외하면? (a10)")

    if result is None:
        st.warning(
            "models/pandemic_station_fixed_results.json 파일이 없습니다. "
            "a10 분석을 먼저 실행해 주세요."
        )
        return

    annual = pd.DataFrame(result["annual_comparison"])
    recovery = result["recovery_2024"]

    st.caption(
        f"{result['analysis_period']['start_year']}~"
        f"{result['analysis_period']['end_year']}년 · "
        f"매년 관측률 {result['min_coverage']:.0%} 이상 · "
        f"고정 노선-역 조합 "
        f"{result['fixed_station_count']:,}개"
    )

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "2024년 전체 역",
        f"{recovery['all_change_pct']:+.2f}%",
    )

    col2.metric(
        "2024년 고정 역",
        f"{recovery['fixed_change_pct']:+.2f}%",
    )

    col3.metric(
        "회복률 차이",
        f"{recovery['difference_pp']:+.2f}%p",
    )

    # 연도별 회복률을 긴 형태로 변환하여 두 기준을 비교
    plot = annual.melt(
        id_vars="year",
        value_vars=[
            "all_change_pct",
            "fixed_change_pct",
        ],
        var_name="집계 기준",
        value_name="2019년 대비 변화율 (%)",
    )

    plot["집계 기준"] = plot["집계 기준"].map(
        {
            "all_change_pct": "전체 역",
            "fixed_change_pct": "고정 역",
        }
    )

    plot["year"] = plot["year"].astype(str)

    chart = (
        alt.Chart(plot)
        .mark_line(point=True)
        .encode(
            x=alt.X(
                "year:O",
                title="연도",
            ),
            y=alt.Y(
                "2019년 대비 변화율 (%):Q",
            ),
            color="집계 기준:N",
            tooltip=[
                "year:O",
                "집계 기준:N",
                alt.Tooltip(
                    "2019년 대비 변화율 (%):Q",
                    format=".2f",
                ),
            ],
        )
        .properties(height=320)
    )

    st.altair_chart(chart, use_container_width=True)

    # 고정 역 선정 기준별 민감도 분석
    sensitivity = pd.DataFrame(result["sensitivity_analysis"])

    sensitivity["관측률 기준"] = sensitivity["threshold"].map(
        lambda value: f"{value:.0%}"
    )

    st.markdown("**고정 역 선정 기준 민감도 분석**")

    table = sensitivity[
        [
            "관측률 기준",
            "fixed_station_count",
            "fixed_change_pct",
            "difference_pp",
        ]
    ].rename(
        columns={
            "fixed_station_count": "고정 역 수",
            "fixed_change_pct": "2024년 변화율 (%)",
            "difference_pp": "전체 역 대비 차이 (%p)",
        }
    )

    st.dataframe(
        table.style.format(
            {
                "2024년 변화율 (%)": "{:+.2f}",
                "전체 역 대비 차이 (%p)": "{:+.2f}",
            },
            na_rep="분석 불가",
        ),
        hide_index=True,
        use_container_width=True,
    )

    st.caption(
        "고정 역 비교는 역 구성 변화의 영향을 줄이기 위한 분석이며, "
        "팬데믹의 인과적 효과를 직접 추정하지 않습니다."
    )


# ============================================================
# [구현 핵심]
# ============================================================

# 1. a6·a7·a9·a10 결과 JSON을 읽어 기존 Streamlit 화면에 추가함
# 2. 기존 데이터·모델·DB를 수정하거나 재학습하지 않음
# 3. 신뢰구간과 분석 조건을 명시해 과도한 인과 해석을 방지함
# 4. 분석 결과가 없는 경우 전체 앱을 중단하지 않고 안내함
#
# [발견한 문제 및 수정]
# 1. a6~a10 결과가 기존 대시보드에 노출되지 않았음
#    → 분석별 시각화 함수를 별도 모듈로 분리함
# 2. a7과 a9의 강우 기준·보정 방법이 달라 직접 비교에 주의가 필요함
#    → 각 분석의 기준을 독립적으로 표시함
#
# [추후 보완 사항]
# 1. 다양한 연도별 구간에 대한 추가 검증
# 2. 이용자 선택 조건에 따른 시각화 필터 확장
