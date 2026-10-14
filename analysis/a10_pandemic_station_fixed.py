# 팬데믹 회복률 분석: 전체 역과 고정 역 집합 비교
# 연도별 역 구성 변화가 2019년 대비 이용량 변화율에 미치는 영향 검증

import json
import re
import sqlite3
from pathlib import Path

import pandas as pd

from _common import DB_PATH


START_YEAR = 2019
END_YEAR = 2024

# 매년 전체 날짜 중 최소 95% 이상 관측된 역을 고정 역으로 선정
MIN_COVERAGE = 0.95

# Streamlit에서 사용할 분석 결과 저장 경로
RESULT_PATH = (
    Path(__file__).resolve().parents[1]
    / "models"
    / "pandemic_station_fixed_results.json"
)


def rule(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def load_station_data():
    # DB 파일이 없으면 sqlite3가 새 DB를 생성하지 않도록 사전 확인
    if not DB_PATH.is_file():
        raise FileNotFoundError(f"SQLite DB 파일이 없습니다: {DB_PATH}")

    # 일반 sqlite3.connect()는 DB 파일이 없으면 새 DB를 생성할 수 있음
    # 기존 DB의 데이터 추가·수정·삭제도 허용되지 않음
    # mode=ro : 기존 DB를 읽기 전용으로 열어 데이터 변경을 방지함
    # uri=True : 연결 문자열을 일반파일명이 아닌 URI로 해석하도록 설정함
    # 이번 분석은 기존 DB를 수정하지 않고 조회만 해야 하므로 읽기 전용 연결을 사용함
    con = sqlite3.connect(
        f"file:{DB_PATH.as_posix()}?mode=ro",
        uri=True,
    )

    try:
        subway = pd.read_sql_query(
            """
            SELECT
                date,
                line,
                station,
                io_type,
                total
            FROM subway_daily
            WHERE date >= '2019-01-01'
              AND date < '2025-01-01'
            """,
            con,
        )
    finally:
        con.close()

    if subway.empty:
        raise ValueError("분석 기간에 해당하는 지하철 데이터가 없습니다.")

    # 날짜와 이용량 자료형 변환
    subway["date"] = pd.to_datetime(subway["date"], errors="coerce")
    subway["total"] = pd.to_numeric(subway["total"], errors="coerce")

    # 분석에 필요한 필수 키 검증
    required = ["date", "line", "station", "io_type"]

    # 첫 번째 any() : 각 컬럼에 결측값이 하나라도 있는지 확인
    # 두 번째 any() : 여러 컬럼 중 결측값이 존재하는 컬럼이 하나라도 있는지 확인
    # 필수 컬럼 전체에서 결측값이 단 하나라도 발견되면 True 반환하도록 함
    if subway[required].isna().any().any():
        raise ValueError("날짜, 노선, 역명, 승하차 구분에 결측값이 있습니다.")

    for col in ["line", "station", "io_type"]:
        subway[col] = subway[col].astype(str).str.strip()

        if subway[col].eq("").any():
            raise ValueError(f"{col} 컬럼에 공백 값이 있습니다.")

    # 기존 clean.py와 동일한 괄호 부기명 제거 방식 사용
    subway["station"] = subway["station"].map(
        lambda name: re.sub(r"\(.*?\)", "", name).strip()
    )

    if subway["station"].eq("").any():
        raise ValueError("역명 표준화 후 빈 역명이 발생했습니다.")

    # 기존 전처리와 동일하게 0 이하 이용량은 결측 처리
    subway.loc[subway["total"] <= 0, "total"] = float("nan")

    subway["year"] = subway["date"].dt.year

    # 동일 날짜·노선·역의 승차와 하차 이용량을 합산
    # 동명역이라도 노선이 다르면 별도 역으로 취급
    station_daily = subway.groupby(
        ["date", "year", "line", "station"],
        as_index=False,
    ).agg(ride=("total", lambda s: s.sum(min_count=1)))

    return station_daily


def make_station_coverage(station_daily):
    # 연도별 날짜 수: 윤년에는 366일
    days_by_year = {
        year: (
            pd.Timestamp(year=year + 1, month=1, day=1)
            - pd.Timestamp(year=year, month=1, day=1)
        ).days
        for year in range(START_YEAR, END_YEAR + 1)
    }

    # 실제 이용량이 존재하는 날짜만 관측일로 인정
    valid = station_daily.dropna(subset=["ride"]).copy()

    # 같은 역에 승하차 데이터가 함께 존재하거나 동일 날짜가 여러 행에
    # 등장하더라도 날짜는 한 번만 계산함
    # 역별 실제 관측일 수를 계산하기 위해 nunique()를 사용함
    coverage = valid.groupby(
        ["year", "line", "station"],
        as_index=False,
    ).agg(observed_days=("date", "nunique"))

    coverage["calendar_days"] = coverage["year"].map(days_by_year)
    coverage["coverage"] = coverage["observed_days"] / coverage["calendar_days"]

    return coverage


def select_fixed_stations(coverage, min_coverage):
    # 각 연도에 관측률 기준을 만족하는 역만 선택
    # coverage["coverage"] : 각 역의 해당 관측률
    # >= min_coverage : 지정한 최소 관측률 이상인지 True/False로 확인
    # 위 조건을 만족하는 coverage 행만 선택해 별도의 DataFrame으로 복사
    # 이 단계에서는 연도별 관측률만 검사함
    # 이후 groupby()와 nunique()를 이용하여 모든 연도에서 기준을
    # 만족했는지 확인함
    eligible = coverage[coverage["coverage"] >= min_coverage].copy()

    # 2019~2024년 모든 연도에서 기준을 만족하는 역
    required_years = END_YEAR - START_YEAR + 1

    # (노선, 역명)별로 관측률 기준을 만족한 연도의 개수를 계산
    # 고정 역 선정에는 서로 다른 연도가 몇 개인지 확인하는 nunique()가 적합함
    station_years = eligible.groupby(["line", "station"])["year"].nunique()

    fixed_keys = station_years[station_years == required_years].index

    return pd.DataFrame(
        fixed_keys.tolist(),
        columns=["line", "station"],
    )


def annual_mean(station_daily):
    # 전체 역 기준: 날짜별 총이용량 계산
    daily_total = station_daily.groupby("date")["ride"].sum(min_count=1)

    # 분석 기간의 전체 날짜를 명시적으로 구성
    # 관측 자체가 없는 날은 임의로 0건 처리하지 않음
    calendar = pd.date_range(
        f"{START_YEAR}-01-01",
        f"{END_YEAR}-12-31",
        freq="D",
    )

    daily_total = daily_total.reindex(calendar)

    # 누락 날짜가 있으면 결과를 조용히 계산하지 않고 오류 발생
    missing = daily_total.isna().sum()

    if missing:
        raise ValueError(
            f"전체 일별 이용량에서 관측되지 않은 날짜가 {missing}일 있습니다."
        )

    # a5와 동일하게 연도별 일평균 이용량 계산
    return daily_total.groupby(daily_total.index.year).mean()


def calculate_comparison(station_daily, fixed_stations):
    if fixed_stations.empty:
        raise ValueError("모든 연도에서 관측률 기준을 만족하는 고정 역이 없습니다.")

    # 기존 방식: 매년 관측된 전체 역
    all_mean = annual_mean(station_daily)

    # 고정 역 목록과 (노선, 역명) 기준으로 결합
    fixed_daily = station_daily.merge(
        fixed_stations,
        on=["line", "station"],
        how="inner",
        validate="many_to_one",
    )

    # 보강 방식: 동일한 고정 역 집합만 사용
    fixed_mean = annual_mean(fixed_daily)

    result = pd.DataFrame(
        {
            "year": all_mean.index,
            "all_mean": all_mean.values,
            "fixed_mean": fixed_mean.reindex(all_mean.index).values,
        }
    )

    # 연도별 집계 결과에는 2019년이 한 행만 존재함
    # 첫 번째 값을 가져오면 2019년 전체 역 일평균 이용량이 됨
    # Series가 아닌 숫자 하나가 필요하기 때문에 iloc[0]을 사용함
    base_all = result.loc[result["year"] == START_YEAR, "all_mean"].iloc[0]

    # 위와 동일한 방식으로 2019년 고정 역 일평균 이용량을 추출함
    # 이후 각 연도의 이용량을 이 기준값으로 나누어 변화율을 계산함
    base_fixed = result.loc[result["year"] == START_YEAR, "fixed_mean"].iloc[0]

    result["all_change_pct"] = (result["all_mean"] / base_all - 1) * 100
    result["fixed_change_pct"] = (result["fixed_mean"] / base_fixed - 1) * 100

    # 동일한 기준연도 대비 변화율의 차이(%p)
    result["difference_pp"] = result["fixed_change_pct"] - result["all_change_pct"]

    return result


def main():
    if not 0 < MIN_COVERAGE <= 1:
        raise ValueError("MIN_COVERAGE는 0 초과 1 이하이어야 합니다.")

    station_daily = load_station_data()
    coverage = make_station_coverage(station_daily)

    years = set(range(START_YEAR, END_YEAR + 1))
    actual_years = set(station_daily["year"].unique())

    if years != actual_years:
        raise ValueError(
            f"필요한 연도 {sorted(years)}와 "
            f"실제 연도 {sorted(actual_years)}가 일치하지 않습니다."
        )

    # ------------------------------------------------------------
    # 1. 연도별 역 구성 확인
    # ------------------------------------------------------------

    rule("① 연도별 역 구성과 관측률")

    print(
        f"고정 역 선정 기준: "
        f"{START_YEAR}~{END_YEAR}년 매년 "
        f"관측률 {MIN_COVERAGE * 100:.0f}% 이상"
    )

    print(f"{'연도':<8}{'역명 수':>12}{'노선-역 조합':>16}{'95% 이상 관측':>18}")

    for year in range(START_YEAR, END_YEAR + 1):
        sub = station_daily[station_daily["year"] == year]
        cov = coverage[coverage["year"] == year]

        # 해당 연도에 등장한 서로 다른 역명의 개수를 계산
        # 같은 역명이 여러 날짜에 반복되더라도 한 번만 계산함
        # 서로 다른 노선에 같은 역명이 있으면 하나의 역명으로 계산됨
        # 따라서 아래의 (노선, 역명) 조합 수와는 결과가 다를 수 있음
        station_names = sub["station"].nunique()
        station_pairs = sub[["line", "station"]].drop_duplicates().shape[0]
        eligible_count = (cov["coverage"] >= MIN_COVERAGE).sum()

        print(f"{year:<8}{station_names:>12,}{station_pairs:>16,}{eligible_count:>18,}")

    # ------------------------------------------------------------
    # 2. 고정 역 집합 선정
    # ------------------------------------------------------------

    rule("② 고정 역 집합 선정")

    fixed_stations = select_fixed_stations(
        coverage,
        MIN_COVERAGE,
    )

    print(f"고정 역 수(노선-역 조합): {len(fixed_stations):,}개")

    if fixed_stations.empty:
        raise ValueError("고정 역 집합이 비어 있습니다.")

    # ------------------------------------------------------------
    # 3. 기존 방식과 고정 역 방식 비교
    # ------------------------------------------------------------

    rule("③ 연도별 일평균 이용량 비교")

    result = calculate_comparison(
        station_daily,
        fixed_stations,
    )

    print(f"{'연도':<8}{'전체 역 일평균':>18}{'고정 역 일평균':>18}")

    for row in result.itertuples(index=False):
        print(f"{row.year:<8}{row.all_mean:>18,.0f}{row.fixed_mean:>18,.0f}")

    rule("④ 2019년 대비 이용량 변화율 비교")

    print(f"{'연도':<8}{'전체 역':>14}{'고정 역':>14}{'차이(%p)':>14}")

    for row in result.itertuples(index=False):
        print(
            f"{row.year:<8}"
            f"{row.all_change_pct:>+13.2f}%"
            f"{row.fixed_change_pct:>+13.2f}%"
            f"{row.difference_pp:>+13.2f}"
        )

    # ------------------------------------------------------------
    # 4. 2024년 회복 수준 확인
    # ------------------------------------------------------------

    rule("⑤ 2024년 회복 수준 비교")

    # 연도별 집계 결과에서 2024년은 한 행이므로 iloc[0]을 이용해
    # 해당 행을 추출함
    # 앞에서는 특정 컬럼의 숫자 하나를 추출
    # 여기서는 2024년의 전체 비교 결과가 담긴 행(Series)을 가져옴
    target = result.loc[result["year"] == END_YEAR].iloc[0]

    print(f"전체 역 기준: {target['all_change_pct']:+.2f}%")
    print(f"고정 역 기준: {target['fixed_change_pct']:+.2f}%")
    print(f"두 방식의 차이: {target['difference_pp']:+.2f}%p")

    print(
        "\n해석 주의: 고정 역 비교는 역 구성 변화의 영향을 "
        "줄이기 위한 분석이며, 팬데믹의 인과적 효과를 "
        "직접 추정하는 것은 아닙니다."
    )

    # ------------------------------------------------------------
    # 5. 관측률 기준 민감도 분석
    # ------------------------------------------------------------

    rule("⑥ 관측률 기준별 민감도 분석")

    print(f"{'기준':<12}{'고정 역 수':>14}{'2024년 변화율':>18}{'전체 대비 차이':>18}")

    # Streamlit용 JSON에 저장할 민감도 분석 결과
    sensitivity_results = []

    for threshold in [0.90, 0.95, 1.00]:
        selected = select_fixed_stations(
            coverage,
            threshold,
        )

        if selected.empty:
            print(f"{threshold * 100:>3.0f}%{'분석 불가':>20}")

            # 분석 불가 조건도 JSON에 기록하여 결과 누락과 구분
            sensitivity_results.append(
                {
                    "threshold": float(threshold),
                    "fixed_station_count": 0,
                    "fixed_change_pct": None,
                    "difference_pp": None,
                }
            )
            continue

        comparison = calculate_comparison(
            station_daily,
            selected,
        )

        last = comparison.loc[comparison["year"] == END_YEAR].iloc[0]

        print(
            f"{threshold * 100:>3.0f}%"
            f"{len(selected):>18,}"
            f"{last['fixed_change_pct']:>+17.2f}%"
            f"{last['difference_pp']:>+17.2f}%p"
        )

        sensitivity_results.append(
            {
                "threshold": float(threshold),
                "fixed_station_count": int(len(selected)),
                "fixed_change_pct": float(last["fixed_change_pct"]),
                "difference_pp": float(last["difference_pp"]),
            }
        )

    # ------------------------------------------------------------
    # 6. Streamlit용 분석 결과 저장
    # ------------------------------------------------------------

    rule("⑦ Streamlit용 분석 결과 저장")

    # JSON에 저장할 수 있도록 pandas/numpy 값을 기본 자료형으로 변환
    annual_results = []

    for row in result.itertuples(index=False):
        annual_results.append(
            {
                "year": int(row.year),
                "all_mean": float(row.all_mean),
                "fixed_mean": float(row.fixed_mean),
                "all_change_pct": float(row.all_change_pct),
                "fixed_change_pct": float(row.fixed_change_pct),
                "difference_pp": float(row.difference_pp),
            }
        )

    result_data = {
        "experiment": "pandemic_station_fixed",
        "analysis_period": {
            "start_year": START_YEAR,
            "end_year": END_YEAR,
        },
        "station_definition": "line + standardized_station",
        "min_coverage": MIN_COVERAGE,
        "fixed_station_count": int(len(fixed_stations)),
        "annual_comparison": annual_results,
        "recovery_2024": {
            "all_change_pct": float(target["all_change_pct"]),
            "fixed_change_pct": float(target["fixed_change_pct"]),
            "difference_pp": float(target["difference_pp"]),
        },
        "sensitivity_analysis": sensitivity_results,
    }

    # models 폴더가 없으면 생성
    RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)

    # 한글이 깨지지 않도록 UTF-8로 저장
    with open(RESULT_PATH, "w", encoding="utf-8") as file:
        json.dump(
            result_data,
            file,
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )

    print(f"분석 결과 저장 완료: {RESULT_PATH}")


if __name__ == "__main__":
    main()


# ============================================================
# [구현 핵심]
# ============================================================

# 1. SQLite DB를 읽기 전용으로 조회하여 기존 데이터와 모델을 변경하지 않음
# 2. (노선, 표준화된 역명)을 역 식별자로 사용하여 동명역 혼동을 방지함
# 3. 각 연도의 실제 관측일 수를 계산하고 고정 역 집합을 선정함
# 4. 2019~2024년 모든 연도에서 동일한 역 집합을 사용함
# 5. 전체 역과 고정 역의 일평균 이용량 및 2019년 대비 변화율을 비교함
# 6. 관측률 90%, 95%, 100% 기준의 민감도 분석을 수행함
# 7. 연도별 비교, 2024년 회복률, 민감도 분석 결과를 JSON으로 저장함
# 8. Streamlit에서 SQLite DB를 다시 조회하지 않고 결과를 사용할 수 있도록 구성함

# [발견한 문제 및 수정]
# 1. 기존 a5 분석은 날짜별 전체 이용량을 사용하여 역 구성 변화를 구분하지 못함
#    → 역 단위 데이터를 다시 집계하여 고정 역 기준 결과를 별도로 계산함

# 2. 연도별로 등장한 역을 단순 교집합으로 선정하면 관측일이 적은 역도 포함됨
#    → 연도별 관측률을 계산하여 일정 기준 이상인 역만 선정함

# 3. 서로 다른 노선에 같은 이름의 역이 존재할 수 있음
#    → 역명만이 아닌 노선과 역명 조합을 기준으로 구분함

# 4. 기존 분석 결과가 콘솔에만 출력되어 Streamlit에서 직접 활용하기 어려웠음
#    → models/pandemic_station_fixed_results.json 저장 기능을 추가함

# 5. 관측률 기준에 따라 고정 역 집합이 비어 있을 수 있음
#    → 분석 불가 결과를 null로 저장하여 정상적인 수치와 구분함

# [추후 보완 사항]
# 1. 역명 변경이나 노선 변경으로 동일 역이 다르게 식별되는 경우 검토
# 2. 고정 역과 제외 역의 연도별 이용량 기여도 추가 분석
# 3. 고정 역 기준 팬데믹 국면별 평일 이용량 및 주말/평일 비율 검증
# 4. Streamlit 팬데믹 변화 메뉴에 전체 역·고정 역 비교 시각화 추가
