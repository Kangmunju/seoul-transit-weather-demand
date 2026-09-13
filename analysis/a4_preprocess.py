# clean.py에서의 전처리 문제를 실제 원본 데이터에서 다시 확인

import sqlite3
from pathlib import Path

import pandas as pd


# 프로젝트 최상위 폴더
BASE_DIR = Path(__file__).resolve().parents[1]

# SQLite 데이터베이스
DB_PATH = BASE_DIR / "data" / "seoul_transit.db"


# 전처리 하기 전에 실제로 어떤 문제가 존재했는지 확인 필요
# load_merged()를 사용하게 되면 clean.py의 전처리 과정을 거친 데이터를 가져오게 됨
# 정제 파이프라인을 통과시키기 전 DB 테이블을 직접 조회해야 함
def load_raw_data():
    # 전처리 전 상태를 확인하기 위해 DB 원본형 데이터를 직접 불러옴
    con = sqlite3.connect(DB_PATH)

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
            """,
            con,
        )

        weather = pd.read_sql_query(
            """
            SELECT
                date,
                precip_mm,
                rain_mm,
                snow_cm
            FROM weather_daily
            """,
            con,
        )

    finally:
        con.close()

    return subway, weather


# 역명 뒤에 붙은 괄호 속 부기명 제거
# clean.py에서 이미 역명 표준화 처리하였음
# 표준화한 역명의 고유 개수를 비교해 중복 취급 될 수 있는 역명이 있는지 실측
def normalize_station_name(series):
    # 문자열 맨 뒤에 있는 (내용) 부분을 찾아 제거
    return (
        series.astype("string").str.replace(r"\([^)]*\)$", "", regex=True).str.strip()
    )


def main():
    # 전처리 전 지하철 데이터, 날찌 데이터를 가저옴
    subway, weather = load_raw_data()

    print("[1. 역명 표기 흔들림]")

    # 원본 고유 역명 수
    raw_station_count = subway["station"].nunique()

    # 괄호 형태의 부기명이 붙은 역명
    station_text = subway["station"].astype("string")

    # 각 역명에 해당 패턴이 들어있는 지 확인
    annotation_mask = station_text.str.contains(
        r"\([^)]*\)",
        regex=True,
        na=False,  # 결측 역명을 부기명이 있는 역으로 잘못 세지 않도록 방지
    )

    # 전체 역명에서 괄호가 있는 것만 선택한 후 중복 제거
    annotated_names = (
        subway.loc[annotation_mask, "station"].dropna().drop_duplicates().sort_values()
    )

    # clean.py와 같은 취지로 부기명을 제거한 역명
    normalized_station = normalize_station_name(subway["station"])
    normalized_station_count = normalized_station.nunique()

    print(f"고유 역명(원본): {raw_station_count:,}개")
    print(f"부기 포함 역명: {len(annotated_names):,}개")
    print(f"표준화 후 고유 역명: {normalized_station_count:,}개")

    print("\n부기 제거 예시")

    example = pd.DataFrame(
        {
            "원본": annotated_names.head(10),
        }
    )

    example["표준화"] = normalize_station_name(example["원본"])

    print(example.to_string(index=False))

    print("\n[2. total <= 0 이상치]")

    total_numeric = pd.to_numeric(
        subway["total"],
        errors="coerce",
    )

    # total이 0 이하인 행 표시
    invalid_total = total_numeric <= 0

    print(f"total <= 0 행: {invalid_total.sum():,}건")
    print(
        "→ 행 전체를 삭제하지 않고 total 값만 결측으로 처리해 "
        "같은 날짜·역의 다른 승하차 정보는 유지"
    )

    print("\n[3. 총강수 precip와 강우 rain의 차이]")

    # precip_mm = 총 강수량(겨울이라면 눈 같은 형태의 강수도 포함됨)
    weather["precip_mm"] = pd.to_numeric(
        weather["precip_mm"],
        errors="coerce",
    )

    # rain_mm = 강우량
    weather["rain_mm"] = pd.to_numeric(
        weather["rain_mm"],
        errors="coerce",
    )

    precip_rain_diff = (
        weather["precip_mm"].notna()
        & weather["rain_mm"].notna()
        & ((weather["precip_mm"] - weather["rain_mm"]).abs() > 1e-9)
    )

    print(f"precip_mm != rain_mm 인 날: {precip_rain_diff.sum():,}일")

    print("→ 총강수(precip)를 그대로 비로 정의하면 눈이 온 날까지 비로 분류될 수 있음")

    print("→ 따라서 비 효과 분석에는 총강수가 아닌 rain_mm을 사용")

    print("\n[4. 데이터 커버리지]")

    lines = subway["line"].dropna().drop_duplicates().sort_values().tolist()

    print(f"노선 수: {len(lines):,}개")
    print("포함 노선:")
    print(lines)

    print(
        "→ 서울·수도권 전철의 여러 노선을 포함하지만 "
        "일부 민자노선까지 모두 포함한 데이터는 아님"
    )

    print(
        "→ 이후 리포트에서는 '수도권 전철 전체'가 아니라 실제 수집 범위를 명확히 밝힘"
    )


if __name__ == "__main__":
    main()


# [1. 역명 표기 흔들림]
# 고유 역명(원본): 545개
# 부기 포함 역명: 61개
# 표준화 후 고유 역명: 540개

# 부기 제거 예시
#          원본 표준화
#  강변(동서울터미널)  강변
# 경복궁(정부서울청사) 경복궁
#     고려대(종암) 고려대
# 공릉(서울과학기술대)  공릉
#    관악산(서울대) 관악산
#    광나루(장신대) 광나루
# 광화문(세종문화회관) 광화문
#     광흥창(서강) 광흥창
#  교대(법원.검찰청)  교대
#    구의(광진구청)  구의

# [2. total <= 0 이상치]
# total <= 0 행: 6,764건
# → 행 전체를 삭제하지 않고 total 값만 결측으로 처리해 같은 날짜·역의 다른 승하차 정보는 유지

# [3. 총강수 precip와 강우 rain의 차이]
# precip_mm != rain_mm 인 날: 102일
# → 총강수(precip)를 그대로 비로 정의하면 눈이 온 날까지 비로 분류될 수 있음
# → 따라서 비 효과 분석에는 총강수가 아닌 rain_mm을 사용

# [4. 데이터 커버리지]
# 노선 수: 27개
# 포함 노선:
# ['1호선', '2호선', '3호선', '4호선', '5호선', '6호선', '7호선', '8호선', '9호선', '9호선2~3단계', '경강선', '경부선', '경원선', '경의선', '경인선', '경춘선', '공항철도 1호선', '과천선', '분당선', '서해선', '수인선', '신림선', '안산선', '우이신설선', '일산선', '장항선', '중앙선']
# → 서울·수도권 전철의 여러 노선을 포함하지만 일부 민자노선까지 모두 포함한 데이터는 아님
# → 이후 리포트에서는 '수도권 전철 전체'가 아니라 실제 수집 범위를 명확히 밝힘


# ============================================================

# clean.py에서 이미 처리한 전처리 로직이 실제 원본 데이터에서 왜 필요했는지 재실측이 필요하다고 판단
# 정제된 merged 데이터가 아니라 SQLite의 원본형 테이블을 직접 조회함

# 역명에 괄호 형태의 부기명이 붙으면 같은 역이 서로 다른 역명처럼 집계될 가능성 있음
# 부기 제거 전후의 고유 역명 수와 실제 변환 예시를 함께 확인하여 처리함

# total <= 0 값을 발견했더라도 행 전체를 삭제하지 않음
# 같은 날짜, 역의 다른 정상 승하차 정보까지 잃을 수 있으므로 문제 있는 total 값만 결측 처리

# 총강수 precip_mm과 실제 강우 rain_mm이 다른 날짜가 존재함을 확인
# 총강수에는 눈이 포함될 수 있음
# 비 효과 분석에서는 precip_mm이 아니라 rain_mm을 사용
# 눈이 온 날을 비 오는 날로 오분류하는 문제를 방지함

# 실제 포함 노선의 수와 목록을 직접 확인해 데이터 커버리지를 점검함
# 현재 데이터는 여러 서울, 수도권 노선을 포함하지만 일부 민자 노선은 빠져 있음
# 이후 README와 리포트에서 실제 수집 범위와 한계를 명확히 밝혀 작성할 것

# 이번 분석은 새로운 전처리를 추가하는 목적이 아님
# 기존 clean.py의 전처리 판단이 실제 데이터에서 필요했음을 수치와 사례로 검증하는 역할을 보인 것
