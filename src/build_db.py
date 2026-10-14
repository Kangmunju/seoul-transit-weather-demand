# python src/build_db.py

import argparse
import csv
import sqlite3
from pathlib import Path

import pandas as pd

from db import SCHEMA, _upsert


# 프로젝트 폴더
BASE_DIR = Path(__file__).resolve().parents[1]

# 원본 데이터 폴더
RAW_DIR = BASE_DIR / "data" / "raw"

# SQLite 파일
DB_PATH = BASE_DIR / "data" / "seoul_transit.db"

# subway_daily 테이블의 UNIQUE 제약조건과 동일한 키
SUBWAY_KEY = ["date", "line", "station", "io_type"]


def read_one(path):
    # 연도마다 인코딩이 달라서 순서대로 확인
    for encoding in ("utf-8-sig", "euc-kr", "cp949"):
        try:
            with open(path, encoding=encoding, newline="") as file:
                reader = csv.reader(file)

                # 첫 줄은 컬럼명
                header = [name.strip() for name in next(reader)]

                # 원래 컬럼 개수
                column_count = len(header)

                rows = []

                for row in reader:
                    # 컬럼이 밀려도 앞쪽 원래 컬럼까지만 사용
                    if len(row) >= column_count:
                        rows.append(row[:column_count])

                return pd.DataFrame(rows, columns=header)

        except UnicodeDecodeError:
            continue

    raise SystemExit(f"지하철 CSV 인코딩을 못 읽었습니다: {path.name}")


def analyze_duplicate_impact(daily, duplicate_mask, duplicate_rows):
    """
    기존 INSERT OR IGNORE 정책으로 제외되는 중복 행의 특성과
    전체 이용량에 대한 잠재적 영향을 분석함.

    실제 DB 적재 데이터는 변경하지 않음.
    """

    print("\n[추가 분석] 중복 데이터 영향도")

    # duplicated(keep="first")에서 True인 행만 분석
    # 같은 키가 처음 등장한 행은 기존 DB에 저장되므로 제외하지 않음
    excluded = daily.loc[duplicate_mask].copy()

    excluded_count = len(excluded)

    if excluded_count == 0:
        print("[분석] 제외 대상 중복 행이 없음")

        return {
            "excluded_rows": 0,
            "excluded_total": 0,
            "excluded_ratio": 0.0,
            "same_collected_at_groups": 0,
            "different_collected_at_groups": 0,
        }

    # 이용량을 숫자로 변환하여 분석
    # 기존 전처리 과정에서 이미 숫자로 변환했지만,
    # 합계 계산 전에 한 번 더 명시적으로 확인함
    all_total = pd.to_numeric(
        daily["total"],
        errors="coerce",
    )

    excluded_total_values = pd.to_numeric(
        excluded["total"],
        errors="coerce",
    )

    # 이용량 결측이 있으면 합계가 실제보다 작아질 수 있으므로
    # 분석 결과를 정상 수치로 출력하지 않고 중단함
    if all_total.isna().any():
        raise ValueError(
            "이용량(total)에 결측 또는 숫자 변환 실패 값이 있어 "
            "중복 영향도를 계산할 수 없습니다."
        )

    # 전체 원본 이용량
    original_total = all_total.sum()

    # 기존 정책에 따라 첫 번째 행만 남긴 이용량
    kept_total = all_total.loc[~duplicate_mask].sum()

    # 중복으로 제외된 행의 이용량
    excluded_total = excluded_total_values.sum()

    # 원본 전체 이용량 대비 제외 이용량 비율
    excluded_ratio = (
        excluded_total / original_total * 100 if original_total != 0 else 0.0
    )

    print(f"[분석] 제외 대상 중복 행: {excluded_count:,}행")
    print(f"[분석] 원본 전체 이용량: {original_total:,.0f}건")
    print(f"[분석] 기존 정책으로 유지되는 이용량: {kept_total:,.0f}건")
    print(f"[분석] 제외된 행의 이용량 합계: {excluded_total:,.0f}건")
    print(f"[분석] 원본 전체 이용량 대비 제외 비율: {excluded_ratio:.6f}%")

    # 원본 전체 이용량 = 유지 이용량 + 제외 이용량인지 확인
    if original_total != kept_total + excluded_total:
        raise ValueError("원본 이용량과 유지·제외 이용량의 합계가 일치하지 않습니다.")

    # --------------------------------------------------------
    # 1. 제외된 행의 이용량 분포
    # --------------------------------------------------------

    print("\n[분석] 제외된 행의 이용량 분포")

    print(f"  최소: {excluded_total_values.min():,.0f}건")

    print(f"  중앙값: {excluded_total_values.median():,.1f}건")

    print(f"  평균: {excluded_total_values.mean():,.2f}건")

    print(f"  최대: {excluded_total_values.max():,.0f}건")

    # 작은 이용량에 중복이 집중되는지 확인
    # 범위가 겹치지 않도록 각 조건을 분리함
    ranges = {
        "0~2건": excluded_total_values.between(0, 2),
        "3~10건": excluded_total_values.between(3, 10),
        "11~100건": excluded_total_values.between(11, 100),
        "101~1,000건": excluded_total_values.between(101, 1000),
        "1,001건 이상": excluded_total_values >= 1001,
    }

    for label, mask in ranges.items():
        count = int(mask.sum())
        ratio = count / excluded_count * 100

        print(f"  {label}: {count:,}행 ({ratio:.2f}%)")

    # --------------------------------------------------------
    # 2. 연도별 중복 발생 및 제외 이용량
    # --------------------------------------------------------

    print("\n[분석] 연도별 제외 행과 이용량")

    excluded["year"] = excluded["date"].astype(str).str[:4]

    year_summary = (
        excluded.groupby("year")
        .agg(
            excluded_rows=("total", "size"),
            excluded_total=("total", "sum"),
        )
        .sort_index()
    )

    for year, row in year_summary.iterrows():
        print(
            f"  {year}: "
            f"{int(row['excluded_rows']):,}행 / "
            f"{row['excluded_total']:,.0f}건"
        )

    # --------------------------------------------------------
    # 3. 노선별 중복 집중도
    # --------------------------------------------------------

    print("\n[분석] 중복 제외 행이 많은 노선 상위 10개")

    line_summary = (
        excluded.groupby("line")
        .agg(
            excluded_rows=("total", "size"),
            excluded_total=("total", "sum"),
        )
        .sort_values(
            ["excluded_rows", "excluded_total"],
            ascending=False,
        )
        .head(10)
    )

    for line, row in line_summary.iterrows():
        print(
            f"  {line}: "
            f"{int(row['excluded_rows']):,}행 / "
            f"{row['excluded_total']:,.0f}건"
        )

    # --------------------------------------------------------
    # 4. 역별 중복 집중도
    # --------------------------------------------------------

    print("\n[분석] 중복 제외 행이 많은 역 상위 10개")

    # 동일한 역명이 서로 다른 노선에 존재할 수 있으므로
    # 노선과 역명을 함께 묶어 분석함
    station_summary = (
        excluded.groupby(["line", "station"])
        .agg(
            excluded_rows=("total", "size"),
            excluded_total=("total", "sum"),
        )
        .sort_values(
            ["excluded_rows", "excluded_total"],
            ascending=False,
        )
        .head(10)
    )

    for (line, station), row in station_summary.iterrows():
        print(
            f"  {line} / {station}: "
            f"{int(row['excluded_rows']):,}행 / "
            f"{row['excluded_total']:,.0f}건"
        )

    # --------------------------------------------------------
    # 5. 승차·하차별 중복 발생 현황
    # --------------------------------------------------------

    print("\n[분석] 승차·하차별 제외 행과 이용량")

    io_summary = (
        excluded.groupby("io_type")
        .agg(
            excluded_rows=("total", "size"),
            excluded_total=("total", "sum"),
        )
        .sort_index()
    )

    for io_type, row in io_summary.iterrows():
        print(
            f"  {io_type}: "
            f"{int(row['excluded_rows']):,}행 / "
            f"{row['excluded_total']:,.0f}건"
        )

    # --------------------------------------------------------
    # 6. 동일 키의 등록일자 일치 여부
    # --------------------------------------------------------

    print("\n[분석] 중복 그룹의 등록일자 비교")

    # 동일 키 그룹에서 등록일자가 몇 종류인지 확인
    collected_at_variations = duplicate_rows.groupby(
        SUBWAY_KEY,
        dropna=False,
    )["collected_at"].nunique(dropna=False)

    same_collected_at_groups = int((collected_at_variations == 1).sum())

    different_collected_at_groups = int((collected_at_variations > 1).sum())

    print(f"  등록일자가 모두 같은 그룹: {same_collected_at_groups:,}개")

    print(f"  등록일자가 서로 다른 그룹: {different_collected_at_groups:,}개")

    # --------------------------------------------------------
    # 7. 가장 큰 이용량이 제외된 사례
    # --------------------------------------------------------

    print("\n[분석] 제외 이용량이 큰 사례 상위 10행")

    print(
        excluded[SUBWAY_KEY + ["total", "collected_at"]]
        .sort_values(
            "total",
            ascending=False,
        )
        .head(10)
        .to_string(index=False)
    )

    # --------------------------------------------------------
    # 분석 결과 해석 시 주의사항
    # --------------------------------------------------------

    print("\n[해석 주의]")

    print("- 제외 이용량은 현재 INSERT OR IGNORE 정책에서 저장되지 않는 행들의 합계임")

    print("- 제외 이용량 전체가 실제 누락 수요라는 뜻은 아님")

    print("- 원본만으로는 중복 집계와 별도 이용량 기록을 확정할 수 없음")

    print("- 이번 분석은 DB 저장 정책을 변경하지 않음")

    return {
        "excluded_rows": excluded_count,
        "original_total": original_total,
        "kept_total": kept_total,
        "excluded_total": excluded_total,
        "excluded_ratio": excluded_ratio,
        "same_collected_at_groups": same_collected_at_groups,
        "different_collected_at_groups": different_collected_at_groups,
    }


def validate_subway_data(daily):
    print("\n[검증] 지하철 원본 데이터 품질")

    # SQLite에서 NOT NULL과 UNIQUE가 적용되는 컬럼 확인
    # 공백만 있는 문자열도 유효한 역명이나 노선명으로 취급하지 않음
    missing_by_column = {}

    for column in SUBWAY_KEY:
        values = daily[column]

        # values.isna()
        # - NaN, None, pd.NA 등 실제 결측값을 True로 판별함
        #
        # values.astype("string")
        # - Pandas 문자열 자료형으로 변환함
        # - 기존 결측값은 문자열 "nan"이 아닌 결측값으로 유지됨
        #
        # .str.strip()
        # - 문자열 양쪽의 공백을 제거함
        # - 예: "  서울역  " → "서울역", "   " → ""
        #
        # .eq("")
        # - 공백 제거 후 빈 문자열인지 확인함
        #
        # .fillna(False)
        # - 문자열 비교에서 발생한 결측 결과를 False로 처리함
        # - 실제 결측값은 앞의 isna()에서 이미 확인함
        #
        # | 연산자
        # - Pandas Series의 요소별 OR 연산자
        # - C의 포인터 연산과 관계없으며, Python의 or와도 다름
        #
        # 실제 결측값이거나 공백만 있는 문자열이면 True
        missing = values.isna() | values.astype("string").str.strip().eq("").fillna(
            False
        )

        missing_by_column[column] = int(missing.sum())

    missing_frame = pd.DataFrame(
        {
            column: (
                daily[column].isna()
                | daily[column].astype("string").str.strip().eq("").fillna(False)
            )
            for column in SUBWAY_KEY
        }
    )

    missing_total = int(missing_frame.any(axis=1).sum())

    print(f"[검증] 전체 데이터: {len(daily):,}행")

    for column, count in missing_by_column.items():
        print(f"[검증] {column} 결측·공백: {count:,}행")

    print(f"[검증] 필수 키 누락 행: {missing_total:,}행")

    # keep='first':
    # 동일 키가 처음 등장한 행은 False,
    # 이후 중복되는 행은 True로 표시함
    duplicate_mask = daily.duplicated(
        subset=SUBWAY_KEY,
        keep="first",
    )

    duplicate_count = int(duplicate_mask.sum())

    # keep=False:
    # 중복 그룹에 속하는 첫 번째 행까지 모두 True로 표시함
    all_duplicate_mask = daily.duplicated(
        subset=SUBWAY_KEY,
        keep=False,
    )

    duplicate_rows = daily.loc[all_duplicate_mask].copy()

    print(f"[검증] UNIQUE 키 기준 중복 초과 행: {duplicate_count:,}행")

    print(f"[검증] 중복 그룹에 속한 전체 행: {len(duplicate_rows):,}행")

    # 동일 키에 서로 다른 이용량이 존재하는지 확인
    conflicting_groups = 0
    conflicting_rows = 0

    if not duplicate_rows.empty:
        group = duplicate_rows.groupby(
            SUBWAY_KEY,
            dropna=False,
            sort=False,
        )

        total_variations = group["total"].nunique(dropna=False)

        conflicting_keys = total_variations[total_variations > 1]

        conflicting_groups = len(conflicting_keys)

        if conflicting_groups:
            conflict_index = pd.MultiIndex.from_frame(duplicate_rows[SUBWAY_KEY])

            conflicting_rows = int(conflict_index.isin(conflicting_keys.index).sum())

        print(f"[검증] 이용량이 다른 중복 키 그룹: {conflicting_groups:,}개")

        print(f"[검증] 이용량 충돌 그룹에 속한 행: {conflicting_rows:,}행")

        # 중복 발생 연도 확인
        duplicate_years = (
            daily.loc[duplicate_mask, "date"]
            .astype(str)
            .str[:4]
            .value_counts()
            .sort_index()
        )

        print("[검증] 연도별 중복 초과 행")

        for year, count in duplicate_years.items():
            print(f"  {year}: {count:,}행")

        # 중복 사례 출력
        print("[검증] 중복 키 사례 (최대 10행)")

        sample_columns = SUBWAY_KEY + [
            "total",
            "collected_at",
        ]

        print(duplicate_rows[sample_columns].head(10).to_string(index=False))

        # 값이 다른 중복이 있다면 별도 사례 출력
        if conflicting_groups:
            print("[주의] 이용량이 서로 다른 중복 키가 존재함")

            first_conflicting_key = conflicting_keys.index[0]

            conflict_sample = duplicate_rows.copy()

            for column, value in zip(
                SUBWAY_KEY,
                first_conflicting_key,
            ):
                if pd.isna(value):
                    conflict_sample = conflict_sample[conflict_sample[column].isna()]
                else:
                    conflict_sample = conflict_sample[conflict_sample[column] == value]

            print(conflict_sample[sample_columns].to_string(index=False))

    else:
        print("[검증] 이용량이 다른 중복 키 그룹: 0개")

    # 필수 키 결측이 존재하면 DB에 적재하지 않음
    # INSERT OR IGNORE가 NOT NULL 위반을 조용히 건너뛰는 문제 방지
    if missing_total:
        raise ValueError(
            "지하철 원본 데이터에 필수 키 결측이 있습니다. DB 적재를 중단합니다."
        )

    # 기존 중복 제거 정책은 변경하지 않고,
    # 중복으로 제외되는 이용량의 영향도만 추가 분석함
    impact = analyze_duplicate_impact(
        daily,
        duplicate_mask,
        duplicate_rows,
    )

    return {
        "source_rows": len(daily),
        "duplicate_rows": duplicate_count,
        "unique_rows": len(daily) - duplicate_count,
        "conflicting_groups": conflicting_groups,
        "conflicting_rows": conflicting_rows,
        "impact": impact,
    }


def build_subway(con):
    subway_list = []

    # 2019~2024 파일 하나씩 읽기
    for year in range(2019, 2025):
        path = RAW_DIR / f"subway_daily_{year}.csv"

        raw = read_one(path)

        print(f"[지하철] {path.name}: {len(raw):,}행")

        subway_list.append(raw)

    # 6개 연도 합치기
    raw = pd.concat(
        subway_list,
        ignore_index=True,
    )

    print(f"[지하철] 6개 파일 결합 → {len(raw):,}행")

    print(f"[지하철] 원본 {len(raw):,}행, 열 {len(raw.columns)}")

    # 컬럼명 앞뒤 공백 제거
    raw = raw.rename(columns=lambda column: column.strip())

    # 승하차 이용량 숫자로 바꾸기
    for column in (
        "승차총승객수",
        "하차총승객수",
    ):
        raw[column] = pd.to_numeric(
            raw[column].astype(str).str.replace(",", ""),
            errors="coerce",
        )

    # 20190101 형태를 2019-01-01 형태로 바꾸기
    date = pd.to_datetime(
        raw["사용일자"].astype(str),
        format="%Y%m%d",
        errors="coerce",
    ).dt.strftime("%Y-%m-%d")

    # 승차 데이터
    board = pd.DataFrame(
        {
            "date": date,
            "line": raw["노선명"],
            "station_id": None,
            "station": raw["역명"],
            "io_type": "승차",
            "total": raw["승차총승객수"],
            "collected_at": raw["등록일자"],
        }
    )

    # 하차 데이터
    alight = pd.DataFrame(
        {
            "date": date,
            "line": raw["노선명"],
            "station_id": None,
            "station": raw["역명"],
            "io_type": "하차",
            "total": raw["하차총승객수"],
            "collected_at": raw["등록일자"],
        }
    )

    # 승차와 하차를 세로로 합치기
    daily = pd.concat(
        [board, alight],
        ignore_index=True,
    )

    # DB 적재 전에 중복·결측·이용량 영향도 검증
    validation = validate_subway_data(daily)

    subway_cols = [
        "date",
        "line",
        "station_id",
        "station",
        "io_type",
        "total",
        "collected_at",
    ]

    # 적재 전 DB 행 수 확인
    before_count = con.execute("SELECT COUNT(*) FROM subway_daily").fetchone()[0]

    # 기존 INSERT OR IGNORE 정책 유지
    inserted, skipped = _upsert(
        con,
        "subway_daily",
        subway_cols,
        daily,
    )

    after_count = con.execute("SELECT COUNT(*) FROM subway_daily").fetchone()[0]

    print(
        f"[지하철] subway_daily: "
        f"{len(daily):,}행 → "
        f"insert {inserted:,} / "
        f"skip {skipped:,}"
    )

    print(f"[검증] 원본 UNIQUE 키 기준 고유 행: {validation['unique_rows']:,}행")

    print(f"[검증] DB 적재 전후 행 수: {before_count:,} → {after_count:,}행")

    # 비어 있는 DB에 처음 적재할 때는
    # 원본 고유 키 수와 DB 행 수가 반드시 일치해야 함
    if before_count == 0:
        expected_count = validation["unique_rows"]

        if after_count != expected_count:
            raise ValueError(
                "빈 DB 적재 결과가 원본 고유 키 수와 다릅니다. "
                f"예상 {expected_count:,}행 / "
                f"실제 {after_count:,}행"
            )

        print("[ok ] 빈 DB 적재 행 수와 원본 UNIQUE 키 수 일치")

        # 신규 DB에서는 원본 순서대로 INSERT OR IGNORE하므로
        # 실제 DB의 이용량 합계와 분석상 유지 이용량도 비교함
        db_total = con.execute("SELECT SUM(total) FROM subway_daily").fetchone()[0]

        expected_total = validation["impact"]["kept_total"]

        if db_total != expected_total:
            raise ValueError(
                "DB 이용량 합계가 중복 분석 결과와 다릅니다. "
                f"예상 {expected_total:,.0f}건 / "
                f"실제 {db_total:,.0f}건"
            )

        print("[ok ] DB 이용량 합계와 중복 제외 후 이용량 일치")

    else:
        # 기존 DB 재실행에서는 이미 저장된 키도 건너뛰므로
        # skipped 전체를 원본 내부의 중복 행으로 해석하지 않음
        print("[참고] 기존 DB 재적재 시 skip에는 기존 저장 데이터와의 키 충돌도 포함됨")

    return validation


def build_weather(con):
    path = RAW_DIR / "weather_daily.csv"

    weather = pd.read_csv(path)

    # Open-Meteo 컬럼명을 DB 컬럼명으로 바꾸기
    weather = weather.rename(
        columns={
            "temperature_2m_mean": "temp_mean",
            "temperature_2m_max": "temp_max",
            "temperature_2m_min": "temp_min",
            "precipitation_sum": "precip_mm",
            "rain_sum": "rain_mm",
            "snowfall_sum": "snow_cm",
            "windspeed_10m_max": "wind_max",
            "relative_humidity_2m_mean": "humidity",
        }
    )

    # 수집 날짜 컬럼 추가
    weather["collected_at"] = None

    weather_cols = [
        "date",
        "temp_mean",
        "temp_max",
        "temp_min",
        "precip_mm",
        "rain_mm",
        "snow_cm",
        "wind_max",
        "humidity",
        "collected_at",
    ]

    inserted, skipped = _upsert(
        con,
        "weather_daily",
        weather_cols,
        weather,
    )

    print(
        f"[날씨] weather_daily: "
        f"{len(weather):,}행 → "
        f"insert {inserted:,} / "
        f"skip {skipped:,}"
    )


def main(db_path=DB_PATH):
    # 지정한 경로에 SQLite 연결
    db_path = Path(db_path)
    db_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    con = sqlite3.connect(db_path)

    try:
        # db.py에 만든 테이블 구조 적용
        con.executescript(SCHEMA)

        # 지하철과 날씨 적재
        validation = build_subway(con)
        build_weather(con)

        # 최종 행 수 확인
        subway_count = con.execute("SELECT COUNT(*) FROM subway_daily").fetchone()[0]

        subway_days = con.execute(
            "SELECT COUNT(DISTINCT date) FROM subway_daily"
        ).fetchone()[0]

        weather_count = con.execute("SELECT COUNT(*) FROM weather_daily").fetchone()[0]

        print(
            f"[검증] subway_daily "
            f"{subway_count:,}행 "
            f"(고유 {subway_days}일) / "
            f"weather_daily {weather_count}행"
        )

        print(
            f"[검증] 원본 중복 초과 행 "
            f"{validation['duplicate_rows']:,}행 / "
            f"이용량 충돌 그룹 "
            f"{validation['conflicting_groups']:,}개"
        )

        print(
            f"[검증] 중복 제외 이용량 "
            f"{validation['impact']['excluded_total']:,.0f}건 "
            f"(원본 대비 "
            f"{validation['impact']['excluded_ratio']:.6f}%)"
        )

    finally:
        con.close()

    print(f"[ok ] {db_path} 적재 완료")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="지하철 및 날씨 CSV를 SQLite DB에 적재"
    )

    parser.add_argument(
        "--db-path",
        type=Path,
        default=DB_PATH,
        help="SQLite DB 저장 경로",
    )

    args = parser.parse_args()

    main(args.db_path)


# ============================================================
# [구현 핵심]
# ============================================================

# 2019~2024년 지하철 CSV와 날씨 CSV를 읽어 SQLite에 적재함
# 승차·하차를 개별 행으로 변환하고 고유 키 기준으로 중복 저장을 방지함
# 기본 DB 경로와 별도로 임시 DB 경로를 지정할 수 있도록 구현함
# 적재 전에 원본 UNIQUE 키 중복, 필수 키 결측 및 이용량 충돌을 검증함
# 빈 DB 적재 결과와 원본 고유 키 수를 비교하여 누락 여부를 확인함
# 중복으로 제외된 행의 이용량과 전체 대비 비율을 분석함
# 연도·노선·역·승하차별 중복 집중도와 등록일자 일치 여부를 확인함
#
# [발견한 문제 및 수정]
# 1. DB 저장 경로가 고정되어 빈 DB 검증 시 기존 DB를 건드릴 위험이 있었음
#    → --db-path 옵션을 추가하여 별도 DB에 적재할 수 있도록 수정함
#
# 2. 파일을 import하면 main()이 자동 실행되는 구조였음
#    → __main__ 조건을 추가하여 직접 실행할 때만 적재하도록 수정함
#
# 3. 최초 적재에서 372행이 제외되는 실행 기록을 확인함
#    → 실제 UNIQUE 키 중복 개수와 DB 적재 건수를 비교하도록 보강함
#
# 4. INSERT OR IGNORE로 건너뛴 행을 모두 중복으로 표시하던 문제
#    → 원본 내부 중복과 DB 적재 과정의 skip을 구분하여 출력함
#
# 5. 동일 키에 서로 다른 이용량이 존재하는 문제를 확인함
#    → 중복 키별 이용량 차이와 충돌 사례를 출력하도록 구현함
#
# 6. NOT NULL 위반도 INSERT OR IGNORE에 의해 무시될 수 있었음
#    → DB 적재 전에 필수 키 결측·공백을 검사하고 오류 시 중단함
#
# 7. 중복 372행의 제외가 전체 수요에 미치는 영향이 불명확했음
#    → 제외 이용량 합계, 분포 및 전체 대비 비율을 분석하도록 추가함
#
# 8. 중복이 특정 노선·역·기간에 집중되는지 확인할 필요가 있었음
#    → 연도·노선·역·승하차별 분포와 등록일자를 비교하도록 추가함
#
# [추후 보완 사항]
# 1. 중복 영향도 분석 결과를 보고서에 기록
# 2. 원본 데이터만으로 확인할 수 없는 중복 발생 원인을 한계로 명시
# 3. 향후 원본 제공처에서 중복 레코드의 의미를 확인할 수 있다면
#    합산 또는 제외 정책을 재검토
