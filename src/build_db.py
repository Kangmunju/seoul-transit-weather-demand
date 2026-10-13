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


def build_subway(con):
    subway_list = []

    # 2019~2024 파일 하나씩 읽기
    for year in range(2019, 2025):
        path = RAW_DIR / f"subway_daily_{year}.csv"

        raw = read_one(path)

        print(f"[지하철] {path.name}: {len(raw):,}행")

        subway_list.append(raw)

    # 6개 연도 합치기
    raw = pd.concat(subway_list, ignore_index=True)

    print(f"[지하철] 6개 파일 결합 → {len(raw):,}행")

    print(f"[지하철] 원본 {len(raw):,}행, 열 {len(raw.columns)}")

    # 컬럼명 앞뒤 공백 제거
    raw = raw.rename(columns=lambda column: column.strip())

    # 승하차 인원 숫자로 바꾸기
    for column in ("승차총승객수", "하차총승객수"):
        raw[column] = pd.to_numeric(
            raw[column].astype(str).str.replace(",", ""), errors="coerce"
        )

    # 20190101 형태를 2019-01-01 형태로 바꾸기
    date = pd.to_datetime(
        raw["사용일자"].astype(str), format="%Y%m%d", errors="coerce"
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
    daily = pd.concat([board, alight], ignore_index=True)

    subway_cols = [
        "date",
        "line",
        "station_id",
        "station",
        "io_type",
        "total",
        "collected_at",
    ]

    inserted, skipped = _upsert(con, "subway_daily", subway_cols, daily)

    print(
        f"[지하철] subway_daily: "
        f"{len(daily):,}행 → "
        f"insert {inserted:,} / "
        f"skip(중복) {skipped:,}"
    )


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

    inserted, skipped = _upsert(con, "weather_daily", weather_cols, weather)

    print(
        f"[날씨] weather_daily: "
        f"{len(weather):,}행 → "
        f"insert {inserted:,} / "
        f"skip {skipped:,}"
    )


def main(db_path=DB_PATH):
    # 지정한 경로에 SQLite 연결
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)

    con = sqlite3.connect(db_path)

    try:
        # db.py에 만든 테이블 구조 적용
        con.executescript(SCHEMA)

        # 지하철과 날씨 적재
        build_subway(con)
        build_weather(con)

        # 최종 행 수 확인
        subway_count = con.execute("SELECT COUNT(*) FROM subway_daily").fetchone()[0]

        subway_days = con.execute(
            "SELECT COUNT(DISTINCT date) FROM subway_daily"
        ).fetchone()[0]

        weather_count = con.execute("SELECT COUNT(*) FROM weather_daily").fetchone()[0]

        print(
            f"[검증] subway_daily {subway_count:,}행 "
            f"(고유 {subway_days}일) / "
            f"weather_daily {weather_count}행"
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
#
# [발견한 문제 및 수정]
# 1. DB 저장 경로가 고정되어 빈 DB 검증 시 기존 DB를 건드릴 위험이 있었음
#    → --db-path 옵션을 추가하여 별도 DB에 적재할 수 있도록 수정함
#
# 2. 파일을 import하면 main()이 자동 실행되는 구조였음
#    → __main__ 조건을 추가하여 직접 실행할 때만 적재하도록 수정함
#
# 3. 최초 적재에서 372행이 중복으로 제외되는 실행 기록을 확인함
#    → 기존 UNIQUE 기준을 유지하고 별도 DB에서 재현 여부를 검증하도록 구성함
#
# [추후 보완 사항]
# 1. 중복 372행의 날짜·노선·역별 발생 원인 분석
# 2. 중복 키에 서로 다른 이용량이 존재하는지 확인
# 3. 중복 발생 시 어떤 값을 보존할지 처리 기준 명문화
