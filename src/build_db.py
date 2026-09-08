# python src/build_db.py

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


def main():
    # SQLite 연결
    con = sqlite3.connect(DB_PATH)

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

    con.close()

    print(f"[ok ] data/seoul_transit.db 적재 완료")


main()


# ============================================================

# [지하철] subway_daily_2019.csv: 215,769행
# [지하철] subway_daily_2020.csv: 217,055행
# [지하철] subway_daily_2021.csv: 219,278행
# [지하철] subway_daily_2022.csv: 219,745행
# [지하철] subway_daily_2023.csv: 222,162행
# [지하철] subway_daily_2024.csv: 225,512행
# [지하철] 6개 파일 결합 → 1,319,521행
# [지하철] 원본 1,319,521행, 열 6
# [지하철] subway_daily: 2,639,042행 → insert 2,638,670 / skip(중복) 372
# [날씨] weather_daily: 2,192행 → insert 2,192 / skip 0
# [검증] subway_daily 2,638,670행 (고유 2192일) / weather_daily 2192행
# [ok ] data/seoul_transit.db 적재 완료

# 같은 적재 코드를 여러 번 실행해도 DB 데이터가 불어나지 않음을 검증함

# ============================================================

# data/raw에 받은 csv를 SQLite 데이터베이스에 넣는 파일

# 1. read_one()
# 지하철 csv를 읽음
# 연도마다 인코딩이 달라도 읽을 수 있게 여러 개를 순서대로 확인
# 행에 쓸데없는 값이 더 붙어 있어도 원래 컬럼 개수까지만 가져옴

# 2. build_subway()
# 2019~2024 지하철 파일을 하나로 합침
# 승차, 하차를 각각 한 행으로 나눠 long 형태로 만들었음
# DB에 넣기 전 행 수는 원본의 2배가 됨

# 3. build_weather()
# Open-Metro 날씨 csv를 읽음
# 긴 원본 컬럼명을 DB에서 쓸 짧은 이름으로 변경하였음

# 4. _upsert()
# db.py에서 만든 함수를 사용
# 이미 있는 데이터는 중복 저장하지 않고 건너뜀

# main()
# SQLite에 연결
# 테이블을 만들고 지하철과 날씨를 넣음
# 마지막에 실제 저장된 행 수 확인

# CSV를 단순히 읽고 끝내는 것이 아니라 인코딩, 컬럼 밀림, 중복 문제 등을 방지
# 분석에 사용할 SQLite DB로 만드는 과정을 거쳤음
