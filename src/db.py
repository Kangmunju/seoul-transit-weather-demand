import pandas as pd


# SQLite 테이블 구조
SCHEMA = """
CREATE TABLE IF NOT EXISTS subway_daily (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    date TEXT NOT NULL,
    line TEXT NOT NULL,
    station_id TEXT,
    station TEXT NOT NULL,
    io_type TEXT NOT NULL,
    total INTEGER,
    collected_at TEXT,
    UNIQUE (date, line, station, io_type)
);

CREATE INDEX IF NOT EXISTS ix_sub_date
ON subway_daily (date);

CREATE INDEX IF NOT EXISTS ix_sub_station
ON subway_daily (station, date);

CREATE TABLE IF NOT EXISTS weather_daily (
    date TEXT PRIMARY KEY,
    temp_mean REAL,
    temp_max REAL,
    temp_min REAL,
    precip_mm REAL,
    rain_mm REAL,
    snow_cm REAL,
    wind_max REAL,
    humidity REAL,
    collected_at TEXT
);
"""


def _upsert(con, table, cols, df):
    # DB에 넣을 컬럼 순서 맞추기
    df = df.reindex(columns=cols)

    # 넣기 전 DB 행 수
    before = con.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]

    # 이미 존재하는 데이터는 건너뛰기
    sql = (
        f"INSERT OR IGNORE INTO {table} ({','.join(cols)}) "
        f"VALUES ({','.join('?' * len(cols))})"
    )

    # pandas의 NaN을 SQLite의 NULL로 변환해서 저장
    con.executemany(
        sql, df.where(pd.notna(df), None).itertuples(index=False, name=None)
    )

    con.commit()

    # 넣은 후 DB 행 수
    after = con.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]

    inserted = after - before

    # 실제로 넣은 행 수, 중복이라 건너뛴 행 수
    return inserted, len(df) - inserted


# ============================================================

# SQLite 데이터베이스의 테이블 구조와 데이터를 중복 없이 저장하는 기능을 정의함

# 1. SCHEMA
# subway_daily : 지하철 일별 승하차 데이터 저장
# wether_daily : 일별 날씨 데이터 저장

# 2. UNIQUE
# 지하철 데이터는 날짜 + 노선 + 역 + 승하차 구분이 동일한 경우 중복으로 판단
# 같은 데이터를 다시 적재해도 DB에 중복 저장되지 않도록 함

# INSERT OR IGNORE
# 새로운 데이터면 저장함
# UNIQUE 조건에 걸리는 기존 데이터는 자동으로 건너뛰도록 함

# _upsert()
# DataFrame의 데이터를 SQLite에 저장하는 함수
# 저장 전후 행 수를 비교해 실제 저장된 행과 건너뛴 행의 개수를 반환

# 결측값 처리
# pandas의 NaN을 None으로 바꿔 SQLite에서는 NULL로 저장되도록 함

# Python에서 중복을 매번 제거하지 않고 위의 조건들로 데이터베이스 자체에서 중복을 막도록 함
