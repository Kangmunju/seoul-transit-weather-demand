import sqlite3
import sys
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# 프로젝트 최상위 폴더
ROOT = Path(__file__).resolve().parents[1]

# src 폴더
SRC = ROOT / "src"

# SQLite 파일
DB_PATH = ROOT / "data" / "seoul_transit.db"

# analysis 폴더에서 실행해도 src의 파일을 불러올 수 있게 함
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from clean import build_clean_data

names = {f.name for f in fm.fontManager.ttflist}

for cand in ["AppleGothic", "Malgun Gothic", "NanumGothic", "NanumBarunGothic"]:
    if cand in names:
        plt.rcParams["font.family"] = cand
        break
else:
    print("[WARN] 한글 폰트를 찾지 못했습니다.")

# 마이너스 기호 깨짐 방지
plt.rcParams["axes.unicode_minus"] = False


def load_merged():
    # SQLite DB에 연결
    con = sqlite3.connect(DB_PATH)

    try:
        # 지하철 데이터 불러오기
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

        # 날씨 데이터 불러오기
        weather = pd.read_sql_query(
            """
            SELECT
                date,
                temp_mean,
                temp_max,
                temp_min,
                precip_mm,
                rain_mm,
                snow_cm,
                wind_max,
                humidity
            FROM weather_daily
            """,
            con,
        )

    finally:
        # 오류가 나더라도 DB 연결은 닫음
        con.close()

    # clean.py의 전체 정제 파이프라인 실행
    merged, sw, rep = build_clean_data(subway, weather)

    return merged, sw, rep
