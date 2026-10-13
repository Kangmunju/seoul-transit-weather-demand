import sqlite3
import sys
from pathlib import Path

import pandas as pd


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


# ============================================================
# [구현 핵심]
# ============================================================

# SQLite에서 지하철 이용량과 날씨 데이터를 불러오고,
# clean.py의 공통 전처리 함수를 재사용하도록 구현함
# 분석 파일마다 데이터 조회 및 정제 코드를 중복 작성하지 않도록 구성함
#
# [발견한 문제 및 수정]
# 1. 실제로 그래프를 그리지 않는데 Matplotlib를 import하고 있었음
#    → 불필요한 import와 한글 폰트 설정 코드를 제거함
#
# 2. requirements.txt에 없는 라이브러리 때문에 실행이 실패할 수 있었음
#    → pandas와 표준 라이브러리만 사용하도록 의존성을 정리함
#
# [추후 보완 사항]
# 1. DB 파일이 없을 때 원인을 명확하게 안내하는 예외 처리 추가
# 2. 데이터 조회 결과의 필수 컬럼 및 결측 여부 검증
