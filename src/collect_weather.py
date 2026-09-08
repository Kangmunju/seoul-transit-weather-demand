import time
from pathlib import Path

import pandas as pd
import requests


# 원본 데이터를 저장할 폴더
RAW_DIR = Path("data/raw")
RAW_DIR.mkdir(parents=True, exist_ok=True)


# 서울 종관기상관측소(ASOS 108번) 위치
SEOUL_LAT = 37.5714
SEOUL_LON = 126.9658


# Open-Meteo 과거 날씨 API
ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"


# 받을 날씨 항목
DAILY_VARS = [
    "temperature_2m_mean",
    "temperature_2m_max",
    "temperature_2m_min",
    "precipitation_sum",
    "rain_sum",
    "snowfall_sum",
    "windspeed_10m_max",
    "relative_humidity_2m_mean",
]


def fetch(start, end):
    # API에 전달할 조건
    params = {
        "latitude": SEOUL_LAT,
        "longitude": SEOUL_LON,
        "start_date": start,
        "end_date": end,
        "daily": ",".join(DAILY_VARS),
        "timezone": "Asia/Seoul",
    }

    # 네트워크 오류가 날 수 있어서 최대 3번 시도
    for attempt in range(3):
        try:
            response = requests.get(ARCHIVE_URL, params=params, timeout=60)

            response.raise_for_status()

            break

        except Exception as error:
            print(f"시도 {attempt + 1}/3 실패: {type(error).__name__}: {error}")

            time.sleep(5)

    else:
        raise SystemExit("Open-Meteo 호출에 3번 실패했습니다. 네트워크를 확인하세요.")

    # JSON에서 daily 부분만 가져오기
    daily = response.json().get("daily")

    # DataFrame으로 변환
    df = pd.DataFrame(daily)

    # time 컬럼 이름을 date로 변경
    df = df.rename(columns={"time": "date"})

    return df


def collect_weather():
    print("[get ] 날씨 ← Open-Meteo 2019-01-01~2024-12-31 (무키)")

    weather = fetch("2019-01-01", "2024-12-31")

    save_path = RAW_DIR / "weather_daily.csv"

    weather.to_csv(save_path, index=False, encoding="utf-8-sig")

    print(f"{len(weather):,}일 수신, 컬럼 {len(weather.columns)}개")

    print(f"[ok ] 날씨 → {save_path.name}")


collect_weather()
