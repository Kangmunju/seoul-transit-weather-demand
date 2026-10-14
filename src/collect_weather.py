import os
import time
import tempfile
from pathlib import Path

import pandas as pd
import requests


# 프로젝트 최상위 폴더
BASE_DIR = Path(__file__).resolve().parents[1]

# 원본 데이터를 저장할 폴더
RAW_DIR = BASE_DIR / "data" / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)


# 서울 종관기상관측소(ASOS 108번) 위치
SEOUL_LAT = 37.5714
SEOUL_LON = 126.9658


# Open-Meteo 과거 날씨 API
ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"


# 분석에 사용하는 날짜 범위
START_DATE = "2019-01-01"
END_DATE = "2024-12-31"


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


# CSV에서는 API의 time 필드를 date로 저장함
REQUIRED_COLUMNS = ["date", *DAILY_VARS]


def validate_weather(weather):
    # API 응답과 기존 CSV를 동일한 기준으로 검증함
    if not isinstance(weather, pd.DataFrame):
        raise ValueError("날씨 데이터가 DataFrame 형식이 아님")

    if weather.empty:
        raise ValueError("날씨 데이터가 비어 있음")

    missing_columns = [
        column for column in REQUIRED_COLUMNS if column not in weather.columns
    ]

    if missing_columns:
        raise ValueError(f"날씨 데이터 필수 컬럼 누락: {missing_columns}")

    # 날짜를 변환하면서 형식 오류를 확인함
    dates = pd.to_datetime(
        weather["date"],
        format="%Y-%m-%d",
        errors="coerce",
    )

    if dates.isna().any():
        raise ValueError(f"날씨 데이터에 잘못된 날짜가 {dates.isna().sum()}행 존재함")

    # 같은 날짜가 두 번 들어오면 일별 병합 결과가 중복될 수 있음
    duplicate_count = dates.duplicated().sum()

    if duplicate_count:
        raise ValueError(f"날씨 데이터의 중복 날짜: {duplicate_count}행")

    # 윤년을 포함하여 기대 날짜 목록을 생성함
    expected_dates = pd.date_range(
        START_DATE,
        END_DATE,
        freq="D",
    )

    actual_dates = pd.DatetimeIndex(dates)

    missing_dates = expected_dates.difference(actual_dates)
    extra_dates = actual_dates.difference(expected_dates)

    if len(missing_dates):
        sample = [value.strftime("%Y-%m-%d") for value in missing_dates[:5]]

        raise ValueError(
            f"날씨 데이터 날짜 누락: {len(missing_dates)}일 (예: {sample})"
        )

    if len(extra_dates):
        sample = [value.strftime("%Y-%m-%d") for value in extra_dates[:5]]

        raise ValueError(f"분석 범위 밖 날짜: {len(extra_dates)}일 (예: {sample})")

    if len(weather) != len(expected_dates):
        raise ValueError(
            f"날씨 데이터 행 수 오류: {len(weather)}행 / 예상 {len(expected_dates)}행"
        )

    # 날씨 변수의 문자열이나 비정상 값을 확인함
    # 관측값 자체의 결측은 별도 전처리 대상이므로
    # 여기서는 숫자로 해석할 수 없는 값만 차단함
    for column in DAILY_VARS:
        values = pd.to_numeric(
            weather[column],
            errors="coerce",
        )

        invalid_mask = weather[column].notna() & values.isna()

        if invalid_mask.any():
            raise ValueError(
                f"{column}에 숫자로 변환할 수 없는 값이 {invalid_mask.sum()}행 존재함"
            )

    print(f"[ok ] 날씨 날짜 검증: {len(weather):,}일 / {START_DATE}~{END_DATE}")

    print("[ok ] 날짜 중복·누락·범위 오류 없음")

    return len(weather)


def validate_weather_file(path):
    # 기존 CSV를 다시 수집하지 않고 검증함
    path = Path(path)

    if not path.is_file():
        raise FileNotFoundError(f"날씨 원본 파일이 없음: {path}")

    try:
        weather = pd.read_csv(
            path,
            encoding="utf-8-sig",
        )

    except (OSError, UnicodeError, pd.errors.ParserError) as error:
        raise ValueError(f"날씨 CSV 읽기 실패: {error}") from error

    return validate_weather(weather)


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
            response = requests.get(
                ARCHIVE_URL,
                params=params,
                timeout=60,
            )

            response.raise_for_status()

            # HTTP 성공이어도 JSON이 잘못되었을 수 있음
            payload = response.json()

            if not isinstance(payload, dict):
                raise ValueError("API 응답이 JSON 객체 형식이 아님")

            daily = payload.get("daily")

            if not isinstance(daily, dict):
                raise ValueError("API 응답에 daily 데이터가 없음")

            weather = pd.DataFrame(daily)

            # time 컬럼 이름을 date로 변경
            weather = weather.rename(columns={"time": "date"})

            # 수집 직후 날짜·컬럼·데이터 형식을 검증함
            validate_weather(weather)

            return weather

        except (
            requests.RequestException,
            ValueError,
            TypeError,
            pd.errors.ParserError,
        ) as error:
            print(f"[WARN] 시도 {attempt + 1}/3 실패: {type(error).__name__}: {error}")

            # 마지막 실패 뒤에는 불필요하게 대기하지 않음
            if attempt < 2:
                time.sleep(5)

    raise RuntimeError("Open-Meteo 날씨 수집 및 검증에 3번 실패했습니다.")


def collect_weather():
    print(f"[get ] 날씨 ← Open-Meteo {START_DATE}~{END_DATE} (무키)")

    weather = fetch(
        START_DATE,
        END_DATE,
    )

    save_path = RAW_DIR / "weather_daily.csv"

    temporary_path = None

    try:
        # 기존 정상 파일을 보호하기 위해 임시 CSV를 먼저 작성함
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8-sig",
            newline="",
            suffix=".csv",
            prefix="weather_daily_",
            dir=RAW_DIR,
            delete=False,
        ) as file:
            temporary_path = Path(file.name)

            weather.to_csv(
                file,
                index=False,
            )

        # 저장된 파일을 다시 읽어 날짜·컬럼 검증
        validate_weather_file(temporary_path)

        # 검증에 성공한 경우에만 최종 파일로 교체
        os.replace(
            temporary_path,
            save_path,
        )

        print(f"{len(weather):,}일 수신, 컬럼 {len(weather.columns)}개")

        print(f"[ok ] 날씨 → {save_path.name}")

    finally:
        # 실패 시 남은 임시 파일 정리
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def main():
    try:
        collect_weather()

    except (
        RuntimeError,
        ValueError,
        OSError,
        requests.RequestException,
    ) as error:
        raise SystemExit(f"[FAIL] 날씨 원본 데이터 준비 중단: {error}") from error


if __name__ == "__main__":
    main()


# ============================================================
# [구현 핵심]
# ============================================================

# Open-Meteo API에서 2019~2024년 서울 날씨 데이터를 수집함
# 기존 날씨 변수와 CSV 저장 형식을 유지함
# API 응답과 기존 CSV를 동일한 검증 함수로 확인함
# 2019~2024년 전체 2,192일의 날짜 누락·중복·범위 오류를 검사함
# 임시 CSV 검증에 성공한 경우에만 최종 원본 파일을 교체함
#
# [발견한 문제 및 수정]
# 1. API 응답이 정상인지 확인하지 않고 DataFrame으로 변환했음
#    → daily 구조 및 필수 컬럼 검사 추가
#
# 2. API가 일부 날짜만 반환해도 CSV를 저장할 수 있었음
#    → 날짜 누락·중복·범위 및 전체 일수 검증 추가
#
# 3. 기존 정상 CSV에 바로 덮어쓰는 방식이었음
#    → 임시 파일 저장 및 재검증 후 최종 파일 교체
#
# 4. 기존 날씨 CSV의 내용을 재검증할 함수가 없었음
#    → validate_weather_file() 추가
#
# 5. 수집 실패 후에도 마지막 재시도에서 대기했음
#    → 마지막 실패 시 불필요한 대기 제거
#
# [추후 보완 사항]
# 1. 날씨 변수별 물리적으로 가능한 값 범위 검증
# 2. API 응답의 데이터 버전 및 수집 시점 기록
# 3. 날씨 결측값의 원인과 대체 방식 추가 검증
