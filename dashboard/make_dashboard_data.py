from pathlib import Path
import sys


# 프로젝트 최상위 경로
BASE_DIR = Path(__file__).resolve().parents[1]

# src 폴더의 기존 코드를 불러올 수 있도록 경로 추가
sys.path.insert(0, str(BASE_DIR / "src"))

from train import load_data
from clean import build_clean_data


# 대시보드용 데이터 저장 위치
OUTPUT_PATH = BASE_DIR / "dashboard" / "dashboard_data.csv"


def main():
    # SQLite DB에서 지하철과 날씨 원본 데이터 불러오기
    subway, weather = load_data()

    # 기존 전처리 파이프라인 적용
    merged, _, _ = build_clean_data(subway, weather)

    # 대시보드에서 사용할 컬럼만 선택
    columns = [
        "date",
        "ride",
        "temp_mean",
        "temp_max",
        "temp_min",
        "precip_mm",
        "rain_mm",
        "snow_cm",
        "wind_max",
        "humidity",
        "is_rain",
        "is_heavy_rain",
        "dow",
        "is_weekend",
        "is_holiday",
        "is_workday",
        "month",
        "year",
        "pandemic_acute",
        "pandemic_recovery",
        "is_pandemic",
    ]

    dashboard_data = merged[columns].copy()

    # CSV 저장
    dashboard_data.to_csv(
        OUTPUT_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    print(f"[ok ] 대시보드 데이터 저장 → {OUTPUT_PATH}")
    print(f"[검증] {len(dashboard_data):,}행 / {len(dashboard_data.columns)}열")
    print(
        f"[기간] {dashboard_data['date'].min().date()} "
        f"~ {dashboard_data['date'].max().date()}"
    )


if __name__ == "__main__":
    main()
