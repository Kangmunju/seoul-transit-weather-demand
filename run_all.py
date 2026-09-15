from pathlib import Path
import subprocess
import sys


# 프로젝트 최상위 폴더
BASE_DIR = Path(__file__).resolve().parent

# 원본 데이터 폴더
RAW_DIR = BASE_DIR / "data" / "raw"

# 필요한 지하철 원본 파일
SUBWAY_FILES = [RAW_DIR / f"subway_daily_{year}.csv" for year in range(2019, 2025)]

# 날씨 원본 파일
WEATHER_FILE = RAW_DIR / "weather_daily.csv"


def rule(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def run_script(script):
    # 현재 run_all.py를 실행한 Python으로 각 파일 실행
    subprocess.run(
        [sys.executable, str(BASE_DIR / script)],
        cwd=BASE_DIR,
        check=True,
    )


def prepare_data():
    # 지하철 원본 파일이 모두 있는지 확인
    subway_ready = all(file.exists() for file in SUBWAY_FILES)

    if subway_ready:
        print("[skip] 지하철 원본 데이터가 이미 존재함")
    else:
        print("[run ] 지하철 원본 데이터 다운로드")
        run_script("src/download_data.py")

    # 날씨 원본 파일 확인
    if WEATHER_FILE.exists():
        print("[skip] 날씨 원본 데이터가 이미 존재함")
    else:
        print("[run ] 날씨 원본 데이터 수집")
        run_script("src/collect_weather.py")


def main():
    rule("1. 원본 데이터 준비")
    prepare_data()

    rule("2. SQLite DB 구축")
    run_script("src/build_db.py")

    rule("3. 모델 학습 및 평가")
    run_script("src/train.py")

    rule("4. EDA")
    run_script("analysis/a1_eda.py")

    rule("5. 강우 영향 분석")
    run_script("analysis/a2_rain_effect.py")

    rule("6. 데이터 누수 및 시계열 검증")
    run_script("analysis/a3_leakage.py")

    rule("7. 전처리 검증")
    run_script("analysis/a4_preprocess.py")

    rule("8. 팬데믹 구조 변화 분석")
    run_script("analysis/a5_pandemic.py")

    rule("전체 실행 완료")
    print("[ok ] 데이터 준비 → DB → 모델 → 분석 전체 과정 완료")


if __name__ == "__main__":
    main()
