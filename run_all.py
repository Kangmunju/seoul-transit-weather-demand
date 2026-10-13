from pathlib import Path
import os
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


def run_script(script, *args):
    # 현재 run_all.py를 실행한 Python으로 각 파일 실행
    # Windows 환경에서도 하위 프로세스 출력 인코딩을 UTF-8로 통일
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"

    subprocess.run(
        [
            sys.executable,
            str(BASE_DIR / script),
            *args,
        ],
        cwd=BASE_DIR,
        env=env,
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

    rule("9. 전체 실행 결과 검증")
    run_script("verify_code.py", "--after-run")

    rule("전체 실행 완료")
    print("[ok ] 데이터 준비 → DB → 모델 → 분석 → 검증 전체 과정 완료")


if __name__ == "__main__":
    main()


# ============================================================
# [구현 핵심]
# ============================================================

# 지하철·날씨 원본 데이터 준비부터 DB 구축, 모델 학습,
# EDA 및 주요 분석까지 순서대로 실행하는 파이프라인을 구현함
# 각 단계의 실행 실패를 확인하고 후속 작업을 중단하도록 구성함
#
# [발견한 문제 및 수정]
# 1. 기존에는 모든 분석 스크립트가 종료되면 전체 실행 성공으로 표시했음
#    → 마지막에 verify_code.py를 실행하여 산출물까지 검증하도록 수정함
#
# 2. Windows 환경에서 하위 Python 프로세스의 출력 인코딩 문제가 발생할 수 있었음
#    → PYTHONIOENCODING을 UTF-8로 설정하여 출력 환경을 통일함
#
# 3. 분석 결과와 대시보드용 JSON 파일의 생성 여부를 확인하지 않았음
#    → 실행 후 검증 단계에서 누락 여부를 확인하도록 연결함
#
# [추후 보완 사항]
# 1. 원본 파일의 존재 여부뿐 아니라 데이터 완전성까지 검증
# 2. 다운로드 실패 시 부분 저장을 방지하는 예외 처리 강화
# 3. 새 가상환경에서 전체 파이프라인 반복 실행 및 재현성 확인
