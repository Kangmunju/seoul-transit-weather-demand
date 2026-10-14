from pathlib import Path
import argparse
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


# *args : 가변 위치 인자(Variable Positional Arguments)
#        함수를 호출할 때 전달한 추가 위치 인자를 개수 제한 없이 받을 수 있음
#        전달된 인자는 함수 내부에서 튜플(tuple)로 묶임
#        *는 여러 위치 인자를 모아서 받는 역할을 함
# cf) C에서 int *p의 *는 포인터 변수를 선언하거나 역참조할 때 사용함
def run_script(script, *args):

    # os.environ.copy() : 현재 프로세스의 환경변수를 일반 딕셔너리로 복사하는 메서드
    #                     복사본을 수정해도 os.environ 자체는 변경되지 않음
    env = os.environ.copy()

    # PYTHONIOENCODING : Python의 표준 입출력 인코딩을 지정하는 환경변수
    env["PYTHONIOENCODING"] = "utf-8"

    # subprocess.run() : 현재 Python 프로그램에서 외부 프로그램을 실행하는 함수
    #                    지정한 프로그램의 실행이 끝날 때까지 기다린 후 결과를 반환함
    #                    여기서는 각 Python 스크립트를 별도의 자식 프로세스로 실행함
    # subprocess.run()은 실행 완료까지 기다리는 동기 방식(별도 프로세스 실행)
    # 호출한 Python 프로그램과 자식 프로그램의 일반 변수는 직접 공유되지 않음
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


def validate_subway_files():
    # 기존 CSV를 다운로드 없이 검사
    from src.download_data import validate_year_file

    for year, path in zip(
        range(2019, 2025),
        SUBWAY_FILES,
    ):
        result = validate_year_file(path, year)

        print(f"[ok ] {path.name}: {result['rows']:,}행 / {result['days']}일")


def prepare_data():
    # 지하철 원본 파일이 모두 있는지 확인
    subway_ready = all(file.is_file() for file in SUBWAY_FILES)

    if subway_ready:
        print("[check] 기존 지하철 원본 데이터 검증")

        try:
            validate_subway_files()

        except (
            ValueError,
            OSError,
        ) as error:
            raise SystemExit(
                "[FAIL] 기존 지하철 원본 데이터 검증 실패: "
                f"{error}\n"
                "불완전한 원본 파일을 확인한 후 "
                "해당 파일을 복구해야 합니다."
            ) from error

        print("[skip] 지하철 원본 데이터가 이미 존재하며 검증 통과")

    else:
        print("[run ] 지하철 원본 데이터 다운로드")
        run_script("src/download_data.py")

        # 새로 다운로드한 파일도 재검증
        print("[check] 다운로드 결과 검증")
        validate_subway_files()

    # 날씨 원본 파일 확인
    if WEATHER_FILE.exists():
        print("[skip] 날씨 원본 데이터가 이미 존재함")
    else:
        print("[run ] 날씨 원본 데이터 수집")
        run_script("src/collect_weather.py")


def main():
    # argparse.ArgumentParser() : 명령행 인자를 해석하는 객체를 생성함
    # --help 또는 -h를 입력하면 사용법을 출력하고 프로그램을 종료함
    # 정의하지 않은 옵션을 입력하면 오류를 출력하고 실행을 중단함
    parser = argparse.ArgumentParser(
        description=(
            "서울 지하철 수요 × 날씨 예측 프로젝트의 "
            "전체 데이터 처리 및 분석 파이프라인 실행"
        )
    )

    # parse_args() : 전달된 명령행 인자가 정의된 형식에 맞는지 검사함
    # 현재 별도 실행 옵션은 없으므로 인자 없이 실행할 때만 아래 단계로 진행함
    # --help는 도움말을 출력한 후 종료하며, 잘못된 옵션은 오류로 처리함
    parser.parse_args()

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

    # 기존 a1~a5 분석에 이어 피드백 보강 분석을 실행
    # 각 분석 스크립트가 오류 없이 끝나야 다음 단계로 진행함

    rule("9. 날씨 변수 기여도 및 Walk-forward 검증")
    run_script("analysis/a6_weather_ablation.py")

    rule("10. 강우 효과 보정 분석")
    run_script("analysis/a7_rain_adjusted.py")

    rule("11. 동일 테스트 기간 데이터 분할 비교")
    run_script("analysis/a8_fixed_test_split.py")

    rule("12. 평일 기준 강우 효과 보정 분석")
    run_script("analysis/a9_rain_workday_adjusted.py")

    rule("13. 고정 역 집합 기준 팬데믹 회복률 검증")
    run_script("analysis/a10_pandemic_station_fixed.py")

    rule("14. 대시보드 데이터 생성")
    run_script("dashboard/make_dashboard_data.py")

    rule("15. 전체 실행 결과 검증")
    run_script("verify_code.py", "--after-run")

    rule("전체 실행 완료")
    print(
        "[ok ] 데이터 준비 → DB → 모델 → 분석(a1~a10) "
        "→ 대시보드 데이터 → 검증 전체 과정 완료"
    )


if __name__ == "__main__":
    main()


# ============================================================
# [구현 핵심]
# ============================================================

# 원본 데이터 준비부터 DB 구축, 모델 학습, 분석,
# 대시보드 데이터 생성 및 최종 검증까지 자동 실행하도록 구현함
# 각 단계에서 오류가 발생하면 실행을 중단하여
# 불완전한 결과를 전체 실행 성공으로 표시하지 않도록 구성함
# 기존 지하철 CSV를 재사용할 때 날짜 범위와 CSV 구조를 검증함
# 피드백 보강 분석 a6~a10도 전체 파이프라인에 포함함
# argparse로 명령행 인자를 검사하여 도움말 요청과 잘못된 옵션을 안전하게 처리함
#
# [발견한 문제 및 수정]
# 1. 전체 실행 과정에 대시보드 CSV 생성 단계가 빠져 있었음
#    → make_dashboard_data.py를 실행하도록 9단계 추가
#
# 2. 기존 대시보드 CSV가 남아 있어도 검증을 통과할 수 있었음
#    → 분석 완료 후 CSV를 다시 생성하고 최종 검증하도록 순서 수정
#
# 3. Windows 환경에서 하위 프로세스 출력 인코딩 문제가 발생할 수 있었음
#    → PYTHONIOENCODING을 UTF-8로 설정하여 실행 환경을 통일함
#
# 4. 기존 원본 파일이 존재하기만 하면 검증 없이 재사용하던 문제
#    → 파일별 필수 컬럼과 날짜 범위를 확인한 후 재사용하도록 수정함
#
# 5. 다운로드한 파일이 불완전해도 후속 단계로 진행할 수 있었음
#    → 다운로드 후 재검증하고 실패 시 전체 파이프라인을 중단함
#
# 6. 피드백 보강 분석 a6~a10이 전체 실행 과정에서 누락되어 있었음
#    → 기존 a1~a5 다음에 추가하여 분석 전체를 순차 실행하도록 수정함
#
# 7. --help 옵션을 입력해도 전체 파이프라인이 실행되는 문제가 있었음
#    → argparse로 명령행 인자를 먼저 검사하도록 수정함
#    → 도움말 요청 또는 잘못된 옵션 입력 시 DB 구축과 모델 학습을 실행하지 않음
#
# [추후 보완 사항]
# 1. 빈 SQLite DB에서 최초 구축 시 데이터 건수와 중복 여부 검증
# 2. 날짜별 역·노선 데이터의 완전성 검사
# 3. 새 가상환경에서 전체 파이프라인 재실행 및 재현성 확인
