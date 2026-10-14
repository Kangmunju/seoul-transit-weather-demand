from pathlib import Path
import json
import argparse


BASE_DIR = Path(__file__).resolve().parent


# 프로젝트에 반드시 있어야 하는 핵심 소스 파일
REQUIRED_FILES = [
    "README.md",
    "requirements.txt",
    "run_all.py",
    "src/db.py",
    "src/download_data.py",
    "src/collect_weather.py",
    "src/build_db.py",
    "src/clean.py",
    "src/features.py",
    "src/train.py",
    "analysis/_common.py",
    "analysis/a1_eda.py",
    "analysis/a2_rain_effect.py",
    "analysis/a3_leakage.py",
    "analysis/a4_preprocess.py",
    "analysis/a5_pandemic.py",
    "analysis/a6_weather_ablation.py",
    "analysis/a7_rain_adjusted.py",
    "analysis/a8_fixed_test_split.py",
    "analysis/a9_rain_workday_adjusted.py",
    "analysis/a10_pandemic_station_fixed.py",
    "reports/analysis_report.md",
    "reports/DATA_ACCESS.md",
    "reports/PITFALLS.md",
    "dashboard/app.py",
    "dashboard/analysis_extensions.py",
    "dashboard/make_dashboard_data.py",
]


# 실행 후 생성되거나 준비되어 있어야 하는 산출물
GENERATED_FILES = [
    "data/seoul_transit.db",
    "models/metrics.json",
    "models/feature_importance.csv",
    "models/test_predictions.csv",
    "models/leakage_results.json",
    "models/weather_ablation_results.json",
    "models/weather_ablation_folds.csv",
    "models/weather_permutation_importance.csv",
    "models/rain_adjusted_results.json",
    "models/fixed_test_split_results.json",
    "models/rain_workday_adjusted_results.json",
    "models/pandemic_station_fixed_results.json",
    "models/holiday_feature_comparison.json",
    "models/holiday_feature_importance.csv",
    "models/holiday_feature_predictions.csv",
    "dashboard/dashboard_data.csv",
]


# 원본 데이터 파일
RAW_FILES = [f"data/raw/subway_daily_{year}.csv" for year in range(2019, 2025)]

RAW_FILES.append("data/raw/weather_daily.csv")


def rule(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def check_files(files):
    missing = []

    for file in files:
        path = BASE_DIR / file

        if path.is_file():
            print(f"[ok ] {file}")
        else:
            print(f"[FAIL] {file}")
            missing.append(file)

    return missing


def check_metrics():
    path = BASE_DIR / "models" / "metrics.json"

    if not path.is_file():
        print("[FAIL] models/metrics.json 없음")
        return False

    try:
        with open(path, "r", encoding="utf-8") as file:
            metrics = json.load(file)

    except (json.JSONDecodeError, OSError) as error:
        print(f"[FAIL] metrics.json 읽기 실패: {error}")
        return False

    required_models = [
        "baseline_lag7",
        "linear_regression",
        "random_forest",
    ]

    missing = [model for model in required_models if model not in metrics]

    if missing:
        print(f"[FAIL] 모델 성능 누락: {missing}")
        return False

    print("[ok ] 세 모델의 성능 결과 확인")
    return True


def check_leakage_results():
    path = BASE_DIR / "models" / "leakage_results.json"

    if not path.is_file():
        print("[FAIL] models/leakage_results.json 없음")
        return False

    try:
        with open(path, "r", encoding="utf-8") as file:
            results = json.load(file)

    except (json.JSONDecodeError, OSError) as error:
        print(f"[FAIL] leakage_results.json 읽기 실패: {error}")
        return False

    required_keys = [
        "split_comparison",
        "walk_forward",
    ]

    missing = [key for key in required_keys if key not in results]

    if missing:
        print(f"[FAIL] 시계열 검증 결과 누락: {missing}")
        return False

    if not isinstance(results["split_comparison"], list):
        print("[FAIL] split_comparison 형식 오류")
        return False

    if not isinstance(results["walk_forward"], list):
        print("[FAIL] walk_forward 형식 오류")
        return False

    if len(results["split_comparison"]) != 2:
        print("[FAIL] 랜덤/시간순 split 비교 결과 개수 오류")
        return False

    if len(results["walk_forward"]) != 5:
        print("[FAIL] Walk-forward 검증 결과 개수 오류")
        return False

    print("[ok ] 시계열 검증 결과 확인")
    return True


def check_analysis_results():
    # a6~a10 및 휴일 피처 비교에서 저장하는 JSON 결과 파일
    # 각 파일의 존재 여부, JSON 형식 및 핵심 항목을 확인함

    analysis_results = [
        "models/weather_ablation_results.json",
        "models/rain_adjusted_results.json",
        "models/fixed_test_split_results.json",
        "models/rain_workday_adjusted_results.json",
        "models/pandemic_station_fixed_results.json",
        "models/holiday_feature_comparison.json",
    ]

    # 분석별로 반드시 포함되어야 하는 핵심 결과 항목
    required_keys = {
        "models/weather_ablation_results.json": [
            "split_comparison",
            "heavy_rain",
            "walk_forward",
        ],
        "models/rain_adjusted_results.json": [
            "simple_comparison",
            "matched_comparison",
            "adjusted_regression",
        ],
        "models/fixed_test_split_results.json": [
            "time_split",
            "mixed_split",
        ],
        "models/rain_workday_adjusted_results.json": [
            "rain_comparison",
            "intensity_comparisons",
            "bootstrap",
        ],
        "models/pandemic_station_fixed_results.json": [
            "annual_comparison",
            "recovery_2024",
            "sensitivity_analysis",
        ],
        "models/holiday_feature_comparison.json": [
            "old_model",
            "new_model",
            "added_features",
        ],
    }

    all_ok = True

    for filename in analysis_results:
        path = BASE_DIR / filename

        if not path.is_file():
            print(f"[FAIL] {filename} 없음")
            all_ok = False
            continue

        try:
            with open(path, "r", encoding="utf-8") as file:
                results = json.load(file)

        except (json.JSONDecodeError, OSError) as error:
            print(f"[FAIL] {filename} 읽기 실패: {error}")
            all_ok = False
            continue

        if not isinstance(results, dict):
            print(f"[FAIL] {filename} 최상위 JSON 형식 오류")
            all_ok = False
            continue

        if not results:
            print(f"[FAIL] {filename} 분석 결과가 비어 있음")
            all_ok = False
            continue

        # 파일별 핵심 결과가 누락되거나 null인지 확인함
        missing_keys = [
            key
            for key in required_keys[filename]
            if key not in results or results[key] is None
        ]

        if missing_keys:
            print(f"[FAIL] {filename} 필수 항목 누락: {missing_keys}")
            all_ok = False
            continue

        # a10은 분석 대상 연도가 모두 포함되었는지 추가 확인함
        if filename == "models/pandemic_station_fixed_results.json":
            annual_comparison = results["annual_comparison"]

            if not isinstance(annual_comparison, list):
                print("[FAIL] a10 연도별 비교 결과 형식 오류")
                all_ok = False
                continue

            if not all(isinstance(item, dict) for item in annual_comparison):
                print("[FAIL] a10 연도별 비교 항목 형식 오류")
                all_ok = False
                continue

            years = {item.get("year") for item in annual_comparison}

            if years != set(range(2019, 2025)):
                print("[FAIL] a10 연도별 비교에 2019~2024년 데이터 누락")
                all_ok = False
                continue

        print(f"[ok ] {filename} 분석 결과 확인")

    return all_ok


def main():
    parser = argparse.ArgumentParser(description="프로젝트 소스 및 실행 산출물 검증")

    parser.add_argument(
        "--after-run",
        action="store_true",
        help="전체 파이프라인 실행 후 원본 데이터와 산출물까지 검증",
    )

    args = parser.parse_args()

    rule("1. 핵심 프로젝트 파일 확인")
    missing_core = check_files(REQUIRED_FILES)

    missing_raw = []
    missing_generated = []

    metrics_ok = True
    leakage_ok = True
    analysis_ok = True

    if args.after_run:
        rule("2. 원본 데이터 확인")
        missing_raw = check_files(RAW_FILES)

        rule("3. 실행 산출물 확인")
        missing_generated = check_files(GENERATED_FILES)

        rule("4. 모델 성능 결과 확인")
        metrics_ok = check_metrics()

        rule("5. 시계열 검증 결과 확인")
        leakage_ok = check_leakage_results()

        rule("6. 피드백 보강 분석 결과 확인")
        analysis_ok = check_analysis_results()

    else:
        print(
            "\n[info] 실행 전 검증 모드: 원본 데이터 및 생성 산출물 검사는 생략합니다."
        )

    rule("최종 검증 결과")

    passed = (
        not missing_core
        and not missing_raw
        and not missing_generated
        and metrics_ok
        and leakage_ok
        and analysis_ok
    )

    if passed:
        if args.after_run:
            print(
                "[PASS] 프로젝트 소스, 원본 데이터, "
                "실행 산출물이 모두 정상적으로 존재함"
            )
        else:
            print("[PASS] 프로젝트 핵심 소스 파일이 모두 존재함")

        return

    print("[FAIL] 프로젝트 검증 실패")

    if missing_core:
        print(f"- 누락된 핵심 파일: {len(missing_core)}개")

    if missing_raw:
        print(f"- 누락된 원본 데이터: {len(missing_raw)}개")

    if missing_generated:
        print(f"- 누락된 실행 산출물: {len(missing_generated)}개")

    if not metrics_ok:
        print("- 모델 성능 결과 확인 필요")

    if not leakage_ok:
        print("- 시계열 검증 결과 확인 필요")

    if not analysis_ok:
        print("- 피드백 보강 분석 결과 확인 필요")

    raise SystemExit(1)


if __name__ == "__main__":
    main()


# ============================================================
# [구현 핵심]
# ============================================================

# 프로젝트 핵심 소스 파일과 실행 산출물의 존재 여부를 검증함
# 실행 전에는 소스 파일만 검사하고,
# 실행 후에는 원본 데이터, DB, 모델 결과, 분석 결과까지 확인하도록 구현함
# 피드백 보강 분석 a6~a10과 휴일 피처 결과 파일까지 검증함
# a10은 2019~2024년의 연도별 결과가 모두 존재하는지 추가 확인함
#
# [발견한 문제 및 수정]
# 1. 새로 클론한 프로젝트에서 생성 전 DB 파일을 필수로 요구했음
#    → 실행 전과 실행 후 검증을 분리하여 불필요한 실패를 방지함
#
# 2. 파일 존재 여부만 확인하여 분석 결과의 누락을 구분하기 어려웠음
#    → 모델 성능과 시계열 검증 JSON의 필수 항목 검사 추가
#
# 3. 피드백 9번에서 추가한 leakage_results.json이 검증 대상에 없었음
#    → 실행 후 산출물 검사에 포함함
#
# 4. 피드백 보강 분석 a6~a10이 핵심 파일 검사에서 누락되어 있었음
#    → REQUIRED_FILES에 a6~a10을 추가함
#
# 5. a6~a9 분석 결과물이 실행 후 검증 대상에 포함되지 않았음
#    → GENERATED_FILES에 결과 JSON 및 CSV를 추가함
#    → JSON 파일의 형식과 비어 있지 않은 구조를 검사하는 함수 추가
#
# 6. a10 JSON과 휴일 피처 산출물이 검증 대상에서 누락되어 있었음
#    → GENERATED_FILES 및 JSON 필수 항목 검사에 추가함
#
# 7. 대시보드 확장 모듈이 핵심 소스 파일 검사에서 누락되어 있었음
#    → dashboard/analysis_extensions.py를 REQUIRED_FILES에 추가함
#
# 8. a10 결과 파일의 존재 여부만으로는 분석 연도 누락을 확인할 수 없었음
#    → annual_comparison에 2019~2024년이 모두 포함되었는지 검사함
#
# [추후 보완 사항]
# 1. CSV 및 DB 내부 데이터 건수와 날짜 범위 검증
# 2. 원본 데이터의 연도별 누락 일수 및 중복 여부 검사
# 3. Streamlit 렌더링 결과와 JSON 표시 수치의 일치 여부 자동 검증
# 4. 분석 JSON 내부의 세부 지표 및 값 범위 검증
