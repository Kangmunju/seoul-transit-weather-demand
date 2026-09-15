from pathlib import Path
import json


BASE_DIR = Path(__file__).resolve().parent


# 프로젝트에 반드시 있어야 하는 핵심 파일
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
    "reports/analysis_report.md",
    "reports/DATA_ACCESS.md",
    "reports/PITFALLS.md",
    "data/seoul_transit.db",
    "models/metrics.json",
    "models/feature_importance.csv",
    "models/test_predictions.csv",
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

        if path.exists():
            print(f"[ok ] {file}")
        else:
            print(f"[FAIL] {file}")
            missing.append(file)

    return missing


def check_metrics():
    path = BASE_DIR / "models" / "metrics.json"

    if not path.exists():
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


def main():
    rule("1. 핵심 프로젝트 파일 확인")
    missing_core = check_files(REQUIRED_FILES)

    rule("2. 원본 데이터 확인")
    missing_raw = check_files(RAW_FILES)

    rule("3. 모델 결과 확인")
    metrics_ok = check_metrics()

    rule("최종 검증 결과")

    if not missing_core and not missing_raw and metrics_ok:
        print("[PASS] 프로젝트 핵심 파일과 산출물이 모두 정상적으로 존재함")
        return

    print("[FAIL] 프로젝트 검증 실패")

    if missing_core:
        print(f"- 누락된 핵심 파일: {len(missing_core)}개")

    if missing_raw:
        print(f"- 누락된 원본 데이터: {len(missing_raw)}개")

    if not metrics_ok:
        print("- 모델 성능 결과 확인 필요")

    raise SystemExit(1)


if __name__ == "__main__":
    main()
