import numpy as np


def mape(y, pred):
    # 실제값과 예측값을 numpy 배열로 변환
    y = np.asarray(y, dtype=float)
    pred = np.asarray(pred, dtype=float)

    # 실제값이 0인 경우 0으로 나누는 문제 방지
    valid = y != 0

    # 유효한 실제값이 없으면 계산할 수 없으므로 NaN 반환
    if not valid.any():
        return float("nan")

    # 유효한 데이터만 사용해 평균 절대 백분율 오차 계산
    return float(np.mean(np.abs((y[valid] - pred[valid]) / y[valid])) * 100)


# ============================================================
# [구현 핵심]
# ============================================================

# 모델 학습과 날씨 변수 기여도 분석에서 사용하는
# MAPE 계산식을 공통 함수로 분리함
# 실제값이 0인 행은 계산에서 제외함
# 기존 계산식과 반환 형식을 유지함
#
# [발견한 문제 및 수정]
# src/train.py와 analysis/a6_weather_ablation.py에
# 동일한 MAPE 계산식이 중복 구현되어 있었음
# → 공통 함수로 분리하여 계산 기준을 통일함
#
# [추후 보완 사항]
# 1. 다른 분석 파일에도 MAPE 계산식이 중복되어 있는지 점검
# 2. 실제값이 모두 0인 경우의 평가 결과 처리 방식 문서화
