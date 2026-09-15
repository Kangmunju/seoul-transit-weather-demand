# 같은 모델 사용
# 랜덤하게 데이터 나누기 VS 데이터를 시간순으로 나누기
# 평가가 어떻게 달라지는 지 확인


import numpy as np

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import TimeSeriesSplit

from _common import load_merged

import features as F


SEED = 42


# 랜덤 포레스트를 여러 번 학습하고 평가할 예정
# 반복해서 전부 작성하면 코드가 너무 길어질 것으로 예상
# 학습/테스트 데이터를 주면 랜덤 포레스트를 학습시키고 성능까지 계산해주는 함수로 묶어서 구현함
def fit_eval(Xtr, ytr, Xte, yte):
    model = RandomForestRegressor(
        n_estimators=400,
        random_state=SEED,
        min_samples_leaf=2,  # 한 데이터 하나에만 맞는 지나치게 세세한 규칙 방지
        n_jobs=-1,
    )
    # Xtr = 학습할 때 모델에게 보여주는 피처
    # ytr = Xtr에 해당하는 실제 이용객 수(정답)
    # Xte = 평가할 때 모델에게 넣어볼 피처
    # yte = Xte에 해당하는 실제 이용객 수(정답)

    model.fit(Xtr, ytr)
    pred = model.predict(Xte)

    mae = mean_absolute_error(yte, pred)
    r2 = r2_score(yte, pred)
    mape = float(np.mean(np.abs((yte - pred) / yte)) * 100)

    return mae, r2, mape


def nearest_gap(dates, train_idx, test_idx):
    train_dates = np.sort(dates[train_idx])

    gaps = []

    for i in test_idx:  # 테스트 데이터로 뽑힌 행 번호 하나씩 반복
        gap = np.min(np.abs((train_dates - dates[i]) / np.timedelta64(1, "D")))
        # 현재 테스트 날짜 하나를 모든 train 날짜와 뺌
        # 날짜 차이를 '며칠'이라는 숫자 단위로 바꿈
        # 절댓값(과거인지 미래인지는 무관, 얼마나 가까운지만 체크할 것)
        # 가장 작은 값을 고름
        gaps.append(gap)

    return np.median(gaps)


def main():
    # _common.py에 만들어 둔 load_merged() 호출
    # merged에 이미 지하철 일별 수요, 날씨, 달력 등의 데이터가 날짜 기준으로 합쳐져 있음
    # 반환되는 값들 중 merged만 실제로 사용
    # sw, rep는 사용하지 않지만 함수 자체가 3개 값 반환하도록 설정하였음으로 일단 같이 받았음
    merged, sw, rep = load_merged()

    d = merged.sort_values("date").reset_index(drop=True).copy()

    d = F.build(d)  # import features as F
    # 기존 데이터에 시계열 피처 추가
    # 이 파일에서 피처 생성 방식을 새로 만들지 않고
    # 실제 모델 학습 때 사용했던 features.py를 재사용

    feature_cols = F.feature_columns(include_pandemic=True)

    # 모델에 넣을 피처 또는 정답 ride가 비어 있는 행 제외
    # lag_7 최초 7일에 과거 7일 전 데이터가 없어 결측 생겼었음
    # 원래 2,192일에서 초반 7일이 빠져 2,185일이 실제 비교에 사용되는 구조
    d = d.dropna(subset=feature_cols + [F.TARGET]).reset_index(drop=True)

    X = d[feature_cols].to_numpy()  # 모델에게 주는 문제
    y = d[F.TARGET].to_numpy()  # 실제 지하철 이용객 수(ride)
    # train 날짜와 test 날짜가 실제 시간상 얼마나 가까운지 계산하기 위해 따로 보관
    dates = d["date"].to_numpy()

    n = len(d)  # 사용할 수 있는 전체 데이터 개수(현재 데이터 기준 2,185일)
    cut = int(n * 0.75)

    print("[같은 데이터·모델·피처, split만 다르게]")

    # 시간순 split
    time_train_idx = np.arange(cut)  # d를 날짜순 정렬했으므로 과거 75% 날짜가 됨
    time_test_idx = np.arange(cut, n)  # 뒤쪽 25%의 미래 데이터

    mae_t, r2_t, mape_t = fit_eval(
        X[time_train_idx],  # 과거 75%의 입력 피처
        y[time_train_idx],  # 과거 75%의 실제 이용객 수
        X[time_test_idx],  # 미래 25%의 입력 피처
        y[time_test_idx],  # 미래 25%의 실제 이용객 수
    )

    # 랜덤 split
    rng = np.random.RandomState(SEED)
    idx = rng.permutation(n)

    random_test_idx = idx[: n - cut]
    random_train_idx = idx[n - cut :]

    mae_r, r2_r, mape_r = fit_eval(
        X[random_train_idx],
        y[random_train_idx],
        X[random_test_idx],
        y[random_test_idx],
    )

    # 같은 데이터와 모델인데 split 방식만 바꾼 경우 평가가 얼마나 달라지는 지 확인
    # 그리고 그 차이가 실제 성능 차이로 이어지는 지 검토
    print(f"랜덤 split  MAE {mae_r:,.0f}  MAPE {mape_r:.2f}%  R² {r2_r:.4f}")
    print(f"시간순 split MAE {mae_t:,.0f}  MAPE {mape_t:.2f}%  R² {r2_t:.4f}")

    print("\n[메커니즘 - test 날과 가장 가까운 train 날의 거리]")

    random_gap = nearest_gap(
        dates,
        random_train_idx,
        random_test_idx,
    )

    time_gap = nearest_gap(
        dates,
        time_train_idx,
        time_test_idx,
    )

    print(f"랜덤 split: 중앙값 {random_gap:.0f}일")
    print(f"시간순 split: 중앙값 {time_gap:.0f}일")

    # 앞에서 시간순으로 한 번 나눠 검증했음
    # 딱 한 기간에서만 평가했기 때문에 여러 시점에서 반복 평가하는 것이 필요하다고 판단
    print("\n[walk-forward 시계열 교차검증]")

    # 시계열 데이터를 5번에 걸쳐 train/test로 나눌 것
    # 뒤로 갈수록 train이 점점 늘어남
    # 지금까지 확보한 과거 데이터로 시간을 앞으로 이동시키며 미래 구간을 계속 검증
    tscv = TimeSeriesSplit(n_splits=5)

    maes = []
    r2s = []

    for k, (train_idx, test_idx) in enumerate(tscv.split(X), 1):
        mae_k, r2_k, mape_k = fit_eval(
            X[train_idx],
            y[train_idx],
            X[test_idx],
            y[test_idx],
        )
        # tscv.split(X)로 train(과거)과 test(미래)에 사용할 행 번호를 만듦
        # enumerate(,1)로 터미널에 fold1부터 자연스럽게 출력할 수 있도록 함
        # 앞에서 만든 fit_eval() 재사용
        # fold2가 되면 또 새로운 랜덤 포레스트를 만들어 다시 학습하는 구조

        maes.append(mae_k)
        r2s.append(r2_k)

        print(
            f"fold{k}: "
            f"train {len(train_idx)}일 → "
            f"test {len(test_idx)}일 "
            f"MAE {mae_k:,.0f} "
            f"MAPE {mape_k:.2f}% "
            f"R² {r2_k:.3f}"
        )

    print(f"\n평균 MAE {np.mean(maes):,.0f} (±{np.std(maes):,.0f})")

    print(f"평균 R² {np.mean(r2s):.3f} (±{np.std(r2s):.3f})")


if __name__ == "__main__":
    main()


# [같은 데이터·모델·피처, split만 다르게]
# 랜덤 split  MAE 349,963  MAPE 3.61%  R² 0.9723
# 시간순 split MAE 347,269  MAPE 2.99%  R² 0.9649

# [메커니즘 - test 날과 가장 가까운 train 날의 거리]
# 랜덤 split: 중앙값 1일
# 시간순 split: 중앙값 274일

# [walk-forward 시계열 교차검증]
# fold1: train 365일 → test 364일 MAE 3,092,521 MAPE 34.01% R² -0.178
# fold2: train 729일 → test 364일 MAE 363,771 MAPE 4.08% R² 0.969
# fold3: train 1093일 → test 364일 MAE 659,678 MAPE 5.49% R² 0.914
# fold4: train 1457일 → test 364일 MAE 384,803 MAPE 3.40% R² 0.956
# fold5: train 1821일 → test 364일 MAE 336,583 MAPE 2.84% R² 0.965

# 평균 MAE 967,471 (±1,068,882)
# 평균 R² 0.725 (±0.452)


# ============================================================

# 같은 데이터, 같은 피처, 같은 RandomForest 모델을 사용
# train/test split 방식만 바꿔 평가 결과가 어떻게 달라지는지 비교함

# 랜덤 split은 시간 순서를 무시함
# test 시점보다 미래의 데이터가 train에 포함될 가능성 존재
# test 날짜와 매우 가까운 날짜가 train에 포함될 가능성 존재
# 시간순 split은 과거로 학습하고 이후 미래를 테스트함
# 실제 미래 수요 예측과 최대한 더 비슷한 평가 구조를 만들고자 하였음

# 시간 split에서도 train/test 인덱스를 따로 저장함
# 모델 평가뿐 아니라 nearest_gap()에서도 같은 인덱스를 사용
# train과 test가 시간상 얼마나 떨어져 있는지 비교 가능

# nearest_gap()으로 각 test 날짜와 가장 가까운 train 날짜의 거리를 계산
# 랜덤 split에서는 train/test가 시간축에서 매우 가깝게 섞이는 것을 확인
# 시간순 split에서는 두 구간이 시간적으로 분리되는 것을 확인
# 단, 날짜가 가깝다는 사실 자체를 데이터 누수라고 단정하지는 않았음
# 랜덤 split이 실제 미래 예측과 다른 평가 환경을 만든다는 점을 확인하는 용도로 사용함

# 실제 비교에서는 랜덤 split이 모든 평가 지표에서 더 좋지는 않은 결과가 나왔음
# 랜덤 split이 항상 성능을 높인다고 단정할 수는 없다는 결론을 내림
# 시계열에서는 실제 예측 상황에 맞는 분할 방식을 사용하는 것이 더 중요하다고 판단함

# 한 번의 시간순 split 결과만으로 모델을 판단하지 않으려고 함
# TimeSeriesSplit으로 과거 -> 미래 순서를 유지한 walk-forward 검증을 5번 수행

# fold마다 새로운 RandomForest를 학습해 시점별 성능을 비교하였음
# 초기 fold에서는 성능이 크게 낮고 이후 fold에서는 비교적 안정적인 모습을 확인함
# 단일 split의 높은 성능만으로 일반화 성능을 판단하기는 어려움
# 여러 미래 시점에서 반복 검증하는 과정이 필요함을 확인하였음
