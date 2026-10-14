import csv
import io
import calendar
import os
import tempfile
from datetime import datetime
from pathlib import Path

import requests


# 프로젝트 최상위 폴더
BASE_DIR = Path(__file__).resolve().parents[1]

# 원본 데이터를 저장할 폴더
RAW_DIR = BASE_DIR / "data" / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)


# 서울 열린데이터광장 지하철 파일 다운로드 주소
SUBWAY_DL = (
    "https://datafile.seoul.go.kr/bigfile/iot/inf/nio_download.do?useCache=false"
)

# 데이터셋 ID
SUBWAY_INF = "OA-12914"

# 사용할 연도
SUBWAY_YEARS = [2019, 2020, 2021, 2022, 2023, 2024]


# 2019~2022는 연도별 파일 1개
YEAR_SEQ = {2019: [110], 2020: [111], 2021: [112], 2022: [113]}

# 2023~2024는 월별 파일 12개
MONTH_SEQ_BASE = {2023: 114, 2024: 126}


def download_seq(seq):
    # 서울 열린데이터광장에 파일 요청
    try:
        response = requests.post(
            SUBWAY_DL,
            timeout=180,
            headers={"User-Agent": "Mozilla/5.0"},
            data={
                "infId": SUBWAY_INF,
                "seq": seq,
                "seqNo": seq,
                "infSeq": 3,
            },
        )

        response.raise_for_status()

    except requests.RequestException as error:
        raise RuntimeError(f"seq={seq} 다운로드 요청 실패: {error}") from error

    # 너무 작은 파일이면 오류 페이지일 가능성이 있음
    if len(response.content) < 1000:
        raise RuntimeError(
            f"seq={seq} 응답 크기가 비정상적으로 작음: {len(response.content)}바이트"
        )

    # 실제 CARD_SUBWAY 파일인지 확인
    content_disposition = response.headers.get("Content-Disposition", "")

    if "CARD_SUBWAY" not in content_disposition.upper():
        raise RuntimeError(f"seq={seq} 응답이 지하철 CSV 파일로 확인되지 않음")

    return response.content


def decode_file(content):
    # 연도마다 CSV 인코딩이 달라서 순서대로 시도
    encodings = ["utf-8-sig", "euc-kr", "cp949"]

    for encoding in encodings:
        try:
            return content.decode(encoding)
        except UnicodeDecodeError:
            continue

    # 인코딩이 확인되지 않은 데이터를 임의로 대체하지 않음
    raise ValueError("CSV 인코딩을 확인할 수 없음: utf-8-sig, euc-kr, cp949")


def parse_csv(text, source):
    # CSV 구조와 필수 컬럼을 확인
    try:
        reader = csv.reader(io.StringIO(text, newline=""))

        header = next(reader, None)

        if not header:
            raise ValueError("제목행이 없음")

        header = [column.strip().lstrip("\ufeff") for column in header]

        required_columns = {
            "사용일자",
            "노선명",
            "역명",
            "승차총승객수",
            "하차총승객수",
            "등록일자",
        }

        missing = required_columns - set(header)

        if missing:
            raise ValueError(f"필수 컬럼 누락: {sorted(missing)}")

        rows = []
        column_count = len(header)

        for line_number, row in enumerate(reader, start=2):
            if not row or not any(value.strip() for value in row):
                continue

            if len(row) < column_count:
                raise ValueError(
                    f"{line_number}행의 컬럼 수 부족: "
                    f"{len(row)}개 / 예상 {column_count}개"
                )

            # 기존 DB 적재 코드와 동일하게
            # 원래 컬럼 수를 초과하는 값은 제외
            rows.append(row[:column_count])

    except csv.Error as error:
        raise ValueError(f"{source} CSV 파싱 실패: {error}") from error

    if not rows:
        raise ValueError(f"{source} 데이터 행이 없음")

    return header, rows


def validate_dates(header, rows, year, month=None):
    # 사용일자 컬럼을 기준으로 실제 날짜 범위를 검증
    date_index = header.index("사용일자")

    actual_dates = set()

    for row_number, row in enumerate(rows, start=2):
        date_text = row[date_index].strip()

        try:
            date_value = datetime.strptime(date_text, "%Y%m%d").date()

        except ValueError as error:
            raise ValueError(
                f"{year}년 {row_number}행 사용일자 오류: {date_text}"
            ) from error

        if date_value.year != year:
            raise ValueError(f"{year}년 파일에 다른 연도 날짜 포함: {date_text}")

        if month is not None and date_value.month != month:
            raise ValueError(
                f"{year}년 {month}월 파일에 다른 월 날짜 포함: {date_text}"
            )

        actual_dates.add(date_value)

    if month is None:
        expected_days = 366 if calendar.isleap(year) else 365

        expected_dates = {
            datetime(year, 1, 1).date() + __import__("datetime").timedelta(days=offset)
            for offset in range(expected_days)
        }

        period_name = f"{year}년"

    else:
        expected_days = calendar.monthrange(year, month)[1]

        expected_dates = {
            datetime(year, month, day).date() for day in range(1, expected_days + 1)
        }

        period_name = f"{year}년 {month:02d}월"

    missing_dates = sorted(expected_dates - actual_dates)

    if missing_dates:
        sample = ", ".join(date.isoformat() for date in missing_dates[:5])

        raise ValueError(
            f"{period_name} 날짜 누락: {len(missing_dates)}일 (예: {sample})"
        )

    if len(actual_dates) != expected_days:
        raise ValueError(
            f"{period_name} 날짜 수 오류: "
            f"{len(actual_dates)}일 / 예상 {expected_days}일"
        )

    return len(actual_dates)


def validate_year_file(path, year):
    # 기존 CSV도 같은 기준으로 검증
    path = Path(path)

    if not path.is_file():
        raise FileNotFoundError(f"지하철 원본 파일이 없음: {path}")

    with open(
        path,
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as file:
        text = file.read()

    header, rows = parse_csv(text, path.name)
    day_count = validate_dates(header, rows, year)

    return {
        "year": year,
        "rows": len(rows),
        "days": day_count,
    }


def save_validated_csv(year, header, rows):
    # 검증을 통과한 데이터만 임시 파일에 저장
    save_path = RAW_DIR / f"subway_daily_{year}.csv"

    temporary_path = None

    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8-sig",
            newline="",
            suffix=".csv",
            prefix=f"subway_daily_{year}_",
            dir=RAW_DIR,
            delete=False,
        ) as file:
            temporary_path = Path(file.name)

            writer = csv.writer(file)
            writer.writerow(header)
            writer.writerows(rows)

        # 실제 저장된 임시 파일까지 다시 검증
        result = validate_year_file(temporary_path, year)

        # 검증 성공 후에만 최종 파일 교체
        os.replace(temporary_path, save_path)

        print(
            f"[ok ] 지하철 {year} → {save_path.name} "
            f"({result['rows']:,}행 / {result['days']}일)"
        )

    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def download_subway():
    for year in SUBWAY_YEARS:
        print(f"[run] 지하철 {year}년 데이터 준비")

        # 2019~2022: 연도별 파일
        if year in YEAR_SEQ:
            seq = YEAR_SEQ[year][0]

            content = download_seq(seq)
            text = decode_file(content)

            header, rows = parse_csv(text, f"{year}년 seq={seq}")

            validate_dates(header, rows, year)

        # 2023~2024: 월별 파일
        else:
            base_seq = MONTH_SEQ_BASE[year]

            header = None
            rows = []

            for month in range(1, 13):
                seq = base_seq + month - 1

                content = download_seq(seq)
                text = decode_file(content)

                month_header, month_rows = parse_csv(
                    text,
                    f"{year}년 {month:02d}월 seq={seq}",
                )

                # 각 월의 날짜 누락 여부 검증
                validate_dates(
                    month_header,
                    month_rows,
                    year,
                    month=month,
                )

                if header is None:
                    header = month_header

                elif month_header != header:
                    raise ValueError(
                        f"{year}년 {month:02d}월 CSV 제목행이 다른 월과 일치하지 않음"
                    )

                rows.extend(month_rows)

                print(f"[ok ] {year}-{month:02d} seq={seq} / {len(month_rows):,}행")

            # 12개월 결합 후 연도 전체 날짜 재검증
            validate_dates(header, rows, year)

        save_validated_csv(year, header, rows)


def main():
    try:
        download_subway()

    except (
        RuntimeError,
        ValueError,
        OSError,
    ) as error:
        raise SystemExit(f"[FAIL] 지하철 원본 데이터 준비 중단: {error}") from error


if __name__ == "__main__":
    main()


# ============================================================
# [구현 핵심]
# ============================================================

# 서울 열린데이터광장의 지하철 원본 CSV를 2019~2024년까지 다운로드함
# 2019~2022년은 연도별 파일, 2023~2024년은 월별 파일을 결합함
# 모든 월과 연도의 날짜 및 CSV 구조를 검증한 후 최종 파일을 저장함
# 임시 파일을 사용하여 다운로드 실패 시 기존 정상 파일을 보존함
#
# [발견한 문제 및 수정]
# 1. 월별 다운로드 실패 시 continue로 다음 월을 처리하던 문제
#    → 다운로드 실패 시 즉시 중단하여 누락된 월이 저장되지 않도록 수정함
#
# 2. 첫 달 다운로드 실패 시 제목행이 누락될 수 있었음
#    → 모든 월의 제목행을 확인하고 일치 여부를 검증함
#
# 3. 불완전한 CSV가 정상 파일로 저장될 수 있었음
#    → 날짜 범위 및 날짜 수 검증 후 임시 파일을 최종 파일로 교체함
#
# 4. 디코딩 실패 시 깨진 문자를 대체하여 저장하던 문제
#    → 지원 인코딩으로 해석할 수 없으면 오류를 발생시키도록 변경함
#
# [추후 보완 사항]
# 1. 월별 파일에서 날짜별 역·노선 누락 여부까지 검증
# 2. 다운로드 재시도 및 네트워크 오류 복구 기능 추가
# 3. 원본 파일의 체크섬 기록을 통한 데이터 버전 추적
