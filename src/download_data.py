import requests
from pathlib import Path


# 원본 데이터를 저장할 폴더
RAW_DIR = Path("data/raw")
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
    response = requests.post(
        SUBWAY_DL,
        timeout=180,
        headers={"User-Agent": "Mozilla/5.0"},
        data={"infId": SUBWAY_INF, "seq": seq, "seqNo": seq, "infSeq": 3},
    )

    # 정상 응답인지 확인
    if response.status_code != 200:
        return None

    # 너무 작은 파일이면 오류 페이지일 가능성이 있음
    if len(response.content) < 1000:
        return None

    # 실제 CARD_SUBWAY 파일인지 확인
    content_disposition = response.headers.get("Content-Disposition", "")

    if "CARD_SUBWAY" not in content_disposition:
        return None

    return response.content


def decode_file(content):
    # 연도마다 CSV 인코딩이 달라서 순서대로 시도
    encodings = ["utf-8-sig", "euc-kr", "cp949"]

    for encoding in encodings:
        try:
            return content.decode(encoding)
        except UnicodeDecodeError:
            continue

    # 위 인코딩이 전부 실패하면 깨진 문자를 대체해서 읽음
    return content.decode("utf-8", errors="replace")


def download_subway():
    for year in SUBWAY_YEARS:
        # 2019~2022
        if year in YEAR_SEQ:
            seq = YEAR_SEQ[year][0]

            content = download_seq(seq)

            if content is None:
                print(f"[FAIL] 지하철 {year} seq={seq} 다운로드 실패")
                continue

            text = decode_file(content)

        # 2023~2024
        else:
            base_seq = MONTH_SEQ_BASE[year]

            month_texts = []

            for month in range(12):
                seq = base_seq + month

                content = download_seq(seq)

                if content is None:
                    print(f"[FAIL] 지하철 {year} seq={seq} 다운로드 실패")
                    continue

                text = decode_file(content)

                lines = text.splitlines()

                # 첫 달은 제목행까지 저장
                if month == 0:
                    month_texts.extend(lines)

                # 두 번째 달부터는 제목행 제외
                else:
                    month_texts.extend(lines[1:])

            text = "\n".join(month_texts)

        # 연도별 CSV 저장
        save_path = RAW_DIR / f"subway_daily_{year}.csv"

        with open(save_path, "w", encoding="utf-8-sig", newline="") as file:
            file.write(text)

        print(f"[ok ] 지하철 {year} → {save_path.name}")


download_subway()
