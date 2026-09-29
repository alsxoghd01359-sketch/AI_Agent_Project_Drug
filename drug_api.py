"""식약처 의약품/건강기능식품 관련 공공데이터 API 클라이언트.

승인된 서비스 4개, 오퍼레이션 총 15개:

- DURPrdlstInfoService03 (https://apis.data.go.kr/1471000/DURPrdlstInfoService03)
    의약품 제품허가 목록/상세/주성분상세 (3개)
    DUR 병용금기/노인주의/DUR품목정보/특정연령대금기/용량주의/투여기간주의/
    효능군중복/서방정분할주의/임부금기 (9개)
- MdcinGrnIdntfcInfoService03 (https://apis.data.go.kr/1471000/MdcinGrnIdntfcInfoService03)
    낱알식별정보 (1개)
- DrbEasyDrugInfoService (https://apis.data.go.kr/1471000/DrbEasyDrugInfoService)
    e약은요 의약품개요정보 (1개)
- HtfsInfoService03 (https://apis.data.go.kr/1471000/HtfsInfoService03)
    건강기능식품 상세정보 (1개, 목록조회는 상세정보의 부분집합이라 생략)
"""
import os
import time
from urllib.parse import urlencode

import requests
from dotenv import load_dotenv

load_dotenv()

SERVICE_KEY = os.getenv("DATA_GO_KR_SERVICE_KEY")
if not SERVICE_KEY:
    raise RuntimeError(".env 파일에 DATA_GO_KR_SERVICE_KEY를 설정해주세요.")

_KEY_ALREADY_ENCODED = "%" in SERVICE_KEY
_session = requests.Session()

BASE_URLS = {
    "prmsn": "https://apis.data.go.kr/1471000/DrugPrdtPrmsnInfoService08",
    "dur": "https://apis.data.go.kr/1471000/DURPrdlstInfoService03",
    "pill": "https://apis.data.go.kr/1471000/MdcinGrnIdntfcInfoService03",
    "easy": "https://apis.data.go.kr/1471000/DrbEasyDrugInfoService",
    "htfs": "https://apis.data.go.kr/1471000/HtfsInfoService03",
}


class DrugApiError(RuntimeError):
    pass


def _call(base_key: str, endpoint: str, params: dict) -> dict:
    base_url = BASE_URLS[base_key]
    other = {"type": "json", **params}

    if _KEY_ALREADY_ENCODED:
        url = f"{base_url}/{endpoint}?serviceKey={SERVICE_KEY}&{urlencode(other)}"
        resp = _session.get(url, timeout=15)
    else:
        resp = _session.get(f"{base_url}/{endpoint}", params={"serviceKey": SERVICE_KEY, **other}, timeout=15)

    resp.raise_for_status()
    data = resp.json()

    header = data.get("response", {}).get("header") or data.get("header") or {}
    result_code = header.get("resultCode")

    if result_code == "03":  # NODATA_ERROR: 검색 결과 없음(정상)
        return {"items": [], "totalCount": 0}

    if result_code not in ("00", None):
        raise DrugApiError(f"{endpoint} 요청 실패: {result_code} {header.get('resultMsg')}")

    body = data.get("response", {}).get("body") or data.get("body") or {}
    items = body.get("items")
    if items is None:
        items = []
    elif isinstance(items, dict):
        items = items.get("item") or []
    elif isinstance(items, list):
        # 일부 서비스(HtfsInfoService03 등)는 items가 [{"item": {...}}, ...] 형태로 옴
        items = [it["item"] if isinstance(it, dict) and "item" in it else it for it in items]
    if isinstance(items, dict):
        items = [items]

    return {"items": items, "totalCount": body.get("totalCount", len(items))}


def fetch_page(base_key: str, endpoint: str, params: dict, page_no: int, num_of_rows: int) -> dict:
    return _call(base_key, endpoint, {**params, "pageNo": page_no, "numOfRows": num_of_rows})


def iter_all_pages(base_key: str, endpoint: str, params: dict, num_of_rows: int = 100,
                    start_page: int = 1, sleep_sec: float = 0.05):
    """(page_no, items, total_count)를 페이지 단위로 순서대로 yield."""
    page_no = start_page
    while True:
        result = fetch_page(base_key, endpoint, params, page_no, num_of_rows)
        items = result["items"]
        total_count = result["totalCount"]

        yield page_no, items, total_count

        if not items or page_no * num_of_rows >= total_count:
            break

        page_no += 1
        time.sleep(sleep_sec)
