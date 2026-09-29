"""6개 승인 API, 14개 오퍼레이션을 전부 페이징하며 원본 데이터를 JSONL로 저장.

중간에 끊겨도 data/_progress.json에 마지막으로 끝낸 페이지가 기록되어 있어서
다시 실행하면 이어서 진행합니다.

사용법: python fetch_drug_data.py
"""
import json
from pathlib import Path

import requests

from drug_api import DrugApiError, fetch_page

DATA_DIR = Path("data")
PROGRESS_FILE = DATA_DIR / "_progress.json"
NUM_OF_ROWS = 500

# (파일명, base_key, endpoint, 고정 요청 파라미터)
OPERATIONS = [
    ("drug_prdt_prmsn_list", "prmsn", "getDrugPrdtPrmsnInq08", {}),
    ("drug_prdt_prmsn_detail", "prmsn", "getDrugPrdtPrmsnDtlInq08", {}),
    ("drug_prdt_mcpn_detail", "prmsn", "getDrugPrdtMcpnDtlInq08", {}),
    ("dur_usjnt_taboo", "dur", "getUsjntTabooInfoList03", {}),
    ("dur_odsn_atent", "dur", "getOdsnAtentInfoList03", {}),
    ("dur_prdlst", "dur", "getDurPrdlstInfoList03", {}),
    ("dur_spcify_agrde_taboo", "dur", "getSpcifyAgrdeTabooInfoList03", {}),
    ("dur_cpcty_atent", "dur", "getCpctyAtentInfoList03", {}),
    ("dur_mdctn_pd_atent", "dur", "getMdctnPdAtentInfoList03", {}),
    ("dur_efcy_dplct", "dur", "getEfcyDplctInfoList03", {}),
    ("dur_seobangjeong_partitn_atent", "dur", "getSeobangjeongPartitnAtentInfoList03", {}),
    ("dur_pwnm_taboo", "dur", "getPwnmTabooInfoList03", {}),
    ("pill_ident", "pill", "getMdcinGrnIdntfcInfoList03", {}),
    ("easy_drug", "easy", "getDrbEasyDrugList", {}),
    ("htfs_item", "htfs", "getHtfsItem01", {}),
]


def _load_progress() -> dict:
    if PROGRESS_FILE.exists():
        return json.loads(PROGRESS_FILE.read_text(encoding="utf-8"))
    return {}


def _save_progress(progress: dict) -> None:
    PROGRESS_FILE.write_text(json.dumps(progress, ensure_ascii=False, indent=2), encoding="utf-8")


def fetch_operation(name: str, base_key: str, endpoint: str, params: dict, progress: dict) -> None:
    out_file = DATA_DIR / f"{name}.jsonl"
    state = progress.get(name, {"last_done_page": 0, "total_count": None})

    if state.get("done"):
        print(f"[{name}] 이미 완료됨 (누적 {state['total_count']}건). 건너뜀.")
        return

    start_page = state["last_done_page"] + 1
    mode = "a" if state["last_done_page"] > 0 else "w"

    print(f"[{name}] {start_page}페이지부터 시작 (numOfRows={NUM_OF_ROWS})")

    with out_file.open(mode, encoding="utf-8") as f:
        page_no = start_page
        while True:
            try:
                result = fetch_page(base_key, endpoint, params, page_no, NUM_OF_ROWS)
            except (DrugApiError, requests.exceptions.RequestException) as e:
                print(f"[{name}] 중단: {page_no}페이지에서 오류 - {e}")
                print(f"         {state['last_done_page']}페이지까지 저장됨. 다시 실행하면 이어서 진행됩니다.")
                return

            items = result["items"]
            total_count = result["totalCount"]

            for item in items:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

            state["last_done_page"] = page_no
            state["total_count"] = total_count
            progress[name] = state
            _save_progress(progress)

            fetched_so_far = page_no * NUM_OF_ROWS
            print(f"  [{name}] {page_no}페이지 완료 ({min(fetched_so_far, total_count)}/{total_count})")

            if not items or fetched_so_far >= total_count:
                state["done"] = True
                progress[name] = state
                _save_progress(progress)
                print(f"[{name}] 완료: 총 {total_count}건 -> {out_file}")
                return

            page_no += 1


def main():
    DATA_DIR.mkdir(exist_ok=True)
    progress = _load_progress()

    for name, base_key, endpoint, params in OPERATIONS:
        fetch_operation(name, base_key, endpoint, params, progress)


if __name__ == "__main__":
    main()
