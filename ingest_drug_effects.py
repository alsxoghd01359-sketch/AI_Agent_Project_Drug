"""효능효과 텍스트만 담은 전용 컬렉션(drug_effects) 적재.

drug_products는 효능효과+용법용량+사용상주의사항+상호작용+이상반응을 다 합쳐서
임베딩했더니, 사용상주의사항의 상투적 문구(임부·수유부는 사용하지 마십시오 등)가
비슷비슷해서 효능효과의 구분력이 묻히는 문제가 있었다.
"두통에 먹는 약" 같은 증상 기반 검색은 효능효과 텍스트만 따로 짧게 임베딩해야
노이즈가 줄어든다.
"""
import json
from pathlib import Path

from openai import RateLimitError

from chroma_setup import get_client, get_or_create_collection
from ingest_chroma import _read_jsonl, doc_xml_to_text, clean_metadata, _add_with_retry

DATA_DIR = Path("data")
PROGRESS_FILE = DATA_DIR / "_chroma_progress.json"
BATCH_SIZE = 30
MAX_DOC_CHARS = 800  # 효능효과만이라 짧아도 충분


def _load_progress() -> dict:
    if PROGRESS_FILE.exists():
        return json.loads(PROGRESS_FILE.read_text(encoding="utf-8"))
    return {}


def _save_progress(progress: dict) -> None:
    PROGRESS_FILE.write_text(json.dumps(progress, ensure_ascii=False, indent=2), encoding="utf-8")


def main():
    client = get_client()
    progress = _load_progress()

    coll_key = "drug_effects"
    if progress.get(coll_key, {}).get("done"):
        print(f"[{coll_key}] 이미 완료됨. 건너뜀.")
        return

    print(f"[{coll_key}] 원본 병합 중...")
    easy_by_seq = {r["itemSeq"]: r for r in _read_jsonl("easy_drug")}
    detail_by_seq = {r["ITEM_SEQ"]: r for r in _read_jsonl("drug_prdt_prmsn_detail")}

    def records():
        for rec in _read_jsonl("drug_prdt_prmsn_list"):
            seq = rec["ITEM_SEQ"]
            easy = easy_by_seq.get(seq, {})
            detail = detail_by_seq.get(seq, {})
            efcy = easy.get("efcyQesitm") or doc_xml_to_text(detail.get("EE_DOC_DATA"))
            if not efcy:
                continue
            yield {
                "item_seq": seq,
                "item_name": rec.get("ITEM_NAME"),
                "entp_name": rec.get("ENTP_NAME"),
                "spclty_pblc": rec.get("SPCLTY_PBLC"),
                "efcy": efcy[:MAX_DOC_CHARS],
            }

    coll = get_or_create_collection(client, coll_key, real_embedding=True)
    start_idx = progress.get(coll_key, {}).get("last_done_idx", -1) + 1

    ids, docs, metas = [], [], []
    idx = -1
    added = 0

    for idx, rec in enumerate(records()):
        if idx < start_idx:
            continue

        ids.append(f"eff_{idx}")
        docs.append(f"제품명: {rec['item_name']}\n효능효과: {rec['efcy']}")
        metas.append(clean_metadata({
            "item_seq": rec["item_seq"],
            "item_name": rec["item_name"],
            "entp_name": rec["entp_name"],
            "spclty_pblc": rec["spclty_pblc"],
        }))

        if len(ids) >= BATCH_SIZE:
            _add_with_retry(coll, ids, docs, metas)
            added += len(ids)
            ids, docs, metas = [], [], []
            progress[coll_key] = {"last_done_idx": idx}
            _save_progress(progress)
            if added % 3000 == 0:
                print(f"  [{coll_key}] {added}건 처리 중...")

    if ids:
        _add_with_retry(coll, ids, docs, metas)
        added += len(ids)

    if idx >= 0:
        progress[coll_key] = {"last_done_idx": idx, "done": True}
        _save_progress(progress)

    print(f"[{coll_key}] 완료: {added}건 -> 누적 {coll.count()}건")


if __name__ == "__main__":
    main()
