"""data/*.jsonl 원본 데이터를 정리해서 ChromaDB 컬렉션 7개에 적재.

- drug_products, health_foods: 텍스트 설명이 있어서 OpenAI 임베딩(진짜 RAG 대상)
- 나머지: 구조화 조회 전용, 더미 임베딩(비용/시간 절약, where 필터로만 조회)

재실행하면 data/_chroma_progress.json에 기록된 지점부터 이어서 적재합니다.
"""
import json
import time
from pathlib import Path

from openai import RateLimitError

from chroma_setup import get_client, get_or_create_collection

DATA_DIR = Path("data")
PROGRESS_FILE = DATA_DIR / "_chroma_progress.json"
BATCH_SIZE = 300
REAL_EMBED_BATCH_SIZE = 30  # OpenAI 요청당 토큰 한도(30만) 넘지 않도록 실제 임베딩 컬렉션은 작게
MAX_DOC_CHARS = 2000  # 문서 하나가 너무 길면 임베딩 요청이 비대해지므로 자름


def _load_progress() -> dict:
    if PROGRESS_FILE.exists():
        return json.loads(PROGRESS_FILE.read_text(encoding="utf-8"))
    return {}


def _save_progress(progress: dict) -> None:
    PROGRESS_FILE.write_text(json.dumps(progress, ensure_ascii=False, indent=2), encoding="utf-8")


def _read_jsonl(name: str):
    with (DATA_DIR / f"{name}.jsonl").open(encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            yield {k.strip(): v for k, v in rec.items()}  # dur_prdlst의 "TYPE_NAME  " 같은 키 공백 제거


def clean_metadata(d: dict) -> dict:
    cleaned = {}
    for k, v in d.items():
        if v is None or v == "":
            continue
        if isinstance(v, (str, int, float, bool)):
            cleaned[k] = v
        else:
            cleaned[k] = str(v)
    return cleaned


def doc_xml_to_text(xml_str):
    from text_extract import doc_xml_to_text as _f
    return _f(xml_str)


def _add_with_retry(collection, ids, docs, metas, max_retries: int = 8):
    delay = 2
    for attempt in range(max_retries):
        try:
            collection.add(ids=ids, documents=docs, metadatas=metas)
            return
        except RateLimitError:
            print(f"  (rate limit, {delay}초 대기 후 재시도 {attempt + 1}/{max_retries})")
            time.sleep(delay)
            delay = min(delay * 2, 60)
    raise RuntimeError("rate limit 재시도 초과")


def ingest_records(collection, coll_key: str, records_iter, to_doc_and_meta, progress: dict, id_prefix: str,
                    batch_size: int = BATCH_SIZE):
    """records_iter: dict 레코드 제너레이터. to_doc_and_meta(rec) -> (document_text, metadata_dict) | None(스킵)."""
    start_idx = progress.get(coll_key, {}).get("last_done_idx", -1) + 1

    ids, docs, metas = [], [], []
    idx = -1
    added = 0

    for idx, rec in enumerate(records_iter):
        if idx < start_idx:
            continue

        result = to_doc_and_meta(rec)
        if result is None:
            continue
        doc_text, meta = result
        doc_text = (doc_text or " ")[:MAX_DOC_CHARS]

        ids.append(f"{id_prefix}_{idx}")
        docs.append(doc_text)
        metas.append(clean_metadata(meta))

        if len(ids) >= batch_size:
            _add_with_retry(collection, ids, docs, metas)
            added += len(ids)
            ids, docs, metas = [], [], []
            progress[coll_key] = {"last_done_idx": idx}
            _save_progress(progress)

    if ids:
        _add_with_retry(collection, ids, docs, metas)
        added += len(ids)

    if idx >= 0:
        progress[coll_key] = {"last_done_idx": idx, "done": True}
        _save_progress(progress)

    print(f"[{coll_key}] {added}건 추가 (누적 {collection.count()}건)")


def main():
    client = get_client()
    progress = _load_progress()

    # ---------- 1) drug_products (실제 임베딩) ----------
    if not progress.get("drug_products", {}).get("done"):
        print("[drug_products] 원본 병합 중...")
        detail_by_seq = {r["ITEM_SEQ"]: r for r in _read_jsonl("drug_prdt_prmsn_detail")}
        easy_by_seq = {r["itemSeq"]: r for r in _read_jsonl("easy_drug")}

        def merged_products():
            for rec in _read_jsonl("drug_prdt_prmsn_list"):
                seq = rec["ITEM_SEQ"]
                merged = {**rec, **detail_by_seq.get(seq, {}), **{"LIST_" + k: v for k, v in rec.items()}}
                merged["_easy"] = easy_by_seq.get(seq, {})
                yield merged

        def product_doc_meta(rec):
            easy = rec.get("_easy", {})
            efcy = easy.get("efcyQesitm") or doc_xml_to_text(rec.get("EE_DOC_DATA"))
            usemethod = easy.get("useMethodQesitm") or doc_xml_to_text(rec.get("UD_DOC_DATA"))
            caution = " ".join(filter(None, [easy.get("atpnWarnQesitm"), easy.get("atpnQesitm")])) \
                or doc_xml_to_text(rec.get("NB_DOC_DATA"))
            intrc = easy.get("intrcQesitm") or ""
            se = easy.get("seQesitm") or ""

            doc = "\n".join(filter(None, [
                f"제품명: {rec.get('ITEM_NAME')}",
                f"효능효과: {efcy}" if efcy else "",
                f"용법용량: {usemethod}" if usemethod else "",
                f"사용상주의사항: {caution}" if caution else "",
                f"상호작용: {intrc}" if intrc else "",
                f"이상반응: {se}" if se else "",
            ]))

            meta = {
                "item_seq": rec.get("ITEM_SEQ"),
                "item_name": rec.get("ITEM_NAME"),
                "entp_name": rec.get("ENTP_NAME"),
                "spclty_pblc": rec.get("SPCLTY_PBLC"),
                "prduct_type": rec.get("PRDUCT_TYPE"),
                "item_ingr_name": rec.get("ITEM_INGR_NAME"),
                "atc_code": rec.get("ATC_CODE"),
                "cancel_name": rec.get("CANCEL_NAME"),
                "edi_code": rec.get("EDI_CODE"),
            }
            return doc, meta

        coll = get_or_create_collection(client, "drug_products", real_embedding=True)
        ingest_records(coll, "drug_products", merged_products(), product_doc_meta, progress, "prod",
                        batch_size=REAL_EMBED_BATCH_SIZE)

    # ---------- 2) health_foods (실제 임베딩) ----------
    if not progress.get("health_foods", {}).get("done"):
        def hf_doc_meta(rec):
            doc = "\n".join(filter(None, [
                f"제품명: {rec.get('PRDUCT')}",
                f"주요기능성: {rec.get('MAIN_FNCTN')}" if rec.get("MAIN_FNCTN") else "",
                f"섭취방법: {rec.get('SRV_USE')}" if rec.get("SRV_USE") else "",
                f"섭취시 주의사항: {rec.get('INTAKE_HINT1')}" if rec.get("INTAKE_HINT1") else "",
                f"성상: {rec.get('SUNGSANG')}" if rec.get("SUNGSANG") else "",
            ]))
            meta = {
                "sttemnt_no": rec.get("STTEMNT_NO"),
                "prduct": rec.get("PRDUCT"),
                "entrps": rec.get("ENTRPS"),
            }
            return doc, meta

        coll = get_or_create_collection(client, "health_foods", real_embedding=True)
        ingest_records(coll, "health_foods", _read_jsonl("htfs_item"), hf_doc_meta, progress, "hf",
                        batch_size=REAL_EMBED_BATCH_SIZE)

    # ---------- 3) drug_ingredients (더미) ----------
    if not progress.get("drug_ingredients", {}).get("done"):
        def ing_doc_meta(rec):
            doc = f"{rec.get('PRDUCT')} - {rec.get('MTRAL_NM')} {rec.get('QNT')}{rec.get('INGD_UNIT_CD') or ''}"
            meta = {
                "item_seq": rec.get("ITEM_SEQ"),
                "mtral_code": rec.get("MTRAL_CODE"),
                "mtral_nm": rec.get("MTRAL_NM"),
                "qnt": rec.get("QNT"),
                "ingd_unit_cd": rec.get("INGD_UNIT_CD"),
            }
            return doc, meta

        coll = get_or_create_collection(client, "drug_ingredients", real_embedding=False)
        ingest_records(coll, "drug_ingredients", _read_jsonl("drug_prdt_mcpn_detail"), ing_doc_meta, progress, "ing")

    # ---------- 4) dur_contraindications (더미) ----------
    if not progress.get("dur_contraindications", {}).get("done"):
        def taboo_doc_meta(rec):
            doc = f"{rec.get('ITEM_NAME')} + {rec.get('MIXTURE_ITEM_NAME')}: {rec.get('PROHBT_CONTENT')}"
            meta = {
                "item_seq": rec.get("ITEM_SEQ"),
                "item_name": rec.get("ITEM_NAME"),
                "ingr_code": rec.get("INGR_CODE"),
                "mixture_item_seq": rec.get("MIXTURE_ITEM_SEQ"),
                "mixture_item_name": rec.get("MIXTURE_ITEM_NAME"),
                "mixture_ingr_code": rec.get("MIXTURE_INGR_CODE"),
                "prohbt_content": rec.get("PROHBT_CONTENT"),
                "notification_date": rec.get("NOTIFICATION_DATE"),
            }
            return doc, meta

        coll = get_or_create_collection(client, "dur_contraindications", real_embedding=False)
        ingest_records(coll, "dur_contraindications", _read_jsonl("dur_usjnt_taboo"), taboo_doc_meta, progress, "taboo")

    # ---------- 5) dur_single_caution (더미, 7개 유형 통합) ----------
    if not progress.get("dur_single_caution", {}).get("done"):
        single_files = [
            "dur_odsn_atent", "dur_cpcty_atent", "dur_mdctn_pd_atent",
            "dur_spcify_agrde_taboo", "dur_pwnm_taboo", "dur_efcy_dplct",
            "dur_seobangjeong_partitn_atent",
        ]

        def all_single_records():
            for fname in single_files:
                yield from _read_jsonl(fname)

        def single_doc_meta(rec):
            doc = f"[{rec.get('TYPE_NAME')}] {rec.get('ITEM_NAME')}: {rec.get('PROHBT_CONTENT') or rec.get('REMARK') or ''}"
            meta = {
                "type_name": rec.get("TYPE_NAME"),
                "item_seq": rec.get("ITEM_SEQ"),
                "item_name": rec.get("ITEM_NAME"),
                "ingr_code": rec.get("INGR_CODE"),
                "ingr_name": rec.get("INGR_NAME"),
                "prohbt_content": rec.get("PROHBT_CONTENT"),
                "remark": rec.get("REMARK"),
            }
            return doc, meta

        coll = get_or_create_collection(client, "dur_single_caution", real_embedding=False)
        ingest_records(coll, "dur_single_caution", all_single_records(), single_doc_meta, progress, "single")

    # ---------- 6) dur_overview (더미) ----------
    if not progress.get("dur_overview", {}).get("done"):
        def overview_doc_meta(rec):
            doc = f"{rec.get('ITEM_NAME')} - DUR유형: {rec.get('TYPE_NAME')}"
            meta = {
                "item_seq": rec.get("ITEM_SEQ"),
                "item_name": rec.get("ITEM_NAME"),
                "entp_name": rec.get("ENTP_NAME"),
                "type_name": rec.get("TYPE_NAME"),
            }
            return doc, meta

        coll = get_or_create_collection(client, "dur_overview", real_embedding=False)
        ingest_records(coll, "dur_overview", _read_jsonl("dur_prdlst"), overview_doc_meta, progress, "ov")

    # ---------- 7) pill_identification (더미) ----------
    if not progress.get("pill_identification", {}).get("done"):
        def pill_doc_meta(rec):
            doc = f"{rec.get('ITEM_NAME')} - {rec.get('DRUG_SHAPE')} {rec.get('COLOR_CLASS1')} " \
                  f"앞:{rec.get('PRINT_FRONT')} 뒤:{rec.get('PRINT_BACK')}"
            meta = {
                "item_seq": rec.get("ITEM_SEQ"),
                "item_name": rec.get("ITEM_NAME"),
                "entp_name": rec.get("ENTP_NAME"),
                "drug_shape": rec.get("DRUG_SHAPE"),
                "color_class1": rec.get("COLOR_CLASS1"),
                "color_class2": rec.get("COLOR_CLASS2"),
                "print_front": rec.get("PRINT_FRONT"),
                "print_back": rec.get("PRINT_BACK"),
                "item_image": rec.get("ITEM_IMAGE"),
            }
            return doc, meta

        coll = get_or_create_collection(client, "pill_identification", real_embedding=False)
        ingest_records(coll, "pill_identification", _read_jsonl("pill_ident"), pill_doc_meta, progress, "pill")

    print("전체 적재 완료.")


if __name__ == "__main__":
    main()
