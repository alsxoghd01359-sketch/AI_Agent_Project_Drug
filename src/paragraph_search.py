"""제품이 특정된 뒤의 생활 질문(음주/임신/졸음 등)에 답하기 위한 문단 단위 하이브리드 검색.

순수 임베딩만 쓰면 실패하는 경우가 있다는 걸 실측으로 확인했다.
- 타이레놀: "술을 마시는 사람이..."처럼 완결된 문장 -> 임베딩이 잘 잡음 (1위, 명확한 격차)
- 용각산: "...진정제, 알코올 등"처럼 단어 나열식 -> 임베딩이 4위로 놓침 (격차도 미미)
게다가 질문은 "술"인데 원문은 "알코올"이라 순수 키워드 매칭도 그대로는 안 통한다.

그래서 동의어 그룹으로 묶은 키워드 매칭을 임베딩과 함께 쓴다:
같은 동의어 그룹에 속한 단어가 질문과 문단에 각각 있으면 그 문단을 최우선으로 올리고,
그 안에서 순위가 더 필요하면 임베딩 유사도로 정렬한다. 동의어 그룹에 안 걸리는 경우엔
임베딩 순위를 그대로 쓴다(완전히 새로운 표현의 질문에도 대응하기 위한 폴백).
"""
import os
import json
from pathlib import Path

import numpy as np
from openai import OpenAI

from text_extract import doc_xml_to_text

DATA_DIR = Path("data")
# 실측 확인: 짧은 한국어 질문("술 마셔도 되나")과 긴 공식 문서 간 유사도를
# text-embedding-3-small로 계산하면, 완전히 무관한 문장("오늘 날씨가 좋네요")이
# 실제 정답 문단보다 유사도가 더 높게 나올 정도로 변별력이 없었다(0.29 vs 0.16).
# text-embedding-3-large는 같은 비교에서 정답(0.40~0.47)이 무관한 문장(0.28)보다
# 뚜렷하게 높게 나와 정상적으로 구분됨을 확인하고 교체함.
EMBED_MODEL = "text-embedding-3-large"

# 자주 나오는 생활질문 주제의 동의어 그룹. 필요에 따라 계속 추가.
TOPIC_SYNONYMS = [
    {"술", "음주", "알코올", "주류", "음주자"},
    {"임신", "임부", "임산부", "수유", "수유부"},
    {"소아", "어린이", "영아", "유아", "아기", "어린아이"},
    {"고령자", "노인", "어르신"},
    {"운전", "기계조작", "졸음", "졸릴", "졸린"},
    {"공복", "식전", "식후", "식사", "빈속"},
    {"두드러기", "발진", "가려움", "알레르기", "과민증"},
    {"신장", "콩팥", "신장애"},
    {"간", "간장애", "간질환"},
]

_client = None
_detail_by_seq = None


def _get_client():
    global _client
    if _client is None:
        _client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    return _client


def _ensure_detail_loaded():
    global _detail_by_seq
    if _detail_by_seq is None:
        _detail_by_seq = {}
        with (DATA_DIR / "drug_prdt_prmsn_detail.jsonl").open(encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                _detail_by_seq[rec["ITEM_SEQ"]] = rec


def _get_paragraphs(item_seq: str, field: str) -> list[str]:
    """field: 'EE'(효능효과) | 'UD'(용법용량) | 'NB'(사용상주의사항)"""
    _ensure_detail_loaded()
    rec = _detail_by_seq.get(item_seq)
    if not rec:
        return []
    text = doc_xml_to_text(rec.get(f"{field}_DOC_DATA"))
    return [p.strip() for p in text.split("\n") if p.strip()]


def _keyword_groups_in(text: str) -> list[set]:
    return [g for g in TOPIC_SYNONYMS if any(term in text for term in g)]


def _embed(texts: list[str]) -> list[np.ndarray]:
    resp = _get_client().embeddings.create(model=EMBED_MODEL, input=texts)
    return [np.array(d.embedding) for d in resp.data]


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def search_paragraphs(item_seq: str, field: str, query: str, top_k: int = 3):
    """하이브리드(동의어 키워드 우선 + 임베딩 보조) 문단 검색.
    반환: [{"text":, "score":, "matched_by": "keyword"|"embedding"}, ...]
    """
    paragraphs = _get_paragraphs(item_seq, field)
    if not paragraphs:
        return []

    query_groups = _keyword_groups_in(query)

    para_vecs = _embed(paragraphs)
    qvec = _embed([query])[0]

    scored = []
    for text, vec in zip(paragraphs, para_vecs):
        sim = _cosine(qvec, vec)
        keyword_hit = any(any(term in text for term in g) for g in query_groups)
        scored.append({"text": text, "score": sim, "matched_by": "keyword" if keyword_hit else "embedding"})

    # 키워드 매칭된 문단을 최우선으로, 그 안에서는 임베딩 유사도로 정렬. 나머지는 임베딩 순위.
    scored.sort(key=lambda r: (r["matched_by"] != "keyword", -r["score"]))

    return scored[:top_k]
