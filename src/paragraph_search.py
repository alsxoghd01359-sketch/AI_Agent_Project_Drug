"""제품이 특정된 뒤의 생활 질문(음주/임신/졸음 등)에 답하기 위한 문단 단위 임베딩 검색.

청킹 전략(전략5, 실측 비교로 채택): PARAGRAPH(문단/하위항목) 단위로 잘게 쪼개되,
각 청크 앞에 그게 속한 상위 번호 항목 제목을 붙여서 맥락을 유지한다.
예: "5) 고령자"만 있던 청크 -> "[9. 고령자에 대한 투여] 5) 고령자".

- ARTICLE 전체를 통째로 묶는 방식(전략4)과 비교: 긴 ARTICLE(예: "4. 이상반응"에
  수십 개 하위 데이터가 섞여 있는 경우) 하나로 뭉치면 임베딩이 희석돼 실제
  질문과 무관한 내용에 묻힌다. 실측으로 운전 관련 질문에서 이 희석 때문에
  전용 조항이 top-8 밖으로 밀려나는 실패를 확인했다. 잘게 쪼개면 이 문제를
  피한다.
- 제목 없이 문단만 쪼개는 이전 방식과 비교: "5) 고령자"처럼 제목이 뭘
  가리키는지 알 수 없는 고립된 조각을 만들지 않는다.
- 답변 근거를 "몇 번째 문단"처럼 구체적으로 인용하기에도 적합하다(청크 하나
  = 근거 문단 하나이므로, 답변을 만들 때 그 청크 원문을 그대로 근거로 쓸 수
  있다).

키워드 동의어 가중치(TOPIC_SYNONYMS)는 폐기했다 — "특정 키워드에 가중치를
주는건 결국 키워드로 검색하는거랑 별반 차이없다"는 판단. text-embedding-3-large로
교체한 뒤 순수 임베딩 변별력만으로 충분함을 실측으로 확인했다(이전 하이브리드
구현은 git 히스토리에 남아있다). 짧은 질문과 긴 공식 문서 간 유사도를
text-embedding-3-small로 계산하면 완전히 무관한 문장이 실제 정답 문단보다
유사도가 더 높게 나올 정도로 변별력이 없었는데(0.29 vs 0.16),
text-embedding-3-large는 정답(0.40~0.47)이 무관한 문장(0.28)보다 뚜렷하게
높게 나와 정상적으로 구분된다.
"""
import os
import json
import re
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
from openai import OpenAI

DATA_DIR = Path("data")
EMBED_MODEL = "text-embedding-3-large"

_TAG_RE = re.compile(r"<[^>]+>")
_NBSP_RE = re.compile(r"&nbsp;")
_WS_RE = re.compile(r"\s+")
_TOP_LEVEL_RE = re.compile(r"^\d+\.\s")  # "1. ", "2. " 등 상위 번호 항목

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


def _clean_noise(text: str) -> str:
    text = _NBSP_RE.sub(" ", text)
    text = _TAG_RE.sub(" ", text)
    text = _WS_RE.sub(" ", text).strip()
    return text


def _iter_articles(xml_str: str | None):
    if not xml_str:
        return
    try:
        root = ET.fromstring(xml_str)
    except ET.ParseError:
        text = _clean_noise(xml_str)
        if text:
            yield ("", [text])
        return
    for article in root.iter("ARTICLE"):
        title = (article.get("title") or "").strip()
        paras = []
        for para in article.findall("PARAGRAPH"):
            raw = "".join(para.itertext()).strip()
            cleaned = _clean_noise(raw)
            if cleaned:
                paras.append(cleaned)
        if title or paras:
            yield (title, paras)


def _chunk_labeled_paragraphs(xml_str: str | None) -> list[str]:
    chunks = []
    for title, paras in _iter_articles(xml_str):
        current_heading = title or ""
        for text in paras:
            if _TOP_LEVEL_RE.match(text):
                current_heading = text
                chunks.append(text)
            elif current_heading:
                chunks.append(f"[{current_heading}] {text}")
            else:
                chunks.append(text)
    return [c for c in chunks if c.strip()]


def _get_paragraphs(item_seq: str, field: str) -> list[str]:
    """field: 'EE'(효능효과) | 'UD'(용법용량) | 'NB'(사용상주의사항)"""
    _ensure_detail_loaded()
    rec = _detail_by_seq.get(item_seq)
    if not rec:
        return []
    return _chunk_labeled_paragraphs(rec.get(f"{field}_DOC_DATA"))


def _embed(texts: list[str]) -> list[np.ndarray]:
    resp = _get_client().embeddings.create(model=EMBED_MODEL, input=texts)
    return [np.array(d.embedding) for d in resp.data]


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


REWRITE_MODEL = "gpt-4o-mini"
_REWRITE_PROMPT = (
    "다음 구어체 질문을 의약품 첨부문서(사용상주의사항) 문체에 맞는 격식체 질문으로 "
    "바꿔라. 질문만 출력하고 다른 말은 하지 마라.\n\n질문: {query}"
)


def _rewrite_formal(query: str) -> str:
    """구어체 질문과 공식 문서 문체 간의 격차 때문에 임베딩 유사도가 떨어지는 문제를
    완화한다. 실측 확인: 일라정 x "콩팥이 안좋은데 먹어도 되나요?" 원본 질문으로는
    정답 문단이 top-20 밖(13위, 그나마도 우연한 키워드 중복)이었는데, 격식체로
    재작성한 질문으로는 진짜 정답 문단이 5위(0.39)까지 올라왔다."""
    resp = _get_client().chat.completions.create(
        model=REWRITE_MODEL,
        temperature=0,
        messages=[{"role": "user", "content": _REWRITE_PROMPT.format(query=query)}],
    )
    return resp.choices[0].message.content.strip()


def search_paragraphs(item_seq: str, field: str, query: str, top_k: int = 8):
    """순수 임베딩 기반 문단(라벨링된 하위항목 단위) 검색.

    원본 질문과 격식체 재작성 질문 두 랭킹을 번갈아 섞어(라운드로빈) 중복 없이
    top_k개를 채운다. 점수로 두 랭킹을 직접 경쟁시키면(예: 둘 중 더 높은 점수
    채택) 한쪽 질문이 전반적으로 더 높은 점수대를 갖는 경우 다른 쪽에서만 상위에
    오른 정답이 묻혀버리는 걸 실측으로 확인했다 — 용각산 x 음주는 원본 질문
    랭킹에서만 정답이 2위(격식체 랭킹에선 top-8 밖으로 밀림), 일라정 x
    "콩팥이 안좋은데 먹어도 되나요"는 격식체 랭킹에서만 정답이 5위(원본
    랭킹에선 안 보임)였다. 순위 교대 방식은 점수 크기와 무관하게 양쪽의
    1위, 2위, ...를 동등하게 반영하므로 이런 비대칭을 보존한다.
    반환: [{"text":, "score":}, ...] (score 내림차순, score는 두 질문 중 더 높은 값)
    """
    paragraphs = _get_paragraphs(item_seq, field)
    if not paragraphs:
        return []

    formal_query = _rewrite_formal(query)
    para_vecs = _embed(paragraphs)
    qvec_orig, qvec_formal = _embed([query, formal_query])

    scores_orig = [_cosine(qvec_orig, v) for v in para_vecs]
    scores_formal = [_cosine(qvec_formal, v) for v in para_vecs]
    rank_orig = sorted(range(len(paragraphs)), key=lambda i: -scores_orig[i])
    rank_formal = sorted(range(len(paragraphs)), key=lambda i: -scores_formal[i])

    seen = set()
    merged = []
    for pos in range(len(paragraphs)):
        for ranking in (rank_orig, rank_formal):
            if len(merged) >= top_k:
                break
            idx = ranking[pos]
            if idx not in seen:
                seen.add(idx)
                merged.append(idx)
        if len(merged) >= top_k:
            break

    result = [
        {"text": paragraphs[i], "score": max(scores_orig[i], scores_formal[i])}
        for i in merged
    ]
    result.sort(key=lambda r: -r["score"])
    return result
