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

import chroma_setup

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


def _chunk_with_parents(xml_str: str | None) -> tuple[list[str], list[str]]:
    """검색 단위(세분화된 하위 문항, 전략5)와 그 검색 단위가 속한 상위 ARTICLE
    전체 텍스트를 나란히(같은 인덱스로) 반환한다.

    실측 확인: 쪼갤수록 임베딩 유사도가 오른다(예: 용각산 "술 마셔도 되나" 기준
    하위항목 단위 0.2764 -> 쉼표로 더 쪼갠 단위 0.3352). 그래서 검색(임베딩·
    재랭킹)은 세분화된 단위로 정밀하게 하되, 재랭킹까지 끝난 뒤 LLM에게 최종
    근거로 보여줄 때는 그 하위항목이 속한 ARTICLE 전체를 준다 — "5) 고령자"
    처럼 한 줄만 보여주면 맥락이 잘리는 문제를, 검색 정밀도를 희생하지 않고
    해결한다(Parent Document Retrieval 패턴: 검색은 작은 단위, 생성은 큰 단위).
    """
    chunks: list[str] = []
    parents: list[str] = []
    for title, paras in _iter_articles(xml_str):
        if not paras:
            continue
        full_article = " ".join(filter(None, [title] + paras))
        current_heading = title or ""
        for text in paras:
            if _TOP_LEVEL_RE.match(text):
                current_heading = text
                chunks.append(text)
            elif current_heading:
                chunks.append(f"[{current_heading}] {text}")
            else:
                chunks.append(text)
            parents.append(full_article)
    return chunks, parents


def _chunk_labeled_paragraphs(xml_str: str | None) -> list[str]:
    chunks, _ = _chunk_with_parents(xml_str)
    return chunks


def _get_paragraphs(item_seq: str, field: str) -> list[str]:
    """field: 'EE'(효능효과) | 'UD'(용법용량) | 'NB'(사용상주의사항)"""
    _ensure_detail_loaded()
    rec = _detail_by_seq.get(item_seq)
    if not rec:
        return []
    return _chunk_labeled_paragraphs(rec.get(f"{field}_DOC_DATA"))


def _get_paragraphs_with_parents(item_seq: str, field: str) -> tuple[list[str], list[str]]:
    _ensure_detail_loaded()
    rec = _detail_by_seq.get(item_seq)
    if not rec:
        return [], []
    return _chunk_with_parents(rec.get(f"{field}_DOC_DATA"))


def _embed(texts: list[str]) -> list[np.ndarray]:
    resp = _get_client().embeddings.create(model=EMBED_MODEL, input=texts)
    return [np.array(d.embedding) for d in resp.data]


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


_CHROMA_COLLECTION = None


def _get_chroma_collection():
    global _CHROMA_COLLECTION
    if _CHROMA_COLLECTION is None:
        client = chroma_setup.get_client()
        _CHROMA_COLLECTION = chroma_setup.get_or_create_collection(
            client, "lifestyle_paragraphs", real_embedding=True
        )
    return _CHROMA_COLLECTION


def _embed_paragraphs_cached(item_seq: str, field: str, paragraphs: list[str]) -> list[np.ndarray]:
    """같은 제품·같은 필드의 문단 임베딩을 ChromaDB에 캐싱한다.

    기존에는 같은 제품에 대한 질문이 들어올 때마다 문서 전체를 매번 다시
    임베딩했다 — 실측으로 질문 1건 비용의 약 60%가 이 "문서 전체 재임베딩"
    이었다(예: 169개 청크 문서에서 23,487 토큰). 같은 제품은 자주 재질문되므로
    한 번 계산한 임베딩을 영구 저장해두고 재사용한다.

    청킹 로직이 바뀌어 문단 수나 내용이 달라지면(예: 전략 변경, 원본 XML
    갱신) 캐시된 문서 텍스트가 현재 파싱 결과와 달라지므로, 그 경우엔 캐시를
    무시하고 다시 계산해서 덮어쓴다 — 오래된 청킹 결과가 조용히 재사용되는
    걸 막기 위함.
    """
    if not paragraphs:
        return []

    collection = _get_chroma_collection()
    ids = [f"{item_seq}:{field}:{i}" for i in range(len(paragraphs))]

    cached = collection.get(ids=ids, include=["embeddings", "documents"])
    returned_ids = cached.get("ids") or []
    cached_embeddings = cached.get("embeddings")
    id_to_doc = dict(zip(returned_ids, cached.get("documents") or []))
    id_to_emb = dict(zip(returned_ids, cached_embeddings)) if cached_embeddings is not None else {}

    if len(returned_ids) == len(ids) and all(id_to_doc.get(i) == p for i, p in zip(ids, paragraphs)):
        return [np.array(id_to_emb[i]) for i in ids]

    vecs = _embed(paragraphs)
    collection.upsert(
        ids=ids,
        embeddings=[v.tolist() for v in vecs],
        documents=paragraphs,
        metadatas=[{"item_seq": item_seq, "field": field} for _ in paragraphs],
    )
    return vecs


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


POOL_SIZE = 40
RERANK_MODEL = "gpt-4o-mini"
_RERANK_PROMPT = """사용자 질문: "{query}"

아래는 의약품 사용상주의사항에서 임베딩 유사도로 1차 검색된 후보 문단들입니다.
각 후보가 이 질문에 실제로 답이 되는 정도에 따라, 관련도가 높은 순서대로
번호만 쉼표로 구분해서 나열하세요(예: "3,7,1"). 질문과 실제로 무관한 후보는
목록에서 빼세요. 관련 있는 후보가 하나도 없으면 "없음"이라고만 답하세요.

[후보]
{candidates}
"""


def _rerank(query: str, pool: list[dict], top_k: int) -> list[dict]:
    """임베딩 1차 검색(POOL_SIZE개)은 재현율(정답을 후보 풀에 넣는 것)을 위한
    단계이고, 여기서 실제 관련도 판단과 정렬을 맡는다. 코사인 유사도는 거대
    문서(100개+ 청크)에서 비슷한 문구가 반복되면 희석되어 순위가 많이 밀리는
    걸 실측으로 확인했다 — 심바로드정(134개 청크) x "간이 안좋은데 먹어도
    되나요?"는 정답 조항이 원본 질문 29위, 격식체 재작성도 11위였다. 임베딩
    없이 LLM이 직접 "이 문단이 이 질문에 답이 되는가"를 판단하면 반복되는
    유사 문구에 흔들리지 않는다."""
    cand_text = "\n".join(f"{i + 1}. {c['text']}" for i, c in enumerate(pool))
    resp = _get_client().chat.completions.create(
        model=RERANK_MODEL,
        temperature=0,
        messages=[{"role": "user", "content": _RERANK_PROMPT.format(query=query, candidates=cand_text)}],
    )
    order = [int(n) for n in re.findall(r"\d+", resp.choices[0].message.content)]

    reranked = []
    seen = set()
    for n in order:
        if 1 <= n <= len(pool) and n not in seen:
            seen.add(n)
            reranked.append(pool[n - 1])
    return reranked[:top_k]


def search_paragraphs(item_seq: str, field: str, query: str, top_k: int = 8):
    """임베딩 1차 검색(재현율) + LLM 재랭킹(정확한 순서) 2단계 문단 검색.

    1단계: 원본 질문과 격식체 재작성 질문 두 랭킹을 번갈아 섞어(라운드로빈)
    중복 없이 POOL_SIZE개를 채운다. 점수로 두 랭킹을 직접 경쟁시키면(예: 둘
    중 더 높은 점수 채택) 한쪽 질문이 전반적으로 더 높은 점수대를 갖는 경우
    다른 쪽에서만 상위인 정답이 묻혀버리는 걸 실측으로 확인했다 — 용각산 x
    음주는 원본 질문 랭킹에서만 정답이 2위(격식체 랭킹에선 top-8 밖), 일라정
    x "콩팥이 안좋은데 먹어도 되나요"는 격식체 랭킹에서만 정답이 5위(원본
    랭킹에선 안 보임)였다. 순위 교대 방식은 점수 크기와 무관하게 양쪽의
    1위, 2위, ...를 동등하게 반영하므로 이런 비대칭을 보존한다.

    2단계: POOL_SIZE개 후보를 LLM에게 보여주고 실제 관련도 순으로 재정렬시켜
    top_k개만 추린다(_rerank 참고).

    반환: [{"text":, "score":}, ...] (score는 임베딩 1차 점수 참고용, 순서는
    재랭킹 결과를 따른다)
    """
    paragraphs, parents = _get_paragraphs_with_parents(item_seq, field)
    if not paragraphs:
        return []

    formal_query = _rewrite_formal(query)
    para_vecs = _embed_paragraphs_cached(item_seq, field, paragraphs)
    qvec_orig, qvec_formal = _embed([query, formal_query])

    scores_orig = [_cosine(qvec_orig, v) for v in para_vecs]
    scores_formal = [_cosine(qvec_formal, v) for v in para_vecs]
    rank_orig = sorted(range(len(paragraphs)), key=lambda i: -scores_orig[i])
    rank_formal = sorted(range(len(paragraphs)), key=lambda i: -scores_formal[i])

    pool_size = min(POOL_SIZE, len(paragraphs))
    seen = set()
    merged = []
    for pos in range(len(paragraphs)):
        for ranking in (rank_orig, rank_formal):
            if len(merged) >= pool_size:
                break
            idx = ranking[pos]
            if idx not in seen:
                seen.add(idx)
                merged.append(idx)
        if len(merged) >= pool_size:
            break

    pool = [
        {"text": paragraphs[i], "parent": parents[i], "score": max(scores_orig[i], scores_formal[i])}
        for i in merged
    ]
    pool.sort(key=lambda r: -r["score"])

    reranked = _rerank(query, pool, top_k)

    # 재랭킹까지는 세분화된 하위항목 텍스트로 정밀하게 판단하되(검색 단위),
    # 최종적으로 LLM에게 보여줄 때는 그 하위항목이 속한 ARTICLE 전체로
    # 확장한다(생성 단위) — "5) 고령자"처럼 한 줄만 보여줘서 맥락이 잘리는
    # 문제를 검색 정밀도 희생 없이 해결한다. 여러 하위항목이 같은 ARTICLE에
    # 속하면 중복 표시하지 않고 가장 순위 높은 것만 남긴다.
    result = []
    seen_parents = set()
    for r in reranked:
        if r["parent"] in seen_parents:
            continue
        seen_parents.add(r["parent"])
        result.append({"text": r["parent"], "score": r["score"]})
    return result
