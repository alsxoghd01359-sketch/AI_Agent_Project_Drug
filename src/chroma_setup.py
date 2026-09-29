"""ChromaDB 클라이언트/임베딩 함수 설정.

- 텍스트가 실제로 의미를 담고 있는 컬렉션(drug_products, health_foods)만 OpenAI 임베딩 사용.
- 나머지(구조화 조회용) 컬렉션은 벡터 유사도 검색을 쓰지 않으므로 더미 임베딩으로
  OpenAI 호출 비용/시간을 아낀다. 조회는 metadata where 필터로만 한다.
"""
import os

import chromadb
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
from dotenv import load_dotenv

load_dotenv()

PERSIST_DIR = "./chroma_db"
OPENAI_EMBED_MODEL = "text-embedding-3-small"


class DummyEmbeddingFunction:
    """벡터 유사도 검색을 쓰지 않는 구조화 컬렉션용. 항상 같은 1차원 벡터를 반환."""

    def __call__(self, input):
        return [[0.0] for _ in input]

    def name(self):
        return "dummy"


def get_client():
    return chromadb.PersistentClient(path=PERSIST_DIR)


def get_openai_embedding_function():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(".env 파일에 OPENAI_API_KEY를 설정해주세요.")
    return OpenAIEmbeddingFunction(api_key=api_key, model_name=OPENAI_EMBED_MODEL)


def get_or_create_collection(client, name: str, real_embedding: bool):
    ef = get_openai_embedding_function() if real_embedding else DummyEmbeddingFunction()
    return client.get_or_create_collection(name=name, embedding_function=ef)
