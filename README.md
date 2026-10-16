# AI 복약 정보 도우미 (AI_Agent_Project_Drug)

식약처(MFDS) 공공데이터에 그라운딩해, 병용금기·DUR 주의사항·판매상태·신규허가 같은
**사실 기반 복약 정보**를 정확하게 답하는 한국어 챗봇입니다. 범용 LLM과 달리 학습
시점에 고정되지 않고, 데이터를 주기적으로 재수집해 최신 상태를 반영합니다.

## 주요 기능

- **제품 정보 조회** (`get_drug_info`): 성분, 효능효과, 용법용량, 사용상주의사항
- **병용금기 확인** (`check_drug_interactions`): 여러 약을 함께 먹어도 되는지, DUR
  병용금기·성분 중복·효능군 중복 여부
- **생활질문 답변** (`ask_lifestyle_question`): 운전·음주·임신·수술·동반 질환 등
  구체적인 상황에서의 주의사항 (7개 DUR 유형 외의 질문)
- **증상 기반 성분 추천** (`search_by_symptom`): 증상 키워드로 일반의약품 성분 후보 검색,
  현재 복용 중인 약과의 병용금기 성분 제외
- **판매상태 확인**: 정상 판매 중인지, 취소/취하됐는지와 날짜

## 데이터 출처

data.go.kr(공공데이터포털)의 식약처 공개 API 13개를 전부 수집해 `data/*.jsonl`로 저장합니다.

| 파일 | 내용 |
|---|---|
| `drug_prdt_prmsn_list` | 품목허가 목록(허가일, 판매상태 등) |
| `drug_prdt_prmsn_detail` | 품목허가 상세(성분, 효능효과, 용법용량, 사용상주의사항 원문) |
| `drug_prdt_mcpn_detail` | 복수성분 상세(성분별 분량) |
| `dur_usjnt_taboo` | DUR 병용금기 |
| `dur_odsn_atent` | DUR 노인주의 |
| `dur_prdlst` | DUR 특정연령대금기 |
| `dur_spcify_agrde_taboo` | DUR 특정연령대금기(추가) |
| `dur_cpcty_atent` | DUR 용량주의 |
| `dur_mdctn_pd_atent` | DUR 투여기간주의 |
| `dur_efcy_dplct` | DUR 효능군 중복 |
| `dur_seobangjeong_partitn_atent` | DUR 서방정분할주의 |
| `dur_pwnm_taboo` | DUR 임부금기 |
| `easy_drug` | e약은요(쉬운 설명) |

## 프로젝트 구조

```
src/
  fetch_drug_data.py   데이터 수집 (data.go.kr API -> data/*.jsonl)
  drug_api.py          API 요청 공통 로직
  drug_lookup.py        제품 조회, 성분/DUR 캐시, 퍼지 매칭
  name_match.py         제품명 fuzzy matching, 모호성 판단
  name_kind.py          제품명/성분명/브랜드명 분류
  paragraph_search.py  사용상주의사항 문단 임베딩 검색 + 재랭킹
  symptom_search.py    증상 -> 약효분류 -> 성분 후보 검색
  multi_condition.py   복합 조건 질문 파싱
  emergency_detect.py  응급 상황 감지
  text_extract.py      XML 사용상주의사항 -> 텍스트 변환
  orchestrator.py      LLM 도구 호출 라우팅, 대화 관리 (핵심)
  chroma_setup.py      ChromaDB 임베딩 인덱스 구성
  ingest_chroma.py / ingest_drug_effects.py  인덱싱 스크립트
  streamlit_app.py     Streamlit 채팅 UI
  chat_cli.py          터미널 CLI
```
