"""
LLM 연결 + RAG 답변 생성 + Tool Calling

- OpenAI 호환 API 사용 (OPENAI_BASE_URL 로 Upstage Solar, Ollama, vLLM 등 교체 가능)
- 매 질문마다 검색(RAG) 결과를 컨텍스트로 넣고,
  LLM 이 필요하면 도구를 호출한다:
    search_documents   : 추가 검색
    get_requirements   : 학과·학번별 항목(id, 배점, 유형) 조회
    check_graduation   : 규칙 엔진으로 졸업요건 충족 여부 계산 (점수 계산은 LLM 이 하지 않음)
- API 키가 없으면 검색 결과만 보여주는 '오프라인 모드'로 동작
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field

from .checker import (check_graduation, format_result_text, get_requirement_items,
                      resolve_department, parse_year, load_requirements)
from .retriever import Retriever, format_context

CHAT_MODEL = os.getenv("CHAT_MODEL", "gpt-4o-mini")
EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-small")

SYSTEM_PROMPT = """너는 순천향대학교 SW융합대학 학생을 돕는 '다면적 졸업인증제 상담 챗봇'이다.

규칙:
1. 반드시 [참고 문서]와 도구 결과에 근거해서만 답한다. 문서에 없는 내용은 추측하지 말고
   "안내문에 없는 내용"이라고 말한 뒤 多Dream비교과센터(향설 1관 RC306호) 문의를 권한다.
2. 답변에 사용한 근거는 [문서 n] 형태로 표시한다.
3. 학과·학번에 따라 기준이 다르므로, 학과/학번을 모르면 먼저 물어본다.
   (학생 프로필이 주어졌다면 그 학과·학번 기준으로 답한다.)
4. 졸업 충족 여부나 점수 계산은 절대 직접 계산하지 말고 check_graduation 도구를 사용한다.
   - 학생이 자신이 들은 과목/참여한 활동을 말하면: get_requirements 로 항목 id 를 확인한 뒤
     activities 를 채워 check_graduation 을 호출한다.
   - 과목명은 courses 리스트에 그대로 넣는다(과목 자동 인식).
   - 학생이 언급하지 않은 항목은 미이수로 두고, 결과에서 '부족한 항목'을 안내한다.
   - 자기주도상담·학과활동·전공상담 등 횟수형(count)은 참여 횟수를 넣는다 (회당 점수는 도구가 계산).
   - 세부 점수가 있는 항목(points)은 get_requirements 의 subitems 이름 그대로 {"subitems": {"SQL개발자(SQLD) 취득": true}} 처럼 넣는다.
5. 결과를 설명할 때: 총점/필요점수, 충족 여부, 부족한 항목과 채우는 방법(회당 점수 등)을 간결하게 정리한다.
6. 공식 확인은 종합정보시스템 → 학부학사행정 → 졸업 → 개인별졸업진도관리 → 비교과일람표 에서 하도록 안내한다.
7. 한국어로, 친근하지만 정확하게 답한다."""

TOOLS = [
    {"type": "function", "function": {
        "name": "search_documents",
        "description": "졸업인증제 안내 문서에서 관련 내용을 검색한다.",
        "parameters": {"type": "object", "properties": {
            "query": {"type": "string", "description": "검색어"}}, "required": ["query"]}}},
    {"type": "function", "function": {
        "name": "get_requirements",
        "description": "학과·학번에 적용되는 졸업인증 항목 목록(id, 이름, 유형, 배점)을 조회한다. check_graduation 호출 전에 항목 id 확인용.",
        "parameters": {"type": "object", "properties": {
            "department": {"type": "string", "description": "학과명 (예: AI·빅데이터학과, 컴소)"},
            "admission_year": {"type": "integer", "description": "입학년도 4자리 (예: 2022)"}},
            "required": ["department", "admission_year"]}}},
    {"type": "function", "function": {
        "name": "check_graduation",
        "description": ("학생의 수강 과목과 비교과 활동으로 다면적 졸업인증 충족 여부를 계산한다. "
                        "activities 는 {항목id: 값}. 값 형식: course/check 항목=true, count 항목=참여횟수(정수), "
                        "points 항목=인정점수(숫자) 또는 {\"subitems\": {세부항목명: 값}}, choose 항목=[과목명], "
                        "all_of 항목=[이수한 부분명], topcit_rule 항목={attempts, level}."),
        "parameters": {"type": "object", "properties": {
            "department": {"type": "string"},
            "admission_year": {"type": "integer"},
            "courses": {"type": "array", "items": {"type": "string"}, "description": "수강(이수)한 과목명 리스트"},
            "activities": {"type": "object", "description": "{항목id: 값}"}},
            "required": ["department", "admission_year"]}}},
]


@dataclass
class Answer:
    text: str
    sources: list = field(default_factory=list)       # [(source, location, score)]
    tool_calls: list = field(default_factory=list)    # [(name, args, result_preview)]
    check_result: dict | None = None


# ----------------------------------------------------------------------------
# 클라이언트 / 임베딩
# ----------------------------------------------------------------------------
def get_client():
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        return None
    try:
        from openai import OpenAI
    except ImportError:
        print("[warn] openai 패키지 미설치 → 오프라인 모드")
        return None
    base = os.getenv("OPENAI_BASE_URL") or None
    return OpenAI(api_key=key, base_url=base)


def make_embedder(client):
    """client 가 있으면 임베딩 함수 반환. USE_EMBEDDINGS=0 이면 끔."""
    if client is None or os.getenv("USE_EMBEDDINGS", "1") == "0":
        return None

    def embed(texts: list[str]) -> list[list[float]]:
        out = []
        for i in range(0, len(texts), 64):
            resp = client.embeddings.create(model=EMBED_MODEL, input=texts[i:i + 64])
            out.extend(d.embedding for d in resp.data)
        return out
    return embed


# ----------------------------------------------------------------------------
# 도구 실행
# ----------------------------------------------------------------------------
def _compact_requirements(dept: str, year: int) -> dict:
    d = resolve_department(dept)
    if not d:
        return {"error": f"학과 '{dept}'를 찾지 못함. 가능: {', '.join(load_requirements()['departments'])}"}
    spec = get_requirement_items(d, year)
    if "error" in spec:
        return spec
    items = []
    for it in spec["items"]:
        x = {"id": it["id"], "name": it["name"], "section": it["section"], "category": it["category"],
             "type": it["type"], "points": it["points"]}
        if it["type"] == "count":
            x["per"] = it["per"]
        if it["type"] == "choose":
            x["options"] = it.get("options") or f"'{it['pattern']}' 포함 과목"
            x["n"] = it["n"]
        if it["type"] == "all_of":
            x["parts"] = it["parts"]
        if it["type"] == "points" and it.get("subitems"):
            x["subitems"] = [{"name": s["name"], "mode": s["mode"],
                              **({"points": s["points"]} if "points" in s else {}),
                              **({"per": s["per"]} if "per" in s else {}),
                              **({"options": s["options"]} if "options" in s else {})} for s in it["subitems"]]
        items.append(x)
    return {"department": d, "year": year, "total": spec["total"], "groups": spec["groups"],
            "sections": spec["sections"], "items": items,
            "unit_rules": spec["unit_rules"], "notes": spec["notes"]}


def run_tool(name: str, args: dict, retriever: Retriever, state: dict) -> str:
    if name == "search_documents":
        res = retriever.search(args.get("query", ""), k=4,
                               department=state.get("department"), year=state.get("year"))
        state["sources"].extend(res)
        return format_context(res)
    if name == "get_requirements":
        return json.dumps(_compact_requirements(args["department"], int(args["admission_year"])), ensure_ascii=False)
    if name == "check_graduation":
        res = check_graduation(args["department"], args["admission_year"],
                               args.get("courses") or [], args.get("activities") or {})
        if isinstance(res, dict):
            return json.dumps(res, ensure_ascii=False)
        state["check_result"] = res.to_dict()
        return format_result_text(res)
    return json.dumps({"error": f"unknown tool {name}"})


# ----------------------------------------------------------------------------
# 답변 생성
# ----------------------------------------------------------------------------
def _profile_block(profile: dict | None, check_text: str | None) -> str:
    s = ""
    if profile and profile.get("department") and profile.get("year"):
        s += f"\n\n[학생 프로필] 학과: {profile['department']}, 학번: {profile['year']}"
    if check_text:
        s += f"\n\n[학생이 자가진단 탭에서 입력한 최근 판정 결과]\n{check_text}"
    return s


def answer(question: str, retriever: Retriever, client=None, history: list[dict] | None = None,
           profile: dict | None = None, check_text: str | None = None, k: int = 5,
           max_tool_rounds: int = 5) -> Answer:
    profile = profile or {}
    dept = resolve_department(profile.get("department", "")) if profile.get("department") else None
    year = parse_year(profile["year"]) if profile.get("year") else None

    retrieved = retriever.search(question, k=k, department=dept, year=year)
    context = format_context(retrieved)
    state = {"department": dept, "year": year, "sources": list(retrieved), "check_result": None}

    if client is None:
        return _offline_answer(question, retrieved, check_text)

    messages = [{"role": "system", "content": SYSTEM_PROMPT + _profile_block(profile, check_text)}]
    for h in (history or [])[-8:]:
        messages.append({"role": h["role"], "content": h["content"]})
    messages.append({"role": "user", "content": f"[참고 문서]\n{context}\n\n[질문]\n{question}"})

    trace = []
    for _ in range(max_tool_rounds):
        resp = client.chat.completions.create(model=CHAT_MODEL, messages=messages, tools=TOOLS,
                                              temperature=0.1)
        msg = resp.choices[0].message
        if not msg.tool_calls:
            return Answer(msg.content or "", _dedup_sources(state["sources"]), trace, state["check_result"])
        messages.append({"role": "assistant", "content": msg.content or "",
                         "tool_calls": [tc.model_dump() for tc in msg.tool_calls]})
        for tc in msg.tool_calls:
            try:
                args = json.loads(tc.function.arguments or "{}")
                out = run_tool(tc.function.name, args, retriever, state)
            except Exception as e:  # 도구 오류도 LLM 에 돌려줘서 스스로 수정하게
                args, out = {}, f"도구 실행 오류: {e}"
            trace.append((tc.function.name, args, out[:600]))
            messages.append({"role": "tool", "tool_call_id": tc.id, "content": out})

    resp = client.chat.completions.create(model=CHAT_MODEL, messages=messages, temperature=0.1)
    return Answer(resp.choices[0].message.content or "", _dedup_sources(state["sources"]), trace,
                  state["check_result"])


def _dedup_sources(results):
    seen, out = set(), []
    for c, s in results:
        key = (c.source, c.location)
        if key not in seen:
            seen.add(key)
            out.append((c.source, c.location, round(s, 3)))
    return out


def _offline_answer(question, retrieved, check_text) -> Answer:
    parts = ["🔌 **오프라인 모드** (OPENAI_API_KEY 없음) — LLM 생성 없이 검색된 근거만 보여줍니다.\n"]
    if check_text and any(w in question for w in ["부족", "충족", "졸업", "가능", "남은"]):
        parts.append("**최근 자가진단 결과**\n```\n" + check_text + "\n```")
    for i, (c, s) in enumerate(retrieved[:3], 1):
        parts.append(f"**[문서 {i}] {c.source} / {c.location}** (유사도 {s:.2f})\n```\n{c.text[:900]}\n```")
    return Answer("\n\n".join(parts), _dedup_sources(retrieved))
