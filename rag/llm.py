"""
LLM 연결 + RAG 답변 생성 + Tool Calling

- OpenAI 호환 API 사용 (OPENAI_BASE_URL 로 Gemini, Upstage Solar, Ollama 등 교체 가능)
- LLM_PROVIDER=local 이면 API 키 없이 무료 오픈소스 모델(Qwen2.5)을 직접 실행 (rag/local_llm.py)
- 매 질문마다 검색(RAG) 결과를 컨텍스트로 넣고,
  LLM 이 필요하면 도구를 호출한다:
    search_documents   : 추가 검색
    get_requirements   : 학과·학번별 항목(id, 배점, 유형) 조회
    check_graduation   : 규칙 엔진으로 졸업요건 충족 여부 계산 (점수 계산은 LLM 이 하지 않음)
- 무료 오픈소스 LLM(작은 모델)은 빠르고 정확하도록 별도 경로(answer_stream):
    · 일반 질문: 도구 없이 한 번에 생성, 학과·학번 기준표와 공식 안내 위주의 짧은 컨텍스트
    · 졸업 판정 요청: 문장 자동 인식 결과 + check_graduation 도구 1개만 제공
    · 생성 중인 답변을 바로 화면에 보여주고(스트리밍), 문서에서 찾은 핵심 문장을 함께 표시
- API 키가 없으면 검색 결과만 보여주는 '오프라인 모드'로 동작
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field

from .checker import (check_graduation, format_result_text, get_requirement_items,
                      resolve_department, parse_year, load_requirements)
from .retriever import Retriever, format_context, detect_profile
from .extract import (extract_inputs, is_check_request, wants_judgement, merge_record, new_record, has_any,
                      describe_record, credit_line, RESET_RE)

CHAT_MODEL = os.getenv("CHAT_MODEL", "gpt-4o-mini")
EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-small")

SYSTEM_PROMPT = """너는 순천향대학교 SW융합대학 학생을 돕는 '다면적 졸업인증제 상담 챗봇'이다.

규칙:
1. 반드시 [참고 문서]와 도구 결과에 근거해서만 답한다. 문서에 없는 내용은 추측하지 말고
   "안내 자료에 없는 내용"이라고 말한 뒤, 다면적 졸업인증·비교과는 多Dream비교과센터(향설 1관 RC306호),
   졸업학점·교육과정 등 학사는 학사팀(041-530-1036) 문의를 권한다.
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


def model_label(client) -> str:
    if client is None:
        return "오프라인 모드 (LLM 없음)"
    if getattr(client, "is_local", False):
        return f"무료 오픈소스 LLM: {client.model_name}"
    return f"API: {CHAT_MODEL}"


@dataclass
class Answer:
    text: str
    sources: list = field(default_factory=list)       # [(source, location, score)]
    tool_calls: list = field(default_factory=list)    # [(name, args, result_preview)]
    check_result: dict | None = None
    points: list = field(default_factory=list)        # [(문서에서 찾은 핵심 문장, 출처)]
    auto_text: str | None = None                      # 문장 자동 인식 + 판정 엔진 결과


# ----------------------------------------------------------------------------
# 클라이언트 / 임베딩
# ----------------------------------------------------------------------------
def get_client():
    """LLM_PROVIDER: auto(기본, 키 있으면 API·없으면 오프라인) / api / local(무료 오픈소스 모델) / off"""
    provider = os.getenv("LLM_PROVIDER", "auto").lower()
    if provider in ("off", "offline", "none"):
        return None
    if provider == "local":
        from .local_llm import LocalLLMClient
        return LocalLLMClient()
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


class LocalEmbedder:
    """무료 공개 문장 임베딩 모델 (API 키 불필요, 한국어 지원).
    intfloat/multilingual-e5-small: 문서는 'passage: ', 질문은 'query: ' 를 앞에 붙여 쓰도록 학습된 모델."""

    def __init__(self, name: str | None = None):
        import torch
        from sentence_transformers import SentenceTransformer
        self.name = name or os.getenv("LOCAL_EMBED_MODEL", "intfloat/multilingual-e5-small")
        self.model = SentenceTransformer(self.name, device="cuda" if torch.cuda.is_available() else "cpu")

    def __call__(self, texts: list[str]) -> list[list[float]]:
        return self.model.encode(["passage: " + t for t in texts], normalize_embeddings=True,
                                 batch_size=32, show_progress_bar=False).tolist()

    def query(self, text: str) -> list[float]:
        return self.model.encode(["query: " + text], normalize_embeddings=True, show_progress_bar=False)[0].tolist()


def make_embedder(client=None):
    """검색용 임베딩 함수. USE_EMBEDDINGS=0 이면 끔(TF-IDF만).
    - API 클라이언트: OpenAI 호환 임베딩 API
    - 그 밖(무료 LLM·오프라인): sentence-transformers 가 설치돼 있으면 무료 공개 모델 multilingual-e5-small"""
    if os.getenv("USE_EMBEDDINGS", "1") == "0":
        return None
    if client is not None and not getattr(client, "is_local", False):
        def embed(texts: list[str]) -> list[list[float]]:
            out = []
            for i in range(0, len(texts), 64):
                resp = client.embeddings.create(model=EMBED_MODEL, input=texts[i:i + 64])
                out.extend(d.embedding for d in resp.data)
            return out
        embed.name = EMBED_MODEL
        return embed
    try:
        import sentence_transformers  # noqa: F401
    except ImportError:
        return None
    try:
        return LocalEmbedder()
    except Exception as e:  # 모델을 못 받으면 TF-IDF 만으로 검색
        print(f"[warn] 임베딩 모델을 불러오지 못해 TF-IDF 검색만 사용합니다: {e}")
        return None


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


LOCAL_SYSTEM = """너는 순천향대학교 SW융합대학 학생을 돕는 '졸업요건(다면적 졸업인증제 중심) 상담 챗봇'이다.
답변 방법:
1. 첫 문장에서 질문에 대한 답을 바로 말한다.
2. 이어서 관련 항목을 '- ' 목록으로 정리한다. 각 항목에 점수, 횟수, 조건, 구분(대학공통/학과기준), 인재상을 문서에 적힌 그대로 쓴다.
3. 학생이 다음에 할 일(채우는 방법, 확인할 곳)을 한 줄로 덧붙인다.
4. [참고 문서]에 있는 내용만 사용한다. 문서에 없거나 '확인하지 못한 내용'에 해당하면 "안내 자료에 없는 내용"이라고 말하고, 다면적 졸업인증·비교과는 多Dream비교과센터(향설 1관 RC306호), 졸업학점·교육과정 등 학사는 학사팀(041-530-1036) 문의를 권한다.
5. 숫자는 문서의 값을 그대로 쓰고, 점수를 직접 더하거나 계산하지 않는다.
6. 근거로 쓴 문서는 [문서 n]으로 표시한다.
7. [학생 프로필]의 학과·학번 기준으로 답한다. 반드시 한국어로 답한다."""

LOCAL_CHECK_RULES = """
졸업 판정 질문 규칙:
- 점수와 충족 여부는 [판정 결과]의 값만 그대로 쓴다. 총점/필요점수, 충족 여부, 부족한 항목과 채우는 방법(회당 점수, 남은 횟수)을 정리한다.
- 학생이 말한 과목·활동이 [판정 결과]의 인식 목록에 빠져 있으면 check_graduation 도구를 호출해 다시 계산한다.
  activities 는 {항목id: 값} (check=true, count=횟수, points=점수). 항목 id 는 [항목 id 목록]을 본다."""

APPENDIX_MARK = "\n\n---\n"   # 이 표시 뒤(핵심 근거·출처)는 대화 기록에서 제외


def _is_structured(c) -> bool:
    return "학번" in c.location or c.location.startswith("대학공통")


def _local_context(retrieved, limit: int = 3600):
    """작은 모델용 컨텍스트: 학과·학번 기준표 → 공식 안내 → PDF 원문 순, 길이 제한."""
    rank = lambda cs: 0 if _is_structured(cs[0]) else (1 if cs[0].source.endswith(".md") else 2)
    picked, blocks, total = sorted(retrieved, key=rank), [], 0
    for c, s in picked:
        t = c.text[:1800 if _is_structured(c) else 900]
        if blocks and total + len(t) > limit:
            continue
        blocks.append((c, s, t))
        total += len(t)
    text = "\n\n".join(f"[문서 {i}] 출처: {c.source} / {c.location}\n{t}" for i, (c, s, t) in enumerate(blocks, 1))
    return text, [(c, s) for c, s, _ in blocks]


_STOP = set("알려 려줘 해야 어떻 떻게 뭐가 있어 하면 는데 이야 할까 해? 가능 능해 나요 인가 건가 하나 나? 거야 에서 으로 하고 내가 해줘 줘? 몇번 몇 번".split())


def _grams(s: str) -> set:
    s = re.sub(r"[\s\-\[\]()·/:,.?!※—]+", "", s)
    return {s[i:i + 2] for i in range(len(s) - 1)}


def _lcs(a: str, b: str) -> int:
    """두 문자열의 가장 긴 공통 부분 문자열 길이 (질문의 '학과활동' 같은 연속 일치를 우대)."""
    best, prev = 0, [0] * (len(b) + 1)
    for i in range(1, len(a) + 1):
        cur = [0] * (len(b) + 1)
        for j in range(1, len(b) + 1):
            if a[i - 1] == b[j - 1]:
                cur[j] = prev[j - 1] + 1
                best = max(best, cur[j])
        prev = cur
    return best


_OVERVIEW = ["전체", "총점", "합계", "몇 점", "몇점", "요건", "기준 알려", "뭐 해야", "뭘 해야"]


def key_points(question: str, retrieved, n: int = 5) -> list[tuple[str, str]]:
    """검색된 문서에서 질문과 가장 관련 있는 문장(줄)을 뽑는다 → (문장, 출처). LLM 과 별개로 근거를 보장."""
    q = _grams(question) - _STOP
    qn = re.sub(r"[\s\-\[\]()·/:,.?!※—]+", "", question)
    out, seen = [], set()
    if any(w in question for w in _OVERVIEW):           # 전체 기준 질문 → 기준표 요약 줄 먼저
        for c, s in retrieved:
            if "학번" in c.location and not c.location.startswith("대학공통"):
                for line in c.text.split("\n"):
                    if line.startswith(("합계", "구분:", "인재상별", "이수현황표")):
                        out.append((line, f"{c.source} / {c.location}"))
                        seen.add(line[:40])
                break
    if not q:
        return out[:n]
    cand = []
    for c, s in retrieved:
        for line in c.text.split("\n"):
            line = line.strip().lstrip("-•· ").strip()
            if (len(line) < 7 or line.startswith(("#", "[문서", "[순천향", "출처", "http"))
                    or line.endswith("(SW융합대학)")):
                continue
            hit = len(q & _grams(line))
            if not hit:
                continue
            ln = re.sub(r"[\s\-\[\]()·/:,.?!※—]+", "", line)
            bonus = (0.2 if _is_structured(c) or c.source.endswith(".md") else 0.0) + 0.3 * min(_lcs(qn, ln), 8)
            cand.append(((hit / (len(q) ** 0.5) + bonus) / (1 + len(line) / 400), line, f"{c.source} / {c.location}"))
    cand.sort(key=lambda x: -x[0])
    if cand:
        best, n_pdf = cand[0][0], 0
        for sc, line, src in cand:
            k = line[:40]
            if sc < best * 0.6 or k in seen:
                continue
            if " / p." in src:                  # PDF 원문 표 조각은 최대 1줄
                if n_pdf:
                    continue
                n_pdf += 1
            seen.add(k)
            out.append((line if len(line) <= 260 else line[:257] + "…", src))
            if len(out) >= n:
                break
    return out[:n]


def _id_hint(dept: str, year: int) -> str:
    spec = get_requirement_items(dept, year)
    if "error" in spec:
        return ""
    rows = []
    for it in spec["items"]:
        extra = f", 회당 {it['per']}점" if it["type"] == "count" else ""
        if it["type"] == "points" and it.get("subitems"):
            extra = ", subitems: " + " / ".join(x["name"] for x in it["subitems"])
        rows.append(f"{it['id']} = {it['name']} ({it['type']}{extra})")
    return "\n".join(rows)


def _clean_history(history):
    out = []
    for h in (history or [])[-4:]:
        c = h["content"] if isinstance(h.get("content"), str) else str(h.get("content"))
        out.append({"role": h["role"], "content": c.split(APPENDIX_MARK)[0][:1200]})
    return out


def _strip_special(text: str) -> str:
    for t in ("<|im_end|>", "<|endoftext|>", "<|im_start|>"):
        text = text.split(t)[0]
    return re.sub(r"<\|[^>]*$", "", text)       # 생성 중간의 '<|im' 같은 조각도 숨김


def chat_judgement(question: str, dept: str | None, year: int | None, record: dict | None = None,
                   memory: dict | None = None):
    """채팅 판정. → (판정 설명 텍스트, CheckResult) 또는 (None, None)
    record: 화면이 이미 누적한 기록 (Colab 화면) / memory: 대화별 누적 기록을 담을 dict (Gradio 등)
    둘 다 없으면 이번 문장만으로 판정한다."""
    if not (dept and year):
        return None, None
    spec = get_requirement_items(dept, year)
    if "error" in spec:
        return None, None
    if record is not None:
        rec = record
    else:
        ext = extract_inputs(question, dept, year)
        key = f"{dept}|{year}"
        base = (memory or {}).get(key) or new_record(dept, year)
        if not wants_judgement(question, ext, base, spec):
            return None, None
        rec = merge_record(base, ext, question, spec)
        if memory is not None:
            memory[key] = rec
        if RESET_RE.search(question) and not has_any(rec):
            return "대화로 말씀하신 이수 기록을 모두 지웠어요. 들은 과목과 활동을 다시 말씀해 주세요.", None
    if not has_any(rec):
        return None, None
    res = check_graduation(dept, year, rec["courses"], rec["activities"])
    if isinstance(res, dict):
        return None, None
    text = "지금까지 말씀하신 내용(대화 누적): " + ", ".join(describe_record(rec, spec)) + "\n" + format_result_text(res)
    cl = credit_line(rec.get("credits"), load_requirements())
    if cl:
        text += "\n" + cl
    return text, res


def answer_stream(question: str, retriever: Retriever, client=None, history: list[dict] | None = None,
                  profile: dict | None = None, check_text: str | None = None, k: int = 5,
                  record: dict | None = None, memory: dict | None = None):
    """화면용 스트리밍 답변. 진행 중에는 문자열(지금까지의 답변)을, 마지막에는 Answer 를 내보낸다.
    무료 오픈소스 LLM 이 아니면 answer() 결과를 한 번에 내보낸다."""
    if client is None or not getattr(client, "is_local", False) or not hasattr(client, "stream"):
        yield answer(question, retriever, client, history, profile, check_text, k, record=record, memory=memory)
        return

    profile = profile or {}
    dept = resolve_department(profile.get("department", "")) if profile.get("department") else None
    year = parse_year(profile["year"]) if profile.get("year") else None
    d2, y2 = detect_profile(question)
    if d2 and d2 != dept:
        dept, year = d2, y2 or year
    elif y2:
        year = y2

    retrieved = retriever.search(question, k=k, department=dept, year=year)
    context, used = _local_context(retrieved)
    state = {"department": dept, "year": year, "sources": list(used), "check_result": None}
    trace = []

    auto, res = chat_judgement(question, dept, year, record, memory)
    is_check = auto is not None
    if is_check:
        state["check_result"] = res.to_dict() if res else None
        trace.append(("문장 자동 인식", {}, auto[:600]))

    system = LOCAL_SYSTEM
    if dept and year:
        system += f"\n\n[학생 프로필] 학과: {dept}, 학번: {year}"
    if check_text and not is_check and any(w in question for w in ["부족", "충족", "졸업", "가능", "남은", "내 점수"]):
        system += f"\n\n[학생이 자가진단 탭에서 입력한 최근 판정 결과]\n{check_text[:1500]}"
    tools = None
    if is_check and auto:
        system += LOCAL_CHECK_RULES + f"\n\n[항목 id 목록]\n{_id_hint(dept, year)}\n\n[판정 결과 (판정 엔진 계산값)]\n{auto}"
        tools = [t for t in TOOLS if t["function"]["name"] == "check_graduation"]

    messages = [{"role": "system", "content": system}] + _clean_history(history)
    messages.append({"role": "user", "content": f"[참고 문서]\n{context}\n\n[질문]\n{question}"})

    yield "⏳ 답변을 만드는 중이에요…"
    for rnd in range(2):
        full = ""
        for piece in client.stream(messages, tools=tools, max_new_tokens=600 if is_check else 512):
            full += piece
            shown = _strip_special(full)
            if not shown.lstrip().startswith("<"):        # 도구 호출 출력은 화면에 보이지 않음
                yield shown
        msg = parse_local(full, tools)
        if not msg.tool_calls:
            break
        messages.append({"role": "assistant", "content": msg.content or "",
                         "tool_calls": [tc.model_dump() for tc in msg.tool_calls]})
        for tc in msg.tool_calls:
            try:
                args = json.loads(tc.function.arguments or "{}")
                args.setdefault("department", dept)
                args.setdefault("admission_year", year)
                out = run_tool(tc.function.name, args, retriever, state)
            except Exception as e:
                args, out = {}, f"도구 실행 오류: {e}"
            trace.append((tc.function.name, args, out[:600]))
            messages.append({"role": "tool", "tool_call_id": tc.id, "content": out})
        tools = None                                      # 두 번째 생성은 설명만
        yield "⏳ 판정 엔진으로 다시 계산했어요. 결과를 정리하는 중…"

    text = (msg.content or "").strip() or "(답변을 만들지 못했어요. 아래 핵심 근거를 확인해 주세요.)"
    mine = [cs for cs in retrieved if not dept or not cs[0].department or cs[0].department == dept]
    yield Answer(text, _dedup_sources(state["sources"]), trace, state["check_result"],
                 points=key_points(question, mine), auto_text=auto)


def parse_local(text: str, tools):
    from .local_llm import parse_output
    msg = parse_output(text)
    if not tools:
        msg.tool_calls = None
    return msg


def answer(question: str, retriever: Retriever, client=None, history: list[dict] | None = None,
           profile: dict | None = None, check_text: str | None = None, k: int = 5,
           max_tool_rounds: int = 5, record: dict | None = None, memory: dict | None = None) -> Answer:
    if client is not None and getattr(client, "is_local", False) and hasattr(client, "stream"):
        final = None
        for ev in answer_stream(question, retriever, client, history, profile, check_text, k, record, memory):
            final = ev
        return final

    profile = profile or {}
    dept = resolve_department(profile.get("department", "")) if profile.get("department") else None
    year = parse_year(profile["year"]) if profile.get("year") else None

    retrieved = retriever.search(question, k=k, department=dept, year=year)
    context = format_context(retrieved)
    state = {"department": dept, "year": year, "sources": list(retrieved), "check_result": None}

    # 졸업 판정 요청이면 문장에서 과목·활동을 규칙으로 먼저 인식해 판정 (작은 모델·오프라인 보완)
    auto = None
    d2, y2 = detect_profile(question)
    if d2 and d2 != dept:
        dept, year = d2, y2 or year
    elif y2:
        year = y2
    auto, res = chat_judgement(question, dept, year, record, memory)
    if auto:
        state["check_result"] = res.to_dict() if res else None

    if client is None:
        return _offline_answer(question, retrieved, check_text, auto, state["check_result"])

    sys_block = _profile_block(profile, check_text)
    if auto:
        sys_block += ("\n\n[학생 문장에서 자동 인식한 입력과 판정 결과 (판정 엔진 계산값)]\n" + auto +
                      "\n→ 이 계산값을 사용해 답하라. 학생이 말했는데 인식되지 않은 항목이 있으면 check_graduation 도구로 다시 계산하라.")
    messages = [{"role": "system", "content": SYSTEM_PROMPT + sys_block}]
    for h in (history or [])[-8:]:
        messages.append({"role": h["role"], "content": h["content"]})
    messages.append({"role": "user", "content": f"[참고 문서]\n{context}\n\n[질문]\n{question}"})

    trace = []
    for _ in range(max_tool_rounds):
        resp = client.chat.completions.create(model=CHAT_MODEL, messages=messages, tools=TOOLS,
                                              temperature=0.1)
        msg = resp.choices[0].message
        if not msg.tool_calls:
            if auto:
                trace.insert(0, ("문장 자동 인식", {}, auto[:600]))
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


def _offline_answer(question, retrieved, check_text, auto=None, check_result=None) -> Answer:
    parts = ["🔌 **오프라인 모드** (LLM 없음) — 판정 엔진 계산과 검색된 근거로 답합니다.\n"]
    if auto:
        parts.append("**말씀하신 내용으로 계산한 졸업인증 판정**\n```\n" + auto + "\n```\n"
                     "빠진 항목이 있으면 '졸업 자가진단' 탭에서 직접 입력해 주세요.")
    elif check_text and any(w in question for w in ["부족", "충족", "졸업", "가능", "남은"]):
        parts.append("**최근 자가진단 결과**\n```\n" + check_text + "\n```")
    for i, (c, s) in enumerate(retrieved[:3], 1):
        parts.append(f"**[문서 {i}] {c.source} / {c.location}** (유사도 {s:.2f})\n```\n{c.text[:900]}\n```")
    return Answer("\n\n".join(parts), _dedup_sources(retrieved), check_result=check_result)
