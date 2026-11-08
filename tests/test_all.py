"""python -m pytest tests -q   (또는 python tests/test_all.py)"""
import json
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rag.checker import (load_requirements, get_requirement_items, check_graduation, parse_course_list,
                         resolve_department, parse_year, DEPT_KINDS)
from rag.retriever import build_index
from rag import llm

REQ = load_requirements()


# ---------------------------------------------------------------- 데이터 무결성
def test_totals_match_pdf():
    """모든 학과·학번의 항목 합이 PDF 표의 소계/합계와 같아야 한다."""
    for dept, info in REQ["departments"].items():
        for t in info["tracks"]:
            for y in range(t["years"][0], t["years"][1] + 1):
                items = get_requirement_items(dept, y, REQ)["items"]
                assert sum(i["points"] for i in items) == t["total"], (dept, y)
                # PDF 표의 '대학공통' = 시스템 기준 대학공통 + 졸업필수활동(학과기준으로 집계)
                uni = sum(i["points"] for i in items if i["group"] == "대학공통")
                grad = sum(i["points"] for i in items if i["kind"] == "졸업필수활동")
                assert uni + grad == t["subtotals"]["대학공통"] and grad in (13, 15), (dept, y)
                for k in [k for k in DEPT_KINDS if k != "졸업필수활동"]:
                    got = sum(i["points"] for i in items if i["group"] == "학과기준" and i["kind"] == k)
                    assert got == t["subtotals"].get(k, 0), (dept, y, k)


def test_alias_and_year():
    assert resolve_department("ai빅데이터") == "AI·빅데이터학과"
    assert resolve_department("컴소") == "컴퓨터소프트웨어공학과"
    assert resolve_department("IoT") == "사물인터넷학과"
    assert parse_year("22학번") == 2022 and parse_year(2024) == 2024


def test_course_parsing():
    got = parse_course_list("1. 인간사랑 A+\n현대기본영어(2학점) B0, 캡스톤디자인Ⅱ\n자기개발과 전공탐색(M&M)")
    assert got == ["인간사랑", "현대기본영어", "캡스톤디자인2", "자기개발과전공탐색"]


# ---------------------------------------------------------------- AI·빅데이터 22학번 시나리오
FULL_AIBD_2022 = dict(
    courses="인간사랑\n현대기본영어\n봉사학습\n자기개발과 전공탐색\n예비창업설계",
    activities={
        "c_asp": ["1학기", "2학기"], "c_selfcounsel": 5, "c_gradact": 15,
        "dept_act": 5, "cert_contest": {"subitems": {"SQL개발자(SQLD) 취득": True}},
        "coding": True, "aisw": True, "free_creative": 9, "core_lang": True, "free_global": 10,
    })


def test_aibd_2022_full_pass():
    r = check_graduation("AI빅데이터", "22학번", **FULL_AIBD_2022)
    assert r.passed and r.total_earned == 95 == r.total_required, r.missing


def test_aibd_2022_partial():
    acts = dict(FULL_AIBD_2022["activities"], dept_act=3, coding=False)
    r = check_graduation("AI·빅데이터학과", 2022, FULL_AIBD_2022["courses"], acts)
    assert not r.passed and r.total_earned == 88
    names = {m["name"]: m["need"] for m in r.missing}
    assert names == {"학과활동": 2, "코딩역량인증제(Lv2 이상)": 5}


def test_by_category():
    r = check_graduation("AI빅데이터", 2022, **FULL_AIBD_2022)
    bc = {k: v["required"] for k, v in r.by_category.items()}
    assert bc == {"소통하는 공감인재": 20, "융합하는 창의인재": 45, "공헌하는 글로벌인재": 30}  # 비교과일람표 기준
    assert all(v["earned"] == v["required"] for v in r.by_category.values())
    for dept, info in REQ["departments"].items():          # 모든 학과·학번: 인재상 합 == 총점
        for t in info["tracks"]:
            r = check_graduation(dept, t["years"][0], "", {})
            assert sum(v["required"] for v in r.by_category.values()) == t["total"]


def test_two_level_sections():
    """AI·빅데이터 22학번: 대학공통 30(교과 20·비교과 10) / 학과기준 65(교과 10·졸업필수활동 15·비교과지정 21·비교과미지정 19)"""
    r = check_graduation("AI빅데이터", 2022, "", {})
    assert {g: v["required"] for g, v in r.by_group.items()} == {"대학공통": 30, "학과기준": 65}
    assert {k: v["required"] for k, v in r.by_kind.items()} == {
        "대학공통·교과": 20, "대학공통·비교과": 10, "학과기준·교과": 10,
        "학과기준·졸업필수활동": 15, "학과기준·비교과지정": 21, "학과기준·비교과미지정": 19}
    # 모든 학과·학번: 졸업필수활동류(졸업필수활동/졸업종합지도/졸업및취·창업지도)는 학과기준으로 집계
    for d, y, uni, grad in [("컴소", 2021, 30, 15), ("정보보호", 2023, 30, 15), ("사물인터넷", 2024, 32, 13),
                            ("메타버스", 2026, 32, 13), ("의료IT", 2022, 30, 15)]:
        r = check_graduation(d, y, "", {})
        assert r.by_group["대학공통"]["required"] == uni, (d, y)
        assert r.by_kind["학과기준·졸업필수활동"]["required"] == grad, (d, y)


def test_status_matrix_matches_school_system():
    """학교 시스템 '다면적졸업인증 이수현황' 필요점수 (AI·빅데이터 22학번 스크린샷 기준)"""
    r = check_graduation("AI빅데이터", 2022, "", {})
    need = {g: {c: v["required"] for c, v in row.items()} for g, row in r.matrix.items()}
    assert need == {"대학공통": {"공감": 15, "창의": 5, "글로벌": 10},
                    "학과기준": {"공감": 5, "창의": 40, "글로벌": 20}}


def test_partial_scores():
    """자기주도상담 회당 1점, 졸업필수활동 부분 점수, ASP 두 학기 모두 필요"""
    r = check_graduation("AI빅데이터", 2022, "", {"c_selfcounsel": 3, "c_gradact": 10, "c_asp": ["1학기"]})
    it = {i["id"]: i for i in r.items}
    assert it["c_selfcounsel"]["earned"] == 3 and not it["c_selfcounsel"]["passed"]
    assert it["c_gradact"]["earned"] == 10
    assert it["c_asp"]["earned"] == 0
    r = check_graduation("AI빅데이터", 2022, "", {"cert_contest": {"subitems": {"경진대회 참가 (3~10점, 인정 점수 입력)": 4}}})
    assert {i["id"]: i for i in r.items}["cert_contest"]["earned"] == 4


def test_topcit_level_select():
    r = check_graduation("메타버스", 2024, "", {"free_creative": {"subitems": {"TOPCIT 수준별 점수": "수준2(도전형) 150~399"}}})
    assert {i["id"]: i for i in r.items}["free_creative"]["earned"] == 2


def test_points_cap_and_count():
    r = check_graduation("AI빅데이터", 2022, "", {"dept_act": 99, "free_global": 30})
    it = {i["id"]: i for i in r.items}
    assert it["dept_act"]["earned"] == 5 and it["free_global"]["earned"] == 10


def test_choose_and_substitute():
    r = check_graduation("컴소", 2018, "전공영어1, 전공영어3, 전공영어기초", {})
    it = {i["id"]: i for i in r.items}
    assert it["major_eng"]["earned"] == 15
    assert any("2019~2022 기준" in w for w in r.warnings)   # 2018학번 대학공통 경고


def test_iot_capstone_pattern():
    r = check_graduation("사물인터넷", 2018, "캡스톤디자인1, 캡스톤디자인2", {})
    assert {i["id"]: i for i in r.items}["capstone2plus"]["earned"] == 10


def test_topcit_rule():
    f = lambda a, l: {i["id"]: i for i in check_graduation("메타버스", 2024, "", {"topcit": {"attempts": a, "level": l}}).items}["topcit"]["passed"]
    assert f(5, 0) and f(3, 2) and f(1, 3) and f(0, 4)
    assert not f(4, 1) and not f(2, 2)


def test_unknown_year():
    assert "error" in check_graduation("AI빅데이터", 2017, "", {})


# ---------------------------------------------------------------- 검색
def test_retrieval_profile_pin():
    r = build_index(verbose=False)
    res = r.search("졸업요건 알려줘", k=3, department="AI·빅데이터학과", year=2022)
    locs = [c.location for c, _ in res]
    assert "AI·빅데이터학과 2022학번" in locs and "대학공통 2019~2022학번" in locs
    res = r.search("권장 자격증 뭐 있어?", k=3, department="AI·빅데이터학과", year=2022)
    assert any("SQL" in c.text for c, _ in res)


def test_retrieval_official_docs():
    """순천향대 공식 안내 문서(docs/*.md) 내용을 프로필 유무와 관계없이 찾아야 한다"""
    r = build_index(verbose=False)
    cases = [("AI·SW전공역량졸업인증제 200점 어떻게 채워?", "전공역량점수"), ("코딩역량인증제 시험은 어떤 언어로 봐?", "Python"),
             ("TOPCIT 응시하면 뭐 줘?", "마일리지"), ("비교과자율활동 점수는 어떻게 받아?", "다문화"),
             ("다면적졸업인증만 채우면 졸업 가능해?", "졸업학점"), ("ASP가 뭐야?", "After School"),
             ("컴소 25학번 학과 MT 회당 몇 점?", "회당 2점")]
    for q, key in cases:
        for d, y in [("AI·빅데이터학과", 2022), (None, None)]:
            assert any(key in c.text for c, _ in r.search(q, 5, d, y)), (q, d)


# ---------------------------------------------------------------- LLM tool-calling 루프 (가짜 클라이언트)
class FakeClient:
    """1턴: get_requirements → 2턴: check_graduation → 3턴: 최종 답변"""
    def __init__(self):
        self.calls = 0
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    def _tc(self, name, args, i):
        fn = SimpleNamespace(name=name, arguments=json.dumps(args, ensure_ascii=False))
        tc = SimpleNamespace(id=f"call_{i}", function=fn, type="function")
        tc.model_dump = lambda: {"id": tc.id, "type": "function", "function": {"name": name, "arguments": fn.arguments}}
        return tc

    def create(self, **kw):
        self.calls += 1
        if self.calls == 1:
            msg = SimpleNamespace(content="", tool_calls=[self._tc("get_requirements", {"department": "AI빅데이터", "admission_year": 2022}, 1)])
        elif self.calls == 2:
            assert "dept_act" in kw["messages"][-1]["content"]
            msg = SimpleNamespace(content="", tool_calls=[self._tc("check_graduation", {
                "department": "AI빅데이터", "admission_year": 2022,
                "courses": ["인간사랑", "예비창업설계"], "activities": {"dept_act": 3}}, 2)])
        else:
            assert "미충족" in kw["messages"][-1]["content"]
            msg = SimpleNamespace(content="아직 졸업요건을 충족하지 못했어요. [문서 1]", tool_calls=None)
        return SimpleNamespace(choices=[SimpleNamespace(message=msg)])


def test_llm_tool_loop():
    r = build_index(verbose=False)
    a = llm.answer("인간사랑, 예비창업설계 들었고 학과활동 3번 했어. 졸업 가능?", r, FakeClient(),
                   profile={"department": "AI·빅데이터학과", "year": 2022})
    assert [t[0] for t in a.tool_calls] == ["문장 자동 인식", "get_requirements", "check_graduation"]
    assert a.check_result and a.check_result["total_earned"] == 13   # 인간사랑5 + 예비창업5 + 학과활동3
    assert a.sources


def test_local_llm_wrapper():
    """무료 오픈소스 LLM 래퍼: Qwen 형식 출력(<tool_call>)을 해석해 RAG·도구 호출 루프가 그대로 동작해야 한다"""
    from rag.local_llm import LocalLLMClient, parse_output

    class T:
        def __init__(self, d): self.data = d
        shape = property(lambda self: (1, len(self.data)))
        def to(self, dev): return self
        def __getitem__(self, s): return self.data[s]
    script = [
        '<tool_call>\n{"name": "check_graduation", "arguments": {"department": "AI빅데이터", "admission_year": 2022, '
        '"courses": ["인간사랑", "예비창업설계"], "activities": {"dept_act": 3, "c_selfcounsel": 2}}}\n</tool_call><|im_end|>',
        '아직 미충족이에요. [문서 1]<|im_end|>']
    class Tok:
        eos_token_id = 0
        def apply_chat_template(self, conv, tools=None, add_generation_prompt=True, tokenize=False):
            for m in conv:   # 도구 호출 인자는 dict 로 변환되어 있어야 함 (Qwen 템플릿이 tojson 으로 렌더)
                for tc in m.get("tool_calls") or []:
                    assert isinstance(tc["function"]["arguments"], dict)
            return json.dumps(conv, ensure_ascii=False)
        def __call__(self, p, return_tensors=None): return {"input_ids": T(list(p))}
        def decode(self, ids, skip_special_tokens=False): return "".join(ids)
    class Model:
        device, i = "cpu", 0
        def generate(self, input_ids=None, **kw):
            out = list(input_ids.data) + list(script[self.i]); self.i += 1; return [out]
    client = LocalLLMClient("Qwen/Qwen2.5-3B-Instruct", _tokenizer=Tok(), _model=Model())
    r = build_index(verbose=False)
    a = llm.answer("인간사랑, 예비창업설계 들었고 학과활동 3번, 자기주도상담 2번 했어. 졸업 가능해?", r, client,
                   profile={"department": "AI·빅데이터학과", "year": 2022})
    assert [t[0] for t in a.tool_calls] == ["문장 자동 인식", "check_graduation"]
    assert a.check_result["total_earned"] == 15 and "미충족" in a.text
    assert llm.make_embedder(client) is None and "오픈소스" in llm.model_label(client)
    m = parse_output('<tool_call>{"name":"search_documents","arguments":"{\\"query\\": \\"TOPCIT\\"}"}</tool_call>')
    assert m.tool_calls[0].function.name == "search_documents" and json.loads(m.tool_calls[0].function.arguments) == {"query": "TOPCIT"}


def test_sentence_extraction():
    """채팅 문장 → 과목·활동 자동 인식 (작은 모델·오프라인에서도 채팅 판정)"""
    from rag.extract import extract_inputs, is_check_request
    def score(q, d, y):
        e = extract_inputs(q, d, y)
        return check_graduation(d, y, e["courses"], e["activities"]).total_earned
    assert score("인간사랑, 현대기본영어, 봉사학습, 예비창업설계 들었고 학과활동 3번, SQLD 땄어. 졸업 가능해?", "AI·빅데이터학과", 2022) == 29
    assert score("자기주도상담 두 번 했고 ASP 1학기만 들었어. 코딩역량인증제는 아직 못 땄어. 졸업필수활동 10점이야", "AI·빅데이터학과", 2022) == 12
    assert score("캡스톤디자인, 산학캡스톤디자인, 전공영어기초 들었고 전공상담 4번, 학술제 발표했어. TOPCIT 2번 응시", "컴퓨터소프트웨어공학과", 2022) == 31
    assert score("TOPCIT 3번 응시해서 수준2 달성했고 맛있는SW 들었어", "메타버스&게임학과", 2024) == 10
    assert score("지도교수상담 2회, 학과활동 5회, 정보처리기사 응시했고 ADsP 취득했어", "사물인터넷학과", 2024) == 2 + 5 + 7
    assert score("학과학술동아리 2학기 했고 C프로그래밍 들었어. 학과활동 열 번", "정보보호학과", 2023) == 21
    assert is_check_request("졸업 가능해?") and not is_check_request("학과활동은 몇 번 해야 해?")


def test_local_streaming_and_key_points():
    """무료 LLM 경로: 일반 질문은 도구 없이 한 번에 생성(스트리밍), 답변과 함께 문서 핵심 문장 제공"""
    from rag.local_llm import LocalLLMClient

    class FakeStream(LocalLLMClient):
        def __init__(self, outs):
            self.outs, self.i, self.is_local, self.model_name, self.seen = outs, 0, True, "fake", []
        def stream(self, messages, tools=None, max_new_tokens=None):
            self.seen.append(bool(tools))
            out = self.outs[self.i]; self.i += 1
            for k in range(0, len(out), 5):
                yield out[k:k + 5]
    r = build_index(verbose=False)
    c = FakeStream(["학과활동은 5회 참여하면 5점이에요 [문서 1].<|im_end|>"])
    evs = list(llm.answer_stream("학과활동은 몇 번 참여해야 해?", r, c, [],
                                 {"department": "AI·빅데이터학과", "year": 2022}))
    partial, a = [e for e in evs if isinstance(e, str)], evs[-1]
    assert len(partial) > 3 and not any("<|" in p for p in partial)      # 글자가 나눠서 표시됨
    assert c.seen == [False] and a.text.startswith("학과활동은 5회")      # 일반 질문은 도구 없이 1회 생성
    assert any("학과활동" in l and "회당 1점" in l for l, _ in a.points)
    pts = llm.key_points("내 학번 졸업인증 기준 전체 알려줘",
                         r.search("내 학번 졸업인증 기준 전체 알려줘", k=5, department="AI·빅데이터학과", year=2022))
    assert pts[0][0].startswith("합계 95점")


def test_colab_ui_bridge():
    """Colab 화면: 한 장짜리 HTML 생성 + 백그라운드 AI 답변 작업(start → poll) 흐름"""
    import time
    from rag import colab_ui
    from rag.local_llm import LocalLLMClient
    page = colab_ui.build_page()
    assert "/*REQ*/" not in page and "/*CORE*/" not in page and "askColab" in page and "invokeFunction" in page

    class FakeStream(LocalLLMClient):
        def __init__(self):
            self.is_local, self.model_name = True, "fake"
        def stream(self, messages, tools=None, max_new_tokens=None):
            for w in ["학과활동은 ", "5회 ", "참여해요.", "<|im_end|>"]:
                time.sleep(0.01); yield w
    r = build_index(verbose=False)
    hits = colab_ui.search_hits(r, "학과활동은 몇 번?", "AI·빅데이터학과", 2022)
    assert hits and all(set(h) == {"id", "score"} for h in hits)
    jid = colab_ui.start_job(FakeStream(), r, "학과활동은 몇 번?", "AI·빅데이터학과", 2022, [], "",
                             {"dept": "AI·빅데이터학과", "year": 2022, "courses": ["인간사랑"], "activities": {}, "credits": None})
    for _ in range(200):
        s = colab_ui.poll_job(jid)
        if s["done"]:
            break
        time.sleep(0.02)
    assert s["done"] and not s["error"] and s["text"] == "학과활동은 5회 참여해요."


def test_extra_document_retrieval():
    """졸업요건 추가 자료(학칙·교육과정)도 검색되어야 한다"""
    r = build_index(verbose=False)
    for q, loc in [("졸업하려면 몇 학점 들어야 해?", "[1] 총 졸업학점"), ("한 학기에 최대 몇 학점까지 수강신청 가능해?", "[2]"),
                   ("중핵교양은 몇 과목 들어야 돼?", "[3] 중핵교양"), ("머신러닝은 몇 학년 과목이야?", "2학년"),
                   ("졸업 최저 평점이 몇이야?", "[6] 확인하지 못한")]:
        res = r.search(q, k=5, department="AI·빅데이터학과", year=2022)
        assert any(c.source == "순천향대_졸업요건_추가자료.md" and loc in c.location for c, _ in res), q


def test_check_intent():
    from rag.extract import is_check_request
    assert not is_check_request("AI·SW전공역량졸업인증제 200점은 어떻게 채워?")
    assert not is_check_request("자가진단 결과에서 부족한 항목을 어떻게 채우면 되는지 알려줘")
    assert is_check_request("인간사랑 들었고 학과활동 3번 했어. 졸업 가능해?")
    assert is_check_request("SQLD 땄는데 어떻게 돼? 졸업 가능해?")


def test_cumulative_chat_and_credits():
    """대화 누적 판정: 여러 메시지에 나눠 말해도 이어서 계산 ('더', '안 했어', '처음부터 다시') + 총 이수학점 130"""
    from rag.extract import extract_credits
    r = build_index(verbose=False)
    mem, prof = {}, {"department": "AI·빅데이터학과", "year": 2022}
    say = lambda q: llm.answer(q, r, None, profile=prof, memory=mem)
    a = say("인간사랑 들었고 학과활동 3번 했어. 졸업 가능해?")
    assert a.check_result["total_earned"] == 8
    a = say("봉사학습도 들었어")
    assert a.check_result["total_earned"] == 13
    a = say("학과활동 2번 더 했어")
    assert a.check_result["total_earned"] == 15                       # 학과활동 3 + 2 = 5회
    a = say("총 115학점이야")
    assert "115 / 130학점" in a.text and "15학점 부족" in a.text
    a = say("봉사학습은 안 들었어")
    assert a.check_result["total_earned"] == 10
    a = say("졸업하려면 총 130학점 들어야 해?")                          # 질문은 기록하지 않음
    assert a.check_result is None and mem["AI·빅데이터학과|2022"]["credits"] == 115
    say("처음부터 다시")
    assert not mem["AI·빅데이터학과|2022"]["courses"] and not mem["AI·빅데이터학과|2022"]["activities"]
    assert extract_credits("지금까지 98학점 들었어") == 98 and extract_credits("130학점 이상이야?") is None


def test_hybrid_embedding_faiss():
    """임베딩 + FAISS 하이브리드 검색 (실제 모델 대신 같은 인터페이스의 가짜 임베딩·FAISS로 연결 확인)"""
    import types
    import numpy as np
    from rag.ingest import build_chunks
    from rag.retriever import Retriever

    class FakeIndexFlatIP:                       # faiss.IndexFlatIP 와 같은 사용법
        def __init__(self, d): self.d, self.x = d, np.zeros((0, d), np.float32)
        def add(self, v): self.x = np.vstack([self.x, v])
        def search(self, q, k):
            s = q @ self.x.T; i = np.argsort(-s, axis=1)[:, :k]
            return np.take_along_axis(s, i, axis=1), i
    sys.modules["faiss"] = types.SimpleNamespace(IndexFlatIP=FakeIndexFlatIP)

    class FakeEmbedder:                          # LocalEmbedder 와 같은 사용법 (문서: __call__, 질문: query)
        name = "fake-e5"
        def _v(self, t):
            v = np.zeros(64, np.float32)
            for i in range(len(t) - 1):
                v[hash(t[i:i + 2]) % 64] += 1
            return v / (np.linalg.norm(v) + 1e-9)
        def __call__(self, texts): return [self._v(t) for t in texts]
        def query(self, t): return self._v(t)
    try:
        r = Retriever(build_chunks(), FakeEmbedder())
        assert r.faiss_index is not None and "FAISS" in r.search_mode and "fake-e5" in r.search_mode
        res = r.search("졸업하려면 몇 학점 들어야 해?", k=5, department="AI·빅데이터학과", year=2022)
        assert any("[1] 총 졸업학점" in c.location for c, _ in res)
        dense = r.dense_scores("학과활동")
        assert dense.shape == (len(r.chunks),) and abs(float(dense.max())) <= 1.0001
    finally:
        sys.modules.pop("faiss", None)


def test_offline_chat_judgement():
    """LLM 없이도 채팅 문장으로 졸업 판정 결과를 답한다"""
    r = build_index(verbose=False)
    a = llm.answer("인간사랑, 예비창업설계 들었고 학과활동 3번 했어. 졸업 가능해?", r, None,
                   profile={"department": "AI·빅데이터학과", "year": 2022})
    assert a.check_result["total_earned"] == 13 and "판정" in a.text


def test_offline_mode():
    r = build_index(verbose=False)
    a = llm.answer("학과활동 몇 번?", r, None, profile={"department": "AI·빅데이터학과", "year": 2022})
    assert "오프라인" in a.text and a.sources


if __name__ == "__main__":
    fails = 0
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            try:
                fn(); print("PASS", name)
            except Exception as e:
                fails += 1; print("FAIL", name, repr(e))
    sys.exit(1 if fails else 0)
