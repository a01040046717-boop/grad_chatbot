"""python -m pytest tests -q   (또는 python tests/test_all.py)"""
import json
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rag.checker import (load_requirements, get_requirement_items, check_graduation, parse_course_list,
                         resolve_department, parse_year, KINDS)
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
                for k in KINDS:
                    assert sum(i["points"] for i in items if i["kind"] == k) == t["subtotals"][k], (dept, y, k)


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
        "dept_act": 5, "free_comm": 5, "cert_contest": {"subitems": {"SQL개발자(SQLD) 취득": True}},
        "coding": True, "aisw": True, "free_creative": 4, "core_lang": True, "free_global": 10,
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
    assert [t[0] for t in a.tool_calls] == ["get_requirements", "check_graduation"]
    assert a.check_result and a.check_result["total_earned"] == 13   # 인간사랑5 + 예비창업5 + 학과활동3
    assert a.sources


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
