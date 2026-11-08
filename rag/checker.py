"""
다면적 졸업인증제 판정기 (규칙 기반)

LLM이 점수를 '추측'하지 않도록, 표의 내용은 data/requirements.json 으로 구조화하고
판정은 이 모듈이 결정적으로 계산한다. LLM은 결과를 설명하는 역할만 한다.

입력(activities) 형식 — 항목 type 별:
  course       : True/False  (수강 리스트에 없어도 이수했으면 True)
  check        : True/False
  all_of       : ["1학기", "2학기"] 처럼 이수한 부분 리스트 (또는 True = 전부)
  count        : 정수 (참여 횟수)
  choose       : ["과목명", ...]  (수강 리스트 외에 추가로 인정할 과목)
  points       : 숫자(총 인정 점수)  또는  {"subitems": {"세부항목명": 값}, "other": 숫자}
                 (세부항목 값: check=True, count=횟수, number=점수, select=선택지 이름)
  topcit_rule  : {"attempts": 응시횟수, "level": 최고 달성 수준(0~5)}
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "requirements.json"
KINDS = ["대학공통", "교과", "비교과지정", "비교과미지정"]
CATEGORIES = ["소통하는 공감인재", "융합하는 창의인재", "공헌하는 글로벌인재"]


# ----------------------------------------------------------------------------
# 데이터 로딩 / 조회
# ----------------------------------------------------------------------------
def load_requirements(path: Path | str = DATA_PATH) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _norm(s: str) -> str:
    """학과명/과목명 비교용 정규화: 공백·기호 제거, 로마숫자→아라비아, 소문자."""
    s = str(s)
    for a, b in {"Ⅰ": "1", "Ⅱ": "2", "Ⅲ": "3", "Ⅳ": "4", "ㆍ": "", "·": "", "•": ""}.items():
        s = s.replace(a, b)
    s = re.sub(r"[\s\-_,./:;'\"`~!?·&]", "", s)
    return s.lower()


def normalize_course(raw: str) -> str:
    """'캡스톤디자인(3학점) A+' 같은 입력에서 과목명만 남긴다."""
    s = str(raw).strip()
    s = re.sub(r"^\s*[\-\*•\d]+[.)]\s*", "", s)          # 목록 번호 "1." "-" 제거
    s = re.sub(r"\[[^\]]*\]", "", s)                       # [전공필수] 등
    s = re.sub(r"\([^)]*\)", "", s)                        # (3학점), (M&M) 등 괄호 제거
    s = re.sub(r"\s+(A\+|A0|A|B\+|B0|B|C\+|C0|C|D\+|D0|D|F|P|NP|S|U)\s*$", "", s, flags=re.I)
    s = re.sub(r"\s*\d+(\.\d+)?\s*학점\s*", "", s)
    return _norm(s)


def parse_course_list(text_or_list: str | list[str]) -> list[str]:
    """줄바꿈/쉼표로 구분된 과목 리스트를 정규화된 과목명 리스트로."""
    if isinstance(text_or_list, str):
        parts = re.split(r"[\n,，;]+", text_or_list)
    else:
        parts = list(text_or_list)
    out = []
    for p in parts:
        n = normalize_course(p)
        if n and n not in out:
            out.append(n)
    return out


def resolve_department(name: str, req: dict | None = None) -> str | None:
    req = req or load_requirements()
    q = _norm(name)
    if not q:
        return None
    for dept, info in req["departments"].items():
        cands = [dept] + info.get("aliases", [])
        if any(_norm(c) == q for c in cands):
            return dept
    for dept, info in req["departments"].items():
        cands = [dept] + info.get("aliases", [])
        if any(_norm(c) in q or q in _norm(c) for c in cands if len(_norm(c)) >= 2):
            return dept
    return None


def parse_year(value: Any) -> int | None:
    """'22학번', 22, '2022' → 2022"""
    m = re.search(r"\d{2,4}", str(value))
    if not m:
        return None
    y = int(m.group())
    if y < 100:
        y += 2000
    return y


def get_track(dept: str, year: int, req: dict | None = None) -> dict | None:
    req = req or load_requirements()
    for t in req["departments"][dept]["tracks"]:
        if t["years"][0] <= year <= t["years"][1]:
            return t
    return None


def get_common(year: int, req: dict | None = None) -> tuple[list[dict], str | None]:
    """대학공통 필수 항목. PDF 표는 2019학번부터만 있으므로 그 이전은 2019~2022 기준 + 경고."""
    req = req or load_requirements()
    for c in req["common"]:
        if c["years"][0] <= year <= c["years"][1]:
            return c["items"], None
    if year < 2019:
        return req["common"][0]["items"], (
            f"{year}학번의 대학공통 세부 항목은 안내문에 없어 2019~2022 기준으로 계산했습니다. "
            "多Dream비교과센터에 확인하세요."
        )
    return [], f"{year}학번 대학공통 기준이 안내문에 없습니다."


def get_requirement_items(dept: str, year: int, req: dict | None = None) -> dict:
    """학과·학번에 적용되는 전체 항목(대학공통 + 학과) 반환."""
    req = req or load_requirements()
    track = get_track(dept, year, req)
    common, warn = get_common(year, req)
    if track is None:
        avail = [f"{t['years'][0]}~{t['years'][1]}" for t in req["departments"][dept]["tracks"]]
        return {"error": f"{dept} {year}학번 기준은 안내문에 없습니다. (안내된 학번: {', '.join(avail)})"}
    return {
        "department": dept,
        "year": year,
        "track_years": track["years"],
        "total": track["total"],
        "subtotals": track["subtotals"],
        "items": common + track["items"],
        "unit_rules": req["departments"][dept].get("unit_rules", ""),
        "notes": req["departments"][dept].get("notes", []),
        "warnings": [warn] if warn else [],
    }


# ----------------------------------------------------------------------------
# 판정
# ----------------------------------------------------------------------------
@dataclass
class ItemResult:
    id: str
    name: str
    category: str
    kind: str
    required: float
    earned: float
    passed: bool
    detail: str = ""


@dataclass
class CheckResult:
    department: str
    year: int
    total_required: float
    total_earned: float
    passed: bool
    by_kind: dict = field(default_factory=dict)
    by_category: dict = field(default_factory=dict)
    items: list = field(default_factory=list)
    missing: list = field(default_factory=list)
    unmatched_courses: list = field(default_factory=list)
    warnings: list = field(default_factory=list)
    notes: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


def _eval_item(item: dict, courses: list[str], act: Any, used: set) -> tuple[float, str]:
    t, pts = item["type"], item["points"]

    if t == "course":
        names = [_norm(m) for m in item.get("match", [item["name"]])]
        hit = next((c for c in courses if c in names), None)
        if hit:
            used.add(hit)
            return pts, "수강 리스트에서 확인"
        if act:
            return pts, "직접 체크"
        return 0, "미이수"

    if t == "check":
        return (pts, "체크됨") if act else (0, "미이수")

    if t == "all_of":
        parts = item["parts"]
        if act is True:
            done = list(parts)
        elif isinstance(act, dict):
            done = [p for p in parts if act.get(p)]
        elif isinstance(act, (list, tuple)):
            done = [p for p in parts if p in act]
        else:
            done = []
        ok = len(done) == len(parts)
        return (pts if ok else 0), f"{', '.join(done) or '없음'} 이수 ({', '.join(parts)} 모두 필요)"

    if t == "count":
        n = int(act or 0)
        earned = min(n * item["per"], pts)
        need = -(-pts // item["per"])  # ceil
        return earned, f"{n}{item.get('unit', '회')} 참여 (회당 {item['per']}점, 필요 {need}{item.get('unit', '회')})"

    if t == "choose":
        n_req = item["n"]
        each = pts / n_req
        if "pattern" in item:
            pat = _norm(item["pattern"])
            done = [c for c in courses if pat in c]
        else:
            opts = [_norm(o) for o in item["options"] + item.get("substitutes", [])]
            done = [c for c in courses if c in opts]
        extra = [normalize_course(x) for x in (act or [])] if isinstance(act, list) else []
        done = list(dict.fromkeys(done + extra))
        used.update(done)
        k = min(len(done), n_req)
        return k * each, f"{len(done)}과목 확인 ({', '.join(done) or '없음'}) / {n_req}과목 필요"

    if t == "points":
        if isinstance(act, (int, float)):
            total = float(act)
            return min(total, pts), f"입력 점수 {total:g}점"
        act = act or {}
        sub_vals = act.get("subitems", {}) if isinstance(act, dict) else {}
        total, parts = 0.0, []
        for s in item.get("subitems", []):
            v = sub_vals.get(s["name"])
            if not v:
                continue
            if s["mode"] == "check":
                p = s["points"]
            elif s["mode"] == "select":
                p = float(s["options"].get(v, v if isinstance(v, (int, float)) else 0))
            elif s["mode"] == "count":
                p = int(v) * s["per"]
            else:
                p = float(v)
            total += p
            parts.append(f"{s['name']} {p:g}점")
        other = float(act.get("other", 0) or 0) if isinstance(act, dict) else 0
        if other:
            total += other
            parts.append(f"{'기타' if item.get('subitems') else '취득'} {other:g}점")
        return min(total, pts), (", ".join(parts) or "인정 점수 없음") + f" (합 {total:g}점)"

    if t == "topcit_rule":
        a = act or {}
        att, lv = int(a.get("attempts", 0) or 0), int(a.get("level", 0) or 0)
        ok = att >= 5 or (att >= 3 and lv >= 2) or (att >= 1 and lv >= 3) or lv >= 4
        return (pts if ok else 0), f"응시 {att}회, 최고 Lv{lv} (기준: 응시5회 / 3회+Lv2 / 1회+Lv3 / Lv4)"

    raise ValueError(f"unknown type {t}")


def check_graduation(department: str, year: Any, courses: str | list[str] | None = None,
                     activities: dict | None = None, req: dict | None = None) -> CheckResult | dict:
    req = req or load_requirements()
    dept = resolve_department(department, req)
    if dept is None:
        return {"error": f"'{department}' 학과를 찾지 못했습니다. 가능: {', '.join(req['departments'])}"}
    y = parse_year(year)
    if y is None:
        return {"error": f"학번 '{year}'을(를) 해석하지 못했습니다."}
    spec = get_requirement_items(dept, y, req)
    if "error" in spec:
        return spec

    course_list = parse_course_list(courses or [])
    activities = activities or {}
    used: set = set()
    results: list[ItemResult] = []
    for it in spec["items"]:
        earned, detail = _eval_item(it, course_list, activities.get(it["id"]), used)
        results.append(ItemResult(it["id"], it["name"], it["category"], it["kind"], it["points"],
                                  round(earned, 2), earned >= it["points"] - 1e-9, detail))

    by_kind = {}
    for k in KINDS:
        req_k = sum(r.required for r in results if r.kind == k)
        ear_k = sum(r.earned for r in results if r.kind == k)
        by_kind[k] = {"required": req_k, "earned": ear_k}

    by_category = {}
    for c in CATEGORIES:
        by_category[c] = {"required": sum(r.required for r in results if r.category == c),
                          "earned": round(sum(r.earned for r in results if r.category == c), 2)}

    total_req = sum(r.required for r in results)
    total_ear = sum(r.earned for r in results)
    warnings = list(spec["warnings"])
    if abs(total_req - spec["total"]) > 1e-9:
        warnings.append(f"항목 합({total_req:g})이 안내문 합계({spec['total']})와 다릅니다.")

    missing = [
        {"name": r.name, "kind": r.kind, "category": r.category,
         "need": round(r.required - r.earned, 2), "detail": r.detail}
        for r in results if not r.passed
    ]
    return CheckResult(
        department=dept, year=y, total_required=total_req, total_earned=round(total_ear, 2),
        passed=all(r.passed for r in results), by_kind=by_kind, by_category=by_category,
        items=[asdict(r) for r in results], missing=missing,
        unmatched_courses=[c for c in course_list if c not in used],
        warnings=warnings, notes=spec["notes"] + ([spec["unit_rules"]] if spec["unit_rules"] else []),
    )


def format_result_text(res: CheckResult | dict) -> str:
    """사람이 읽는 요약 (CLI / LLM 컨텍스트용)."""
    if isinstance(res, dict) and "error" in res:
        return "⚠️ " + res["error"]
    lines = [
        f"[{res.department} {res.year}학번] 다면적 졸업인증제 판정",
        f"총점 {res.total_earned:g} / {res.total_required:g}  →  {'✅ 충족' if res.passed else '❌ 미충족'}",
        "구분별: " + ", ".join(f"{k} {v['earned']:g}/{v['required']:g}" for k, v in res.by_kind.items()),
        "인재상별: " + ", ".join(f"{k} {v['earned']:g}/{v['required']:g}" for k, v in res.by_category.items()),
    ]
    if res.missing:
        lines.append("부족한 항목:")
        for m in res.missing:
            lines.append(f"  - [{m['kind']}/{m['category']}] {m['name']}: {m['need']:g}점 부족 ({m['detail']})")
    if res.unmatched_courses:
        lines.append("졸업인증 항목과 무관한 수강 과목: " + ", ".join(res.unmatched_courses))
    for w in res.warnings:
        lines.append("※ " + w)
    return "\n".join(lines)
