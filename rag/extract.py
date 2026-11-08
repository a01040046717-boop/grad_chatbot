"""
문장 → 졸업 판정 입력 자동 인식 (규칙 기반)

"인간사랑, 예비창업설계 들었고 학과활동 3번, 자기주도상담 2회 했어. SQLD 땄어" 같은 문장에서
수강 과목과 비교과 활동을 찾아 check_graduation 의 입력(courses, activities)으로 만든다.
LLM 이 작거나(무료 모델) 없을 때(오프라인)도 채팅으로 졸업 판정이 되도록 하는 보조 장치이며,
LLM 이 있으면 이 결과를 참고 자료로 함께 받는다.
"""
from __future__ import annotations

import re

from .checker import _norm, get_requirement_items

KNUM = {"한": 1, "두": 2, "세": 3, "네": 4, "다섯": 5, "여섯": 6, "일곱": 7, "여덟": 8, "아홉": 9, "열": 10}
NUM = r"(\d+|다섯|여섯|일곱|여덟|아홉|한|두|세|네|열)"
NEG = re.compile(r"^.{0,6}?(안\s*했|안\s*함|못|아직|없|안\s*땄|안\s*받|불합격|떨어)")
CHECK_INTENT = re.compile(r"졸업\s*(가능|할\s*수|돼|되)|충족|부족|모자|남은|남았|몇\s*점|판정|채웠|채워|계산")

# check 항목: 이름에 들어 있는 핵심어 → 문장에서 찾을 단어들
CHECK_KEYS = [
    ("코딩역량", ["코딩역량"]),
    ("AI·SW전공역량", ["ai·sw전공역량", "aisw전공역량", "aisw역량", "전공역량졸업인증", "aisw"]),
    ("학술제", ["학술제"]),
    ("중핵교양", ["중핵교양"]),
    ("학습윤리-기본", ["학습윤리기본", "학습윤리-기본"]),
    ("학습윤리-심화", ["학습윤리심화", "학습윤리-심화"]),
    ("국제박람회", ["국제박람회", "박람회"]),
    ("진로(취업)지도", ["진로지도", "취업지도"]),
    ("간담회", ["간담회"]),
]
# count 항목: 이름 핵심어 → 단어들 (긴 것부터 검사)
COUNT_KEYS = [
    ("자기주도상담", ["자기주도상담"]), ("지도교수상담", ["지도교수상담", "교수상담"]), ("졸업상담", ["졸업상담"]),
    ("진로 및 취업상담", ["진로및취업상담", "취업상담", "진로상담"]), ("전공상담", ["전공상담"]),
    ("학과활동", ["학과활동"]), ("MT", ["mt", "엠티"]), ("TOPCIT", ["topcit", "탑싯"]),
]
GOT = r"(취득|땄|따|합격|획득|붙|있)"
TOOK = r"(응시|봤|보|쳤|치)"


def _num(s: str) -> int:
    return int(s) if s.isdigit() else KNUM.get(s, 0)


def _pat(word: str) -> str:
    """'학과활동' → '학\\s*과\\s*활\\s*동' (띄어쓰기 허용)"""
    return r"\s*".join(re.escape(ch) for ch in word if not ch.isspace())


def _find_count(text: str, words: list[str], units=r"(번|회|차례|학기)") -> int | None:
    for w in words:
        m = re.search(_pat(w) + r"[^0-9\n.,]{0,10}?" + NUM + r"\s*" + units, text, re.I)
        if m:
            return _num(m.group(1))
    return None


def _mentioned(text: str, words: list[str]) -> bool:
    low = text.lower()
    for w in words:
        for m in re.finditer(_pat(w), low, re.I):
            if not NEG.search(low[m.end():m.end() + 12]):
                return True
    return False


def extract_inputs(text: str, dept: str, year: int) -> dict:
    spec = get_requirement_items(dept, year)
    if "error" in spec:
        return {"courses": [], "activities": {}, "recognized": []}
    nt = _norm(text)
    courses, acts, recog = [], {}, []

    # 1) 교과목: 긴 이름부터 찾아서 겹치는 짧은 이름(캡스톤디자인 ⊂ 산학캡스톤디자인)은 제외
    names = set()
    for it in spec["items"]:
        if it["type"] == "course":
            names.update(it.get("match", [it["name"]]))
        if it["type"] == "choose":
            names.update(it.get("options", []) + it.get("substitutes", []))
    rest = nt
    for n in sorted(names, key=lambda x: -len(_norm(x))):
        k = _norm(n)
        if k and k in rest:
            courses.append(n)
            recog.append(f"과목 {n}")
            rest = rest.replace(k, "■")
    for it in spec["items"]:      # 'XX 캡스톤디자인 2과목' 같은 패턴 과목(사물인터넷)
        if it["type"] == "choose" and it.get("pattern"):
            m = re.search(_pat(it["pattern"]) + r"[^0-9]{0,6}" + NUM + r"\s*(과목|개)", text)
            if m:
                for j in range(_num(m.group(1))):
                    courses.append(f"{it['pattern']}{j + 1}")
                recog.append(f"{it['pattern']} {_num(m.group(1))}과목")

    for it in spec["items"]:
        iid, t, name = it["id"], it["type"], it["name"]
        nn = _norm(name)
        if t == "count":
            for core, words in COUNT_KEYS:
                if _norm(core) in nn:
                    n = _find_count(text, words)
                    if n is not None:
                        acts[iid] = n
                        recog.append(f"{re.sub(r'\s*\d+회$', '', name)} {n}회")
                    break
        elif t == "check":
            for core, words in CHECK_KEYS:
                if _norm(core) in nn:
                    if _mentioned(text, words):
                        acts[iid] = True
                        recog.append(name)
                    break
        elif t == "all_of":   # ASP
            low = text.lower()
            m = re.search(r"asp", low)
            if m and not NEG.search(low[m.end():m.end() + 12]):
                win = low[m.end():m.end() + 20]
                parts = [p for p in it["parts"] if p in win and "모두" not in win and "다 " not in win]
                acts[iid] = parts if parts and len(parts) < len(it["parts"]) and "만" in win else list(it["parts"])
                recog.append(f"{name} ({', '.join(acts[iid])})")
        elif t == "topcit_rule":
            n = _find_count(text, ["topcit", "탑싯"], r"(번|회)")
            lv = re.search(r"(수준|레벨|level|lv)\s*(\d)", text, re.I)
            if n is not None or lv:
                acts[iid] = {"attempts": n or 0, "level": int(lv.group(2)) if lv else 0}
                recog.append(f"TOPCIT 응시 {n or 0}회, 수준 {lv.group(2) if lv else 0}")
        elif t == "points":
            subs, other = {}, 0.0
            for s in it.get("subitems", []):
                sname = s["name"]
                base = re.sub(r"\s*\(.*?\)", "", sname)
                abbr = re.findall(r"\(([A-Za-z]+)\)", sname)
                core = re.sub(r"\s*(취득|응시|이상|\(.*)$", "", base).strip()
                words = [w for w in [core] + abbr if w]
                if s["mode"] == "check":
                    if sname.startswith("토익스피킹"):
                        m = re.search(r"토익\s*스피킹\D{0,6}(\d{2,3})", text)
                        ok = bool(m and int(m.group(1)) >= 100)
                    elif sname.startswith("토익"):
                        m = re.search(r"토익(?!\s*스피킹)\D{0,6}(\d{3})", text)
                        ok = bool(m and int(m.group(1)) >= 500)
                    elif sname.lower().startswith("opic"):
                        ok = bool(re.search(r"opic\D{0,6}(im|ih|al|m1|m2|m3)", text, re.I))
                    elif sname.endswith("취득"):
                        ok = any(re.search(_pat(w) + r"[^,.\n고]{0,10}?" + GOT, text, re.I) for w in words) and _mentioned(text, words)
                    elif sname.endswith("응시"):
                        ok = any(re.search(_pat(w) + r"[^,.\n고]{0,10}?" + TOOK, text, re.I) for w in words)
                    else:
                        ok = _mentioned(text, words)
                    if ok:
                        subs[sname] = True
                        recog.append(sname)
                elif s["mode"] == "count":
                    n = _find_count(text, words)
                    if n is not None:
                        subs[sname] = n
                        recog.append(f"{core} {n}{'학기' if '학기' in sname else '회'}")
                elif s["mode"] == "number":
                    m = re.search(_pat(core) + r"\D{0,8}?(\d+)\s*점", text)
                    if m:
                        subs[sname] = float(m.group(1))
                        recog.append(f"{core} {m.group(1)}점")
            if not it.get("subitems"):   # 졸업필수활동 10점 같은 직접 점수
                core = re.sub(r"\s*\(.*$", "", name)
                m = re.search(_pat(core) + r"\D{0,8}?(\d+)\s*점", text)
                if m:
                    other = float(m.group(1))
                    recog.append(f"{core} {m.group(1)}점")
            if subs or other:
                acts[iid] = {"subitems": subs, "other": other}
    return {"courses": courses, "activities": acts, "recognized": recog}


def is_check_request(text: str) -> bool:
    return bool(CHECK_INTENT.search(text))
