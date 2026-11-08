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
NEG = re.compile(r"^.{0,6}?(안\s*했|안\s*함|안\s*들|안\s*봤|안\s*해|못|아직|없|안\s*땄|안\s*받|불합격|떨어)")
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
            rest = rest.replace(k, "■")
            if negated(text, n):          # '봉사학습은 안 들었어'
                continue
            courses.append(n)
            recog.append(f"과목 {n}")
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
                        short = re.sub(r"\s*\d+회$", "", name)
                        recog.append(f"{short} {n}회")
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
    credits = extract_credits(text)
    if credits is not None:
        recog.append(f"총 이수학점 {credits}학점")
    return {"courses": courses, "activities": acts, "recognized": recog, "credits": credits}


# ----------------------------------------------------------------------------
# 총 이수학점 (학칙 제33조: 130학점 이상)
# ----------------------------------------------------------------------------
CREDIT_RE = re.compile(r"(?:총|지금까지|현재|이수한|들은|딴|취득한)\s*(\d{2,3})\s*학점(?!\s*(?:이상|필요|들어야|이수해야|채워야|인가|인지|맞))"
                       r"|(\d{2,3})\s*학점\s*(?:들었|이수했|채웠|이야|야\b|있어|땄|받았|취득했|했)")


def extract_credits(text: str) -> int | None:
    m = CREDIT_RE.search(text)
    if not m:
        return None
    n = int(m.group(1) or m.group(2))
    return n if 0 < n <= 200 else None


def credit_line(credits, req: dict | None = None) -> str | None:
    if credits is None:
        return None
    total = ((req or {}).get("meta", {}).get("grad_credits") or {}).get("total", 130)
    left = total - credits
    return (f"졸업학점(학칙 제33조, {total}학점 이상): {credits:g} / {total}학점 → "
            + ("충족" if left <= 0 else f"{left:g}학점 부족") + " (다면적 졸업인증과 별개의 졸업요건)")


# ----------------------------------------------------------------------------
# 대화 누적: 여러 메시지에 나눠 말한 과목·활동을 하나의 기록으로 합친다
# ----------------------------------------------------------------------------
HYPO = re.compile(r"하면|한다면|했다면|으면|라면")
RESET_RE = re.compile(r"처음부터\s*다시|초기화|리셋|다\s*지워|전부\s*지워|새로\s*시작")
MORE_RE = re.compile(r"(\d+|한|두|세|네|다섯)\s*(번|회)\s*더|더\s*(\d+|한|두|세|네|다섯)\s*(번|회)")


def negated(text: str, word: str) -> bool:
    """문장에서 word 바로 뒤에 부정 표현이 오는가 ('봉사학습은 안 했어')"""
    low = text.lower()
    for m in re.finditer(_pat(word), low, re.I):
        if NEG.search(low[m.end():m.end() + 12]):
            return True
    return False


def new_record(dept: str, year: int) -> dict:
    return {"dept": dept, "year": year, "courses": [], "activities": {}, "credits": None}


def has_any(rec: dict | None) -> bool:
    return bool(rec and (rec["courses"] or rec["activities"] or rec.get("credits") is not None))


def wants_judgement(text: str, ext: dict, rec: dict | None = None, spec: dict | None = None) -> bool:
    """판정 요청이거나, 이수 사실을 말했거나('봉사학습도 들었어'), 총 이수학점을 말했거나,
    이미 기록된 항목을 부정했거나('봉사학습은 안 들었어'), 초기화 요청이면 판정한다"""
    if RESET_RE.search(text) or is_check_request(text) or ext.get("credits") is not None:
        return True
    if DONE_SIGNAL.search(text) and ext["recognized"] and not HYPO.search(text):   # '통과하면 어떻게 돼?' 는 가정
        return True
    return bool(rec and spec and recorded_negation(text, rec, spec))


def recorded_negation(text: str, rec: dict, spec: dict) -> bool:
    items = {it["id"]: it for it in spec["items"]}
    if any(negated(text, c) for c in rec["courses"]):
        return True
    return any(iid in items and negated(text, re.sub(r"\s*\(.*$", "", items[iid]["name"])) for iid in rec["activities"])


def merge_record(rec: dict, ext: dict, text: str, spec: dict) -> dict:
    """이전 기록 + 이번 문장. '2번 더'는 횟수에 더하고, '안 했어'는 기록에서 뺀다."""
    if RESET_RE.search(text):
        rec = new_record(rec["dept"], rec["year"])
    rec = {"dept": rec["dept"], "year": rec["year"], "courses": list(rec["courses"]),
           "activities": dict(rec["activities"]), "credits": rec.get("credits")}
    items = {it["id"]: it for it in spec["items"]}
    more = bool(MORE_RE.search(text))
    for c in ext["courses"]:
        if c not in rec["courses"]:
            rec["courses"].append(c)
    for iid, v in ext["activities"].items():
        old, t = rec["activities"].get(iid), items[iid]["type"]
        if t == "count" and more and isinstance(old, (int, float)):
            rec["activities"][iid] = old + v
        elif t == "points" and isinstance(old, dict) and isinstance(v, dict):
            subs = dict(old.get("subitems", {}))
            for k, x in v.get("subitems", {}).items():
                prev = subs.get(k)
                add = more and not isinstance(x, bool) and isinstance(prev, (int, float)) and not isinstance(prev, bool)
                subs[k] = prev + x if add else x
            rec["activities"][iid] = {"subitems": subs, "other": v.get("other") or old.get("other", 0)}
        else:
            rec["activities"][iid] = v
    if ext.get("credits") is not None:
        rec["credits"] = ext["credits"]
    for c in list(rec["courses"]):
        if negated(text, c):
            rec["courses"].remove(c)
    for iid in list(rec["activities"]):
        if iid in items and negated(text, re.sub(r"\s*\(.*$", "", items[iid]["name"])) and iid not in ext["activities"]:
            del rec["activities"][iid]
    return rec


def describe_record(rec: dict, spec: dict) -> list[str]:
    """기록을 사람이 읽는 목록으로"""
    items = {it["id"]: it for it in spec["items"]}
    out = [f"과목 {c}" for c in rec["courses"]]
    for iid, v in rec["activities"].items():
        it = items.get(iid)
        if not it:
            continue
        name, t = it["name"], it["type"]
        short = re.sub(r"\s*\d+회$", "", name)
        core = re.sub(r"\s*\(.*$", "", name)
        if t == "count":
            out.append(f"{short} {v:g}회")
        elif t == "all_of":
            out.append(f"{name} ({', '.join(v)})")
        elif t == "topcit_rule":
            out.append(f"TOPCIT 응시 {v.get('attempts', 0)}회, 수준 {v.get('level', 0)}")
        elif t == "points" and isinstance(v, dict):
            for k, x in v.get("subitems", {}).items():
                out.append(k if x is True else f"{k} {x:g}" if isinstance(x, (int, float)) else f"{k} {x}")
            if v.get("other"):
                out.append(f"{core} {v['other']:g}점")
        elif t == "points":
            out.append(f"{name} {v:g}점")
        else:
            out.append(name)
    if rec.get("credits") is not None:
        out.append(f"총 이수학점 {rec['credits']:g}학점")
    return out


HOWTO_INTENT = re.compile(r"어떻게|방법|뭐야|뭔가|무엇|어디서")
DONE_SIGNAL = re.compile(r"졸업\s*(가능|할\s*수|돼|되)|판정|들었|했어|했고|했는데|했음|땄|취득했|이수했|참여했|응시했|받았|통과|다녀왔|마쳤|끝냈|수료|합격")


def is_check_request(text: str) -> bool:
    """졸업 판정 요청인가? ('…어떻게 채워?' 같은 방법 질문은 제외, 단 들었/했 등 이수 사실을 말하면 판정)"""
    if not CHECK_INTENT.search(text):
        return False
    return not (HOWTO_INTENT.search(text) and not DONE_SIGNAL.search(text))
