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
NEG = re.compile(r"^.{0,6}?(안\s*했|안\s*함|안\s*들|안\s*봤|안\s*해|못|아직|없|안\s*땄|안\s*받|불합격|떨어|빼고|제외|말고)")
CHECK_INTENT = re.compile(r"졸업\s*(가능|할\s*수|돼|되)|충족|부족|모자|남은|남았|(내|지금|현재)\s*(점수|몇\s*점)|판정|채웠|채워|계산")

# check 항목: 이름에 들어 있는 핵심어 → 문장에서 찾을 단어들
CHECK_KEYS = [
    ("코딩역량", ["코딩역량", "코딩인증"]),
    ("AI·SW전공역량", ["ai·sw전공역량", "aisw전공역량", "aisw역량", "전공역량졸업인증", "aisw"]),
    ("학술제", ["학술제"]),
    ("중핵교양", ["중핵교양", "국제사회언어"]),
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
    ("학과활동", ["학과활동", "학과행사"]), ("MT", ["mt", "엠티"]), ("TOPCIT", ["topcit", "탑싯"]),
]
GOT = r"(취득|땄|따|합격|획득|붙|있)"
# 과목 줄임말 → 정식 과목명
COURSE_ALIASES = {"자기개발과 전공탐색": ["전공탐색"], "자기개발과 전공탐색(M&M)": ["전공탐색"], "예비창업설계": ["예창"]}
COURSE_ALIASES = {_norm(k): v for k, v in COURSE_ALIASES.items()}      # 띄어쓰기·괄호 무관하게 찾기
FULL_RE = re.compile(r"^.{0,8}?((다|모두|전부)\s*(했|채웠|채움|참여|받았|완료|끝)|완료|만점)")   # '자기주도상담 다 했어'
EXCEPT_ALL = re.compile(r"(빼고|제외하고|말고)\s*(나머지\s*)?(다|모두|전부)")                      # '봉사학습 빼고 다 들었어'
CONTEST_JOIN = re.compile(r"(경진대회|공모전|해커톤)[^.,\n]{0,12}?(참가|나갔|출전|참여|나감)")
CONTEST_WIN = re.compile(r"(경진대회|공모전|해커톤)[^.,\n]{0,12}?(수상|입상|상\s*받|상을\s*받)")
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


def _full(text: str, words: list[str]) -> bool:
    """'학과활동 다 채웠어' 처럼 단어 바로 뒤에 '다 했다/완료' 가 오는가"""
    low = text.lower()
    for w in words:
        for m in re.finditer(_pat(w), low, re.I):
            if FULL_RE.search(low[m.end():m.end() + 14]) and not NEG.search(low[m.end():m.end() + 12]):
                return True
    return False


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
    for n in sorted(names, key=lambda x: (-len(_norm(x)), x)):      # 길이 → 이름 순 (항상 같은 순서)
        k = _norm(n)
        if k and k in rest:
            rest = rest.replace(k, "■")
            if negated(text, n):          # '봉사학습은 안 들었어'
                continue
            courses.append(n)
            recog.append(f"과목 {n}")
    for n in sorted(names):                       # 줄임말 ('전공탐색', '예창')
        for a in COURSE_ALIASES.get(_norm(n), []):
            if n not in courses and _norm(a) in rest and not negated(text, a):
                rest = rest.replace(_norm(a), "■")
                courses.append(n)
                recog.append(f"과목 {n}")
    if EXCEPT_ALL.search(text):                   # 'X 빼고 다 들었어' → 나머지 교과목 모두
        for it in spec["items"]:
            if it["type"] == "course":
                cands = it.get("match", [it["name"]]) + COURSE_ALIASES.get(_norm(it["name"]), [])
                if not any(c in courses for c in cands) and not any(negated(text, c) for c in cands):
                    courses.append(it["name"])
                    recog.append(f"과목 {it['name']}")
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
                    short = re.sub(r"\s*\d+회$", "", name)
                    if n is not None:
                        acts[iid] = n
                        recog.append(f"{short} {n}회")
                    elif _full(text, words):
                        acts[iid] = -(-it["points"] // it["per"])        # 필요 횟수만큼
                        recog.append(f"{short} {acts[iid]}회 (다 했다고 해서 필요 횟수로 계산)")
                    break
        elif t == "check":
            for core, words in CHECK_KEYS:
                if _norm(core) in nn:
                    if _mentioned(text, words):
                        lv = re.search(r"(레벨|level|lv\.?|단계)\s*(\d)", text, re.I) if core == "코딩역량" else None
                        if lv and int(lv.group(2)) < 2:                  # Lv2 이상만 인정
                            recog.append(f"코딩역량인증 레벨{lv.group(2)} (Lv2 이상이어야 인정 → 미인정)")
                        else:
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
                    lo = re.search(r"\((\d+)~", sname)
                    if m:
                        subs[sname] = float(m.group(1))
                        recog.append(f"{core} {m.group(1)}점")
                    elif lo and ((core.endswith("참가") and CONTEST_JOIN.search(text) and not CONTEST_WIN.search(text))
                                 or (core.endswith("수상") and CONTEST_WIN.search(text))):
                        subs[sname] = float(lo.group(1))                 # 점수를 말하지 않으면 최소 점수
                        recog.append(f"{core} {lo.group(1)}점 (점수를 말하지 않아 최소 점수로 계산)")
            if not it.get("subitems"):   # 졸업필수활동 10점 같은 직접 점수
                core = re.sub(r"\s*\(.*$", "", name)
                cat = re.search(r"\((공감|창의|글로벌)\)", name)
                words = [core] + (["자율활동", "자율"] if "자율활동" in core else [])
                if cat:          # 비교과자율활동(창의)/(글로벌): 인재상 단어 뒤의 점수만 해당 항목에
                    m = re.search(cat.group(1) + r"\D{0,6}?(\d+)\s*점", text) if "자율" in text else None
                    if m:
                        other = float(m.group(1))
                        recog.append(f"{name} {m.group(1)}점")
                else:
                    m = next((x for x in (re.search(_pat(w) + r"\D{0,8}?(\d+)\s*점", text) for w in words) if x), None)
                    if m:
                        other = float(m.group(1))
                        recog.append(f"{core} {m.group(1)}점")
                    elif _full(text, [core]):
                        other = float(it["points"])
                        recog.append(f"{core} {it['points']:g}점 (다 했다고 해서 만점으로 계산)")
            if subs or other:
                acts[iid] = {"subitems": subs, "other": other}
    has_topcit = any(it["type"] == "topcit_rule" or (it["type"] == "count" and "topcit" in _norm(it["name"]))
                     or any("topcit" in _norm(s["name"]) for s in it.get("subitems", [])) for it in spec["items"])
    creative = next((it for it in spec["items"] if it["type"] == "points" and not it.get("subitems")
                     and "자율활동" in it["name"] and "창의" in it["name"]), None)
    if not has_topcit and creative:   # TOPCIT 응시 회당 1점은 비교과자율활동(창의) 점수로 인정 (공식 안내)
        n = _find_count(text, ["topcit", "탑싯"], r"(번|회)")
        if n:
            a = acts.setdefault(creative["id"], {"subitems": {}, "other": 0.0})
            a["other"] = float(a.get("other", 0)) + n
            recog.append(f"TOPCIT 응시 {n}회 → {creative['name']} {n}점")

    # '상담 3번' 처럼 짧게 말했을 때: 상담 횟수 항목이 하나뿐이면 그 항목
    sang = [it for it in spec["items"] if it["type"] == "count" and "상담" in it["name"]]
    if len(sang) == 1 and sang[0]["id"] not in acts:
        n = _find_count(text, ["상담"])
        if n is not None:
            acts[sang[0]["id"]] = n
            recog.append(f"{sang[0]['name']} {n}회")

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
QUESTION = re.compile(r"\?|몇|어떻게|뭐|무엇|언제|어디|왜|나요|까요|인가|인지|(?<!졸업)필수|해야")
DONE_EXTRA = re.compile(r"완료|들음|받음|채움|봤어|갔어|나갔|땄음|수료")
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
    이미 기록된 항목을 부정했거나('봉사학습은 안 들었어'), 초기화 요청이면 판정한다.
    '인간사랑 수강했어?' 처럼 이수 여부를 묻는 질문은 기록하지 않는다 (is_status_question)."""
    if is_status_question(text):
        return False
    if RESET_RE.search(text) or is_check_request(text) or ext.get("credits") is not None:
        return True
    if ext["recognized"] and not HYPO.search(text) and (               # '통과하면 어떻게 돼?' 는 가정
            DONE_SIGNAL.search(text) or DONE_EXTRA.search(text) or not QUESTION.search(text)):
        return True        # '학과활동 5번 참여', '인간사랑 이수 완료' 처럼 '했어' 가 없어도 이수 사실 진술이면 판정
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
        if t == "course" and any(_norm(c) == _norm(name) for c in rec["courses"]):
            continue                                   # 과목 리스트에 이미 있음
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


# ----------------------------------------------------------------------------
# 질문에 나온 항목의 기준 ('졸업필수활동은 몇 점?', '자격증 점수는?') → 항목별로 따로 정리
# ----------------------------------------------------------------------------
_ITEM_EXTRA = {"코딩역량인증제": ["코딩역량", "코딩인증"], "sw전공역량졸업인증제": ["전공역량"], "자기주도상담": ["자기주도"],
               "비교과자율활동": ["자율활동"], "경진대회참가": ["경진대회", "공모전"], "경진대회수상": ["경진대회", "수상"],
               "학과활동": ["학과행사"], "자기개발과전공탐색": ["전공탐색"], "예비창업설계": ["예창"], "자격증": ["자격"]}
_ITEM_STOP = {"과목", "택1", "택2", "택3", "이상", "lv2", "ai", "sw", "it", "학기", "12학기"}


def item_keys(it: dict) -> set:
    base = re.sub(r"\(.*?\)", " ", it["name"])
    keys = {_norm(t) for t in re.split(r"[/·\-\s]+", base) if len(_norm(t)) >= 2 and _norm(t) not in _ITEM_STOP}
    for k in list(keys):
        keys.update(_norm(x) for x in _ITEM_EXTRA.get(k, []))
    if "asp" in _norm(it["name"]):
        keys.add("asp")
    return keys


def _short_cat(c: str) -> str:
    return re.sub(r"^(소통하는|융합하는|공헌하는)\s*", "", c)


def item_fact(it: dict) -> str:
    t, pts = it["type"], it["points"]
    head = f"{it['name']}: {pts:g}점 ({it['section']}, {_short_cat(it['category'])})"
    if t == "count":
        head += f" — 회당 {it['per']:g}점, {-(-pts // it['per']):g}회 필요"
    elif t == "all_of":
        head += f" — {'·'.join(it['parts'])} 모두 이수해야 인정"
    elif t == "course":
        head += " — 해당 과목 이수"
    elif t == "choose":
        head += f" — {', '.join(it.get('options') or [it.get('pattern', '')])} 중 {it['n']}과목"
    elif t == "points" and it.get("subitems"):
        by_pts, rest = {}, []
        for s in it["subitems"]:
            if s["mode"] == "check" and "points" in s:
                by_pts.setdefault(s["points"], []).append(re.sub(r"\s*취득$", "", s["name"]))
            else:
                rest.append(s["name"] + (f" 회당 {s['per']:g}점" if s.get("per") else ""))
        parts = [f"{' / '.join(v)} 중 하나 취득 {k:g}점" for k, v in by_pts.items()] + rest
        head += " — 인정: " + "; ".join(parts) + f" (이 항목은 합쳐서 최대 {pts:g}점)"
    elif t == "points":
        head += " — 세부 활동 점수 합산 (비교과일람표에 인정된 점수)"
    note = (it.get("note") or "").split(" · ")[0]
    if note and t not in ("count", "all_of") and not note.startswith("세부 프로그램"):
        head += f" ※ {note}"
    return head


def item_facts(text: str, dept: str, year: int) -> list[str]:
    """질문에 이름이 나온 항목들의 기준을 한 줄씩 (항목마다 점수가 다르므로 섞이지 않게)"""
    spec = get_requirement_items(dept, year)
    if "error" in spec:
        return []
    nt = _norm(text)
    return [item_fact(it) for it in spec["items"] if any(k and k in nt for k in item_keys(it))]


# ----------------------------------------------------------------------------
# '인간사랑 수강했어?' → 물어본 항목만, 지금까지의 기록(자가진단 탭 + 대화)으로 답한다
# ----------------------------------------------------------------------------
ASKED_RE = re.compile(r"(들었|했|이수했|수강했|땄|받았|통과했|참여했|봤)(어|어요|나|나요|니|냐|지|죠|을까)?\s*[?？]"
                      r"|(들었|했|이수했|수강했|땄|받았|통과했|참여했)(나|니|냐|는지|는\s*지)")


def is_status_question(text: str) -> bool:
    return bool(ASKED_RE.search(text)) and not re.search(r"졸업\s*(가능|할\s*수)", text)


def asked_item_ids(text: str, spec: dict) -> list[str]:
    nt = _norm(text)
    return [it["id"] for it in spec["items"] if any(k and k in nt for k in item_keys(it))]


def status_lines(res, ids: list[str]) -> list[str]:
    """CheckResult 에서 물어본 항목의 상태. 기록이 없으면 '미이수'가 아니라 '아직 확인되지 않음'"""
    out = []
    from types import SimpleNamespace
    for r in res.items:
        r = SimpleNamespace(**r) if isinstance(r, dict) else r       # CheckResult.items 는 dict 목록
        if r.id not in ids:
            continue
        short = re.sub(r"\s*\(.*$", "", r.name)
        if r.passed:
            out.append(f"✅ {r.name}: 이수로 확인됨 ({r.earned:g}/{r.required:g}점 · {r.detail})")
        elif r.earned > 0:
            out.append(f"🔶 {r.name}: 일부만 확인됨 ({r.earned:g}/{r.required:g}점 · {r.detail})")
        else:
            out.append(f"⬜ {r.name}: 아직 확인되지 않음 (입력·언급된 기록 없음) — 자가진단 탭에서 체크하거나 "
                       f"'{short} 들었어/했어'라고 말해 주세요")
    return out
