"""
문서 수집 → 청크 분할

두 종류의 청크를 만든다.
 1) 구조화 청크: data/requirements.json(PDF 표를 구조화한 것)을 학과·학번별 자연어 문단으로 변환.
    표 형태 PDF는 텍스트 추출 시 열이 뒤섞여 검색 품질이 나쁘므로, 표 1칸 = 청크 1개 수준으로 정리.
 2) 원문 청크: docs/ 폴더의 모든 문서(PDF, TXT, MD, DOCX)를 페이지/문단 단위로 분할.
    → '안내받은 다른 문서'는 docs/ 에 넣기만 하면 자동으로 포함된다.
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, asdict
from pathlib import Path

from .checker import load_requirements

ROOT = Path(__file__).resolve().parent.parent
DOCS_DIR = ROOT / "docs"
SUPPORTED = {".pdf", ".txt", ".md", ".docx"}


@dataclass
class Chunk:
    id: str
    text: str
    source: str          # 파일명 또는 'requirements.json'
    location: str        # 페이지 / 섹션
    department: str = "" # 메타데이터 필터·부스팅용
    years: tuple = ()

    def to_dict(self):
        return asdict(self)


def _cid(*parts) -> str:
    return hashlib.md5("|".join(map(str, parts)).encode()).hexdigest()[:12]


# ----------------------------------------------------------------------------
# 1) 구조화 청크
# ----------------------------------------------------------------------------
def _yr(y0, y1) -> str:
    return f"{y0}" if y0 == y1 else f"{y0}~{y1}"


def _item_line(it: dict) -> str:
    sec = it.get("section") or f"대학공통·{'교과' if it['type'] == 'course' else '비교과'}"
    s = f"- [{sec}/{it['category']}] {it['name']} ({it['points']:g}점)"
    if it["type"] == "count":
        s += f" — 회당 {it['per']}점, {-(-it['points'] // it['per'])}{it.get('unit', '회')} 필요"
    if it["type"] == "all_of":
        s += f" — {', '.join(it['parts'])} 모두 이수"
    if it["type"] == "choose":
        src = it.get("options") or [f"'{it['pattern']}' 포함 과목"]
        s += f" — {', '.join(src)} 중 {it['n']}과목"
        if it.get("substitutes"):
            s += f" (대체 인정: {', '.join(it['substitutes'])})"
    if it["type"] == "points" and it.get("subitems"):
        subs = []
        for x in it["subitems"]:
            if x["mode"] == "check":
                subs.append(f"{x['name']} {x['points']}점")
            elif x["mode"] == "select":
                subs.append(f"{x['name']}(" + ", ".join(f"{k} {v}점" for k, v in x["options"].items() if v) + ")")
            elif x["mode"] == "count":
                subs.append(f"{x['name']} 회당 {x['per']}점")
            else:
                subs.append(x["name"])
        s += " — 인정 예: " + ", ".join(subs)
    if it.get("note"):
        s += f" ※ {it['note']}"
    return s


def _category_line(dept: str, year: int, req: dict) -> str:
    from .checker import get_requirement_items, CATEGORIES
    items = get_requirement_items(dept, year, req)["items"]
    line = "인재상별 배점(대학공통 포함): " + ", ".join(
        f"{c} {sum(i['points'] for i in items if i['category'] == c):g}점" for c in CATEGORIES)
    short = {"소통하는 공감인재": "공감", "융합하는 창의인재": "창의", "공헌하는 글로벌인재": "글로벌"}
    table = " / ".join(
        f"{g} " + " · ".join(f"{short[c]} {sum(i['points'] for i in items if i['group'] == g and i['category'] == c):g}"
                             for c in CATEGORIES) for g in ("대학공통", "학과기준"))
    return line + "\n이수현황표 필요점수(구분×인재상): " + table


def structured_chunks() -> list[Chunk]:
    req = load_requirements()
    meta = req["meta"]
    out: list[Chunk] = []

    out.append(Chunk(
        _cid("meta"),
        "다면적 졸업인증제 일반 안내\n"
        f"- 비교과 신청: {meta['apply_path']}\n"
        f"- 이수내역 확인: {meta['check_path']}\n"
        f"- 상담: 다면적졸업인증제에 대한 자세한 상담이 필요한 학생은 {meta['contact']}로 방문\n"
        "- 졸업인증 점수는 '대학공통 필수(45점)' + '학과 기준(교과, 비교과지정, 비교과미지정)'의 합계로 구성되며, "
        "인재상은 소통하는 공감인재 / 융합하는 창의인재 / 공헌하는 글로벌인재 3가지이다.\n"
        "- 적용 학과(SW융합대학): " + ", ".join(req["departments"]),
        meta["source"], "일반 안내"))

    for c in req["common"]:
        y0, y1 = c["years"]
        txt = (f"대학공통 필수 항목 ({_yr(y0, y1)}학번, 모든 학과 공통, 합계 45점, 교과·비교과로 구분)\n"
               + "\n".join(_item_line(i) for i in c["items"]))
        out.append(Chunk(_cid("common", y0), txt, meta["source"], f"대학공통 {_yr(y0, y1)}학번", "", (y0, y1)))

    from .checker import get_requirement_items
    for dept, info in req["departments"].items():
        for t in info["tracks"]:
            y0, y1 = t["years"]
            spec = get_requirement_items(dept, y0, req)
            sec = spec["sections"]
            parts = []
            for g in ("대학공통", "학과기준"):
                subs = [f"{k.split('·')[1]} {v:g}" for k, v in sec.items() if k.startswith(g)]
                parts.append(f"{g} {spec['groups'][g]:g}점({' + '.join(subs)})")
            dept_items = [i for i in spec["items"] if i["group"] == "학과기준"]
            moved = [i for i in dept_items if i["id"].startswith("c_")]
            txt = (f"{dept} {_yr(y0, y1)}학번 다면적 졸업인증제 기준 (SW융합대학)\n"
                   f"합계 {t['total']}점 = " + " + ".join(parts) + "\n"
                   "구분: 대학공통(교과·비교과) / 학과기준(교과·졸업필수활동·비교과지정·비교과미지정)\n"
                   + _category_line(dept, y0, req) + "\n"
                   "학과 기준 항목:\n" + "\n".join(_item_line(i) for i in dept_items))
            for m in moved:
                txt += f"\n※ {m['name']}은(는) 이 학과·학번에서 대학공통이 아닌 학과기준({m['kind']})으로 분류됩니다."
            if info.get("unit_rules"):
                txt += f"\n점수 산정: {info['unit_rules']}"
            for n in info.get("notes", []):
                txt += f"\n※ {n}"
            out.append(Chunk(_cid(dept, y0), txt, meta["source"], f"{dept} {_yr(y0, y1)}학번", dept, (y0, y1)))
    return out


# ----------------------------------------------------------------------------
# 2) 원문 문서 청크
# ----------------------------------------------------------------------------
def _split(text: str, size: int = 700, overlap: int = 120) -> list[str]:
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    if len(text) <= size:
        return [text] if text else []
    paras = re.split(r"\n\s*\n|\n(?=#)", text)
    chunks, cur = [], ""
    for p in paras:
        if len(cur) + len(p) + 1 <= size:
            cur = f"{cur}\n{p}" if cur else p
            continue
        if cur:
            chunks.append(cur)
        while len(p) > size:                 # 긴 문단은 고정 길이로 자르되 겹치게
            chunks.append(p[:size])
            p = p[size - overlap:]
        cur = (chunks[-1][-overlap:] + "\n" + p) if chunks else p
    if cur:
        chunks.append(cur)
    return chunks


def _guess_dept(text: str) -> str:
    req = load_requirements()
    found = [d for d in req["departments"] if d.replace("·", "") in text.replace("·", "").replace("ㆍ", "")]
    return found[0] if len(found) == 1 else ""


def read_document(path: Path) -> list[tuple[str, str]]:
    """(위치, 텍스트) 리스트."""
    ext = path.suffix.lower()
    if ext == ".pdf":
        from pypdf import PdfReader
        return [(f"p.{i + 1}", p.extract_text() or "") for i, p in enumerate(PdfReader(str(path)).pages)]
    if ext == ".md":   # 마크다운은 '## 제목' 단위로 나눠서 표·목록이 잘리지 않게
        text = path.read_text(encoding="utf-8", errors="ignore")
        parts = re.split(r"\n(?=## )", text)
        title = parts[0].strip().splitlines()[0].lstrip("# ").strip() if parts else path.stem
        out = []
        for sec in parts[1:] or parts:
            head = sec.strip().splitlines()[0].lstrip("# ").strip()
            out.append((head, f"[{title}]\n{sec.strip()}"))
        return out
    if ext == ".txt":
        return [("본문", path.read_text(encoding="utf-8", errors="ignore"))]
    if ext == ".docx":
        try:
            import docx  # python-docx
        except ImportError:
            print(f"[skip] {path.name}: python-docx 미설치")
            return []
        d = docx.Document(str(path))
        parts = [p.text for p in d.paragraphs]
        for tb in d.tables:
            for row in tb.rows:
                parts.append(" | ".join(c.text.strip() for c in row.cells))
        return [("본문", "\n".join(parts))]
    return []


def document_chunks(docs_dir: Path = DOCS_DIR) -> list[Chunk]:
    out = []
    for path in sorted(docs_dir.glob("**/*")):
        if path.suffix.lower() not in SUPPORTED or path.name.startswith((".", "_")):  # _README 등 안내 파일 제외
            continue
        for loc, text in read_document(path):
            for j, piece in enumerate(_split(text, size=1800 if path.suffix.lower() == ".md" else 700)):
                dept = _guess_dept(piece) if path.suffix.lower() == ".pdf" else ""  # 일반 안내 문서는 전 학과 공통
                out.append(Chunk(_cid(path.name, loc, j, piece[:50]), piece, path.name,
                                 f"{loc}" + (f" #{j + 1}" if j else ""), dept))
    return out


def build_chunks(docs_dir: Path = DOCS_DIR) -> list[Chunk]:
    return structured_chunks() + document_chunks(docs_dir)


if __name__ == "__main__":
    cs = build_chunks()
    print(f"총 {len(cs)}개 청크")
    for c in cs[:3]:
        print("-" * 60, f"\n[{c.source} / {c.location}]\n{c.text[:400]}")
