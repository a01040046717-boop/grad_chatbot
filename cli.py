"""
터미널용 챗봇 (Streamlit 없이 테스트)

  python cli.py                         # 대화 모드
  python cli.py --dept AI빅데이터 --year 22
  python cli.py --rebuild               # docs/ 다시 인덱싱
명령:  /check  → 졸업 자가진단(질문에 답하며 입력),  /quit
"""
import argparse

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from rag.checker import (get_requirement_items, check_graduation, format_result_text, resolve_department,
                         parse_year, parse_course_list, _norm)
from rag.retriever import load_or_build, build_index
from rag.llm import answer, get_client, make_embedder


def interactive_check(dept: str, year: int) -> str:
    spec = get_requirement_items(dept, year)
    if "error" in spec:
        return spec["error"]
    print("\n수강한 과목을 쉼표로 입력하세요 (없으면 엔터):")
    courses = input("> ")
    parsed = parse_course_list(courses)
    acts = {}
    for it in spec["items"]:
        t, name = it["type"], f"{it['name']} ({it['points']:g}점)"
        if t == "check":
            acts[it["id"]] = input(f"[{it['section']}] {name} 이수/참여? (y/N) ").strip().lower() == "y"
        elif t == "course" and not any(_norm(m) in parsed for m in it.get("match", [])):
            acts[it["id"]] = input(f"[{it['section']}] {name} — 목록에 없는데 이수했나요? (y/N) ").strip().lower() == "y"
        elif t == "all_of":
            acts[it["id"]] = [p for p in it["parts"]
                              if input(f"[{it['section']}] {it['name']} — {p} 이수? (y/N) ").strip().lower() == "y"]
        elif t == "count":
            v = input(f"[{it['section']}] {name} 참여 횟수 (회당 {it['per']}점): ").strip()
            acts[it["id"]] = int(v) if v.isdigit() else 0
        elif t == "points":
            v = input(f"[{it['section']}] {name} 인정 점수 합계: ").strip()
            acts[it["id"]] = float(v) if v.replace(".", "").isdigit() else 0
        elif t == "topcit_rule":
            a = input("TOPCIT 응시 횟수: ").strip()
            l = input("TOPCIT 최고 수준(0~5): ").strip()
            acts[it["id"]] = {"attempts": int(a or 0), "level": int(l or 0)}
    res = check_graduation(dept, year, courses, acts)
    return format_result_text(res)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dept", default="AI·빅데이터학과")
    ap.add_argument("--year", default="2022")
    ap.add_argument("--rebuild", action="store_true")
    args = ap.parse_args()

    client = get_client()
    emb = make_embedder(client)
    retriever = build_index(embedder=emb) if args.rebuild else load_or_build(embedder=emb)
    dept, year = resolve_department(args.dept), parse_year(args.year)
    profile = {"department": dept, "year": year}
    print(f"🎓 졸업요건 챗봇 — {dept} {year}학번 | LLM: {'ON' if client else 'OFF(오프라인)'}")
    print("질문을 입력하세요. (/check: 자가진단, /quit: 종료)\n")

    history, check_text = [], None
    while True:
        try:
            q = input("나> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not q:
            continue
        if q == "/quit":
            break
        if q == "/check":
            check_text = interactive_check(dept, year)
            print("\n" + check_text + "\n")
            continue
        a = answer(q, retriever, client, history, profile, check_text)
        print("\n봇>", a.text)
        if a.tool_calls:
            print("   (도구:", ", ".join(t[0] for t in a.tool_calls) + ")")
        print("   (근거:", "; ".join(f"{s}/{l}" for s, l, _ in a.sources[:3]) + ")\n")
        history += [{"role": "user", "content": q}, {"role": "assistant", "content": a.text}]


if __name__ == "__main__":
    main()
