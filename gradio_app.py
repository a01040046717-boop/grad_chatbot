"""
Gradio 버전 UI (Colab / Hugging Face Spaces 용)
  - Colab:  노트북 마지막 셀에서 build_demo().launch(share=True)
  - 로컬:   python gradio_app.py
"""
from __future__ import annotations

import gradio as gr

from rag.checker import load_requirements, get_requirement_items, check_graduation, format_result_text
from rag.retriever import load_or_build
from rag.llm import answer_stream, get_client, make_embedder, model_label, APPENDIX_MARK

REQ = load_requirements()
DEPTS = list(REQ["departments"])


def _text(content) -> str:
    """Gradio 버전에 따라 메시지 content 가 문자열 또는 블록 리스트일 수 있음."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return " ".join(c.get("text", "") if isinstance(c, dict) else str(c) for c in content)
    if isinstance(content, dict):
        return content.get("text", "")
    return str(content)


def _result_markdown(res) -> str:
    if isinstance(res, dict):
        return "⚠️ " + res["error"]
    head = "## ✅ 졸업인증 충족" if res.passed else "## ❌ 아직 미충족"
    cols = [(g, c) for g in res.matrix for c in res.matrix[g]]
    mx = ["### 다면적 졸업인증 이수현황",
          "| 구분 | " + " | ".join(f"{g}<br>{c}" for g, c in cols) + " | 졸업점수 |",
          "|---|" + "---|" * (len(cols) + 1)]
    for lab, f in [("필요점수 ⓐ", lambda v: v["required"]), ("이수점수 ⓑ", lambda v: v["earned"]),
                   ("미이수점수 ⓐ-ⓑ", lambda v: max(v["required"] - v["earned"], 0))]:
        tot = {"required": res.total_required, "earned": res.total_earned}
        mx.append(f"| {lab} | " + " | ".join(f"{f(res.matrix[g][c]):g}" for g, c in cols) + f" | {f(tot):g} |")
    lines = [head, f"**총점 {res.total_earned:g} / {res.total_required:g}점**", "", *mx, "",
             "| 구분 | 취득 | 필요 |", "|---|---|---|"]
    for g, gv in res.by_group.items():
        lines.append(f"| **{g}** | **{gv['earned']:g}** | **{gv['required']:g}** |")
        lines += [f"| └ {k.split('·')[1]} | {v['earned']:g} | {v['required']:g} |" for k, v in res.by_kind.items() if k.startswith(g)]
    lines += ["", "| 인재상 | 취득 | 필요 |", "|---|---|---|"]
    lines += [f"| {k} | {v['earned']:g} | {v['required']:g} |" for k, v in res.by_category.items()]
    if res.missing:
        lines += ["", "### 부족한 항목", "| 구분 | 인재상 | 항목 | 부족 | 현재 상태 |", "|---|---|---|---|---|"]
        lines += [f"| {m['section']} | {m['category']} | {m['name']} | {m['need']:g}점 | {m['detail']} |" for m in res.missing]
    if res.unmatched_courses:
        lines += ["", "ℹ️ 졸업인증 항목과 매칭되지 않은 과목: " + ", ".join(res.unmatched_courses)]
    lines += [""] + [f"※ {w}" for w in res.warnings + res.notes]
    lines += ["", "_공식 확인: 종합정보시스템 → 학부학사행정 → 졸업 → 개인별졸업진도관리 → 비교과일람표_"]
    return "\n".join(lines)


def build_demo(default_dept: str = "AI·빅데이터학과", default_year: int = 2022, client=None, retriever=None):
    """client/retriever 를 넘기면 그대로 사용 (노트북에서 모델을 두 번 불러오지 않도록)"""
    if client is None:
        client = get_client()
    if retriever is None:
        retriever = load_or_build(embedder=make_embedder(client))
    mode = model_label(client)
    sources = sorted({c.source for c in retriever.chunks})

    with gr.Blocks(title="SW융합대학 졸업요건 챗봇") as demo:
        gr.Markdown(f"# 🎓 SW융합대학 다면적 졸업인증제 챗봇\n"
                    f"상태: **{mode}** · 검색: {getattr(retriever, 'search_mode', 'TF-IDF')} · "
                    f"참고 문서: {', '.join(sources)} · 청크 {len(retriever.chunks)}개")
        with gr.Row():
            dept = gr.Dropdown(DEPTS, value=default_dept, label="학과")
            year = gr.Number(value=default_year, precision=0, label="학번(입학년도)")
        check_state = gr.State("")
        memory = gr.State({})          # 채팅으로 말한 이수 기록 (대화 누적, 학과|학번 별)

        # ------------------------------------------------------------- Q&A 탭
        with gr.Tab("💬 졸업요건 Q&A"):
            try:
                chatbot = gr.Chatbot(type="messages", height=480, label="챗봇")
            except TypeError:  # Gradio 6+ 는 messages 형식이 기본
                chatbot = gr.Chatbot(height=480, label="챗봇")
            msg = gr.Textbox(placeholder="예) 학과활동은 몇 번 해야 해? / 인간사랑, 예비창업설계 들었는데 졸업 가능해?",
                             label="질문", lines=2)
            with gr.Row():
                send = gr.Button("보내기", variant="primary")
                clear = gr.Button("대화 초기화")
            gr.Examples(["내 학번 졸업인증 기준 전체 알려줘", "학과활동은 몇 번 참여해야 해?",
                         "권장 자격증은 뭐가 있어?", "AI·SW전공역량졸업인증제 200점은 어떻게 채워?",
                         "인간사랑, 현대기본영어, 봉사학습, 예비창업설계 들었고 학과활동 3번, SQLD 땄어. 졸업 가능해?"],
                        inputs=msg)

            def _final_text(a) -> str:
                text = a.text
                extra = []
                if a.points:
                    extra.append("📌 **문서에서 찾은 핵심 내용**\n" + "\n".join(f"- {l}  <sub>({src})</sub>" for l, src in a.points))
                if a.auto_text:
                    extra.append("<details><summary>📊 판정 엔진 계산 결과 보기</summary>\n\n```\n" + a.auto_text + "\n```\n</details>")
                tail = []
                if a.tool_calls:
                    tail.append("🛠️ 도구 호출: " + " → ".join(t[0] for t in a.tool_calls))
                if a.sources:
                    tail.append("📎 근거: " + " · ".join(f"{s} / {l}" for s, l, _ in a.sources[:4]))
                if tail:
                    extra.append("<sub>" + "<br>".join(tail) + "</sub>")
                return text + (APPENDIX_MARK + "\n\n".join(extra) if extra else "")

            def respond(m, hist, d, y, chk, mem):
                mem = mem if isinstance(mem, dict) else {}
                if not m.strip():
                    yield "", hist, mem
                    return
                hist = hist or []
                h = [{"role": x["role"], "content": _text(x["content"])} for x in hist if isinstance(x, dict)]
                base = hist + [{"role": "user", "content": m}]
                yield "", base + [{"role": "assistant", "content": "⏳ 문서를 찾는 중이에요…"}], mem
                try:
                    for ev in answer_stream(m, retriever, client, h, {"department": d, "year": int(y)}, chk or None,
                                            memory=mem):
                        if isinstance(ev, str):
                            yield "", base + [{"role": "assistant", "content": ev + " ▌"}], mem
                        else:
                            yield "", base + [{"role": "assistant", "content": _final_text(ev)}], mem
                except Exception as e:
                    yield "", base + [{"role": "assistant", "content": f"⚠️ 오류: {e}"}], mem

            msg.submit(respond, [msg, chatbot, dept, year, check_state, memory], [msg, chatbot, memory])
            send.click(respond, [msg, chatbot, dept, year, check_state, memory], [msg, chatbot, memory])
            clear.click(lambda: ([], {}), None, [chatbot, memory])

        # ------------------------------------------------------------- 자가진단 탭
        with gr.Tab("✅ 졸업 자가진단"):
            with gr.Row():
                with gr.Column(scale=2):
                    courses = gr.Textbox(lines=12, label="📘 수강한 교과목 리스트 (한 줄에 한 과목)",
                                         placeholder="인간사랑\n현대기본영어\n봉사학습\n예비창업설계")
                    result = gr.Markdown("왼쪽/오른쪽을 입력하고 **판정하기**를 누르세요.")
                with gr.Column(scale=3):
                    @gr.render(inputs=[dept, year])
                    def build_items(d, y):
                        try:
                            spec = get_requirement_items(d, int(y))
                        except Exception as e:
                            gr.Markdown(f"⚠️ {e}")
                            return
                        if "error" in spec:
                            gr.Markdown("⚠️ " + spec["error"])
                            return
                        gr.Markdown(f"### 🏃 {d} {int(y)}학번 · 기준 {spec['total']}점")
                        comps, meta = [], []
                        for sec, sec_pts in spec["sections"].items():
                            items = [i for i in spec["items"] if i["section"] == sec]
                            gr.Markdown(f"**{sec}** ({sec_pts:g}점)")
                            for it in items:
                                lab = f"{it['name']} ({it['points']:g}점)"
                                t = it["type"]
                                if t in ("course", "check"):
                                    info = "수강 리스트에 있으면 자동 인정" if t == "course" else it.get("note")
                                    comps.append(gr.Checkbox(label=lab, info=info))
                                    meta.append((it["id"], "bool"))
                                elif t == "all_of":
                                    comps.append(gr.CheckboxGroup(it["parts"], label=f"{lab} — 이수한 부분 (모두 필요)",
                                                                  info=it.get("note")))
                                    meta.append((it["id"], "list"))
                                elif t == "count":
                                    comps.append(gr.Number(0, precision=0, minimum=0,
                                                           label=f"{lab} — 참여 횟수 (회당 {it['per']}점)"))
                                    meta.append((it["id"], "int"))
                                elif t == "choose":
                                    opts = (it.get("options") or []) + it.get("substitutes", [])
                                    if opts:
                                        comps.append(gr.CheckboxGroup(opts, label=f"{lab} — {it['n']}과목 필요"))
                                        meta.append((it["id"], "list"))
                                    else:
                                        gr.Markdown(f"- {lab}: 수강 리스트에서 '{it['pattern']}' 포함 과목을 셉니다.")
                                elif t == "points" and it.get("subitems"):
                                    gr.Markdown(f"**{lab}**" + (f" — {it['note']}" if it.get("note") else ""))
                                    for s_ in it["subitems"]:
                                        if s_["mode"] == "check":
                                            comps.append(gr.Checkbox(label=f"{s_['name']} (+{s_['points']}점)"))
                                        elif s_["mode"] == "count":
                                            comps.append(gr.Number(0, precision=0, minimum=0, label=f"{s_['name']} 횟수 (회당 {s_['per']}점)"))
                                        elif s_["mode"] == "select":
                                            comps.append(gr.Dropdown(list(s_["options"]), value=list(s_["options"])[0], label=s_["name"]))
                                        else:
                                            comps.append(gr.Number(0, minimum=0, label=s_["name"]))
                                        meta.append((it["id"], ("sub", s_["name"])))
                                    comps.append(gr.Number(0, minimum=0, label="기타 비교과 인정 점수"))
                                    meta.append((it["id"], "other"))
                                elif t == "points":
                                    ex = ", ".join(
                                        f"{s['name']}" + (f" {s['points']}점" if s.get("points") else
                                                          f" 회당 {s['per']}점" if s.get("per") else "")
                                        for s in it.get("subitems", []))
                                    comps.append(gr.Number(0, minimum=0, label=f"{lab} — 취득 점수",
                                                           info=it.get("note") or ("예: " + ex if ex else "비교과일람표의 인정 점수")))
                                    meta.append((it["id"], "float"))
                                elif t == "topcit_rule":
                                    comps.append(gr.Number(0, precision=0, minimum=0, label="TOPCIT 응시 횟수"))
                                    meta.append((it["id"], "topcit_att"))
                                    comps.append(gr.Dropdown([0, 1, 2, 3, 4, 5], value=0, label="TOPCIT 최고 수준"))
                                    meta.append((it["id"], "topcit_lv"))
                        btn = gr.Button("판정하기", variant="primary")

                        def run(course_text, *vals):
                            acts = {}
                            for (iid, kind), v in zip(meta, vals):
                                if kind == "bool":
                                    acts[iid] = bool(v)
                                elif kind == "int":
                                    acts[iid] = int(v or 0)
                                elif kind == "float":
                                    acts[iid] = float(v or 0)
                                elif kind == "list":
                                    acts[iid] = list(v or [])
                                elif isinstance(kind, tuple):
                                    acts.setdefault(iid, {"subitems": {}, "other": 0})["subitems"][kind[1]] = v
                                elif kind == "other":
                                    acts.setdefault(iid, {"subitems": {}, "other": 0})["other"] = float(v or 0)
                                elif kind == "topcit_att":
                                    acts.setdefault(iid, {})["attempts"] = int(v or 0)
                                elif kind == "topcit_lv":
                                    acts.setdefault(iid, {})["level"] = int(v or 0)
                            res = check_graduation(d, int(y), course_text or "", acts)
                            return _result_markdown(res), format_result_text(res)

                        btn.click(run, [courses] + comps, [result, check_state])
            gr.Markdown("판정 후 **Q&A 탭**에서 \"부족한 부분 어떻게 채워?\"라고 물으면 판정 결과를 참고해 답합니다.")
    return demo


if __name__ == "__main__":
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass
    build_demo().launch()
