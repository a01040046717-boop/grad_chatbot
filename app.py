"""
다면적 졸업인증제 RAG 챗봇 (Streamlit)

실행:  streamlit run app.py
"""
from __future__ import annotations

import os
from pathlib import Path

import streamlit as st

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Streamlit Cloud 의 Secrets 값을 환경변수로 옮김 (OPENAI_API_KEY, OPENAI_BASE_URL, CHAT_MODEL, EMBED_MODEL 등)
try:
    for _k, _v in st.secrets.items():
        if isinstance(_v, (str, int, float)) and _k not in os.environ:
            os.environ[_k] = str(_v)
except Exception:  # secrets 파일이 없는 로컬 실행
    pass

from rag.checker import (load_requirements, get_requirement_items, check_graduation,
                         format_result_text, parse_course_list, _norm)
from rag.ingest import DOCS_DIR, SUPPORTED
from rag.retriever import load_or_build, build_index
from rag.llm import answer, get_client, make_embedder, model_label

st.set_page_config(page_title="SW융합대학 졸업요건 챗봇", page_icon="🎓", layout="wide")

REQ = load_requirements()
DEPTS = list(REQ["departments"])
DEFAULT_DEPT, DEFAULT_YEAR = "AI·빅데이터학과", 2022


# ----------------------------------------------------------------------------
# 리소스
# ----------------------------------------------------------------------------
@st.cache_resource(show_spinner="문서 인덱싱 중…")
def get_resources(_version: int):
    client = get_client()
    try:
        retriever = load_or_build(embedder=make_embedder(client))
    except Exception as e:  # 임베딩 API 오류 시 TF-IDF 만으로
        st.warning(f"임베딩 생성 실패 → TF-IDF 검색만 사용합니다. ({e})")
        retriever = load_or_build(embedder=None)
    return client, retriever


if "index_version" not in st.session_state:
    st.session_state.index_version = 0
client, retriever = get_resources(st.session_state.index_version)

# ----------------------------------------------------------------------------
# 사이드바: 프로필 + 문서 관리
# ----------------------------------------------------------------------------
with st.sidebar:
    st.header("👤 내 정보")
    dept = st.selectbox("학과", DEPTS, index=DEPTS.index(DEFAULT_DEPT))
    year = st.number_input("학번(입학년도)", min_value=2016, max_value=2026, value=DEFAULT_YEAR, step=1)
    profile = {"department": dept, "year": int(year)}

    st.divider()
    st.header("⚙️ 상태")
    if client:
        st.success(f"LLM 연결됨 — {model_label(client)}")
    else:
        st.warning("OPENAI_API_KEY 없음 → 오프라인 모드\n(검색 결과만 표시)")
    st.caption(f"검색: {'임베딩 + ' if retriever.dense is not None else ''}TF-IDF 하이브리드 · 청크 {len(retriever.chunks)}개")

    st.divider()
    st.header("📚 참고 문서")
    sources = sorted({c.source for c in retriever.chunks})
    for s in sources:
        st.markdown(f"- {s}")
    ups = st.file_uploader("안내 문서 추가 (PDF/TXT/MD/DOCX)", type=[e[1:] for e in SUPPORTED],
                           accept_multiple_files=True)
    if ups and st.button("추가하고 인덱스 다시 만들기", type="primary"):
        for f in ups:
            (DOCS_DIR / f.name).write_bytes(f.getbuffer())
        build_index(embedder=make_embedder(client), verbose=False)
        st.session_state.index_version += 1
        st.rerun()

st.title("🎓 SW융합대학 다면적 졸업인증제 챗봇")
tab_chat, tab_check = st.tabs(["💬 졸업요건 Q&A", "✅ 졸업 자가진단"])

# ----------------------------------------------------------------------------
# 탭 2: 자가진단 (먼저 정의해서 결과를 챗봇이 참조할 수 있게)
# ----------------------------------------------------------------------------
with tab_check:
    spec = get_requirement_items(dept, int(year), REQ)
    if "error" in spec:
        st.error(spec["error"])
    else:
        st.subheader(f"{dept} {year}학번 · 기준 {spec['total']}점")
        st.caption(" / ".join(
            f"{g} {spec['groups'][g]:g}점 (" + ", ".join(f"{k.split('·')[1]} {v:g}" for k, v in spec["sections"].items()
                                                  if k.startswith(g)) + ")" for g in ("대학공통", "학과기준")))
        st.caption("인재상별: " + " · ".join(
            f"{c} {sum(i['points'] for i in spec['items'] if i['category'] == c):g}점"
            for c in ["소통하는 공감인재", "융합하는 창의인재", "공헌하는 글로벌인재"]))
        for w in spec["warnings"]:
            st.warning(w)

        kp = f"{dept}_{year}_"  # 학과·학번이 바뀌면 위젯 초기화
        col_l, col_r = st.columns([1, 1.4], gap="large")

        with col_l:
            st.markdown("#### 📘 수강한 교과목 리스트")
            course_text = st.text_area(
                "한 줄에 한 과목씩 과목명을 적어 주세요 (쉼표로 구분해도 됩니다).",
                height=260, key=kp + "courses",
                placeholder="예)\n인간사랑\n현대기본영어\n봉사학습\n예비창업설계\n자기개발과 전공탐색")
            parsed = parse_course_list(course_text)
            if parsed:
                st.caption("인식된 과목: " + ", ".join(parsed))

        activities = {}
        with col_r:
            st.markdown("#### 🏃 비교과 활동 / 이수 항목 체크")
            for sec, sec_pts in spec["sections"].items():
                items = [i for i in spec["items"] if i["section"] == sec]
                with st.expander(f"{sec} ({sec_pts:g}점)", expanded=True):
                    for it in items:
                        key = kp + it["id"]
                        label = f"{it['name']} ({it['points']:g}점)"
                        t = it["type"]
                        if t == "course":
                            auto = any(_norm(m) in parsed for m in it.get("match", []))
                            v = st.checkbox(label + ("  ✅ 리스트에서 인식" if auto else ""), value=auto,
                                            key=key + ("_a" if auto else ""), disabled=auto,
                                            help="수강 리스트에 있으면 자동 체크됩니다. 이름이 달라 인식 안 되면 직접 체크하세요.")
                            activities[it["id"]] = v
                        elif t == "check":
                            activities[it["id"]] = st.checkbox(label, key=key, help=it.get("note"))
                        elif t == "all_of":
                            activities[it["id"]] = st.multiselect(f"{label} — 이수한 학기 선택 (모두 필요)",
                                                                  it["parts"], key=key, help=it.get("note"))
                        elif t == "count":
                            need = -(-it["points"] // it["per"])
                            activities[it["id"]] = st.number_input(
                                f"{label} — 참여 횟수 (회당 {it['per']}점, {need}회 필요)",
                                min_value=0, max_value=50, value=0, key=key)
                        elif t == "choose":
                            opts = it.get("options") or []
                            if opts:
                                extra = st.multiselect(
                                    f"{label} — {it['n']}과목 필요 (리스트에 없으면 선택)",
                                    opts + it.get("substitutes", []), key=key)
                            else:
                                st.caption(f"{label} — 수강 리스트에서 '{it['pattern']}' 포함 과목을 셉니다.")
                                extra = []
                            activities[it["id"]] = extra
                        elif t == "points":
                            st.markdown(f"**{label}**" + (f"  \n<small>{it['note']}</small>" if it.get("note") else ""),
                                        unsafe_allow_html=True)
                            subs = {}
                            for s in it.get("subitems", []):
                                sk = key + "_" + s["name"]
                                if s["mode"] == "check":
                                    subs[s["name"]] = st.checkbox(f"· {s['name']} (+{s['points']}점)", key=sk)
                                elif s["mode"] == "select":
                                    subs[s["name"]] = st.selectbox(f"· {s['name']}", list(s["options"]), key=sk)
                                elif s["mode"] == "count":
                                    subs[s["name"]] = st.number_input(f"· {s['name']} 횟수 (회당 {s['per']}점)",
                                                                      0, 50, 0, key=sk)
                                else:
                                    subs[s["name"]] = st.number_input(f"· {s['name']}", 0.0, 100.0, 0.0,
                                                                      step=1.0, key=sk)
                            other = st.number_input("· 취득 점수" if not it.get("subitems") else "· 기타 비교과 인정 점수 (비교과일람표 기준)", 0.0, 100.0, 0.0,
                                                    step=1.0, key=key + "_other")
                            activities[it["id"]] = {"subitems": subs, "other": other}
                        elif t == "topcit_rule":
                            c1, c2 = st.columns(2)
                            att = c1.number_input(f"{label} — 응시 횟수", 0, 20, 0, key=key + "_att")
                            lv = c2.selectbox("최고 달성 수준", [0, 1, 2, 3, 4, 5], key=key + "_lv")
                            activities[it["id"]] = {"attempts": att, "level": lv}

        st.divider()
        res = check_graduation(dept, int(year), course_text, activities, REQ)
        st.session_state.check_text = format_result_text(res)

        if isinstance(res, dict):
            st.error(res["error"])
        else:
            m1, m2, m3 = st.columns(3)
            m1.metric("총점", f"{res.total_earned:g} / {res.total_required:g}")
            m2.metric("판정", "충족 ✅" if res.passed else "미충족 ❌")
            m3.metric("부족 항목", f"{len(res.missing)}개")
            st.markdown("**다면적 졸업인증 이수현황** (학교 시스템 이수현황표와 같은 형식)")
            cols = [(g, c) for g in res.matrix for c in res.matrix[g]]
            hdr = [f"{g}·{c}" for g, c in cols] + ["졸업점수"]
            need = [res.matrix[g][c]["required"] for g, c in cols] + [res.total_required]
            got = [res.matrix[g][c]["earned"] for g, c in cols] + [res.total_earned]
            st.dataframe({"구분": ["필요점수 ⓐ", "이수점수 ⓑ", "미이수점수 ⓐ-ⓑ"],
                          **{h: [n, e, max(n - e, 0)] for h, n, e in zip(hdr, need, got)}},
                         use_container_width=True, hide_index=True)
            col_k, col_c = st.columns(2)
            with col_k:
                st.markdown("**구분별**")
                for g, gv in res.by_group.items():
                    ratio = gv["earned"] / gv["required"] if gv["required"] else 1
                    st.progress(min(ratio, 1.0), text=f"**{g}**: {gv['earned']:g} / {gv['required']:g}")
                    for k, v in res.by_kind.items():
                        if k.startswith(g):
                            r_ = v["earned"] / v["required"] if v["required"] else 1
                            st.progress(min(r_, 1.0), text=f"　└ {k.split('·')[1]}: {v['earned']:g} / {v['required']:g}")
            with col_c:
                st.markdown("**인재상별**")
                for k, v in res.by_category.items():
                    ratio = v["earned"] / v["required"] if v["required"] else 1
                    st.progress(min(ratio, 1.0), text=f"{k}: {v['earned']:g} / {v['required']:g}")

            if res.missing:
                st.markdown("#### ❗ 부족한 항목")
                st.dataframe(
                    [{"구분": m["section"], "인재상": m["category"], "항목": m["name"],
                      "부족 점수": m["need"], "현재 상태": m["detail"]} for m in res.missing],
                    use_container_width=True, hide_index=True)
            else:
                st.balloons()
                st.success("모든 항목을 충족했습니다! 공식 확인은 종합정보시스템 → 비교과일람표에서 하세요.")
            with st.expander("전체 항목 상세"):
                st.dataframe([{"구분": i["section"], "인재상": i["category"], "항목": i["name"], "필요": i["required"], "취득": i["earned"],
                               "충족": "✅" if i["passed"] else "❌", "근거": i["detail"]} for i in res.items],
                             use_container_width=True, hide_index=True)
            if res.unmatched_courses:
                st.caption("ℹ️ 졸업인증 항목과 매칭되지 않은 과목: " + ", ".join(res.unmatched_courses))
            for n in res.notes:
                st.caption("※ " + n)
            if st.button("💬 이 결과로 챗봇에게 '뭘 더 해야 하는지' 물어보기"):
                st.session_state.pending_q = "자가진단 결과에서 부족한 항목을 어떻게 채우면 되는지 우선순위대로 알려줘."
                st.toast("Q&A 탭에서 답변을 확인하세요!")

# ----------------------------------------------------------------------------
# 탭 1: 챗봇
# ----------------------------------------------------------------------------
with tab_chat:
    if "messages" not in st.session_state:
        st.session_state.messages = []

    if not st.session_state.messages:
        st.info(f"현재 프로필: **{dept} {year}학번** (사이드바에서 변경)\n\n"
                "수강 과목·활동을 말로 설명하면 졸업 충족 여부도 계산해 드려요.")
        examples = [
            "내 학번 졸업인증 기준 전체 알려줘",
            "학과활동은 몇 번 참여해야 해?",
            "AI·SW전공역량졸업인증제 200점은 어떻게 채워?",
            "인간사랑, 현대기본영어, 봉사학습, 예비창업설계 들었고 학과활동 3번, SQLD 땄어. 졸업 가능해?",
        ]
        cols = st.columns(len(examples))
        for c, ex in zip(cols, examples):
            if c.button(ex, use_container_width=True):
                st.session_state.pending_q = ex

    for m in st.session_state.messages:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])
            if m.get("sources"):
                with st.expander("📎 근거 문서"):
                    for s, loc, sc in m["sources"]:
                        st.markdown(f"- {s} / {loc} (score {sc})")
            if m.get("tools"):
                with st.expander("🛠️ 도구 호출 기록"):
                    for name, args, out in m["tools"]:
                        st.markdown(f"**{name}** `{args}`")
                        st.code(out)

    q = st.chat_input("졸업요건에 대해 물어보세요")
    q = q or st.session_state.pop("pending_q", None)
    if q:
        st.session_state.messages.append({"role": "user", "content": q})
        with st.chat_message("user"):
            st.markdown(q)
        with st.chat_message("assistant"):
            with st.spinner("문서 검색 및 답변 생성 중…"):
                history = [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages[:-1]]
                try:
                    a = answer(q, retriever, client, history, profile, st.session_state.get("check_text"))
                except Exception as e:
                    from rag.llm import Answer
                    a = Answer(f"⚠️ 답변 생성 중 오류가 발생했습니다: {e}")
            st.markdown(a.text)
            if a.sources:
                with st.expander("📎 근거 문서"):
                    for s, loc, sc in a.sources:
                        st.markdown(f"- {s} / {loc} (score {sc})")
            if a.tool_calls:
                with st.expander("🛠️ 도구 호출 기록"):
                    for name, args, out in a.tool_calls:
                        st.markdown(f"**{name}** `{args}`")
                        st.code(out)
        st.session_state.messages.append({"role": "assistant", "content": a.text,
                                          "sources": a.sources, "tools": a.tool_calls})

    if st.session_state.messages and st.button("대화 초기화"):
        st.session_state.messages = []
        st.rerun()
