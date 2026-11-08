"""
Colab 챗봇 화면

웹 버전과 같은 화면(web/template.html + web/core.js)을 Colab 출력 칸에 띄우고,
'질문하기'의 AI 답변은 노트북 커널에서 실행 중인 무료 오픈소스 LLM(rag/llm.answer_stream)이 만든다.

  - 검색·졸업 판정·이수현황표는 화면(JavaScript)에서 즉시 계산 → 기다림 없음
  - AI 답변은 파이썬에서 생성하고, 화면이 0.3초마다 진행 상황을 받아 글자 단위로 표시
  - 연결: google.colab.kernel.invokeFunction ↔ google.colab.output.register_callback
"""
from __future__ import annotations

import html
import json
import threading
import uuid
from pathlib import Path

from .ingest import build_chunks
from .llm import answer_stream, model_label

ROOT = Path(__file__).resolve().parent.parent
JOBS: dict[str, dict] = {}


def build_page(root: Path = ROOT) -> str:
    """template.html 에 졸업요건 데이터·문서 청크·판정 엔진(core.js)을 넣어 한 장짜리 HTML 을 만든다."""
    t = (root / "web/template.html").read_text(encoding="utf-8")
    req = json.loads((root / "data/requirements.json").read_text(encoding="utf-8"))
    chunks = [{"id": c.id, "text": c.text, "source": c.source, "location": c.location,
               "department": c.department, "years": list(c.years)} for c in build_chunks(root / "docs")]
    esc = lambda s: s.replace("</", "<\\/")
    return (t.replace("/*REQ*/", esc(json.dumps(req, ensure_ascii=False)))
             .replace("/*CHUNKS*/", esc(json.dumps(chunks, ensure_ascii=False)))
             .replace("/*CORE*/", (root / "web/core.js").read_text(encoding="utf-8")))


def start_job(client, retriever, question, dept, year, history=None, check_text="") -> str:
    """AI 답변 생성을 백그라운드에서 시작하고 작업 id 를 돌려준다."""
    jid = uuid.uuid4().hex[:10]
    job = JOBS[jid] = {"text": "", "done": False, "error": None, "tools": []}

    def run():
        try:
            hist = [{"role": h.get("role"), "content": h.get("content", "")} for h in (history or [])
                    if isinstance(h, dict) and h.get("role") in ("user", "assistant")]
            for ev in answer_stream(question, retriever, client, hist,
                                    {"department": dept, "year": int(year)}, check_text or None):
                if isinstance(ev, str):
                    job["text"] = ev
                else:
                    job["text"] = ev.text
                    job["tools"] = [t[0] for t in ev.tool_calls]
        except Exception as e:  # 화면에 오류를 보여주기 위해 저장
            job["error"] = f"{type(e).__name__}: {e}"
        finally:
            job["done"] = True

    threading.Thread(target=run, daemon=True).start()
    return jid


def poll_job(jid: str) -> dict:
    job = JOBS.get(jid)
    if job is None:
        return {"done": True, "error": "알 수 없는 작업", "text": ""}
    out = dict(job)
    if job["done"]:
        JOBS.pop(jid, None)
    return out


def launch(client, retriever, height: int = 1000):
    """Colab 출력 칸에 챗봇 화면을 띄운다."""
    from google.colab import output
    from IPython.display import HTML, JSON, display

    label = None if client is None else model_label(client).replace("무료 오픈소스 LLM: ", "")
    output.register_callback("grad.info", lambda: JSON({"model": label}))
    output.register_callback("grad.start", lambda q, d, y, h=None, c="": JSON(
        {"job": start_job(client, retriever, q, d, y, h, c)}))
    output.register_callback("grad.poll", lambda jid: JSON(poll_job(jid)))
    page = build_page()
    display(HTML(
        f'<iframe srcdoc="{html.escape(page, quote=True)}" '
        f'style="width:100%;height:{height}px;border:1px solid #dbe0ea;border-radius:12px;background:#f3f4f7"></iframe>'
        f'<script>try{{google.colab.output.setIframeHeight({height + 30}, true)}}catch(e){{}}</script>'))
    print(f"챗봇 화면을 띄웠어요. AI: {label or '꺼짐 (검색 모드)'}")
