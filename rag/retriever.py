"""
하이브리드 검색기
  - 희소(Sparse): 문자 n-gram TF-IDF  → 한국어 형태소 분석기 없이도 '학술제발표' / '학술제 발표' 같은 변형을 잡는다.
  - 밀집(Dense) : 문장 임베딩 + FAISS 벡터 인덱스
                  · 무료 공개 모델 intfloat/multilingual-e5-small (한국어 지원, API 키 불필요, rag/llm.py LocalEmbedder)
                  · 또는 OpenAI 호환 임베딩 API
                  → '졸업하려면 몇 학점?' 처럼 문서와 단어가 달라도 뜻이 비슷한 내용을 찾는다.
  - 두 점수는 질문마다 0~1 로 맞춘 뒤 반반 섞는다 (점수 범위가 서로 달라서).
  - 메타데이터 부스팅: 질문에서 학과·학번을 인식하면 해당 청크 점수를 올리고 다른 학과 청크는 낮춘다.
"""
from __future__ import annotations

import hashlib
import os
import pickle
import re
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from .checker import load_requirements, resolve_department, parse_year, _norm
from .ingest import Chunk, build_chunks, DOCS_DIR

INDEX_PATH = Path(__file__).resolve().parent.parent / "index" / "index.pkl"


def detect_profile(query: str) -> tuple[str | None, int | None]:
    """질문에서 학과/학번 추출. 예) 'AI빅데이터 22학번' → ('AI·빅데이터학과', 2022)"""
    req = load_requirements()
    dept = None
    q = _norm(query)
    best = 0
    for d, info in req["departments"].items():
        for a in [d] + info.get("aliases", []):
            na = _norm(a)
            if len(na) >= 2 and na in q and len(na) > best:
                dept, best = d, len(na)
    m = re.search(r"(20)?(\d{2})\s*학번", query)
    year = parse_year(m.group(0)) if m else None
    if year is None:
        m = re.search(r"\b(20[12]\d)\b", query)
        year = int(m.group(1)) if m else None
    return dept, year


class Retriever:
    def __init__(self, chunks: list[Chunk], embedder=None, dense=None):
        self.chunks = chunks
        self.embedder = embedder
        self.tfidf = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), sublinear_tf=True)
        self.sparse = self.tfidf.fit_transform([c.text for c in chunks])
        self.dense, self.faiss_index = None, None
        if embedder is not None:
            if dense is None:
                dense = np.array(embedder([c.text for c in chunks]), dtype=np.float32)
            self.dense = self._normalize(np.asarray(dense, dtype=np.float32))
            self.faiss_index = self._build_faiss(self.dense)

    @staticmethod
    def _build_faiss(dense):
        """FAISS 내적(=코사인, 정규화된 벡터) 인덱스. faiss 가 없으면 numpy 로 같은 계산을 한다."""
        try:
            import faiss
        except ImportError:
            return None
        index = faiss.IndexFlatIP(dense.shape[1])
        index.add(np.ascontiguousarray(dense, dtype=np.float32))
        return index

    @property
    def search_mode(self) -> str:
        if self.dense is None:
            return "TF-IDF(문자 n-gram)"
        name = getattr(self.embedder, "name", "임베딩 API")
        return f"TF-IDF + 임베딩({name}{', FAISS' if self.faiss_index is not None else ''})"

    def dense_scores(self, query: str):
        """질문 임베딩과 모든 청크의 코사인 유사도"""
        q = self.embedder.query(query) if hasattr(self.embedder, "query") else self.embedder([query])[0]
        qv = self._normalize(np.array(q, dtype=np.float32))
        if self.faiss_index is not None:
            D, I = self.faiss_index.search(qv[None, :], len(self.chunks))
            out = np.zeros(len(self.chunks), dtype=np.float32)
            out[I[0]] = D[0]
            return out
        return self.dense @ qv

    @staticmethod
    def _normalize(m):
        return m / (np.linalg.norm(m, axis=-1, keepdims=True) + 1e-9)

    @property
    def fingerprint(self) -> str:
        return hashlib.md5("".join(c.id for c in self.chunks).encode()).hexdigest()

    def search(self, query: str, k: int = 5, department: str | None = None, year: int | None = None):
        d2, y2 = detect_profile(query)
        # 질문에 학과·학번이 적혀 있으면 그것을 우선, 없으면 사이드바 프로필 사용
        if d2 and d2 != department:
            department, year = d2, (y2 or None)
        else:
            department, year = department or d2, y2 or year

        sparse = (self.sparse @ self.tfidf.transform([query]).T).toarray().ravel()
        score = sparse
        if self.dense is not None and self.embedder is not None:
            dense = self.dense_scores(query)
            # 임베딩 유사도(보통 0.7~0.9)와 TF-IDF(0~0.3)는 범위가 달라 질문마다 0~1 로 맞춘 뒤 섞고,
            # 기존 TF-IDF 크기로 되돌려 메타데이터 보정·임계값이 그대로 동작하게 한다
            dn = (dense - dense.min()) / (np.ptp(dense) + 1e-9)
            sn = sparse / (sparse.max() + 1e-9)
            score = (0.5 * dn + 0.5 * sn) * max(float(sparse.max()), 0.1)

        # 메타데이터 보정: 다른 학과/학번 자료는 감점, 같은 학과는 소폭 가점 (검색 점수를 뒤집지 않을 정도로)
        boost = np.zeros(len(self.chunks))
        for i, c in enumerate(self.chunks):
            if department and c.department:
                boost[i] += 0.04 if c.department == department else -0.08
            if year and c.years:
                boost[i] += 0.03 if c.years[0] <= year <= c.years[1] else -0.05
        final = score + boost

        # 학번이 인식되면 해당 학번의 대학공통 청크, 학과+학번이면 해당 학과 기준 청크를 반드시 포함
        pinned = []
        if year:
            for i, c in enumerate(self.chunks):
                if c.source and c.years and c.years[0] <= year <= c.years[1]:
                    if (c.department == "" ) or (department and c.department == department):
                        pinned.append(i)
        order = [i for i in np.argsort(-final) if i not in pinned]
        n_free = max(k - len(pinned), 2)
        free = order[:n_free]
        # 출처 다양성: 아직 결과에 없는 문서의 최고 점수 청크가 충분히 관련 있으면 포함
        if order:
            best = final[order[0]]
            shown = {self.chunks[i].source for i in pinned + free}
            for i in order[n_free:]:
                src = self.chunks[i].source
                if src not in shown and final[i] >= 0.6 * best:
                    free = free[:-1] + [i] if len(free) >= n_free else free + [i]
                    shown.add(src)
        top = sorted(pinned, key=lambda i: -final[i]) + sorted(free, key=lambda i: -final[i])
        return [(self.chunks[i], float(final[i])) for i in top if final[i] > 0 or i in pinned]


# ----------------------------------------------------------------------------
# 인덱스 빌드 / 캐시
# ----------------------------------------------------------------------------
def build_index(docs_dir: Path = DOCS_DIR, embedder=None, path: Path = INDEX_PATH, verbose=True) -> Retriever:
    chunks = build_chunks(docs_dir)
    r = Retriever(chunks, embedder)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump({"chunks": chunks, "dense": r.dense, "fp": r.fingerprint, "embed": _embed_name(embedder),
                     "docs": _docs_signature(docs_dir)}, f)
    if verbose:
        srcs = sorted({c.source for c in chunks})
        print(f"[index] {len(chunks)}개 청크, 검색: {r.search_mode}")
        print("[index] 문서:", ", ".join(srcs))
    return r


def _embed_name(embedder) -> str | None:
    return None if embedder is None else getattr(embedder, "name", "api")


def _docs_signature(docs_dir: Path) -> str:
    items = sorted((p.name, p.stat().st_mtime, p.stat().st_size) for p in docs_dir.glob("**/*") if p.is_file())
    data_file = Path(__file__).resolve().parent.parent / "data" / "requirements.json"
    items.append(("requirements.json", data_file.stat().st_mtime, data_file.stat().st_size))
    return hashlib.md5(repr(items).encode()).hexdigest()


def load_or_build(docs_dir: Path = DOCS_DIR, embedder=None, path: Path = INDEX_PATH) -> Retriever:
    """docs/ 나 requirements.json 이 바뀌었으면 자동 재빌드."""
    if path.exists():
        with open(path, "rb") as f:
            saved = pickle.load(f)
        same_docs = saved.get("docs") == _docs_signature(docs_dir)
        same_mode = saved.get("embed") == _embed_name(embedder) and (saved["dense"] is not None) == (embedder is not None)
        if same_docs and same_mode:
            return Retriever(saved["chunks"], embedder, dense=saved["dense"])
    return build_index(docs_dir, embedder, path, verbose=False)


def format_context(results) -> str:
    blocks = []
    for i, (c, s) in enumerate(results, 1):
        blocks.append(f"[문서 {i}] 출처: {c.source} / {c.location}\n{c.text}")
    return "\n\n".join(blocks)


if __name__ == "__main__":
    r = build_index()
    for q in ["AI빅데이터 22학번 졸업요건 알려줘", "TOPCIT 몇 번 봐야 해?", "비교과 이수내역 어디서 확인해?"]:
        print("\n### ", q)
        for c, s in r.search(q, k=3):
            print(f"  {s:.3f}  {c.source} / {c.location}")
