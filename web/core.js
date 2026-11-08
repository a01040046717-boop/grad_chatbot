/* ===== 졸업요건 RAG 코어 (Python rag/ 모듈의 브라우저 이식) ===== */
const GROUPS = ["대학공통", "학과기준"];
const SECTIONS = [["대학공통", "교과"], ["대학공통", "비교과"], ["학과기준", "교과"], ["학과기준", "졸업필수활동"], ["학과기준", "비교과지정"], ["학과기준", "비교과미지정"]];
const KINDS = SECTIONS.map(([g, k]) => `${g}·${k}`);
const CATEGORIES = ["소통하는 공감인재", "융합하는 창의인재", "공헌하는 글로벌인재"];

function norm(s) {
  s = String(s ?? "");
  const map = { "Ⅰ": "1", "Ⅱ": "2", "Ⅲ": "3", "Ⅳ": "4", "ㆍ": "", "·": "", "•": "" };
  s = s.replace(/[ⅠⅡⅢⅣㆍ·•]/g, (m) => map[m]);
  s = s.replace(/[\s\-_,./:;'"`~!?·&]/g, "");
  return s.toLowerCase();
}

function normalizeCourse(raw) {
  let s = String(raw).trim();
  s = s.replace(/^\s*[-*•\d]+[.)]\s*/, "");
  s = s.replace(/\[[^\]]*\]/g, "");
  s = s.replace(/\([^)]*\)/g, "");
  s = s.replace(/\s+(A\+|A0|A|B\+|B0|B|C\+|C0|C|D\+|D0|D|F|P|NP|S|U)\s*$/i, "");
  s = s.replace(/\s*\d+(\.\d+)?\s*학점\s*/g, "");
  return norm(s);
}

function parseCourseList(input) {
  const parts = Array.isArray(input) ? input : String(input || "").split(/[\n,，;]+/);
  const out = [];
  for (const p of parts) { const n = normalizeCourse(p); if (n && !out.includes(n)) out.push(n); }
  return out;
}

function resolveDepartment(name, REQ) {
  const q = norm(name);
  if (!q) return null;
  const depts = Object.entries(REQ.departments);
  for (const [d, info] of depts) if ([d, ...(info.aliases || [])].some((c) => norm(c) === q)) return d;
  for (const [d, info] of depts)
    if ([d, ...(info.aliases || [])].some((c) => norm(c).length >= 2 && (q.includes(norm(c)) || norm(c).includes(q)))) return d;
  return null;
}

function parseYear(v) {
  const m = String(v ?? "").match(/\d{2,4}/);
  if (!m) return null;
  let y = parseInt(m[0], 10);
  if (y < 100) y += 2000;
  return y;
}

const yr = (a, b) => (a === b ? `${a}` : `${a}~${b}`);

function getRequirementItems(dept, year, REQ) {
  const track = REQ.departments[dept].tracks.find((t) => t.years[0] <= year && year <= t.years[1]);
  if (!track) {
    const avail = REQ.departments[dept].tracks.map((t) => yr(t.years[0], t.years[1])).join(", ");
    return { error: `${dept} ${year}학번 기준은 안내문에 없습니다. (안내된 학번: ${avail})` };
  }
  let common = REQ.common.find((c) => c.years[0] <= year && year <= c.years[1]);
  const warnings = [];
  if (!common) {
    common = REQ.common[0];
    warnings.push(`${year}학번의 대학공통 세부 항목은 안내문에 없어 2019~2022 기준으로 계산했습니다. 多Dream비교과센터에 확인하세요.`);
  }
  const info = REQ.departments[dept];
  // 구분 2단계: 대학공통(교과·비교과) / 학과기준(교과·졸업필수활동·비교과지정·비교과미지정)
  const ov = track.common_overrides || {};
  const items = [
    ...common.items.map((it) => ({ ...it, ...(it.group === "학과기준" ? {} : { group: "대학공통", kind: it.type === "course" ? "교과" : "비교과" }), ...(ov[it.id] || {}) })),
    ...track.items.map((it) => ({ ...it, group: "학과기준" })),
  ].map((it) => ({ ...it, section: `${it.group}·${it.kind}` }));
  const groups = Object.fromEntries(GROUPS.map((g) => [g, items.filter((i) => i.group === g).reduce((a, i) => a + i.points, 0)]));
  const sections = {};
  for (const sec of KINDS) { const v = items.filter((i) => i.section === sec).reduce((a, i) => a + i.points, 0); if (v) sections[sec] = v; }
  return { department: dept, year, trackYears: track.years, total: track.total, subtotals: track.subtotals, groups, sections,
    items, unitRules: info.unit_rules || "", notes: info.notes || [], warnings };
}

function evalItem(it, courses, act, used) {
  const pts = it.points;
  switch (it.type) {
    case "course": {
      const names = (it.match || [it.name]).map(norm);
      const hit = courses.find((c) => names.includes(c));
      if (hit) { used.add(hit); return [pts, "수강 리스트에서 확인"]; }
      return act ? [pts, "직접 체크"] : [0, "미이수"];
    }
    case "check": return act ? [pts, "체크됨"] : [0, "미이수"];
    case "all_of": {
      const parts = it.parts;
      const done = act === true ? [...parts] : Array.isArray(act) ? parts.filter((p) => act.includes(p))
        : act && typeof act === "object" ? parts.filter((p) => act[p]) : [];
      return [done.length === parts.length ? pts : 0, `${done.join(", ") || "없음"} 이수 (${parts.join(", ")} 모두 필요)`];
    }
    case "count": {
      const n = parseInt(act || 0, 10) || 0;
      const need = Math.ceil(pts / it.per);
      return [Math.min(n * it.per, pts), `${n}${it.unit || "회"} 참여 (회당 ${it.per}점, 필요 ${need}${it.unit || "회"})`];
    }
    case "choose": {
      let done;
      if (it.pattern) { const p = norm(it.pattern); done = courses.filter((c) => c.includes(p)); }
      else { const opts = [...it.options, ...(it.substitutes || [])].map(norm); done = courses.filter((c) => opts.includes(c)); }
      const extra = Array.isArray(act) ? act.map(normalizeCourse) : [];
      done = [...new Set([...done, ...extra])];
      done.forEach((c) => used.add(c));
      const k = Math.min(done.length, it.n);
      return [k * (pts / it.n), `${done.length}과목 확인 (${done.join(", ") || "없음"}) / ${it.n}과목 필요`];
    }
    case "points": {
      if (typeof act === "number") return [Math.min(act, pts), `입력 점수 ${act}점`];
      const a = act || {}, sv = a.subitems || {};
      let total = 0; const parts = [];
      for (const s of it.subitems || []) {
        const v = sv[s.name];
        if (!v) continue;
        const p = s.mode === "check" ? s.points
          : s.mode === "select" ? (typeof v === "number" ? v : s.options[v] || 0)
          : s.mode === "count" ? (parseInt(v, 10) || 0) * s.per : parseFloat(v) || 0;
        total += p; parts.push(`${s.name} ${p}점`);
      }
      const other = parseFloat(a.other || 0) || 0;
      if (other) { total += other; parts.push(`${(it.subitems || []).length ? "기타" : "취득"} ${other}점`); }
      return [Math.min(total, pts), (parts.join(", ") || "인정 점수 없음") + ` (합 ${total}점)`];
    }
    case "topcit_rule": {
      const a = act || {}, att = parseInt(a.attempts || 0, 10) || 0, lv = parseInt(a.level || 0, 10) || 0;
      const ok = att >= 5 || (att >= 3 && lv >= 2) || (att >= 1 && lv >= 3) || lv >= 4;
      return [ok ? pts : 0, `응시 ${att}회, 최고 Lv${lv} (기준: 응시5회 / 3회+Lv2 / 1회+Lv3 / Lv4)`];
    }
  }
  throw new Error("unknown type " + it.type);
}

function checkGraduation(department, year, courses, activities, REQ) {
  const dept = resolveDepartment(department, REQ);
  if (!dept) return { error: `'${department}' 학과를 찾지 못했습니다. 가능: ${Object.keys(REQ.departments).join(", ")}` };
  const y = parseYear(year);
  if (!y) return { error: `학번 '${year}'을(를) 해석하지 못했습니다.` };
  const spec = getRequirementItems(dept, y, REQ);
  if (spec.error) return spec;
  const list = parseCourseList(courses || []);
  activities = activities || {};
  const used = new Set();
  const items = spec.items.map((it) => {
    const [earned, detail] = evalItem(it, list, activities[it.id], used);
    return { id: it.id, name: it.name, category: it.category, group: it.group, kind: it.kind, section: it.section, required: it.points,
      earned: Math.round(earned * 100) / 100, passed: earned >= it.points - 1e-9, detail };
  });
  const sum = (arr, f) => Math.round(arr.reduce((a, r) => a + r[f], 0) * 100) / 100;
  const byGroup = Object.fromEntries(GROUPS.map((g) => { const rs = items.filter((r) => r.group === g); return [g, { required: sum(rs, "required"), earned: sum(rs, "earned") }]; }));
  const byKind = {};
  for (const sec of Object.keys(spec.sections)) { const rs = items.filter((r) => r.section === sec); byKind[sec] = { required: sum(rs, "required"), earned: sum(rs, "earned") }; }
  const byCategory = {};
  for (const c of CATEGORIES) byCategory[c] = {
    required: items.filter((r) => r.category === c).reduce((a, r) => a + r.required, 0),
    earned: Math.round(items.filter((r) => r.category === c).reduce((a, r) => a + r.earned, 0) * 100) / 100 };
  // 학교 시스템 '다면적졸업인증 이수현황' 표: 구분(대학공통/학과기준) × 인재상(공감/창의/글로벌)
  const SHORT = { "소통하는 공감인재": "공감", "융합하는 창의인재": "창의", "공헌하는 글로벌인재": "글로벌" };
  const matrix = Object.fromEntries(GROUPS.map((g) => [g, Object.fromEntries(CATEGORIES.map((c) => {
    const rs = items.filter((r) => r.group === g && r.category === c);
    return [SHORT[c], { required: sum(rs, "required"), earned: sum(rs, "earned") }]; }))]));
  const totalReq = items.reduce((a, r) => a + r.required, 0);
  const totalEarned = Math.round(items.reduce((a, r) => a + r.earned, 0) * 100) / 100;
  return { department: dept, year: y, totalRequired: totalReq, totalEarned, passed: items.every((r) => r.passed),
    byGroup, byKind, byCategory, matrix, items,
    missing: items.filter((r) => !r.passed).map((r) => ({ name: r.name, kind: r.kind, group: r.group, section: r.section, category: r.category,
      need: Math.round((r.required - r.earned) * 100) / 100, detail: r.detail })),
    unmatchedCourses: list.filter((c) => !used.has(c)), warnings: spec.warnings,
    notes: [...spec.notes, ...(spec.unitRules ? [spec.unitRules] : [])] };
}

function formatResultText(r) {
  if (r.error) return "⚠️ " + r.error;
  const L = [`[${r.department} ${r.year}학번] 다면적 졸업인증제 판정`,
    `총점 ${r.totalEarned} / ${r.totalRequired}  →  ${r.passed ? "✅ 충족" : "❌ 미충족"}`,
    "구분별: " + GROUPS.map((g) => `${g} ${r.byGroup[g].earned}/${r.byGroup[g].required} (` +
      Object.entries(r.byKind).filter(([k]) => k.startsWith(g)).map(([k, v]) => `${k.split("·")[1]} ${v.earned}/${v.required}`).join(", ") + ")").join(" / "),
    "인재상별: " + CATEGORIES.map((c) => `${c} ${r.byCategory[c].earned}/${r.byCategory[c].required}`).join(", "),
    "이수현황표(이수/필요): " + GROUPS.map((g) => `${g} ` + Object.entries(r.matrix[g]).map(([c, v]) => `${c} ${v.earned}/${v.required}`).join(", ")).join(" / ")];
  if (r.missing.length) { L.push("부족한 항목:"); r.missing.forEach((m) => L.push(`  - [${m.section}/${m.category}] ${m.name}: ${m.need}점 부족 (${m.detail})`)); }
  if (r.unmatchedCourses.length) L.push("졸업인증 항목과 무관한 수강 과목: " + r.unmatchedCourses.join(", "));
  r.warnings.forEach((w) => L.push("※ " + w));
  return L.join("\n");
}

function compactRequirements(department, year, REQ) {
  const d = resolveDepartment(department, REQ);
  if (!d) return { error: `학과 '${department}'를 찾지 못함. 가능: ${Object.keys(REQ.departments).join(", ")}` };
  const spec = getRequirementItems(d, parseYear(year), REQ);
  if (spec.error) return spec;
  return { department: d, year: spec.year, total: spec.total, groups: spec.groups, sections: spec.sections, unit_rules: spec.unitRules, notes: spec.notes,
    items: spec.items.map((it) => {
      const x = { id: it.id, name: it.name, section: it.section, category: it.category, type: it.type, points: it.points };
      if (it.type === "count") x.per = it.per;
      if (it.type === "choose") { x.options = it.options || `'${it.pattern}' 포함 과목`; x.n = it.n; }
      if (it.type === "all_of") x.parts = it.parts;
      if (it.type === "points" && it.subitems?.length) x.subitems = it.subitems.map((s) => {
        const o = { name: s.name, mode: s.mode };
        if ("points" in s) o.points = s.points; if ("per" in s) o.per = s.per; if (s.options) o.options = s.options;
        return o; });
      return x;
    }) };
}

/* ---------- 검색: 문자 n-gram(2~4, 단어 경계) TF-IDF + 학과/학번 부스팅 ---------- */
function charWbNgrams(text) {
  const grams = new Map();
  for (const w of String(text).toLowerCase().split(/\s+/).filter(Boolean)) {
    const p = ` ${w} `;
    for (let n = 2; n <= 4; n++) {
      if (p.length < n) break;
      for (let i = 0; i + n <= p.length; i++) { const g = p.slice(i, i + n); grams.set(g, (grams.get(g) || 0) + 1); }
    }
  }
  return grams;
}

function buildIndex(chunks) {
  const N = chunks.length, df = new Map();
  const tfs = chunks.map((c) => { const g = charWbNgrams(c.text); for (const k of g.keys()) df.set(k, (df.get(k) || 0) + 1); return g; });
  const idf = new Map(); for (const [k, v] of df) idf.set(k, Math.log((1 + N) / (1 + v)) + 1);
  const vec = (g) => { const v = new Map(); let s = 0;
    for (const [k, tf] of g) { const w = idf.get(k); if (!w) continue; const x = (1 + Math.log(tf)) * w; v.set(k, x); s += x * x; }
    s = Math.sqrt(s) || 1; for (const k of v.keys()) v.set(k, v.get(k) / s); return v; };
  return { chunks, vecs: tfs.map(vec), vec };
}

function detectProfile(q, REQ) {
  let dept = null, best = 0; const nq = norm(q);
  for (const [d, info] of Object.entries(REQ.departments))
    for (const a of [d, ...(info.aliases || [])]) { const na = norm(a); if (na.length >= 2 && nq.includes(na) && na.length > best) { dept = d; best = na.length; } }
  let m = q.match(/(20)?(\d{2})\s*학번/), year = m ? parseYear(m[0]) : null;
  if (!year) { m = q.match(/\b(20[12]\d)\b/); year = m ? parseInt(m[1], 10) : null; }
  return [dept, year];
}

function search(index, q, k, department, year, REQ) {
  const [d2, y2] = detectProfile(q, REQ);
  if (d2 && d2 !== department) { department = d2; year = y2 || null; }   // 질문에 적힌 학과·학번 우선
  else { department = department || d2; year = y2 || year; }
  const qv = index.vec(charWbNgrams(q));
  const scores = index.chunks.map((c, i) => {
    let s = 0; const dv = index.vecs[i];
    for (const [g, w] of qv) { const x = dv.get(g); if (x) s += x * w; }
    if (department && c.department) s += c.department === department ? 0.04 : -0.08;
    if (year && c.years.length) s += c.years[0] <= year && year <= c.years[1] ? 0.03 : -0.05;
    return s;
  });
  const pinned = [];
  if (year) index.chunks.forEach((c, i) => {
    if (c.years.length && c.years[0] <= year && year <= c.years[1] && (c.department === "" || (department && c.department === department))) pinned.push(i);
  });
  const order = scores.map((s, i) => i).sort((a, b) => scores[b] - scores[a]).filter((i) => !pinned.includes(i));
  const nFree = Math.max(k - pinned.length, 2);
  let free = order.slice(0, nFree);
  if (order.length) {   // 출처 다양성: 결과에 없는 문서의 최고 청크가 충분히 관련 있으면 포함
    const best = scores[order[0]];
    const shown = new Set([...pinned, ...free].map((i) => index.chunks[i].source));
    for (const i of order.slice(nFree)) {
      const src = index.chunks[i].source;
      if (!shown.has(src) && scores[i] >= 0.6 * best) { free = free.length >= nFree ? [...free.slice(0, -1), i] : [...free, i]; shown.add(src); }
    }
  }
  const byScore = (a, b) => scores[b] - scores[a];
  const top = [...pinned.sort(byScore), ...free.sort(byScore)];
  return top.filter((i) => scores[i] > 0 || pinned.includes(i)).map((i) => ({ chunk: index.chunks[i], score: scores[i] }));
}

function formatContext(hits) {
  return hits.map((h, i) => `[문서 ${i + 1}] 출처: ${h.chunk.source} / ${h.chunk.location}\n${h.chunk.text}`).join("\n\n");
}

/* ---------- 문장 → 판정 입력 자동 인식 (Python rag/extract.py 이식) ---------- */
const KNUM = { "한": 1, "두": 2, "세": 3, "네": 4, "다섯": 5, "여섯": 6, "일곱": 7, "여덟": 8, "아홉": 9, "열": 10 };
const NUMP = "(\\d+|다섯|여섯|일곱|여덟|아홉|한|두|세|네|열)";
const NEGP = /^.{0,6}?(안\s*했|안\s*함|안\s*들|안\s*봤|안\s*해|못|아직|없|안\s*땄|안\s*받|불합격|떨어|빼고|제외|말고)/;
const CHECK_INTENT = /졸업\s*(가능|할\s*수|돼|되)|충족|부족|모자|남은|남았|(내|지금|현재)\s*(점수|몇\s*점)|판정|채웠|채워|계산/;
const CHECK_KEYS = [["코딩역량", ["코딩역량", "코딩인증"]], ["AI·SW전공역량", ["ai·sw전공역량", "aisw전공역량", "aisw역량", "전공역량졸업인증", "aisw"]],
  ["학술제", ["학술제"]], ["중핵교양", ["중핵교양", "국제사회언어"]], ["학습윤리-기본", ["학습윤리기본", "학습윤리-기본"]], ["학습윤리-심화", ["학습윤리심화", "학습윤리-심화"]],
  ["국제박람회", ["국제박람회", "박람회"]], ["진로(취업)지도", ["진로지도", "취업지도"]], ["간담회", ["간담회"]]];
const COUNT_KEYS = [["자기주도상담", ["자기주도상담"]], ["지도교수상담", ["지도교수상담", "교수상담"]], ["졸업상담", ["졸업상담"]],
  ["진로 및 취업상담", ["진로및취업상담", "취업상담", "진로상담"]], ["전공상담", ["전공상담"]], ["학과활동", ["학과활동", "학과행사"]],
  ["MT", ["mt", "엠티"]], ["TOPCIT", ["topcit", "탑싯"]]];
const GOT = "(취득|땄|따|합격|획득|붙|있)", TOOK = "(응시|봤|보|쳤|치)";
// 과목 줄임말 → 정식 과목명 (키는 norm 값)
const COURSE_ALIASES = { "자기개발과전공탐색": ["전공탐색"], "자기개발과전공탐색mm": ["전공탐색"], "예비창업설계": ["예창"] };
const FULL_RE = /^.{0,8}?((다|모두|전부)\s*(했|채웠|채움|참여|받았|완료|끝)|완료|만점)/;   // '자기주도상담 다 했어'
const EXCEPT_ALL = /(빼고|제외하고|말고)\s*(나머지\s*)?(다|모두|전부)/;                    // '봉사학습 빼고 다 들었어'
const CONTEST_JOIN = /(경진대회|공모전|해커톤)[^.,\n]{0,12}?(참가|나갔|출전|참여|나감)/;
const CONTEST_WIN = /(경진대회|공모전|해커톤)[^.,\n]{0,12}?(수상|입상|상\s*받|상을\s*받)/;
const knum = (s) => (/^\d+$/.test(s) ? parseInt(s, 10) : KNUM[s] || 0);
const pat = (w) => [...w].filter((c) => !/\s/.test(c)).map((c) => c.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")).join("\\s*");
function findCount(text, words, units = "(번|회|차례|학기)") {
  for (const w of words) { const m = text.match(new RegExp(pat(w) + "[^0-9\\n.,]{0,10}?" + NUMP + "\\s*" + units, "i")); if (m) return knum(m[1]); }
  return null;
}
function full(text, words) {
  const low = text.toLowerCase();
  for (const w of words) { const re = new RegExp(pat(w), "gi"); let m;
    while ((m = re.exec(low))) { const after = low.slice(m.index + m[0].length);
      if (FULL_RE.test(after.slice(0, 14)) && !NEGP.test(after.slice(0, 12))) return true; } }
  return false;
}
function mentioned(text, words) {
  const low = text.toLowerCase();
  for (const w of words) { const re = new RegExp(pat(w), "gi"); let m;
    while ((m = re.exec(low))) { if (!NEGP.test(low.slice(m.index + m[0].length, m.index + m[0].length + 12))) return true; } }
  return false;
}
function extractInputs(text, dept, year, REQ) {
  const spec = getRequirementItems(dept, year, REQ);
  if (spec.error) return { courses: [], activities: {}, recognized: [] };
  const courses = [], acts = {}, recog = [];
  const names = new Set();
  for (const it of spec.items) {
    if (it.type === "course") (it.match || [it.name]).forEach((n) => names.add(n));
    if (it.type === "choose") [...(it.options || []), ...(it.substitutes || [])].forEach((n) => names.add(n));
  }
  let rest = norm(text);
  for (const n of [...names].sort((a, b) => norm(b).length - norm(a).length || (a < b ? -1 : a > b ? 1 : 0))) {
    const k = norm(n);
    if (k && rest.includes(k)) { rest = rest.split(k).join("■"); if (negated(text, n)) continue;   // '봉사학습은 안 들었어'
      courses.push(n); recog.push(`과목 ${n}`); }
  }
  for (const n of [...names].sort()) for (const a of COURSE_ALIASES[norm(n)] || []) {   // 줄임말 ('전공탐색', '예창')
    if (!courses.includes(n) && rest.includes(norm(a)) && !negated(text, a)) { rest = rest.split(norm(a)).join("■"); courses.push(n); recog.push(`과목 ${n}`); }
  }
  if (EXCEPT_ALL.test(text)) for (const it of spec.items) if (it.type === "course") {   // 'X 빼고 다 들었어'
    const cands = [...(it.match || [it.name]), ...(COURSE_ALIASES[norm(it.name)] || [])];
    if (!cands.some((c) => courses.includes(c)) && !cands.some((c) => negated(text, c))) { courses.push(it.name); recog.push(`과목 ${it.name}`); }
  }
  for (const it of spec.items) if (it.type === "choose" && it.pattern) {
    const m = text.match(new RegExp(pat(it.pattern) + "[^0-9]{0,6}" + NUMP + "\\s*(과목|개)"));
    if (m) { for (let j = 0; j < knum(m[1]); j++) courses.push(`${it.pattern}${j + 1}`); recog.push(`${it.pattern} ${knum(m[1])}과목`); }
  }
  for (const it of spec.items) {
    const nn = norm(it.name);
    if (it.type === "count") {
      for (const [core, words] of COUNT_KEYS) if (nn.includes(norm(core))) {
        const n = findCount(text, words), short = it.name.replace(/\s*\d+회$/, "");
        if (n !== null) { acts[it.id] = n; recog.push(`${short} ${n}회`); }
        else if (full(text, words)) { acts[it.id] = Math.ceil(it.points / it.per); recog.push(`${short} ${acts[it.id]}회 (다 했다고 해서 필요 횟수로 계산)`); }
        break; }
    } else if (it.type === "check") {
      for (const [core, words] of CHECK_KEYS) if (nn.includes(norm(core))) {
        if (mentioned(text, words)) {
          const lv = core === "코딩역량" ? text.match(/(레벨|level|lv\.?|단계)\s*(\d)/i) : null;
          if (lv && +lv[2] < 2) recog.push(`코딩역량인증 레벨${lv[2]} (Lv2 이상이어야 인정 → 미인정)`);   // Lv2 이상만 인정
          else { acts[it.id] = true; recog.push(it.name); }
        }
        break; }
    } else if (it.type === "all_of") {
      const low = text.toLowerCase(), i = low.indexOf("asp");
      if (i >= 0 && !NEGP.test(low.slice(i + 3, i + 15))) {
        const win = low.slice(i + 3, i + 23);
        const parts = it.parts.filter((p) => win.includes(p) && !win.includes("모두") && !win.includes("다 "));
        acts[it.id] = parts.length && parts.length < it.parts.length && win.includes("만") ? parts : [...it.parts];
        recog.push(`${it.name} (${acts[it.id].join(", ")})`);
      }
    } else if (it.type === "topcit_rule") {
      const n = findCount(text, ["topcit", "탑싯"], "(번|회)"); const lv = text.match(/(수준|레벨|level|lv)\s*(\d)/i);
      if (n !== null || lv) { acts[it.id] = { attempts: n || 0, level: lv ? +lv[2] : 0 }; recog.push(`TOPCIT 응시 ${n || 0}회, 수준 ${lv ? lv[2] : 0}`); }
    } else if (it.type === "points") {
      const subs = {}; let other = 0;
      for (const s of it.subitems || []) {
        const sname = s.name, base = sname.replace(/\s*\(.*?\)/g, "");
        const abbr = [...sname.matchAll(/\(([A-Za-z]+)\)/g)].map((m) => m[1]);
        const core = base.replace(/\s*(취득|응시|이상|\(.*)$/, "").trim();
        const words = [core, ...abbr].filter(Boolean);
        if (s.mode === "check") {
          let ok;
          if (sname.startsWith("토익스피킹")) { const m = text.match(/토익\s*스피킹\D{0,6}(\d{2,3})/); ok = !!(m && +m[1] >= 100); }
          else if (sname.startsWith("토익")) { const m = text.match(/토익(?!\s*스피킹)\D{0,6}(\d{3})/); ok = !!(m && +m[1] >= 500); }
          else if (sname.toLowerCase().startsWith("opic")) ok = /opic\D{0,6}(im|ih|al|m1|m2|m3)/i.test(text);
          else if (sname.endsWith("취득")) ok = words.some((w) => new RegExp(pat(w) + "[^,.\\n고]{0,10}?" + GOT, "i").test(text)) && mentioned(text, words);
          else if (sname.endsWith("응시")) ok = words.some((w) => new RegExp(pat(w) + "[^,.\\n고]{0,10}?" + TOOK, "i").test(text));
          else ok = mentioned(text, words);
          if (ok) { subs[sname] = true; recog.push(sname); }
        } else if (s.mode === "count") {
          const n = findCount(text, words); if (n !== null) { subs[sname] = n; recog.push(`${core} ${n}${sname.includes("학기") ? "학기" : "회"}`); }
        } else if (s.mode === "number") {
          const m = text.match(new RegExp(pat(core) + "\\D{0,8}?(\\d+)\\s*점")), lo = sname.match(/\((\d+)~/);
          if (m) { subs[sname] = +m[1]; recog.push(`${core} ${m[1]}점`); }
          else if (lo && ((core.endsWith("참가") && CONTEST_JOIN.test(text) && !CONTEST_WIN.test(text)) || (core.endsWith("수상") && CONTEST_WIN.test(text)))) {
            subs[sname] = +lo[1]; recog.push(`${core} ${lo[1]}점 (점수를 말하지 않아 최소 점수로 계산)`); }
        }
      }
      if (!(it.subitems || []).length) {   // 졸업필수활동 10점 같은 직접 점수
        const core = it.name.replace(/\s*\(.*$/, ""), cat = it.name.match(/\((공감|창의|글로벌)\)/);
        const words = [core, ...(core.includes("자율활동") ? ["자율활동", "자율"] : [])];
        if (cat) {   // 비교과자율활동(창의)/(글로벌): 인재상 단어 뒤의 점수만 해당 항목에
          const m = text.includes("자율") ? text.match(new RegExp(cat[1] + "\\D{0,6}?(\\d+)\\s*점")) : null;
          if (m) { other = +m[1]; recog.push(`${it.name} ${m[1]}점`); }
        } else {
          const m = words.map((w) => text.match(new RegExp(pat(w) + "\\D{0,8}?(\\d+)\\s*점"))).find((x) => x);
          if (m) { other = +m[1]; recog.push(`${core} ${m[1]}점`); }
          else if (full(text, [core])) { other = it.points; recog.push(`${core} ${fmtg(it.points)}점 (다 했다고 해서 만점으로 계산)`); }
        }
      }
      if (Object.keys(subs).length || other) acts[it.id] = { subitems: subs, other };
    }
  }
  const hasTopcit = spec.items.some((it) => it.type === "topcit_rule" || (it.type === "count" && norm(it.name).includes("topcit"))
    || (it.subitems || []).some((s) => norm(s.name).includes("topcit")));
  const creative = spec.items.find((it) => it.type === "points" && !(it.subitems || []).length && it.name.includes("자율활동") && it.name.includes("창의"));
  if (!hasTopcit && creative) {   // TOPCIT 응시 회당 1점은 비교과자율활동(창의) 점수로 인정 (공식 안내)
    const n = findCount(text, ["topcit", "탑싯"], "(번|회)");
    if (n) { const a = acts[creative.id] || (acts[creative.id] = { subitems: {}, other: 0 }); a.other = (+a.other || 0) + n;
      recog.push(`TOPCIT 응시 ${n}회 → ${creative.name} ${n}점`); }
  }
  const sang = spec.items.filter((it) => it.type === "count" && it.name.includes("상담"));   // '상담 3번' (상담 항목이 하나뿐일 때)
  if (sang.length === 1 && !(sang[0].id in acts)) { const n = findCount(text, ["상담"]); if (n !== null) { acts[sang[0].id] = n; recog.push(`${sang[0].name} ${n}회`); } }
  const credits = extractCredits(text);
  if (credits !== null) recog.push(`총 이수학점 ${credits}학점`);
  return { courses, activities: acts, recognized: recog, credits };
}

/* ---------- 총 이수학점 (학칙 제33조: 130학점 이상) ---------- */
const fmtg = (x) => String(+(+x).toFixed(2));   // 파이썬 :g 와 같은 표기 (115, 2.5)
const CREDIT_RE = /(?:총|지금까지|현재|이수한|들은|딴|취득한)\s*(\d{2,3})\s*학점(?!\s*(?:이상|필요|들어야|이수해야|채워야|인가|인지|맞))|(\d{2,3})\s*학점\s*(?:들었|이수했|채웠|이야|야(?![가-힣])|있어|땄|받았|취득했|했)/;
function extractCredits(text) {
  const m = text.match(CREDIT_RE); if (!m) return null;
  const n = parseInt(m[1] || m[2], 10); return n > 0 && n <= 200 ? n : null;
}
function creditLine(credits, REQ) {
  if (credits === null || credits === undefined) return null;
  const total = ((REQ.meta || {}).grad_credits || {}).total || 130, left = total - credits;
  return `졸업학점(학칙 제33조, ${total}학점 이상): ${fmtg(credits)} / ${total}학점 → ` + (left <= 0 ? "충족" : `${fmtg(left)}학점 부족`) + " (다면적 졸업인증과 별개의 졸업요건)";
}

/* ---------- 대화 누적: 여러 메시지에 나눠 말한 과목·활동을 하나의 기록으로 (Python rag/extract.py 와 같은 규칙) ---------- */
const HYPO = /하면|한다면|했다면|으면|라면/;
const QUESTION = /\?|몇|어떻게|뭐|무엇|언제|어디|왜|나요|까요|인가|인지|(?<!졸업)필수|해야/;
const DONE_EXTRA = /완료|들음|받음|채움|봤어|갔어|나갔|땄음|수료/;
const RESET_RE = /처음부터\s*다시|초기화|리셋|다\s*지워|전부\s*지워|새로\s*시작/;
const MORE_RE = /(\d+|한|두|세|네|다섯)\s*(번|회)\s*더|더\s*(\d+|한|두|세|네|다섯)\s*(번|회)/;
function negated(text, word) {
  const low = text.toLowerCase(), re = new RegExp(pat(word), "gi"); let m;
  while ((m = re.exec(low))) { if (NEGP.test(low.slice(m.index + m[0].length, m.index + m[0].length + 12))) return true; }
  return false;
}
const newRecord = (dept, year) => ({ dept, year, courses: [], activities: {}, credits: null });
const hasAny = (r) => !!(r && (r.courses.length || Object.keys(r.activities).length || (r.credits !== null && r.credits !== undefined)));
const coreName = (name) => name.replace(/\s*\(.*$/, "");
function recordedNegation(text, rec, spec) {
  const items = Object.fromEntries(spec.items.map((i) => [i.id, i]));
  return rec.courses.some((c) => negated(text, c)) || Object.keys(rec.activities).some((id) => items[id] && negated(text, coreName(items[id].name)));
}
function wantsJudgement(text, ext, rec, spec) {
  if (isStatusQuestion(text)) return false;   // '인간사랑 수강했어?' 는 기록하지 않음
  if (RESET_RE.test(text) || isCheckRequest(text) || (ext.credits !== null && ext.credits !== undefined)) return true;
  if (ext.recognized.length && !HYPO.test(text) && (DONE_SIGNAL.test(text) || DONE_EXTRA.test(text) || !QUESTION.test(text))) return true;
  return !!(rec && spec && recordedNegation(text, rec, spec));
}
function mergeRecord(rec, ext, text, spec) {
  if (RESET_RE.test(text)) rec = newRecord(rec.dept, rec.year);
  rec = { dept: rec.dept, year: rec.year, courses: [...rec.courses], activities: { ...rec.activities }, credits: rec.credits ?? null };
  const items = Object.fromEntries(spec.items.map((i) => [i.id, i])), more = MORE_RE.test(text);
  for (const c of ext.courses) if (!rec.courses.includes(c)) rec.courses.push(c);
  for (const [id, v] of Object.entries(ext.activities)) {
    const old = rec.activities[id], t = items[id].type;
    if (t === "count" && more && typeof old === "number") rec.activities[id] = old + v;
    else if (t === "points" && old && typeof old === "object" && v && typeof v === "object") {
      const subs = { ...(old.subitems || {}) };
      for (const [k, x] of Object.entries(v.subitems || {})) {
        const prev = subs[k], add = more && typeof x === "number" && typeof prev === "number";
        subs[k] = add ? prev + x : x;
      }
      rec.activities[id] = { subitems: subs, other: v.other || old.other || 0 };
    } else rec.activities[id] = v;
  }
  if (ext.credits !== null && ext.credits !== undefined) rec.credits = ext.credits;
  rec.courses = rec.courses.filter((c) => !negated(text, c));
  for (const id of Object.keys(rec.activities))
    if (items[id] && !(id in ext.activities) && negated(text, coreName(items[id].name))) delete rec.activities[id];
  return rec;
}
function describeRecord(rec, spec) {
  const items = Object.fromEntries(spec.items.map((i) => [i.id, i]));
  const out = rec.courses.map((c) => `과목 ${c}`);
  for (const [id, v] of Object.entries(rec.activities)) {
    const it = items[id]; if (!it) continue;
    const name = it.name, t = it.type;
    if (t === "course" && rec.courses.some((c) => norm(c) === norm(name))) continue;   // 과목 리스트에 이미 있음
    if (t === "count") out.push(`${name.replace(/\s*\d+회$/, "")} ${fmtg(v)}회`);
    else if (t === "all_of") out.push(`${name} (${v.join(", ")})`);
    else if (t === "topcit_rule") out.push(`TOPCIT 응시 ${v.attempts || 0}회, 수준 ${v.level || 0}`);
    else if (t === "points" && v && typeof v === "object") {
      for (const [k, x] of Object.entries(v.subitems || {})) out.push(x === true ? k : typeof x === "number" ? `${k} ${fmtg(x)}` : `${k} ${x}`);
      if (v.other) out.push(`${coreName(name)} ${fmtg(v.other)}점`);
    } else if (t === "points") out.push(`${name} ${fmtg(v)}점`);
    else out.push(name);
  }
  if (rec.credits !== null && rec.credits !== undefined) out.push(`총 이수학점 ${fmtg(rec.credits)}학점`);
  return out;
}
const HOWTO_INTENT = /어떻게|방법|뭐야|뭔가|무엇|어디서/;
const DONE_SIGNAL = /졸업\s*(가능|할\s*수|돼|되)|판정|들었|했어|했고|했는데|했음|땄|취득했|이수했|참여했|응시했|받았|통과|다녀왔|마쳤|끝냈|수료|합격/;
// '…어떻게 채워?' 같은 방법 질문은 판정 요청이 아님 (들었/했 등 이수 사실을 말하면 판정)
const isCheckRequest = (t) => CHECK_INTENT.test(t) && !(HOWTO_INTENT.test(t) && !DONE_SIGNAL.test(t));

/* ---------- 질문에 나온 항목의 기준 (항목별로 따로) + '인간사랑 수강했어?' 상태 질문 (Python rag/extract.py 와 같은 규칙) ---------- */
const ITEM_EXTRA = { "코딩역량인증제": ["코딩역량", "코딩인증"], "sw전공역량졸업인증제": ["전공역량"], "자기주도상담": ["자기주도"],
  "비교과자율활동": ["자율활동"], "경진대회참가": ["경진대회", "공모전"], "경진대회수상": ["경진대회", "수상"],
  "학과활동": ["학과행사"], "자기개발과전공탐색": ["전공탐색"], "예비창업설계": ["예창"], "자격증": ["자격"] };
const ITEM_STOP = new Set(["과목", "택1", "택2", "택3", "이상", "lv2", "ai", "sw", "it", "학기", "12학기"]);
function itemKeys(it) {
  const base = it.name.replace(/\(.*?\)/g, " ");
  const keys = new Set(base.split(/[/·\-\s]+/).map(norm).filter((t) => t.length >= 2 && !ITEM_STOP.has(t)));
  for (const k of [...keys]) (ITEM_EXTRA[k] || []).forEach((x) => keys.add(norm(x)));
  if (norm(it.name).includes("asp")) keys.add("asp");
  return keys;
}
const shortCat = (c) => c.replace(/^(소통하는|융합하는|공헌하는)\s*/, "");
function itemFact(it) {
  const t = it.type, pts = it.points;
  let head = `${it.name}: ${fmtg(pts)}점 (${it.section}, ${shortCat(it.category)})`;
  if (t === "count") head += ` — 회당 ${fmtg(it.per)}점, ${Math.ceil(pts / it.per)}회 필요`;
  else if (t === "all_of") head += ` — ${it.parts.join("·")} 모두 이수해야 인정`;
  else if (t === "course") head += " — 해당 과목 이수";
  else if (t === "choose") head += ` — ${(it.options || [it.pattern || ""]).join(", ")} 중 ${it.n}과목`;
  else if (t === "points" && (it.subitems || []).length) {
    const byPts = new Map(), rest = [];
    for (const s of it.subitems) {
      if (s.mode === "check" && s.points !== undefined) { if (!byPts.has(s.points)) byPts.set(s.points, []); byPts.get(s.points).push(s.name.replace(/\s*취득$/, "")); }
      else rest.push(s.name + (s.per ? ` 회당 ${fmtg(s.per)}점` : ""));
    }
    const parts = [...[...byPts].map(([k, v]) => `${v.join(" / ")} 중 하나 취득 ${fmtg(k)}점`), ...rest];
    head += " — 인정: " + parts.join("; ") + ` (이 항목은 합쳐서 최대 ${fmtg(pts)}점)`;
  } else if (t === "points") head += " — 세부 활동 점수 합산 (비교과일람표에 인정된 점수)";
  const note = (it.note || "").split(" · ")[0];
  if (note && t !== "count" && t !== "all_of" && !note.startsWith("세부 프로그램")) head += ` ※ ${note}`;
  return head;
}
function itemFacts(text, dept, year, REQ) {
  const spec = getRequirementItems(dept, year, REQ); if (spec.error) return [];
  const nt = norm(text);
  return spec.items.filter((it) => [...itemKeys(it)].some((k) => k && nt.includes(k))).map(itemFact);
}
const ASKED_RE = /(들었|했|이수했|수강했|땄|받았|통과했|참여했|봤)(어|어요|나|나요|니|냐|지|죠|을까)?\s*[?？]|(들었|했|이수했|수강했|땄|받았|통과했|참여했)(나|니|냐|는지|는\s*지)/;
const isStatusQuestion = (t) => ASKED_RE.test(t) && !/졸업\s*(가능|할\s*수)/.test(t);
const askedItemIds = (text, spec) => { const nt = norm(text); return spec.items.filter((it) => [...itemKeys(it)].some((k) => k && nt.includes(k))).map((it) => it.id); };
function statusLines(res, ids) {
  return res.items.filter((r) => ids.includes(r.id)).map((r) => {
    const short = r.name.replace(/\s*\(.*$/, "");
    if (r.passed) return `✅ ${r.name}: 이수로 확인됨 (${fmtg(r.earned)}/${fmtg(r.required)}점 · ${r.detail})`;
    if (r.earned > 0) return `🔶 ${r.name}: 일부만 확인됨 (${fmtg(r.earned)}/${fmtg(r.required)}점 · ${r.detail})`;
    return `⬜ ${r.name}: 아직 확인되지 않음 (입력·언급된 기록 없음) — 자가진단 탭에서 체크하거나 '${short} 들었어/했어'라고 말해 주세요`;
  });
}

if (typeof module !== "undefined") module.exports = { itemFacts, isStatusQuestion, askedItemIds, statusLines, extractInputs, isCheckRequest, wantsJudgement, mergeRecord, newRecord, hasAny, describeRecord, creditLine, extractCredits, RESET_RE, CATEGORIES, GROUPS, SECTIONS, norm, parseCourseList, resolveDepartment, parseYear, getRequirementItems,
  checkGraduation, formatResultText, compactRequirements, buildIndex, search, detectProfile, KINDS };
