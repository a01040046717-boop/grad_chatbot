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

if (typeof module !== "undefined") module.exports = { CATEGORIES, GROUPS, SECTIONS, norm, parseCourseList, resolveDepartment, parseYear, getRequirementItems,
  checkGraduation, formatResultText, compactRequirements, buildIndex, search, detectProfile, KINDS };
