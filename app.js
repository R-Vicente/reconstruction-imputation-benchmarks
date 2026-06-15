/* ============================================================
   app.js — interactive tables, code viewer, syntax highlight, nav
   Data comes from data.js (window.TABLES) and code_data.js (window.CODE_FILES)
   ============================================================ */

const REPO_URL = "https://github.com/R-Vicente/reconstruction-imputation-benchmarks";

document.querySelectorAll("[data-repo]").forEach(a => { a.href = REPO_URL; });

const esc = s => String(s)
  .replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");

const isNumeric = v => /^[-+]?[\d.,±%<>≈/x\s]+$|^N\/A$/.test(String(v).trim()) && /\d/.test(String(v));

/* ------------------------------------------------------------
   TABLES
   ------------------------------------------------------------ */
(function buildTables() {
  const TABLES = window.TABLES || [];
  const listEl  = document.getElementById("tableList");
  const panelEl = document.getElementById("tablePanel");
  if (!listEl || !panelEl) return;

  let active = TABLES[0]?.id;
  let query = "";

  listEl.innerHTML = TABLES.map((t, i) => `
    <button data-id="${t.id}">
      <span class="tl-id">${String(i + 1).padStart(2, "0")}</span>
      <span class="tl-name">${esc(shortName(t))}</span>
    </button>`).join("");

  function shortName(t) {
    // "Table 1. Dataset characteristics and ..." 
    return t.name.replace(/\.csv$/, "").replace(/^Table\s*/i, "Table ");
  }

  function csvEscape(v) {
    const s = String(v);
    return /[",\n]/.test(s) ? '"' + s.replace(/"/g, '""') + '"' : s;
  }

  function downloadCSV(t) {
    const lines = [t.columns.map(csvEscape).join(",")];
    t.rows.forEach(r => lines.push(r.map(csvEscape).join(",")));
    const blob = new Blob(["\ufeff" + lines.join("\r\n")], { type: "text/csv;charset=utf-8;" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = t.name.replace(/\.csv$/, "") + ".csv";
    document.body.appendChild(a); a.click(); a.remove();
    URL.revokeObjectURL(url);
  }

  function highlightCell(text, q) {
    const s = esc(text);
    if (!q) return s;
    const i = s.toLowerCase().indexOf(q.toLowerCase());
    if (i < 0) return s;
    return s.slice(0, i) + "<mark>" + s.slice(i, i + q.length) + "</mark>" + s.slice(i + q.length);
  }

  function render() {
    const t = TABLES.find(x => x.id === active);
    if (!t) return;
    const q = query.trim().toLowerCase();
    const rows = q
      ? t.rows.filter(r => r.some(c => String(c).toLowerCase().includes(q)))
      : t.rows;

    const thead = "<tr>" + t.columns.map(c => `<th>${esc(c)}</th>`).join("") + "</tr>";
    const tbody = rows.map(r => "<tr>" + r.map((c, i) => {
      const cls = i > 0 && isNumeric(c) ? ' class="num"' : "";
      return `<td${cls}>${highlightCell(c, query.trim())}</td>`;
    }).join("") + "</tr>").join("");

    panelEl.innerHTML = `
      <h3>${esc(t.title)}</h3>
      <p class="caption">${esc(t.caption)}</p>
      <div class="table-controls">
        <div class="search-box">
          <span class="icon">⌕</span>
          <input type="text" id="tableSearch" placeholder="Filter rows — try a dataset or method…" value="${esc(query)}">
        </div>
        <span class="row-count">${rows.length} / ${t.rows.length} rows · ${t.columns.length} cols</span>
        <button class="btn" id="csvBtn">↓ &nbsp;CSV</button>
      </div>
      <div class="table-scroll">
        ${rows.length
          ? `<table class="data"><thead>${thead}</thead><tbody>${tbody}</tbody></table>`
          : `<div class="table-empty">No rows match “${esc(query)}”.</div>`}
      </div>`;

    const search = document.getElementById("tableSearch");
    search.addEventListener("input", e => {
      query = e.target.value;
      const pos = e.target.selectionStart;
      render();
      const s2 = document.getElementById("tableSearch");
      s2.focus(); s2.setSelectionRange(pos, pos);
    });
    document.getElementById("csvBtn").addEventListener("click", () => downloadCSV(t));
  }

  listEl.addEventListener("click", e => {
    const btn = e.target.closest("button");
    if (!btn) return;
    active = btn.dataset.id;
    query = "";
    listEl.querySelectorAll("button").forEach(b => b.classList.toggle("active", b.dataset.id === active));
    render();
  });

  listEl.querySelector("button")?.classList.add("active");
  render();
})();

/* ------------------------------------------------------------
   SYNTAX HIGHLIGHTING
   ------------------------------------------------------------ */
const PY_KW = ["def","return","import","from","for","in","if","elif","else","while",
  "try","except","finally","with","as","class","None","True","False","and","or",
  "not","is","lambda","continue","break","pass","raise","yield","global","assert",
  "del","async","await"].join("|");

function highlightPython(code) {
  const src = esc(code);
  const re = new RegExp(
    "(\"\"\"[\\s\\S]*?\"\"\"|'''[\\s\\S]*?''')" +      // 1 triple string
    "|(#[^\\n]*)" +                                       // 2 comment
    "|(\"(?:[^\"\\\\]|\\\\.)*\"|'(?:[^'\\\\]|\\\\.)*')" + // 3 string
    "|(@[A-Za-z_][\\w.]*)" +                             // 4 decorator
    "|\\b(def|class)\\s+([A-Za-z_]\\w*)" +              // 5 kw, 6 name
    "|\\b(\\d+\\.?\\d*(?:e[-+]?\\d+)?)\\b" +            // 7 number
    "|\\b(" + PY_KW + ")\\b",                            // 8 keyword
    "g");
  return src.replace(re, (m, g1, g2, g3, g4, g5, g6, g7, g8) => {
    if (g1) return `<span class="tok-string">${g1}</span>`;
    if (g2) return `<span class="tok-comment">${g2}</span>`;
    if (g3) return `<span class="tok-string">${g3}</span>`;
    if (g4) return `<span class="tok-deco">${g4}</span>`;
    if (g5) return `<span class="tok-keyword">${g5}</span> <span class="tok-func">${g6}</span>`;
    if (g7) return `<span class="tok-number">${g7}</span>`;
    if (g8) return `<span class="tok-keyword">${g8}</span>`;
    return m;
  });
}

function highlightMarkdown(code) {
  return esc(code).split("\n").map(line => {
    if (/^\s*#{1,6}\s/.test(line)) return `<span class="tok-md-h">${line}</span>`;
    if (/^\s*\|/.test(line)) return `<span class="tok-md-rule">${line}</span>`;
    if (/^\s*[-*]\s/.test(line)) return line.replace(/^(\s*)([-*])(\s)/, '$1<span class="tok-keyword">$2</span>$3');
    // inline `code`
    return line.replace(/(`[^`]+`)/g, '<span class="tok-md-code">$1</span>');
  }).join("\n");
}

function highlightText(code) {
  return esc(code).split("\n").map(line =>
    /^\s*#/.test(line)
      ? `<span class="tok-comment">${line}</span>`
      : line.replace(/(==|>=|<=)/g, '<span class="tok-number">$1</span>')
  ).join("\n");
}

function highlight(file) {
  if (file.lang === "python") return highlightPython(file.code);
  if (file.lang === "markdown") return highlightMarkdown(file.code);
  return highlightText(file.code);
}

/* ------------------------------------------------------------
   CODE VIEWER
   ------------------------------------------------------------ */
(function buildCode() {
  const FILES = window.CODE_FILES || [];
  const listEl  = document.getElementById("fileList");
  const panelEl = document.getElementById("codePanel");
  if (!listEl || !panelEl) return;

  let active = FILES[0]?.id;

  listEl.innerHTML = FILES.map(f => `
    <button data-id="${f.id}">
      <span class="fl-name">${esc(f.name)}</span>
      <span class="fl-desc">${esc(f.desc)}</span>
    </button>`).join("");

  function downloadFile(f) {
    const blob = new Blob([f.code], { type: "text/plain;charset=utf-8;" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url; a.download = f.name;
    document.body.appendChild(a); a.click(); a.remove();
    URL.revokeObjectURL(url);
  }

  function render() {
    const f = FILES.find(x => x.id === active);
    if (!f) return;
    const nLines = f.code.split("\n").length;
    const gutter = Array.from({ length: nLines }, (_, i) => `<span>${i + 1}</span>`).join("");

    panelEl.innerHTML = `
      <div class="code-bar">
        <span class="cb-name">${esc(f.name)}</span>
        <span class="cb-lang">${esc(f.lang)}</span>
        <span class="cb-actions">
          <button id="copyBtn">⧉ &nbsp;Copy</button>
          <button id="dlBtn">↓ &nbsp;Download</button>
        </span>
      </div>
      <div class="code-view">
        <div class="code-gutter">${gutter}</div>
        <div class="code-scroll"><pre><code>${highlight(f)}</code></pre></div>
      </div>`;

    document.getElementById("dlBtn").addEventListener("click", () => downloadFile(f));
    document.getElementById("copyBtn").addEventListener("click", async (e) => {
      try {
        await navigator.clipboard.writeText(f.code);
        const b = e.currentTarget;
        const t = b.innerHTML; b.innerHTML = "✓ &nbsp;Copied";
        setTimeout(() => { b.innerHTML = t; }, 1400);
      } catch (_) {}
    });
  }

  listEl.addEventListener("click", e => {
    const btn = e.target.closest("button");
    if (!btn) return;
    active = btn.dataset.id;
    listEl.querySelectorAll("button").forEach(b => b.classList.toggle("active", b.dataset.id === active));
    render();
  });

  listEl.querySelector("button")?.classList.add("active");
  render();
})();

/* ------------------------------------------------------------
   NAV SCROLLSPY + cite copy
   ------------------------------------------------------------ */
(function scrollspy() {
  const links = [...document.querySelectorAll(".nav-links a[data-spy]")];
  const map = new Map(links.map(l => [l.getAttribute("href").slice(1), l]));
  const sections = [...map.keys()].map(id => document.getElementById(id)).filter(Boolean);
  if (!sections.length) return;
  const obs = new IntersectionObserver(entries => {
    entries.forEach(en => {
      if (en.isIntersecting) {
        links.forEach(l => l.classList.remove("active"));
        map.get(en.target.id)?.classList.add("active");
      }
    });
  }, { rootMargin: "-45% 0px -50% 0px", threshold: 0 });
  sections.forEach(s => obs.observe(s));
})();

document.getElementById("citeCopy")?.addEventListener("click", async (e) => {
  const tx = document.getElementById("bibtex").innerText;
  try {
    await navigator.clipboard.writeText(tx);
    e.currentTarget.textContent = "Copied ✓";
    setTimeout(() => { e.currentTarget.textContent = "Copy"; }, 1400);
  } catch (_) {}
});
