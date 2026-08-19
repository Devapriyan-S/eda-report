/*
 * EDA Report — UI controller and report writer.
 *
 * The page and the downloadable file are built from the same render functions,
 * so the exported report is the page rather than a stripped-down copy of it.
 * The download is a single self-contained HTML file: styles inlined, SVG
 * inlined, no scripts, no network requests. It opens on a machine that has
 * never heard of this tool.
 */

const HEAT_LOW = "#151d2e";
const HEAT_HIGH = "#1fa8a3";

const $ = (s) => document.querySelector(s);
const el = (tag, cls, html) => {
  const n = document.createElement(tag);
  if (cls) n.className = cls;
  if (html !== undefined) n.innerHTML = html;
  return n;
};
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) =>
  ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const fmt = (v, d = 3) => {
  if (v === null || v === undefined || Number.isNaN(v)) return "—";
  const a = Math.abs(v);
  if (a !== 0 && (a >= 1e9 || a < 1e-4)) return Number(v).toExponential(2);
  if (a >= 1000) return Number(v).toLocaleString("en-US", { maximumFractionDigits: 1 });
  return String(Number(Number(v).toFixed(d)));
};

const NS = "http://www.w3.org/2000/svg";
const sv = (tag, attrs = {}) => {
  const n = document.createElementNS(NS, tag);
  for (const [k, v] of Object.entries(attrs)) if (v !== null) n.setAttribute(k, v);
  return n;
};

/* ── Worker ───────────────────────────────────────────────── */

const worker = new Worker("js/worker.js");
let nextId = 0;
const pending = new Map();
const call = (action, args = {}) => new Promise((res) => {
  const id = ++nextId; pending.set(id, res);
  worker.postMessage({ id, action, ...args });
});

worker.onmessage = ({ data }) => {
  if (data.type === "boot") {
    $("#boot-stage").textContent = data.stage;
    $("#boot-bar").style.width = `${data.pct * 100}%`;
  } else if (data.type === "ready") {
    $("#boot-bar").style.width = "100%";
    $("#runtime-badge").textContent = data.versions;
    setTimeout(() => { $("#boot").hidden = true; $("#app").hidden = false; }, 300);
  } else if (data.type === "bootError") {
    $("#boot-stage").textContent = "Could not start Python";
    $("#boot-error").hidden = false;
    $("#boot-error").textContent = `${data.error}\n\nIf this page was just ` +
      `redeployed the CDN may still be propagating — wait a moment and reload.`;
  } else if (data.type === "result") {
    const r = pending.get(data.id);
    if (r) { pending.delete(data.id); r(data.result); }
  } else if (data.type === "log") {
    (data.isError ? console.warn : console.log)("[python]", data.line);
  }
};

/* ── Load ─────────────────────────────────────────────────── */

let lastReport = null;

const dz = $("#dropzone");
dz.addEventListener("click", () => $("#file-input").click());
dz.addEventListener("keydown", (e) => {
  if (e.key === "Enter" || e.key === " ") { e.preventDefault(); $("#file-input").click(); }
});
["dragenter", "dragover"].forEach((ev) =>
  dz.addEventListener(ev, (e) => { e.preventDefault(); dz.classList.add("over"); }));
["dragleave", "drop"].forEach((ev) =>
  dz.addEventListener(ev, (e) => { e.preventDefault(); dz.classList.remove("over"); }));
dz.addEventListener("drop", (e) => e.dataTransfer.files[0] && readFile(e.dataTransfer.files[0]));
$("#file-input").addEventListener("change", (e) => e.target.files[0] && readFile(e.target.files[0]));
document.querySelectorAll("[data-sample]").forEach((b) =>
  b.addEventListener("click", () => analyse(SAMPLES[b.dataset.sample], `${b.dataset.sample}.csv`)));

function readFile(file) {
  const r = new FileReader();
  r.onerror = () => showError(`Could not read ${file.name}.`);
  r.onload = () => analyse(r.result, file.name);
  r.readAsText(file);
}
const showError = (m) => { $("#error").hidden = false; $("#error").textContent = m; };

async function analyse(text, name) {
  $("#error").hidden = true;
  dz.querySelector(".dz-title").textContent = "Analysing…";
  const res = await call("analyse", { text, name });
  dz.querySelector(".dz-title").textContent = "Drop a CSV here, or click to browse";

  if (!res.ok) { showError(res.error); console.error(res.traceback); return; }
  lastReport = res;
  render(res);
  $("#report").hidden = false;
  $("#report").scrollIntoView({ behavior: "smooth", block: "start" });
}

/* ── Render ───────────────────────────────────────────────── */

function render(r) {
  $("#report-title").textContent = r.name;
  // Each section renders independently: a failure in one should degrade that
  // section, not leave the whole report hidden behind a half-finished DOM.
  const sections = [
    ["#summary",  () => summaryBlock(r)],
    ["#findings", () => findingsBlock(r.findings)],
    ["#assoc",    () => associationBlock(r.association)],
    ["#columns",  () => columnsBlock(r.profiles)],
    ["#preview",  () => previewBlock(r.preview)],
  ];
  for (const [sel, build] of sections) {
    try {
      $(sel).replaceChildren(build());
    } catch (err) {
      console.error(`section ${sel} failed`, err);
      $(sel).replaceChildren(el("div", "alert alert-error",
        `This section could not be rendered: ${esc(err.message)}`));
    }
  }
}

function summaryBlock(r) {
  const box = el("div");
  const cells = [
    [r.shape.rows.toLocaleString(), "Rows"],
    [r.shape.columns, "Columns"],
    [r.total_missing.toLocaleString(), "Missing cells"],
    [r.duplicates.toLocaleString(), "Duplicate rows"],
    [`${r.memory_mb} MB`, "In memory"],
  ];
  const grid = el("div", "kpi-grid");
  cells.forEach(([v, l]) => {
    const s = el("div", "stat");
    s.append(el("div", "stat-val", esc(v)), el("div", "stat-lab", esc(l)));
    grid.append(s);
  });
  box.append(grid);

  const roles = Object.entries(r.roles).sort((a, b) => b[1] - a[1]);
  const chips = el("p", "hint");
  chips.textContent = "Column types: " + roles.map(([k, v]) => `${v} ${k}`).join(" · ");
  box.append(chips);
  return box;
}

const SEV = {
  critical: { pill: "pill-bad",  icon: "✕", label: "Critical" },
  warning:  { pill: "pill-warn", icon: "⚠", label: "Warning" },
  note:     { pill: "pill-info", icon: "·", label: "Note" },
  info:     { pill: "pill-ok",   icon: "i", label: "Info" },
};

function findingsBlock(findings) {
  const box = el("div");
  const counts = findings.reduce((a, f) => ({ ...a, [f.severity]: (a[f.severity] ?? 0) + 1 }), {});
  const line = el("p", "hint");
  line.textContent = Object.entries(SEV)
    .filter(([k]) => counts[k])
    .map(([k, v]) => `${counts[k]} ${v.label.toLowerCase()}`)
    .join(" · ") || "nothing flagged";
  box.append(line);

  findings.forEach((f) => {
    const s = SEV[f.severity] ?? SEV.note;
    const card = el("div", `finding finding-${f.severity}`);
    const head = el("div", "finding-head");
    head.append(el("span", `pill ${s.pill}`, esc(s.label)));
    head.append(el("span", "finding-title", esc(f.title)));
    card.append(head);
    card.append(el("p", "finding-detail", esc(f.detail)));
    if (f.columns.length) {
      card.append(el("p", "finding-cols",
        f.columns.slice(0, 8).map((c) => `<code>${esc(c)}</code>`).join(" ")));
    }
    box.append(card);
  });
  return box;
}

function associationBlock(a) {
  const box = el("div");
  if (a.columns.length < 2) {
    box.append(el("p", "hint", "Fewer than two comparable columns — nothing to correlate."));
    return box;
  }

  box.append(el("p", "hint",
    "Every pair on one 0–1 scale: rank correlation between numeric columns, " +
    "Cramér's V between categoricals, and the correlation ratio for mixed pairs. " +
    "A plain <code>df.corr()</code> would show only the first of those three."));
  if (a.truncated) {
    box.append(el("div", "alert alert-warn",
      `${a.truncated} lower-variation column(s) omitted from the matrix to keep the ` +
      `pairwise computation tractable.`));
  }

  const n = a.columns.length;
  const CELL = Math.max(16, Math.min(34, Math.floor(760 / (n + 6))));
  const LABEL = 132;
  const W = LABEL + n * CELL + 8, H = LABEL + n * CELL + 8;
  const svg = sv("svg", { class: "chart heat", viewBox: `0 0 ${W} ${H}`, role: "img" });

  const mix = (t) => {
    const lo = [21, 29, 46], hi = [31, 168, 163];
    // Square-root ramp: linear interpolation makes mid-strength associations
    // nearly invisible against the dark background.
    const e = Math.sqrt(Math.max(0, Math.min(1, t)));
    return `rgb(${lo.map((c, i) => Math.round(c + (hi[i] - c) * e)).join(",")})`;
  };

  a.columns.forEach((c, i) => {
    const short = c.length > 17 ? c.slice(0, 16) + "…" : c;
    const rowLabel = sv("text", { x: LABEL - 6, y: LABEL + i * CELL + CELL / 2 + 3.5,
                                  "text-anchor": "end", class: "tick" });
    rowLabel.textContent = short;
    // Node.append() returns undefined, so the title must be built separately —
    // chaining .textContent onto it throws and takes the whole render with it.
    const rowTitle = sv("title");
    rowTitle.textContent = c;
    rowLabel.append(rowTitle);
    svg.append(rowLabel);

    const cx = LABEL + i * CELL + CELL / 2;
    const colLabel = sv("text", { x: cx, y: LABEL - 8, "text-anchor": "start", class: "tick",
                                  transform: `rotate(-60 ${cx} ${LABEL - 8})` });
    colLabel.textContent = short;
    svg.append(colLabel);
  });

  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) {
      const v = a.matrix[i][j];
      const rect = sv("rect", {
        x: LABEL + j * CELL, y: LABEL + i * CELL,
        // 1px gap so adjacent cells stay distinguishable.
        width: CELL - 1, height: CELL - 1, rx: 2,
        fill: i === j ? "var(--surface-2)" : mix(v),
      });
      const t = sv("title");
      t.textContent = `${a.columns[i]} ↔ ${a.columns[j]}\n${v.toFixed(3)}`;
      rect.append(t);
      svg.append(rect);
      // Only label strong cells; a number in every cell is unreadable.
      if (i !== j && v >= 0.5 && CELL >= 26) {
        const lab = sv("text", {
          x: LABEL + j * CELL + CELL / 2 - 0.5, y: LABEL + i * CELL + CELL / 2 + 3,
          "text-anchor": "middle", class: "cellv",
        });
        lab.textContent = v.toFixed(2).replace("0.", ".");
        svg.append(lab);
      }
    }
  }
  box.append(svg);

  const strong = a.pairs.filter((p) => p.strength >= 0.3).slice(0, 12);
  if (strong.length) {
    const table = el("table");
    const thead = el("thead");
    const hr = el("tr");
    ["Pair", "Strength", "Measure", "Reading"].forEach((h) => hr.append(el("th", null, esc(h))));
    thead.append(hr);
    const tbody = el("tbody");
    const METHOD = { spearman: "Spearman", cramers_v: "Cramér's V", correlation_ratio: "η (ratio)" };
    strong.forEach((p) => {
      const tr = el("tr");
      tr.append(el("td", null, `<code>${esc(p.a)}</code> ↔ <code>${esc(p.b)}</code>`));
      tr.append(el("td", "num", p.strength.toFixed(3)));
      tr.append(el("td", null, `<span class="muted">${esc(METHOD[p.method])}</span>`));
      let reading = "";
      if (p.method === "spearman") {
        const lin = Math.abs(p.pearson ?? 0);
        reading = p.strength - lin > 0.2
          ? `monotone but curved — Pearson only ${lin.toFixed(2)}`
          : "close to linear";
      } else if (p.method === "correlation_ratio") {
        reading = `${esc(p.grouping)} explains the variance`;
      } else {
        reading = "largely the same information";
      }
      tr.append(el("td", null, `<span class="muted">${reading}</span>`));
      tbody.append(tr);
    });
    table.append(thead, tbody);
    const scroll = el("div", "table-scroll");
    scroll.append(table);
    box.append(el("h4", "sub", "Strongest relationships"), scroll);
  }
  return box;
}

function sparkline(hist, w = 190, h = 40) {
  const svg = sv("svg", { class: "spark", viewBox: `0 0 ${w} ${h}` });
  if (!hist || !hist.counts?.length) return svg;
  const max = Math.max(...hist.counts, 1);
  const bw = w / hist.counts.length;
  hist.counts.forEach((c, i) => {
    const bh = (c / max) * (h - 2);
    svg.append(sv("rect", {
      x: (i * bw).toFixed(2), y: (h - bh).toFixed(2),
      width: Math.max(1, bw - 1).toFixed(2), height: bh.toFixed(2),
      rx: 1, fill: HEAT_HIGH,
    }));
  });
  return svg;
}

function columnsBlock(profiles) {
  const box = el("div", "col-grid");
  profiles.forEach((p) => {
    const card = el("div", `col-card role-${p.role}`);
    card.append(el("div", "col-name", esc(p.name)));
    card.append(el("div", "col-role", `${esc(p.role)} · ${esc(p.dtype)}`));

    const bits = [`${p.n_unique.toLocaleString()} unique`];
    if (p.n_missing) bits.push(`${p.missing_pct}% missing`);
    card.append(el("div", "col-meta", bits.join(" · ")));

    if (p.role === "numeric" && p.stats.mean !== undefined) {
      if (p.histogram) card.append(sparkline(p.histogram));
      card.append(el("div", "col-meta",
        `${fmt(p.stats.min)} — ${fmt(p.stats.median)} — ${fmt(p.stats.max)}`));
      card.append(el("div", "col-meta", `mean ${fmt(p.stats.mean)} ± ${fmt(p.stats.sd)}`));
      const flags = [];
      if (Math.abs(p.stats.skew) > 1) flags.push(`skew ${fmt(p.stats.skew, 1)}`);
      if (p.stats.n_outliers) flags.push(`${p.stats.n_outliers} outliers`);
      if (p.stats.n_negative) flags.push(`${p.stats.n_negative} negative`);
      if (flags.length) card.append(el("div", "col-warn", flags.join(" · ")));
    } else if (p.stats.levels) {
      const list = el("div", "freq");
      const total = p.n_valid;
      p.stats.levels.slice(0, 5).forEach((lv) => {
        const row = el("div", "freq-row");
        row.append(el("span", "freq-label", esc(lv.value.length > 20 ? lv.value.slice(0, 19) + "…" : lv.value)));
        const track = el("span", "freq-track");
        const bar = el("span", "freq-bar");
        bar.style.width = `${(lv.count / total) * 100}%`;
        track.append(bar);
        row.append(track, el("span", "freq-count", lv.count.toLocaleString()));
        list.append(row);
      });
      card.append(list);
    } else if (p.role === "datetime" && p.stats.min) {
      card.append(el("div", "col-meta",
        `${esc(p.stats.min)} → ${esc(p.stats.max)} (${p.stats.span_days.toLocaleString()} days)`));
    }
    box.append(card);
  });
  return box;
}

function previewBlock(preview) {
  const table = el("table");
  const thead = el("thead");
  const hr = el("tr");
  preview.columns.forEach((c) => hr.append(el("th", null, esc(c))));
  thead.append(hr);
  const tbody = el("tbody");
  preview.rows.forEach((row) => {
    const tr = el("tr");
    row.forEach((v) => tr.append(el("td", typeof v === "number" ? "num" : null,
      v === null ? '<span style="opacity:.35">null</span>' : esc(String(v).slice(0, 60)))));
    tbody.append(tr);
  });
  table.append(thead, tbody);
  const scroll = el("div", "table-scroll");
  scroll.append(table);
  return scroll;
}

/* ── Download a self-contained report ─────────────────────── */

$("#download").addEventListener("click", () => {
  if (!lastReport) return;

  // Inline the stylesheet: the exported file has no access to css/app.css, and
  // a report that opens unstyled on someone else's machine is not a report.
  const styles = [...document.styleSheets]
    .flatMap((sheet) => {
      try { return [...sheet.cssRules].map((r) => r.cssText); }
      catch { return []; }          // cross-origin sheets are unreadable
    })
    .join("\n");

  const body = $("#report").cloneNode(true);
  body.removeAttribute("hidden");
  // Strip the page furniture that means nothing in a standalone file: the
  // download button itself, its caption, and the step numbers, which would
  // start at 2 and read as if the report were missing its first section.
  body.querySelectorAll("button, #download + .hint, .step-num").forEach((n) => n.remove());

  const generated = new Date().toISOString().slice(0, 16).replace("T", " ");
  const html = `<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>EDA report — ${esc(lastReport.name)}</title>
<style>${styles}
/* Re-stated after the extracted rules. Serialising a CSSRule that mixes the
   background shorthand with background-image emits "background-color: ;" --
   an empty value, which invalidates the whole declaration. Without this the
   exported report renders light text on a white page. */
html, body { background-color: #0b0f17 !important; color: #e8edf7; }
body { padding: 2rem 0; margin: 0; }
.report-meta { color: var(--text-mute); font-size: .8rem; margin-bottom: 1.5rem; }
.step-head { gap: 0; }
[hidden] { display: none !important; }
</style></head>
<body><div class="wrap">
<h1 style="margin-bottom:.2rem">Exploratory data report</h1>
<p class="report-meta">${esc(lastReport.name)} · ${lastReport.shape.rows.toLocaleString()} rows ×
${lastReport.shape.columns} columns · generated ${esc(generated)} by
<a href="https://devapriyan-s.github.io/eda-report/">EDA Report</a></p>
${body.innerHTML}
</div></body></html>`;

  const blob = new Blob([html], { type: "text/html;charset=utf-8" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = `eda-${lastReport.name.replace(/\.csv$/i, "")}.html`;
  document.body.append(a);
  a.click();
  a.remove();
  URL.revokeObjectURL(url);
});
