// ytb page. Built with `npm run build` in frontend/ into src/ytb/static/main.js (committed, so the
// Python app needs no Node at runtime). Element ids are also used by scripts/manual_session.py.
const STORE = "ytb.runs.v1";
const KEEP = 20; // runs kept in this browser
const SHOW = 4; // runs listed
const DASH = "—";
const ERROR_PREFIX = {
    422: "Check the link or text",
    404: "No captions or audio for this video",
    413: "Too long (max 150,000 characters, or 60 minutes for videos without captions)",
    503: "YouTube is rate-limiting this machine, try again later",
};
const SOURCE_WORDS = { captions: "captions", whisper: "local transcription" };
const COUNT_WORDS = ["No", "One", "Two", "Three", "Four", "Five", "Six", "Seven", "Eight", "Nine"];
const nf = new Intl.NumberFormat("en");
const languageNames = new Intl.DisplayNames(["en"], { type: "language" });
// ---------- small helpers ----------
function el(id) {
    const node = document.getElementById(id);
    if (!node)
        throw new Error(`missing #${id}`);
    return node;
}
function h(tag, props = {}, ...children) {
    const node = Object.assign(document.createElement(tag), props);
    node.append(...children);
    return node;
}
const countWord = (n) => COUNT_WORDS[n] ?? nf.format(n);
const plural = (n, one, many) => (n === 1 ? one : many);
function duration(s) {
    if (s === 0)
        return "0\u00a0s";
    if (s < 10)
        return `${s.toFixed(1)}\u00a0s`;
    if (s < 60)
        return `${Math.round(s)}\u00a0s`;
    const m = Math.floor(s / 60);
    return `${m}\u00a0min ${String(Math.round(s - m * 60)).padStart(2, "0")}\u00a0s`; // taste-skill-ignore: clock format "2 min 05 s"
}
function ago(at) {
    const s = Math.max(0, Math.round((Date.now() - at) / 1000));
    if (s < 60)
        return "just now";
    if (s < 3600)
        return `${Math.floor(s / 60)} min ago`;
    if (s < 86400)
        return `${Math.floor(s / 3600)} h ago`;
    return new Date(at).toISOString().slice(0, 10);
}
function language(code) {
    if (!code)
        return DASH;
    try {
        return languageNames.of(code) ?? code;
    }
    catch {
        return code;
    }
}
function modelName(model) {
    const last = model.split("/").filter(Boolean).pop() ?? model;
    return model.startsWith("Qwen/") ? `${last} (base)` : `${last} (fine-tuned)`;
}
const isBase = (model) => model.startsWith("Qwen/");
const thumbUrl = (id) => `https://i.ytimg.com/vi/${id}/mqdefault.jpg`;
const watchUrl = (id) => `https://www.youtube.com/watch?v=${id}`;
function label(run) {
    if (run.res.video_id)
        return run.res.video_id;
    const t = run.input.replace(/\s+/g, " ");
    return `“${t.slice(0, 44)}${t.length > 44 ? "…" : ""}”`;
}
// ---------- runs kept in this browser ----------
function isRun(x) {
    const r = x;
    return !!r && typeof r.at === "number" && typeof r.input === "string" && !!r.res
        && typeof r.res.summary === "string" && Array.isArray(r.res.chunk_tokens);
}
function loadRuns() {
    try {
        const parsed = JSON.parse(localStorage.getItem(STORE) ?? "[]");
        return Array.isArray(parsed) ? parsed.filter(isRun) : [];
    }
    catch {
        return [];
    }
}
function saveRuns() {
    try {
        localStorage.setItem(STORE, JSON.stringify(runs));
    }
    catch {
        // storage blocked (private window): the history lives for this page only
    }
}
let runs = loadRuns();
let selected = null;
let lastBody = null;
// ---------- title and counts ----------
function renderTitle() {
    const n = runs.length;
    el("greeting").textContent = n === 0 ? "Nothing read yet." : `${countWord(n)} ${plural(n, "summary", "summaries")} so far.`;
    const last = runs[0];
    el("lede").textContent = last
        ? `Last: ${label(last)}, ${last.res.chunks === 1 ? "read in one pass" : `read in ${last.res.chunks} parts`}, `
            + `${duration(last.res.seconds)}, ${modelName(last.res.model)}.`
        : "Paste a YouTube link. Captions are used when a video has them; otherwise its audio is transcribed on this machine.";
}
function setTile(id, value, caption) {
    el(`${id}-value`).textContent = value;
    el(`${id}-cap`).textContent = caption;
}
function renderTiles() {
    const sum = (pick) => runs.reduce((a, run) => a + pick(run.res), 0);
    const n = runs.length;
    setTile("t-tokens", nf.format(sum((r) => r.input_tokens)), n ? `across ${countWord(n).toLowerCase()} ${plural(n, "summary", "summaries")}` : "nothing read yet");
    setTile("t-time", duration(sum((r) => r.seconds)), "spent summarizing");
    setTile("t-parts", nf.format(sum((r) => r.chunks)), "of long transcripts");
    const whisper = runs.filter((r) => r.res.source === "whisper").length;
    setTile("t-whisper", nf.format(whisper), "videos with no captions");
}
// ---------- the answer ----------
function show(view) {
    el("empty").hidden = view !== "empty";
    el("loading").hidden = view !== "loading";
    el("error").hidden = view !== "error";
    el("result").hidden = view !== "result";
    el("copy").disabled = view !== "result";
}
function finding(r) {
    const what = r.video_id
        ? `${language(r.language)} ${r.source === "whisper" ? "audio transcribed on this machine" : "captions"}`
        : "pasted text";
    const how = r.chunks === 1
        ? "in one pass"
        : `in ${r.chunks} parts, combined in ${r.rounds} ${plural(r.rounds, "round", "rounds")}`;
    return `Read ${nf.format(r.input_tokens)} tokens of ${what} ${how}, in ${duration(r.seconds)}.`;
}
function renderStrip(r) {
    const segments = r.chunk_tokens.map((tokens, i) => {
        const seg = h("span", { className: "seg", title: `Part ${i + 1} of ${r.chunks}: ${nf.format(tokens)} tokens` });
        seg.style.flexGrow = String(tokens);
        return seg;
    });
    el("strip").replaceChildren(...segments);
    el("strip-cap").textContent = r.chunks === 1
        ? `Fit in one pass: ${nf.format(r.input_tokens)} tokens.`
        : `${r.chunks} overlapping parts of up to ${nf.format(Math.max(...r.chunk_tokens))} tokens, each summarized, then combined.`;
}
function renderDetails(r) {
    const rows = [
        ["Video", r?.video_id ?? DASH],
        ["Language", r ? language(r.language) : DASH],
        ["Source", r ? (r.source ? SOURCE_WORDS[r.source] : "pasted text") : DASH],
        ["Parts", r ? nf.format(r.chunks) : DASH],
        ["Tokens", r ? `${nf.format(r.input_tokens)} in → ${nf.format(r.output_tokens)} out` : DASH],
        ["Server time", r ? duration(r.seconds) : DASH],
        ["Model", r ? modelName(r.model) : DASH],
    ];
    // before any run the panel keeps its frame and says one thing instead of seven dashes
    el("stats-empty").hidden = r !== null;
    el("stats").hidden = r === null;
    el("stats").replaceChildren(...rows.flatMap(([k, v]) => [
        h("dt", {}, k),
        h("dd", { className: v === DASH ? "dash" : "num" }, v),
    ]));
}
function renderResult(run) {
    const r = run.res;
    const link = el("thumb-link");
    const img = el("thumb");
    if (r.video_id) {
        link.href = watchUrl(r.video_id);
        img.src = thumbUrl(r.video_id);
        img.alt = `Thumbnail of YouTube video ${r.video_id}`;
        link.hidden = false;
    }
    else {
        link.hidden = true;
        img.removeAttribute("src");
    }
    el("finding").textContent = finding(r);
    el("summary").textContent = r.summary;
    renderStrip(r);
    renderDetails(r);
    show("result");
}
// ---------- history and comparison ----------
function renderHistory() {
    el("hist-count").textContent = nf.format(runs.length);
    el("clear").hidden = runs.length === 0;
    const list = el("history");
    if (runs.length === 0) {
        list.replaceChildren(h("li", { className: "empty" }, "Nothing summarized in this browser yet."));
        return;
    }
    const items = runs.slice(0, SHOW).map((run) => {
        const media = run.res.video_id
            ? h("img", { src: thumbUrl(run.res.video_id), alt: "", loading: "lazy", width: 64, height: 36 })
            : h("span", { className: "text-tile" }, "Text");
        const source = run.res.source ? SOURCE_WORDS[run.res.source] : "pasted text";
        const meta = `${source} · ${run.res.chunks} ${plural(run.res.chunks, "part", "parts")} · ${duration(run.res.seconds)} · ${modelName(run.res.model)}`;
        const row = h("button", { type: "button", className: "row-card" }, media, h("span", { className: "row-main" }, h("span", { className: "row-title" }, label(run)), h("span", { className: "row-meta" }, meta)), h("span", { className: "row-when" }, ago(run.at)));
        if (run.at === selected)
            row.setAttribute("aria-current", "true");
        row.addEventListener("click", () => {
            selected = run.at;
            renderResult(run);
            renderHistory();
        });
        return h("li", {}, row);
    });
    if (runs.length > SHOW)
        items.push(h("li", { className: "more" }, `${nf.format(runs.length - SHOW)} older in this browser`));
    list.replaceChildren(...items);
}
function compareColumn(run, missingLabel) {
    if (!run) {
        return h("div", { className: "pair-col" }, h("h3", {}, missingLabel), h("p", { className: "empty" }, `Not run on the ${missingLabel.toLowerCase()} yet.`));
    }
    return h("div", { className: "pair-col" }, h("h3", {}, modelName(run.res.model)), h("p", { className: "pair-text" }, run.res.summary), h("p", { className: "row-meta" }, `${duration(run.res.seconds)} · ${nf.format(run.res.output_tokens)} tokens out`));
}
function renderCompare() {
    const box = el("compare");
    // the newest video run on both models; otherwise the newest video, with the missing model's slot left open
    const paired = runs.find((run) => run.res.video_id
        && runs.some((o) => o.res.video_id === run.res.video_id && isBase(o.res.model) !== isBase(run.res.model)));
    const anchor = paired ?? runs.find((run) => run.res.video_id);
    if (!anchor || !anchor.res.video_id) {
        box.replaceChildren(h("p", { className: "empty" }, "Run one video on both models to compare them here."));
        return;
    }
    const id = anchor.res.video_id;
    const base = runs.find((o) => o.res.video_id === id && isBase(o.res.model));
    const tuned = runs.find((o) => o.res.video_id === id && !isBase(o.res.model));
    box.replaceChildren(h("p", { className: "panel-sub" }, `Video ${id}, newest run of each model.`), h("div", { className: "pair" }, compareColumn(base, "Base model"), compareColumn(tuned, "Fine-tuned model")));
}
function renderAll() {
    renderTitle();
    renderTiles();
    renderHistory();
    renderCompare();
}
// ---------- actions ----------
async function submit(body) {
    lastBody = body;
    const go = el("go");
    const status = el("status");
    go.disabled = true;
    show("loading");
    renderDetails(null); // the details belong to the answer on screen; none while this one is pending or failed
    const t0 = performance.now();
    const tick = () => {
        status.textContent = `Working… ${Math.round((performance.now() - t0) / 1000)} s. `
            + "Videos without captions are transcribed on this machine and can take a few minutes.";
    };
    tick();
    const timer = window.setInterval(tick, 500);
    try {
        const res = await fetch("/api/summarize", {
            method: "POST",
            headers: { "content-type": "application/json" },
            body: JSON.stringify(body),
        });
        const data = await res.json().catch(() => ({}));
        if (!res.ok) {
            const detail = data.detail;
            el("error-text").textContent = `${ERROR_PREFIX[res.status] ?? "Something went wrong"}: ${typeof detail === "string" ? detail : "the request was not valid."}`;
            show("error");
            status.textContent = "";
            return;
        }
        const run = { at: Date.now(), input: body.url ?? body.text ?? "", res: data };
        runs = [run, ...runs].slice(0, KEEP);
        selected = run.at;
        saveRuns();
        renderResult(run);
        renderAll();
        status.textContent = `Done in ${duration((performance.now() - t0) / 1000)}.`;
    }
    catch {
        el("error-text").textContent = "Cannot reach the backend. Is the server running?";
        show("error");
        status.textContent = "";
    }
    finally {
        window.clearInterval(timer);
        go.disabled = false;
    }
}
el("form").addEventListener("submit", (ev) => {
    ev.preventDefault();
    const url = el("url").value.trim();
    const text = el("text").value.trim();
    const body = {};
    if (url)
        body.url = url;
    if (text)
        body.text = text;
    void submit(body);
});
el("retry").addEventListener("click", () => {
    if (lastBody)
        void submit(lastBody);
});
el("copy").addEventListener("click", async () => {
    const btn = el("copy");
    try {
        await navigator.clipboard.writeText(el("summary").textContent ?? "");
        btn.textContent = "Copied";
    }
    catch {
        btn.textContent = "Copy failed";
    }
    window.setTimeout(() => (btn.textContent = "Copy"), 1500);
});
el("clear").addEventListener("click", () => {
    if (!window.confirm(`Clear the ${runs.length} ${plural(runs.length, "summary", "summaries")} saved in this browser?`))
        return;
    runs = [];
    selected = null;
    saveRuns();
    renderAll();
    renderDetails(null);
    show("empty");
});
function renderHealth(hh) {
    const chip = el("model-chip");
    const score = el("model-score");
    if (!hh?.model) {
        chip.dataset.state = "offline";
        el("model").textContent = "backend offline";
        el("model-kind").textContent = "";
        score.hidden = true;
        return;
    }
    chip.dataset.state = hh.status === "ok" ? "ready" : "loading";
    const last = hh.model.split("/").filter(Boolean).pop() ?? hh.model;
    el("model").textContent = last;
    el("model-kind").textContent = isBase(hh.model) ? "base" : "fine-tuned";
    const r1 = hh.rouge?.rouge1;
    score.hidden = typeof r1 !== "number";
    if (typeof r1 === "number")
        score.textContent = `ROUGE-1 ${r1.toFixed(1)}`;
}
fetch("/api/health")
    .then((r) => r.json())
    .then(renderHealth)
    .catch(() => renderHealth(null));
// ---------- first paint ----------
renderAll();
const latest = runs[0];
if (latest) {
    selected = latest.at;
    renderResult(latest);
}
else {
    renderDetails(null);
    show("empty");
}
export {};
