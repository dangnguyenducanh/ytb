// ytb page. Built with `npm run build` in frontend/ into src/ytb/static/main.js (committed, so the
// Python app needs no Node at runtime). Element ids are also used by scripts/manual_session.py.

type Source = "captions" | "whisper";

interface SummaryResponse {
  summary: string;
  video_id: string | null;
  language: string | null;
  source: Source | null;
  chunks: number;
  chunk_tokens: number[];
  input_tokens: number;
  output_tokens: number;
  seconds: number;
  model: string;
  // added later: runs saved in this browser before then lack them, so every use has a fallback
  title?: string | null;
  channel?: string | null;
  duration?: number | null; // seconds of video, when YouTube's captions or the audio gave it
  words_in?: number;
  words_out?: number;
}

interface Run {
  at: number; // epoch ms
  input: string; // the link or the text as submitted
  res: SummaryResponse;
}

export {}; // a module: keeps these names out of the DOM's global scope (it has its own `Body`)

type RequestBody = { url?: string; text?: string };
type View = "empty" | "loading" | "error" | "result";

const STORE = "ytb.runs.v1";
const KEEP = 20; // runs kept in this browser
const SHOW = 4; // runs listed
const DASH = "—";
const SPEAK_WPM = 150; // typical speaking rate: the video's length when it was not recorded
const READ_WPM = 238; // typical silent reading rate
const ERROR_PREFIX: Record<number, string> = {
  422: "Check the link or text",
  404: "No captions or audio for this video",
  413: "Too long (max 150,000 characters, or 60 minutes for videos without captions)",
  503: "YouTube is rate-limiting this machine, try again later",
};
const COUNT_WORDS = ["No", "One", "Two", "Three", "Four", "Five", "Six", "Seven", "Eight", "Nine"];

const nf = new Intl.NumberFormat("en");
const languageNames = new Intl.DisplayNames(["en"], { type: "language" });

// ---------- small helpers ----------
function el<T extends HTMLElement = HTMLElement>(id: string): T {
  const node = document.getElementById(id);
  if (!node) throw new Error(`missing #${id}`);
  return node as T;
}

function h<K extends keyof HTMLElementTagNameMap>(
  tag: K,
  props: Partial<HTMLElementTagNameMap[K]> = {},
  ...children: (Node | string)[]
): HTMLElementTagNameMap[K] {
  const node = Object.assign(document.createElement(tag), props);
  node.append(...children);
  return node;
}

const countWord = (n: number): string => COUNT_WORDS[n] ?? nf.format(n);
const plural = (n: number, one: string, many: string): string => (n === 1 ? one : many);

function duration(s: number): string {
  if (s < 60) return `${Math.round(s)} s`;
  const m = Math.floor(s / 60);
  return `${m} min ${String(Math.round(s - m * 60)).padStart(2, "0")} s`; // taste-skill-ignore: clock format "2 min 05 s"
}

function minutes(s: number): string {
  const m = Math.round(s / 60);
  if (m < 60) return `${m} min`;
  return `${Math.floor(m / 60)} h ${String(m % 60).padStart(2, "0")} min`; // taste-skill-ignore: clock format "1 h 05 min"
}

function clock(s: number): string {
  const t = Math.round(s);
  const mm = String(Math.floor((t % 3600) / 60));
  const ss = String(t % 60).padStart(2, "0"); // taste-skill-ignore: clock format "7:05"
  return t >= 3600 ? `${Math.floor(t / 3600)}:${mm.padStart(2, "0")}:${ss}` : `${mm}:${ss}`; // taste-skill-ignore: clock format "1:07:05"
}

function ago(at: number): string {
  const s = Math.max(0, Math.round((Date.now() - at) / 1000));
  if (s < 60) return "just now";
  if (s < 3600) return `${Math.floor(s / 60)} min ago`;
  if (s < 86400) return `${Math.floor(s / 3600)} h ago`;
  return new Date(at).toISOString().slice(0, 10);
}

function language(code: string | null): string {
  if (!code) return DASH;
  try {
    return languageNames.of(code) ?? code;
  } catch {
    return code;
  }
}

const thumbUrl = (id: string): string => `https://i.ytimg.com/vi/${id}/mqdefault.jpg`;
const watchUrl = (id: string): string => `https://www.youtube.com/watch?v=${id}`;

// ---------- what a summary means for the reader ----------
const wordsIn = (r: SummaryResponse): number => r.words_in ?? Math.round(r.input_tokens * 0.75);
const wordsOut = (r: SummaryResponse): number => r.words_out ?? Math.round(r.output_tokens * 0.75);
const readSeconds = (r: SummaryResponse): number => (wordsOut(r) / READ_WPM) * 60;

/** The video's length: exact when known, else estimated from the words spoken. Null for pasted text. */
function videoLength(r: SummaryResponse): { s: number; exact: boolean } | null {
  if (!r.video_id) return null;
  if (typeof r.duration === "number" && r.duration > 0) return { s: r.duration, exact: true };
  return { s: (wordsIn(r) / SPEAK_WPM) * 60, exact: false };
}

const about = (exact: boolean): string => (exact ? "" : "about ");

function readTime(s: number): string {
  if (s < 45) return "under a minute";
  const m = Math.max(1, Math.round(s / 60));
  return `${m} ${plural(m, "minute", "minutes")}`;
}

function lengthAdjective(s: number): string {
  return s < 60 ? `${Math.round(s)}-second` : `${Math.round(s / 60)}-minute`;
}

function label(run: Run): string {
  if (run.res.title) return run.res.title;
  if (run.res.video_id) return `YouTube video ${run.res.video_id}`;
  const t = run.input.replace(/\s+/g, " ");
  return `“${t.slice(0, 44)}${t.length > 44 ? "…" : ""}”`;
}

// ---------- runs kept in this browser ----------
function isRun(x: unknown): x is Run {
  const r = x as Run;
  return !!r && typeof r.at === "number" && typeof r.input === "string" && !!r.res
    && typeof r.res.summary === "string" && Array.isArray(r.res.chunk_tokens);
}

function loadRuns(): Run[] {
  try {
    const parsed: unknown = JSON.parse(localStorage.getItem(STORE) ?? "[]");
    return Array.isArray(parsed) ? parsed.filter(isRun) : [];
  } catch {
    return [];
  }
}

function saveRuns(): void {
  try {
    localStorage.setItem(STORE, JSON.stringify(runs));
  } catch {
    // storage blocked (private window): the history lives for this page only
  }
}

let runs: Run[] = loadRuns();
let selected: number | null = null;
let lastBody: RequestBody | null = null;

// ---------- the headline: what this summary did for this clip ----------
function renderTitle(run: Run | null): void {
  if (!run) {
    el("greeting").textContent = "Any YouTube video, read in a minute.";
    el("lede").textContent = "Paste a link. Captions are used when a video has them; otherwise its audio is transcribed on this machine.";
    return;
  }
  const r = run.res;
  const v = videoLength(r);
  const read = readTime(readSeconds(r));
  el("greeting").textContent = v
    ? `${v.exact ? "A" : "About a"} ${lengthAdjective(v.s)} video, read in ${read}.`
    : `${nf.format(wordsIn(r))} words, read in ${read}.`;
  const name = r.title ? `“${r.title}”${r.channel ? ` from ${r.channel}` : ""}` : v ? "This video" : "Your text";
  el("lede").textContent = v
    ? `${name}: ${nf.format(wordsIn(r))} spoken words became ${nf.format(wordsOut(r))} to read, `
      + `so you save ${about(v.exact)}${minutes(Math.max(0, v.s - readSeconds(r)))} of watching.`
    : `${name}: ${nf.format(wordsIn(r))} words became ${nf.format(wordsOut(r))} to read.`;
}

function setTile(id: string, value: string, caption: string): void {
  el(`${id}-value`).textContent = value;
  el(`${id}-cap`).textContent = caption;
}

function renderTiles(): void {
  const videos = runs.map((run) => run.res).filter((r) => r.video_id);
  const watch = videos.reduce((a, r) => a + (videoLength(r)?.s ?? 0), 0);
  const read = videos.reduce((a, r) => a + readSeconds(r), 0);
  const estimated = videos.some((r) => !videoLength(r)?.exact);
  const n = videos.length;
  setTile("t-saved", minutes(Math.max(0, watch - read)),
    n ? `versus watching ${countWord(n).toLowerCase()} ${plural(n, "video", "videos")}${estimated ? ", partly estimated" : ""}` : "no videos yet");
  setTile("t-watch", minutes(watch), n ? `read in ${minutes(read)}` : "no videos yet");
  const inW = runs.reduce((a, run) => a + wordsIn(run.res), 0);
  const outW = runs.reduce((a, run) => a + wordsOut(run.res), 0);
  setTile("t-words", nf.format(inW), runs.length ? `condensed to ${nf.format(outW)}` : "nothing read yet");
  const speed = read > 0 ? watch / read : 0;
  setTile("t-speed", speed ? `${nf.format(Math.round(speed))}×` : DASH,
    speed ? `one minute of reading per ${nf.format(Math.round(speed))} of video` : "no videos yet");
}

// ---------- the answer ----------
function show(view: View): void {
  el("empty").hidden = view !== "empty";
  el("loading").hidden = view !== "loading";
  el("error").hidden = view !== "error";
  el("result").hidden = view !== "result";
  el<HTMLButtonElement>("copy").disabled = view !== "result";
}

function sourceWords(r: SummaryResponse): string {
  if (!r.video_id) return "pasted text";
  return r.source === "whisper" ? `${language(r.language)} audio, transcribed on this machine` : `${language(r.language)} captions`;
}

function finding(r: SummaryResponse): string {
  return r.chunks === 1 ? `From ${sourceWords(r)}, read in one pass.` : `From ${sourceWords(r)}, read in ${r.chunks} parts.`;
}

function renderStrip(r: SummaryResponse): void {
  // the parts as stretches of the video: equal token parts, so their times are proportional to tokens
  const v = videoLength(r);
  const total = r.chunk_tokens.reduce((a, t) => a + t, 0) || 1;
  let done = 0;
  const ranges = r.chunk_tokens.map((tokens) => {
    const from = done;
    done += tokens;
    return v ? `${clock((v.s * from) / total)}–${clock((v.s * done) / total)}` : "";
  });
  const segments = r.chunk_tokens.map((tokens, i) => {
    const seg = h("span", { className: "seg", title: v ? `Part ${i + 1}: ${ranges[i]}` : `Part ${i + 1} of ${r.chunks}` });
    seg.style.flexGrow = String(tokens);
    return seg;
  });
  el("strip-fig").hidden = r.chunks < 2;
  el("strip").replaceChildren(...segments);
  el("strip-cap").textContent = v
    ? `Key points from each part, in order: ${ranges.join(", ")}${v.exact ? "" : " (times approximate)"}.`
    : `Key points from each of ${r.chunks} equal parts, in order.`;
}

type Row = [string, string | Node];

function group(title: string, note: string, rows: Row[], lead?: Node): HTMLElement {
  return h("section", { className: "kv-group" },
    h("h3", {}, title),
    h("p", { className: "kv-note" }, note),
    ...(lead ? [h("p", { className: "kv-lead" }, lead)] : []),
    h("dl", { className: "kv" }, ...rows.flatMap(([k, v]) => [h("dt", {}, k), h("dd", {}, v)])));
}

function renderDetails(r: SummaryResponse | null): void {
  // before any run the panel keeps its frame and says one thing instead of a column of dashes
  el("details-h").textContent = r && !r.video_id ? "About this text" : "About this video";
  el("stats-empty").hidden = r !== null;
  el("stats").hidden = r === null;
  if (!r) {
    el("stats").replaceChildren();
    return;
  }
  const v = videoLength(r);
  const read = readSeconds(r);
  const groups: HTMLElement[] = [];
  if (r.video_id) {
    const rows: Row[] = [];
    if (r.channel) rows.push(["Channel", r.channel]);
    rows.push(["Length", v ? (v.exact ? duration(v.s) : `about ${minutes(v.s)}`) : DASH], ["Language", language(r.language)]);
    const title = h("a", { href: watchUrl(r.video_id), target: "_blank", rel: "noopener" }, r.title ?? "Watch on YouTube");
    groups.push(group("Video", "The clip you sent.", rows, title));
  } else {
    groups.push(group("Text", "What you pasted.", [["Length", `${nf.format(wordsIn(r))} words`]]));
  }
  const yours: Row[] = [["Reading time", readTime(read)], ["Words", `${nf.format(wordsIn(r))} → ${nf.format(wordsOut(r))}`]];
  if (v) yours.push(["Time saved", `${about(v.exact)}${minutes(Math.max(0, v.s - read))}`]);
  groups.push(group("Your summary", "What you read instead of watching.", yours));
  groups.push(group("How it was read", "Where the words came from.", [
    ["Source", r.source === "whisper" ? "Audio, transcribed here" : r.source === "captions" ? "The video’s captions" : "Pasted text"],
    ["Coverage", r.chunks === 1 ? "All at once" : `${r.chunks} parts`],
  ]));
  el("stats").replaceChildren(...groups);
}

function renderResult(run: Run): void {
  const r = run.res;
  const link = el<HTMLAnchorElement>("thumb-link");
  const img = el<HTMLImageElement>("thumb");
  if (r.video_id) {
    link.href = watchUrl(r.video_id);
    img.src = thumbUrl(r.video_id);
    img.alt = r.title ? `Thumbnail of “${r.title}”` : `Thumbnail of YouTube video ${r.video_id}`;
    link.hidden = false;
  } else {
    link.hidden = true;
    img.removeAttribute("src");
  }
  renderTitle(run);
  el("finding").textContent = finding(r);
  el("summary").textContent = r.summary;
  renderStrip(r);
  renderDetails(r);
  show("result");
}

// ---------- history ----------
function renderHistory(): void {
  el("hist-count").textContent = nf.format(runs.length);
  el<HTMLButtonElement>("clear").hidden = runs.length === 0;
  const list = el("history");
  if (runs.length === 0) {
    list.replaceChildren(h("li", { className: "empty" }, "Nothing summarized in this browser yet."));
    return;
  }
  const items = runs.slice(0, SHOW).map((run) => {
    const r = run.res;
    const media = r.video_id
      ? h("img", { src: thumbUrl(r.video_id), alt: "", loading: "lazy", width: 64, height: 36 })
      : h("span", { className: "text-tile" }, "Text");
    const v = videoLength(r);
    const size = v ? `${about(v.exact)}${minutes(v.s)} video` : `${nf.format(wordsIn(r))} words`;
    const meta = `${size} · read in ${readTime(readSeconds(r))}${r.channel ? ` · ${r.channel}` : ""}`;
    const row = h("button", { type: "button", className: "row-card" },
      media,
      h("span", { className: "row-main" }, h("span", { className: "row-title" }, label(run)), h("span", { className: "row-meta" }, meta)),
      h("span", { className: "row-when" }, ago(run.at)));
    if (run.at === selected) row.setAttribute("aria-current", "true");
    row.addEventListener("click", () => {
      selected = run.at;
      renderResult(run);
      renderHistory();
    });
    return h("li", {}, row);
  });
  if (runs.length > SHOW) items.push(h("li", { className: "more" }, `${nf.format(runs.length - SHOW)} older in this browser`));
  list.replaceChildren(...items);
}

function renderAll(): void {
  renderTiles();
  renderHistory();
}

// ---------- actions ----------
async function submit(body: RequestBody): Promise<void> {
  lastBody = body;
  const go = el<HTMLButtonElement>("go");
  const status = el("status");
  go.disabled = true;
  show("loading");
  // the headline and the details belong to the answer on screen; neither describes an older clip while this one is pending or failed
  renderTitle(null);
  renderDetails(null);
  const t0 = performance.now();
  const tick = (): void => {
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
    const data: unknown = await res.json().catch(() => ({}));
    if (!res.ok) {
      const detail = (data as { detail?: unknown }).detail;
      el("error-text").textContent = `${ERROR_PREFIX[res.status] ?? "Something went wrong"}: ${typeof detail === "string" ? detail : "the request was not valid."}`;
      show("error");
      status.textContent = "";
      return;
    }
    const run: Run = { at: Date.now(), input: body.url ?? body.text ?? "", res: data as SummaryResponse };
    runs = [run, ...runs].slice(0, KEEP);
    selected = run.at;
    saveRuns();
    renderResult(run);
    renderAll();
    status.textContent = `Done in ${duration((performance.now() - t0) / 1000)}.`;
  } catch {
    el("error-text").textContent = "Cannot reach the backend. Is the server running?";
    show("error");
    status.textContent = "";
  } finally {
    window.clearInterval(timer);
    go.disabled = false;
  }
}

el("form").addEventListener("submit", (ev) => {
  ev.preventDefault();
  const url = el<HTMLInputElement>("url").value.trim();
  const text = el<HTMLTextAreaElement>("text").value.trim();
  const body: RequestBody = {};
  if (url) body.url = url;
  if (text) body.text = text;
  void submit(body);
});

el("retry").addEventListener("click", () => {
  if (lastBody) void submit(lastBody);
});

el("copy").addEventListener("click", async () => {
  const btn = el<HTMLButtonElement>("copy");
  try {
    await navigator.clipboard.writeText(el("summary").textContent ?? "");
    btn.textContent = "Copied";
  } catch {
    btn.textContent = "Copy failed";
  }
  window.setTimeout(() => (btn.textContent = "Copy"), 1500);
});

el("clear").addEventListener("click", () => {
  if (!window.confirm(`Clear the ${runs.length} ${plural(runs.length, "summary", "summaries")} saved in this browser?`)) return;
  runs = [];
  selected = null;
  saveRuns();
  renderAll();
  renderTitle(null);
  renderDetails(null);
  show("empty");
});

// ---------- service status in the top bar ----------
interface Health { status?: string }

function renderHealth(hh: Health | null): void {
  const state = !hh ? "offline" : hh.status === "ok" ? "ready" : "loading";
  const text = state === "ready" ? "Ready" : state === "offline" ? "Offline" : "Starting…";
  el("service").dataset.state = state;
  el("service").title = text; // on a phone only the dot shows
  el("service-text").textContent = text;
}

fetch("/api/health")
  .then((r) => r.json() as Promise<Health>)
  .then(renderHealth)
  .catch(() => renderHealth(null));

// ---------- light / dark ----------
// index.html sets data-theme before first paint (from ?theme= or the saved choice); without one the system decides.
const THEME_KEY = "ytb.theme";
const systemDark = window.matchMedia("(prefers-color-scheme: dark)");

function theme(): "light" | "dark" {
  const t = document.documentElement.dataset.theme;
  return t === "light" || t === "dark" ? t : systemDark.matches ? "dark" : "light";
}

function renderThemeButton(): void {
  const dark = theme() === "dark";
  const text = dark ? "Switch to light theme" : "Switch to dark theme";
  const btn = el<HTMLButtonElement>("theme");
  btn.setAttribute("aria-label", text);
  btn.title = text;
  // the icon shows the theme a click switches to; SVG elements have no .hidden property, so set the attribute
  el("icon-sun").toggleAttribute("hidden", !dark);
  el("icon-moon").toggleAttribute("hidden", dark);
}

el("theme").addEventListener("click", () => {
  const next = theme() === "dark" ? "light" : "dark";
  document.documentElement.dataset.theme = next;
  try {
    localStorage.setItem(THEME_KEY, next);
  } catch {
    // storage blocked: the choice lasts for this page
  }
  renderThemeButton();
});
systemDark.addEventListener("change", renderThemeButton);
renderThemeButton();

// ---------- first paint ----------
renderAll();
const latest = runs[0];
if (latest) {
  selected = latest.at;
  renderResult(latest);
} else {
  renderTitle(null);
  renderDetails(null);
  show("empty");
}
