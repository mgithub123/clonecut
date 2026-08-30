/* clonecut app - talks to the local server in app.py.
   Every action starts a job that runs one of the project's CLI stages; we poll
   it and stream the output into the console at the bottom. */
"use strict";

const $  = (s) => document.querySelector(s);
const $$ = (s) => Array.from(document.querySelectorAll(s));

let STATE = null;
let POLL = null;
let SELECTED_VIDEO_ID = null;

/* ---------------------------------------------------------------- utils */

function human(bytes) {
  if (bytes == null) return "";
  const u = ["B", "KB", "MB", "GB"];
  let i = 0, n = bytes;
  while (n >= 1024 && i < u.length - 1) { n /= 1024; i++; }
  return (i === 0 ? n : n.toFixed(n < 10 ? 1 : 0)) + " " + u[i];
}

/* "20260830-183236" -> "30 Aug, 18:32" so two runs of the same idea are told apart. */
function prettyStamp(s) {
  const m = /^(\d{4})(\d{2})(\d{2})-(\d{2})(\d{2})/.exec(s || "");
  if (!m) return null;
  const months = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"];
  return `${+m[3]} ${months[+m[2] - 1]}, ${m[4]}:${m[5]}`;
}

function toast(msg, bad) {
  const t = $("#toast");
  t.textContent = msg;
  t.classList.toggle("bad", !!bad);
  t.hidden = false;
  clearTimeout(toast._t);
  toast._t = setTimeout(() => { t.hidden = true; }, bad ? 6000 : 3000);
}

async function api(path, opts) {
  const res = await fetch(path, opts);
  const text = await res.text();
  let data;
  try { data = text ? JSON.parse(text) : {}; }
  catch (e) { throw new Error(text.slice(0, 300) || res.statusText); }
  if (!res.ok) throw new Error(data.error || res.statusText);
  return data;
}

const post = (path, body) => api(path, {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify(body || {}),
});

/* --------------------------------------------------------------- console */

const csl = {
  el: $("#console"),
  body: $("#console-body"),
  title: $("#console-title"),
  set(state, title) {
    this.el.dataset.state = state;
    if (title) this.title.textContent = title;
  },
  clear() { this.body.textContent = ""; },
  write(lines) {
    if (!lines || !lines.length) return;
    this.body.textContent += lines.join("\n") + "\n";
    this.body.scrollTop = this.body.scrollHeight;
  },
  open(yes) {
    this.el.classList.toggle("open", yes !== false);
    $("#console-toggle").setAttribute("aria-expanded", String(yes !== false));
  },
};

$("#console-toggle").addEventListener("click", () => {
  csl.open(!csl.el.classList.contains("open"));
});

/* Run a job and resolve with its final snapshot. */
function runJob(jobId, label) {
  csl.clear();
  csl.open(true);
  csl.set("running", label + "…");
  return new Promise((resolve) => {
    let since = 0;
    clearInterval(POLL);
    POLL = setInterval(async () => {
      let snap;
      try { snap = await api(`/api/job/${jobId}?since=${since}`); }
      catch (e) { return; }
      since = snap.total_lines;
      csl.write(snap.lines);
      if (snap.status === "running") return;
      clearInterval(POLL);
      const ok = snap.status === "ok";
      csl.set(ok ? "ok" : "failed",
        `${label} — ${ok ? "done" : "failed"} in ${snap.elapsed}s`);
      if (!ok) {
        toast(`${label} failed — see the console at the bottom`, true);
        csl.open(true);
      }
      resolve(snap);
    }, 450);
  });
}

async function startAndRun(path, body, label) {
  let r;
  try { r = await post(path, body); }
  catch (e) { toast(e.message, true); csl.set("failed", label + " — " + e.message); return null; }
  const snap = await runJob(r.job, label);
  await refresh();
  return snap;
}

/* ----------------------------------------------------------------- tabs */

$$(".step").forEach((btn) => btn.addEventListener("click", () => showPanel(btn.dataset.panel)));

function showPanel(id) {
  $$(".step").forEach((b) => b.classList.toggle("is-active", b.dataset.panel === id));
  $$(".panel").forEach((p) => p.classList.toggle("is-active", p.id === id));
  window.scrollTo({ top: 0, behavior: "instant" });
}

/* --------------------------------------------------------------- uploads */

function wireDrop(zoneSel, inputSel, dir) {
  const zone = $(zoneSel), input = $(inputSel);
  zone.addEventListener("click", () => input.click());
  zone.addEventListener("keydown", (e) => {
    if (e.key === "Enter" || e.key === " ") { e.preventDefault(); input.click(); }
  });
  input.addEventListener("change", () => { upload(Array.from(input.files), dir); input.value = ""; });
  ["dragenter", "dragover"].forEach((ev) =>
    zone.addEventListener(ev, (e) => { e.preventDefault(); zone.classList.add("over"); }));
  ["dragleave", "drop"].forEach((ev) =>
    zone.addEventListener(ev, (e) => { e.preventDefault(); zone.classList.remove("over"); }));
  zone.addEventListener("drop", (e) => {
    if (e.dataTransfer && e.dataTransfer.files.length) {
      upload(Array.from(e.dataTransfer.files), dir);
    }
  });
}

function uploadOne(file, dir, onProgress) {
  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    xhr.open("POST", `/api/upload?dir=${dir}&name=${encodeURIComponent(file.name)}`);
    xhr.upload.addEventListener("progress", (e) => {
      if (e.lengthComputable) onProgress(e.loaded / e.total);
    });
    xhr.onload = () => {
      let data = {};
      try { data = JSON.parse(xhr.responseText); } catch (e) { /* ignore */ }
      if (xhr.status >= 200 && xhr.status < 300) resolve(data);
      else reject(new Error(data.error || `upload failed (${xhr.status})`));
    };
    xhr.onerror = () => reject(new Error("upload failed - is the app still running?"));
    xhr.send(file);
  });
}

async function upload(files, dir) {
  const box = $("#uploads");
  for (const file of files) {
    const row = document.createElement("div");
    row.className = "up";
    row.innerHTML = `<span class="nm"></span><span class="bar"><i></i></span><span class="pc"></span>`;
    row.querySelector(".nm").textContent = file.name;
    box.appendChild(row);
    const bar = row.querySelector("i"), pc = row.querySelector(".pc");
    try {
      await uploadOne(file, dir, (f) => {
        bar.style.width = (f * 100).toFixed(0) + "%";
        pc.textContent = (f * 100).toFixed(0) + "%";
      });
      bar.style.width = "100%";
      pc.textContent = "done";
      setTimeout(() => row.remove(), 1200);
    } catch (e) {
      pc.textContent = "failed";
      toast(`${file.name}: ${e.message}`, true);
      setTimeout(() => row.remove(), 5000);
    }
  }
  await refresh();
}

wireDrop("#drop-raw", "#file-raw", "raw");
wireDrop("#drop-music", "#file-music", "music");

/* ------------------------------------------------------------- rendering */

function fileList(el, items, kind, checkType) {
  el.innerHTML = "";
  if (!items.length) {
    const li = document.createElement("li");
    li.className = "empty";
    li.textContent = kind === "raw" ? "no clips yet" : "no music yet";
    el.appendChild(li);
    return;
  }
  for (const f of items) {
    const li = document.createElement("li");
    const label = document.createElement("label");
    const input = document.createElement("input");
    input.type = checkType;
    input.name = kind;
    input.value = f.path;
    input.addEventListener("change", updateIngestButton);
    const nm = document.createElement("span");
    nm.className = "nm"; nm.textContent = f.name;
    const sz = document.createElement("span");
    sz.className = "sz"; sz.textContent = human(f.size);
    label.append(input, nm);
    const x = document.createElement("button");
    x.className = "x"; x.type = "button"; x.title = "Remove"; x.textContent = "×";
    x.addEventListener("click", async () => {
      if (!confirm(`Remove ${f.name}? This deletes the file from ${kind}/.`)) return;
      try { await post("/api/delete", { path: f.path }); await refresh(); }
      catch (e) { toast(e.message, true); }
    });
    li.append(label, sz, x);
    el.appendChild(li);
  }
}

function checkedValues(name) {
  return $$(`input[name="${name}"]:checked`).map((i) => i.value);
}

function updateIngestButton() {
  const clips = checkedValues("raw");
  const track = checkedValues("music");
  const ok = clips.length > 0 && track.length === 1;
  $("#btn-ingest").disabled = !ok || !STATE || !STATE.ffmpeg;
  $("#ingest-note").textContent = !STATE || !STATE.ffmpeg
    ? "ffmpeg is not installed — see the banner at the top."
    : ok ? `${clips.length} clip(s) + ${track[0].split("/")[1]}`
         : "Tick at least one clip and exactly one music track.";
}

function renderProfiles() {
  const sel = $("#profile");
  const prev = sel.value;
  sel.innerHTML = "";
  for (const p of STATE.profiles) {
    const o = document.createElement("option");
    o.value = p.path;
    o.textContent = p.name + (p.bpm ? `  (${Math.round(p.bpm)} BPM)` : "");
    sel.appendChild(o);
  }
  if (prev) sel.value = prev;
  const has = STATE.profiles.length > 0;
  $("#btn-prompt").disabled = !has;
  const cur = STATE.profiles.find((p) => p.path === sel.value);
  $("#profile-hint").textContent = has
    ? (cur ? `${cur.clips.length} clip(s) · ${cur.duration ? cur.duration.toFixed(1) + "s track" : ""}` : "")
    : "Nothing analysed yet — do step 1 first.";
}

function renderPlans() {
  const ul = $("#plans");
  ul.innerHTML = "";
  if (!STATE.plans.length) {
    const li = document.createElement("li");
    li.className = "empty-note";
    li.textContent = "No plans yet. Do step 2 to get some from Claude.";
    ul.appendChild(li);
    $("#btn-render").disabled = true;
    return;
  }
  for (const p of STATE.plans) {
    const li = document.createElement("li");
    const cb = document.createElement("input");
    cb.type = "checkbox"; cb.name = "edl"; cb.value = p.path;
    cb.addEventListener("change", () => {
      $("#btn-render").disabled = checkedValues("edl").length === 0;
    });
    const body = document.createElement("div");
    body.className = "pl-body";
    const title = document.createElement("div");
    title.className = "pl-title";
    const b = document.createElement("b"); b.textContent = p.variant;
    const meta = document.createElement("span");
    meta.className = "meta";
    meta.textContent = [
      p.segments != null ? `${p.segments} cuts` : null,
      p.captions != null ? `${p.captions} captions` : null,
      p.duration != null ? `${p.duration}s` : null,
      prettyStamp(p.stamp),
    ].filter(Boolean).join(" · ");
    title.append(b, meta);
    const why = document.createElement("p");
    why.className = "why"; why.textContent = p.notes || "";
    body.append(title, why);
    li.append(cb, body);
    ul.appendChild(li);
  }
  $("#btn-render").disabled = checkedValues("edl").length === 0;
}

function renderOutputs() {
  const box = $("#outputs");
  box.innerHTML = "";
  if (!STATE.outputs.length) {
    const d = document.createElement("div");
    d.className = "empty-note";
    d.textContent = "Nothing rendered yet.";
    box.appendChild(d);
    return;
  }
  for (const o of STATE.outputs) {
    const card = document.createElement("div");
    card.className = "out";
    const v = document.createElement("video");
    v.src = `/media?path=${encodeURIComponent(o.path)}`;
    v.controls = true; v.preload = "metadata"; v.playsInline = true;
    const bd = document.createElement("div");
    bd.className = "out-body";
    const nm = document.createElement("div");
    nm.className = "nm"; nm.textContent = o.name;
    const row = document.createElement("div");
    row.className = "row";
    const sheet = document.createElement("button");
    sheet.className = "secondary"; sheet.type = "button";
    sheet.textContent = o.sheet ? "View frames" : "Check frames";
    sheet.addEventListener("click", async () => {
      if (o.sheet) { window.open(`/media?path=${encodeURIComponent(o.sheet)}`, "_blank"); return; }
      await startAndRun("/api/contact-sheet", { video: o.path }, "Contact sheet");
    });
    const save = document.createElement("a");
    save.className = "secondary";
    save.href = `/media?path=${encodeURIComponent(o.path)}`;
    save.download = o.name;
    save.textContent = "Save";
    row.append(sheet, save);
    bd.append(nm, row);
    card.append(v, bd);
    box.appendChild(card);
  }
}

function renderLogged() {
  const tb = $("#logged tbody");
  tb.innerHTML = "";
  if (!STATE.logged.length) {
    const tr = document.createElement("tr");
    const td = document.createElement("td");
    td.colSpan = 8; td.className = "muted"; td.textContent = "Nothing logged yet.";
    tr.appendChild(td); tb.appendChild(tr);
  }
  for (const v of STATE.logged) {
    const tr = document.createElement("tr");
    const cells = [
      v.id, v.variant_name, v.posted_at || "—", v.hook_type || "—",
      v.song_section || "—", v.views != null ? v.views.toLocaleString() : "—", v.pulls,
    ];
    cells.forEach((c, i) => {
      const td = document.createElement("td");
      td.textContent = c;
      if (i === 0 || i === 5 || i === 6) td.className = "num";
      tr.appendChild(td);
    });
    const td = document.createElement("td");
    const btn = document.createElement("button");
    btn.className = "secondary"; btn.type = "button"; btn.textContent = "Add numbers";
    btn.addEventListener("click", () => openMetrics(v));
    td.appendChild(btn); tr.appendChild(td);
    tb.appendChild(tr);
  }

  const sel = $("#post-video");
  const prev = sel.value;
  sel.innerHTML = "";
  const logged = new Set(STATE.logged.map((v) => v.video_path));
  for (const o of STATE.outputs) {
    const opt = document.createElement("option");
    opt.value = o.path;
    opt.textContent = o.name + (logged.has(o.path) ? "  (already logged)" : "");
    sel.appendChild(opt);
  }
  if (prev) sel.value = prev;
  $("#btn-post").disabled = STATE.outputs.length === 0;
}

function openMetrics(v) {
  SELECTED_VIDEO_ID = v.id;
  $("#metrics-for").textContent = `#${v.id} ${v.variant_name}`;
  $$("#metrics-form input").forEach((i) => { i.value = ""; });
  $('#metrics-form input[data-m="pulled_at"]').value = new Date().toISOString().slice(0, 10);
  $("#metrics-form").hidden = false;
  $("#metrics-form").scrollIntoView({ behavior: "smooth", block: "center" });
}

/* --------------------------------------------------------------- refresh */

async function refresh() {
  try { STATE = await api("/api/state"); }
  catch (e) { toast("Lost the app server — is the terminal window still open?", true); return; }

  const pill = $("#ffmpeg-pill");
  pill.textContent = STATE.ffmpeg ? "ffmpeg ready" : "ffmpeg missing";
  pill.className = "pill " + (STATE.ffmpeg ? "ok" : "bad");
  pill.title = STATE.ffmpeg_path || "Install ffmpeg, then restart the app";

  $("#clip-count").textContent = STATE.clips.length ? `(${STATE.clips.length})` : "";
  $("#track-count").textContent = STATE.tracks.length ? `(${STATE.tracks.length})` : "";
  $("#minhist").textContent = STATE.min_history;

  const keepRaw = new Set(checkedValues("raw"));
  const keepMusic = new Set(checkedValues("music"));
  fileList($("#clips"), STATE.clips, "raw", "checkbox");
  fileList($("#tracks"), STATE.tracks, "music", "radio");
  $$('input[name="raw"]').forEach((i) => { i.checked = keepRaw.has(i.value); });
  $$('input[name="music"]').forEach((i) => { i.checked = keepMusic.has(i.value); });
  if (!keepMusic.size && STATE.tracks.length === 1) {
    const only = $('input[name="music"]');
    if (only) only.checked = true;
  }
  updateIngestButton();

  renderProfiles();
  renderPlans();
  renderOutputs();
  renderLogged();

  const meta = STATE.latest_plan;
  if (meta && meta.prompt_exists) showHandoff(meta);
}

function showHandoff(meta) {
  $("#handoff").hidden = false;
  $("#prompt-size").textContent =
    `${meta.prompt_chars.toLocaleString()} characters · ${meta.stamp}`;
  $("#kf-size").textContent = `${meta.keyframe_count} images as a .zip — unzip, then attach them all`;
  $("#btn-kf").href = `/api/keyframes.zip?stamp=${encodeURIComponent(meta.stamp)}`;
  $("#btn-copy-prompt").dataset.stamp = meta.stamp;
}

/* --------------------------------------------------------------- actions */

$("#profile").addEventListener("change", renderProfiles);

$("#btn-ingest").addEventListener("click", async () => {
  await startAndRun("/api/ingest", {
    videos: checkedValues("raw"),
    audio: checkedValues("music")[0],
    no_transcript: $("#no-transcript").checked,
    force: $("#force").checked,
  }, "Analysing footage");
  if (STATE && STATE.profiles.length) showPanel("p2");
});

$("#btn-prompt").addEventListener("click", async () => {
  const snap = await startAndRun("/api/plan/prompt", {
    profile: $("#profile").value,
    notes: $("#notes").value,
  }, "Writing the prompt");
  if (snap && snap.status === "ok" && snap.result && snap.result.meta) {
    showHandoff(snap.result.meta);
    $("#handoff").scrollIntoView({ behavior: "smooth", block: "start" });
  }
});

$("#btn-copy-prompt").addEventListener("click", async (e) => {
  const btn = e.currentTarget;
  const stamp = btn.dataset.stamp;
  if (!stamp) return;
  try {
    const { text } = await api(`/api/prompt?stamp=${encodeURIComponent(stamp)}`);
    await navigator.clipboard.writeText(text);
    btn.textContent = "Copied ✓";
    setTimeout(() => { btn.textContent = "Copy prompt"; }, 2000);
  } catch (err) {
    toast("Couldn't copy automatically. Open plans/" + stamp + "-prompt.md and copy it by hand.", true);
  }
});

$("#btn-read").addEventListener("click", async () => {
  const snap = await startAndRun("/api/plan/ingest", {
    profile: $("#profile").value,
    reply: $("#reply").value,
  }, "Reading the plans");
  if (snap && snap.status === "ok") {
    const n = (snap.result && snap.result.new_plans || []).length;
    toast(n ? `${n} plan(s) ready to render` : "No new plans — check the console");
    if (n) { $("#reply").value = ""; showPanel("p3"); }
  }
});

$("#btn-render").addEventListener("click", async () => {
  await startAndRun("/api/render", { edls: checkedValues("edl") }, "Rendering");
});

$("#btn-post").addEventListener("click", async () => {
  const snap = await startAndRun("/api/log/post", {
    video: $("#post-video").value,
    posted_at: $("#post-date").value || undefined,
    caption: $("#post-caption").value || undefined,
    hashtags: $("#post-tags").value || undefined,
  }, "Logging the post");
  if (snap && snap.status === "ok") {
    $("#post-caption").value = ""; $("#post-tags").value = "";
    toast("Logged");
  }
});

$("#btn-metrics").addEventListener("click", async () => {
  const values = {};
  let pulled;
  $$("#metrics-form input").forEach((i) => {
    if (i.dataset.m === "pulled_at") pulled = i.value;
    else values[i.dataset.m] = i.value;
  });
  const snap = await startAndRun("/api/log/metrics", {
    video_id: SELECTED_VIDEO_ID, values, pulled_at: pulled || undefined,
  }, "Saving the numbers");
  if (snap && snap.status === "ok") {
    $("#metrics-form").hidden = true;
    loadReport();
  }
});

$("#btn-metrics-cancel").addEventListener("click", () => { $("#metrics-form").hidden = true; });

async function loadReport() {
  try {
    const { text } = await api("/api/report");
    $("#report").textContent = text.trim() || "Nothing logged yet.";
  } catch (e) { $("#report").textContent = e.message; }
}
$("#btn-report").addEventListener("click", loadReport);

$("#quit").addEventListener("click", async () => {
  if (!confirm("Quit the app? The window stays open but stops working.")) return;
  try { await post("/api/shutdown"); } catch (e) { /* server is gone, expected */ }
  document.body.innerHTML =
    '<div style="padding:3rem;text-align:center;font:15px system-ui">' +
    "clonecut has stopped. Close this tab.</div>";
});

$("#post-date").value = new Date().toISOString().slice(0, 10);

refresh().then(loadReport);
