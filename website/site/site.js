// Shared helpers for index.html and tasks.html.
// Data is bundled in media/data/data.js (works over file:// too); `url` names the dataset.
function loadJSON(url) { return Promise.resolve(window.SITE_DATA[url.split("/").pop().replace(".json", "")]); }
function pct(x) { return (x * 100).toFixed(1) + "%"; }

// Tab groups: one "on" button at a time; `isTab` lets a group hold plain buttons too.
function tabs(id, onPick, isTab = () => true) {
  const root = document.getElementById(id);
  root.querySelectorAll(".tab").forEach(b => b.addEventListener("click", () => {
    if (isTab(b)) { root.querySelectorAll(".tab").forEach(x => { if (isTab(x)) x.classList.remove("on"); }); b.classList.add("on"); }
    onPick(b);
  }));
}

// Play videos only while they are on screen; each loops on its own.
function autoplayVisible() {
  const io = autoplayVisible.io || (autoplayVisible.io = new IntersectionObserver(es => es.forEach(e => {
    if (e.isIntersecting) e.target.play().catch(() => {}); else e.target.pause();
  }), { rootMargin: "120px" }));
  document.querySelectorAll("video").forEach(v => { if (!v.dataset.watched) { v.dataset.watched = 1; v.muted = true; io.observe(v); } });
}

const LEVELS = { L0: "Hold", L1: "Trace", L2: "Write", L3: "Brush", L4: "Sketch", L5: "Paint", L6: "Chopsticks", E2: "Zero-shot" };
function card(r) {
  const badge = r.zero_shot ? '<span class="badge zs">ZERO-SHOT</span>' : r.paper ? '<span class="badge paper">IN THE PAPER</span>' : '<span class="badge next">COMING NEXT</span>';
  const who = { rl: "final RL policy", grasp: "learned grasp", scripted: "scripted preview" }[r.policy];
  return `<div class="card" data-level="${r.level}" data-status="${r.paper ? "paper" : "next"}">
    <video src="media/tasks/${r.id}.mp4" poster="media/tasks/${r.id}.jpg" muted loop playsinline preload="none"></video>
    <div class="body">${badge}<span class="lvl">${r.level} · ${LEVELS[r.level] || ""}</span>
      <h4>${r.id}</h4><p>${escapeHTML(r.instruction)}</p>
      <div class="meta">${r.tool} · time limit ${r.duration} s · ${who}</div></div></div>`;
}
function escapeHTML(s) { return s.replace(/[&<>"']/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c])); }
