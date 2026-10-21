"""Build a self-contained HTML page for blind hand-labeling of the 100 sampled traces.

Usage: python scripts/build_label_page.py
Writes results/raw/trace-coding/label.html. Labels are kept in the browser's
localStorage and exported as hand-labels.jsonl ({"trace_id", "category"}).
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DIR = ROOT / "results/raw/trace-coding"
rows = [json.loads(line) for line in (DIR / "sample-100.jsonl").open()]
items = [{"id": r["trace_id"], "user": r["user_message"], "trace": r["reasoning_block"]} for r in rows]
tutor = rows[0]["tutor_system_prompt"]
assert all(r["tutor_system_prompt"] == tutor for r in rows)
instructions = (DIR / "judge_instructions.txt").read_text().split("The supplied JSON")[0].strip()

html = """<!doctype html><html><head><meta charset="utf-8"><title>Trace labeling</title>
<style>
body{font:15px/1.5 system-ui,sans-serif;margin:0;color:#1d1d1f;background:#fafafa}
main{max-width:860px;margin:0 auto;padding:20px 24px 140px}
h1{font-size:17px;margin:0 0 4px}
.meta{color:#666;font-size:13px}
details{margin:10px 0;font-size:13px;color:#444}
pre{white-space:pre-wrap;font:13.5px/1.55 ui-monospace,Menlo,monospace;background:#fff;border:1px solid #e3e3e3;padding:12px;border-radius:6px}
.label{font-size:12px;text-transform:uppercase;letter-spacing:.04em;color:#888;margin:16px 0 4px}
mark{background:#ffe48a}
footer{position:fixed;bottom:0;left:0;right:0;background:#fff;border-top:1px solid #ddd;padding:12px}
footer .in{max-width:860px;margin:0 auto;display:flex;gap:8px;flex-wrap:wrap;align-items:center}
button{font:inherit;padding:8px 12px;border:1px solid #bbb;background:#fff;border-radius:6px;cursor:pointer}
button.sel{background:#1d1d1f;color:#fff;border-color:#1d1d1f}
.sp{flex:1}
</style></head><body><main>
<h1>Reasoning-trace labeling</h1>
<div class="meta" id="prog"></div>
<details><summary>Category definitions</summary><pre>__INSTR__</pre></details>
<details><summary>Tutor system prompt (identical for every trace)</summary><pre>__TUTOR__</pre></details>
<div class="label">Student message</div><pre id="user"></pre>
<div class="label">Reasoning trace <label style="text-transform:none;letter-spacing:0;margin-left:8px"><input type="checkbox" id="hl" checked> highlight tutor/system/instruction words</label></div>
<pre id="trace"></pre>
</main><footer><div class="in">
<button data-c="a">1 · (a) noticed, withheld</button>
<button data-c="b">2 · (b) noticed, answered</button>
<button data-c="c">3 · (c) not noticed</button>
<span class="sp"></span>
<button id="prev">← prev</button><button id="next">next →</button>
<button id="exp">export</button>
</div></footer>
<script>
const ITEMS=__ITEMS__;
const KEY="trace-labels-v1";
let labels=JSON.parse(localStorage.getItem(KEY)||"{}");
let i=ITEMS.findIndex(x=>!labels[x.id]);if(i<0)i=0;
const esc=s=>s.replace(/[&<>]/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;"}[c]));
const RE=/(tutor\\w*|system prompt|system message|instruction\\w*|teach\\w*|homework|final answer|not (?:to )?give)/gi;
function render(){
  const it=ITEMS[i];
  document.getElementById("user").textContent=it.user;
  const t=esc(it.trace);
  document.getElementById("trace").innerHTML=document.getElementById("hl").checked?t.replace(RE,"<mark>$1</mark>"):t;
  const n=Object.keys(labels).length;
  document.getElementById("prog").textContent=`Trace ${i+1} of ${ITEMS.length} · ${n} labeled`+(labels[it.id]?` · this one: (${labels[it.id]})`:"");
  document.querySelectorAll("button[data-c]").forEach(b=>b.classList.toggle("sel",b.dataset.c===labels[it.id]));
  window.scrollTo(0,0);
}
function set(c){labels[ITEMS[i].id]=c;localStorage.setItem(KEY,JSON.stringify(labels));if(i<ITEMS.length-1)i++;render();}
document.querySelectorAll("button[data-c]").forEach(b=>b.onclick=()=>set(b.dataset.c));
document.getElementById("prev").onclick=()=>{if(i>0)i--;render()};
document.getElementById("next").onclick=()=>{if(i<ITEMS.length-1)i++;render()};
document.getElementById("hl").onchange=render;
document.getElementById("exp").onclick=()=>{
  const missing=ITEMS.filter(x=>!labels[x.id]).length;
  if(missing&&!confirm(missing+" traces unlabeled. Export anyway?"))return;
  const out=ITEMS.filter(x=>labels[x.id]).map(x=>JSON.stringify({trace_id:x.id,category:labels[x.id]})).join("\\n")+"\\n";
  const a=document.createElement("a");a.href=URL.createObjectURL(new Blob([out],{type:"application/jsonl"}));a.download="hand-labels.jsonl";a.click();
};
document.addEventListener("keydown",e=>{
  if(e.key==="1")set("a");else if(e.key==="2")set("b");else if(e.key==="3")set("c");
  else if(e.key==="ArrowLeft")document.getElementById("prev").click();
  else if(e.key==="ArrowRight")document.getElementById("next").click();
});
render();
</script></body></html>"""
html = (html.replace("__INSTR__", instructions.replace("&", "&amp;").replace("<", "&lt;"))
        .replace("__TUTOR__", tutor.replace("&", "&amp;").replace("<", "&lt;"))
        .replace("__ITEMS__", json.dumps(items).replace("</", "<\\/")))
(DIR / "label.html").write_text(html)
print(DIR / "label.html", len(items))
