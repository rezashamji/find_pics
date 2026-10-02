"""Local HTML review page: the receipt + every returned item + the random audit sample, with accept/reject toggles.

Opens from disk (no server). Clicks are kept in the browser and exported as reviews.json; feeding that back gives the
"checked by a person" count in the completeness statement. Thumbnails are written next to the page (small JPEGs), so
the page never touches the originals.
"""
from __future__ import annotations

import html
import json
from pathlib import Path

from .contact import thumb

CSS = """
:root{--bg:#fafaf8;--fg:#1d1d1b;--muted:#6b6b66;--card:#fff;--line:#e4e4df;--ok:#1f7a3a;--bad:#b3261e}
@media (prefers-color-scheme:dark){:root{--bg:#161615;--fg:#ecece8;--muted:#a3a39c;--card:#20201e;--line:#34342f;--ok:#5fc27e;--bad:#f2867d}}
body{margin:0;padding:16px;background:var(--bg);color:var(--fg);font:15px/1.45 system-ui,sans-serif;max-width:1200px;margin:auto}
h1{font-size:22px;margin:8px 0}h2{font-size:18px;margin:28px 0 8px}pre{white-space:pre-wrap;background:var(--card);border:1px solid var(--line);padding:10px;border-radius:8px;font-size:13px}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(150px,1fr));gap:8px}
.it{background:var(--card);border:1px solid var(--line);border-radius:8px;overflow:hidden}
.it img{width:100%;height:150px;object-fit:cover;display:block}.meta{font-size:12px;color:var(--muted);padding:4px 6px}
.it.ok{outline:3px solid var(--ok)}.it.bad{outline:3px solid var(--bad);opacity:.6}
button{font:inherit;padding:6px 10px;border-radius:6px;border:1px solid var(--line);background:var(--card);color:var(--fg);cursor:pointer}
"""

JS = """
const KEY='findpics-reviews-'+document.title;let R={};try{R=JSON.parse(localStorage.getItem(KEY)||'{}')}catch(e){}
function paint(){document.querySelectorAll('.it').forEach(d=>{d.classList.remove('ok','bad');const v=R[d.dataset.id];if(v)d.classList.add(v)})}
document.addEventListener('click',e=>{const d=e.target.closest('.it');if(!d)return;const v=R[d.dataset.id];
R[d.dataset.id]=v==='ok'?'bad':v==='bad'?undefined:'ok';if(!R[d.dataset.id])delete R[d.dataset.id];
try{localStorage.setItem(KEY,JSON.stringify(R))}catch(e){};paint()});
function refineCmd(){const ins=document.getElementById('ins').value.replace(/"/g,"'");
const sel=Object.keys(R).filter(k=>R[k]==='bad'||R[k]==='ok');
const cmd=`findpics refine "${document.body.dataset.dir}" "${ins}"`+(sel.length?` --selected ${sel.join(',')}`:'');
document.getElementById('cmd').textContent=cmd;try{navigator.clipboard.writeText(cmd)}catch(e){}}
function exportR(){const b=new Blob([JSON.stringify(R,null,1)],{type:'application/json'});const a=document.createElement('a');a.href=URL.createObjectURL(b);a.download='reviews.json';a.click()}
paint();
"""


def write_review_page(out_dir: str | Path, request: str, plan: dict, albums: list[dict], max_items: int = 400) -> Path:
    """albums: [{name, report, items: [{item_id, path, label}], audit: [{item_id, path, label}]}]"""
    out = Path(out_dir); th = out / "thumbs"; th.mkdir(parents=True, exist_ok=True)

    def card(it):
        tp = th / (str(it["item_id"]).replace("/", "_") + ".jpg")
        if not tp.exists():
            try:
                thumb(it["path"], 300).save(tp, quality=80)
            except Exception:
                pass
        return (f'<div class="it" data-id="{html.escape(str(it["item_id"]))}"><img loading="lazy" src="thumbs/{tp.name}" alt="">'
                f'<div class="meta">{html.escape(str(it.get("label", "")))}</div></div>')

    parts = [f"<!doctype html><html><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>"
             f"<title>find_pics review</title><style>{CSS}</style></head><body data-dir='{html.escape(str(Path(out_dir).resolve()))}'>",
             f"<h1>find_pics</h1><p><b>Request:</b> {html.escape(request)}</p>",
             "<p>Click a photo once = correct (green), twice = wrong (red), three times = clear. "
             "<button onclick='exportR()'>Export reviews.json</button></p>",
             "<p>Change the albums in words: <input id='ins' size='60' placeholder='e.g. remove the blurry ones / add more like these'>"
             " <button onclick='refineCmd()'>Copy refine command</button> (clicked photos are passed as the selection)</p>"
             "<pre id='cmd'></pre>",
             f"<details><summary>Plan the model made from your sentence</summary><pre>{html.escape(json.dumps(plan, indent=1))}</pre></details>"]
    for a in albums:
        parts.append(f"<h2>{html.escape(a['name'])} ({len(a['items'])})</h2><pre>{html.escape(a['report'])}</pre><div class='grid'>")
        parts += [card(it) for it in a["items"][:max_items]]
        parts.append("</div>")
        if a.get("audit"):
            parts.append(f"<h3>Random audit sample the judge said NO to ({len(a['audit'])} shown). "
                         "Mark any that should have been included: each one is a miss the bound must account for.</h3><div class='grid'>")
            parts += [card(it) for it in a["audit"][:200]]
            parts.append("</div>")
    parts.append(f"<script>{JS}</script></body></html>")
    p = out / "index.html"
    p.write_text("\n".join(parts))
    return p
