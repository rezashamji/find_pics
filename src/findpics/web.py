"""The chat box as a web page: one text box, albums that fill in round by round, tap a photo to mark it wrong, the face
sheet when someone is unknown. Same engine as `findpics chat` (models load once; messages run one at a time).

    findpics web --index <index_dir> --out <session_dir> [--me "Your Name"] [--port 8800]

Prints the URL with a random key. Every request must carry that key: the server listens on the cluster network, and
without the key nobody else on the cluster can see the photos. Nothing here deletes or changes a photo; "wrong" taps
only keep an item out of later answers (same as reviews.json in the CLI).
"""

import io
import json
import queue
import secrets
import socket
import threading
import time
from contextlib import redirect_stdout
from pathlib import Path


class _Tee(io.TextIOBase):
    """Captures what a turn prints, line by line, so the page can show progress while the search runs."""

    def __init__(self, sink: list):
        self.sink, self.buf = sink, ""

    def write(self, s):
        self.buf += s
        while "\n" in self.buf:
            line, self.buf = self.buf.split("\n", 1)
            self.sink.append(line)
        return len(s)


def _t(v):
    """frame_t from a manifest -> JSON-safe (NaN for photos -> None)."""
    try:
        v = float(v)
    except (TypeError, ValueError):
        return None
    return v if v == v else None


def _reply_lines(log: list[str]) -> list[str]:
    """The lines a person wants to read (not the JSON plan or model logs)."""
    keep = ("[round", "Album '", "I don't know", "  reply e.g.", "Stopped", "Redoing:", "' = face group", "Could not",
            "I could not", "There is no group", "Judge answers reused", "  Time of day", "  Library:", "  After removing",
            "'s:", "person '", "reference photos")
    return [l for l in log if l.startswith(keep) or "at least" in l and l.startswith("  ")][-40:]


def serve(a):
    from fastapi import FastAPI, HTTPException, Request
    from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
    import uvicorn
    from .cli import _load, _turn

    ctx = _load(a)
    S, idx = ctx["S"], ctx["idx"]
    key = secrets.token_urlsafe(16)
    out = Path(a.out)
    chat: list[dict] = [dict(text=m, log=[], done=True) for m in S.state["messages"]]
    jobs: queue.Queue = queue.Queue()
    state = dict(busy=False)
    paths = dict(zip(idx.items.item_id.astype(str), idx.items.path))
    media = dict(zip(idx.items.item_id.astype(str), idx.items.media))
    row_of = {k: i for i, k in enumerate(idx.items.item_id.astype(str))}
    bursts: dict = {}   # (turn, album, item ids) -> burst group per item; computed once, the page polls every 2 s

    def worker():
        while True:
            entry = jobs.get()
            state["busy"] = True
            try:
                with redirect_stdout(_Tee(entry["log"])):
                    _turn(ctx, entry["text"], a)
            except SystemExit as e:
                entry["log"].append(f"Could not run this: {e}")
            except Exception as e:   # keep the page alive; show what went wrong
                entry["log"].append(f"Could not run this: {type(e).__name__}: {e}")
            entry["done"] = True
            state["busy"] = False

    threading.Thread(target=worker, daemon=True).start()
    app = FastAPI()

    def check(request: Request):
        if request.query_params.get("key") != key:
            raise HTTPException(403, "missing or wrong key")

    @app.get("/", response_class=HTMLResponse)
    def page(request: Request):
        check(request)
        return HTMLResponse(PAGE.replace("__KEY__", key))

    @app.post("/api/message")
    async def message(request: Request):
        check(request)
        text = str((await request.json()).get("text", "")).strip()[:500]
        if not text:
            raise HTTPException(400, "empty message")
        entry = dict(text=text, log=[], done=False)
        chat.append(entry); jobs.put(entry)
        return JSONResponse(dict(ok=True))

    @app.post("/api/wrong")
    async def wrong(request: Request):
        check(request)
        iid = str((await request.json()).get("item_id", ""))
        if iid not in paths:
            raise HTTPException(404, "unknown item")
        S.add_reviews({iid: "bad"}); S.save()      # kept out of every later answer; the photo itself is untouched
        return JSONResponse(dict(ok=True))

    @app.get("/api/state")
    def api_state(request: Request):
        check(request)
        turn = S.turn
        albums, sheet = [], None
        sj = out / f"turn_{turn}" / "summary.json"
        bad = set(S.state.get("exclude_ids", []))
        if sj.exists():
            sm = json.loads(sj.read_text())
            for al in sm["albums"]:
                man = out / f"turn_{turn}" / al["album"] / "manifest.json"
                items = json.loads(man.read_text())["items"] if man.exists() else []
                items = [i for i in items[:400] if str(i["item_id"]) not in bad]
                ids = tuple(str(i["item_id"]) for i in items)
                bk = (turn, al["album"], ids)
                if bk not in bursts:
                    from .bursts import burst_ids
                    bursts[bk] = burst_ids(idx, [row_of.get(i, 0) for i in ids])
                albums.append(dict(name=al["album"], n=al["n"], report=al["report"],
                                   items=[dict(id=str(i["item_id"]), t=_t(i.get("frame_t")), g=int(g),
                                              thumb=f"/thumb/{turn}/{str(i['item_id']).replace('/', '_')}.jpg")
                                          for i, g in zip(items, bursts[bk])]))
            finished = sm.get("finished", False)
        else:
            finished = False
        if any("I don't know" in l for e in chat[-1:] for l in e["log"]) and (Path(idx.root) / "people_groups.jpg").exists():
            sheet = "/people.jpg"
        return JSONResponse(dict(busy=state["busy"], finished=finished, turn=turn, sheet=sheet,
                                 chat=[dict(text=e["text"], reply=_reply_lines(e["log"]), done=e["done"]) for e in chat],
                                 albums=albums))

    @app.get("/thumb/{turn}/{name}")
    def thumb(turn: int, name: str, request: Request):
        check(request)
        p = (out / f"turn_{turn}" / "thumbs" / Path(name).name).resolve()
        if not str(p).startswith(str(out.resolve())) or not p.exists():
            raise HTTPException(404)
        return FileResponse(p)

    @app.get("/photo/{item_id:path}")
    def photo(item_id: str, request: Request, t: float | None = None):
        check(request)
        if item_id not in paths:          # only items of this library, never an arbitrary file path
            raise HTTPException(404)
        from .media import load_image, video_frame_at
        if media.get(item_id) == "video":   # the frame the search matched (t), else the middle (no player yet)
            im = video_frame_at(paths[item_id], t)
            if im is None:
                raise HTTPException(404)
        else:
            im = load_image(paths[item_id])
        im.thumbnail((1600, 1600))
        buf = io.BytesIO(); im.convert("RGB").save(buf, "JPEG", quality=88)
        from fastapi.responses import Response
        return Response(buf.getvalue(), media_type="image/jpeg")

    @app.get("/people.jpg")
    def people(request: Request):
        check(request)
        return FileResponse(Path(idx.root) / "people_groups.jpg")

    host = socket.gethostname()
    print(f"Open http://{host}:{a.port}/?key={key}  (from a browser on the cluster, e.g. the rcood remote desktop)", flush=True)
    (out / "web_url.txt").write_text(f"http://{host}:{a.port}/?key={key}\n")
    uvicorn.run(app, host="0.0.0.0", port=a.port, log_level="warning")


PAGE = r"""<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>find pics</title><style>
:root{--bg:#fafafa;--fg:#111;--mut:#666;--card:#fff;--line:#e3e3e3;--acc:#2563eb;--bad:#dc2626}
@media (prefers-color-scheme:dark){:root{--bg:#111;--fg:#eee;--mut:#999;--card:#1b1b1b;--line:#333;--acc:#60a5fa;--bad:#f87171}}
*{box-sizing:border-box}html,body{overflow-x:hidden}body{margin:0;font:15px/1.45 system-ui,sans-serif;background:var(--bg);color:var(--fg);overflow-wrap:anywhere}
main{max-width:1100px;margin:0 auto;padding:16px}
#chat .me{font-weight:600;margin-top:14px}#chat .re{color:var(--mut);white-space:pre-wrap;font-size:13px}
main{padding-bottom:84px}
form{position:fixed;left:0;right:0;bottom:0;background:var(--bg);border-top:1px solid var(--line);padding:10px max(16px,calc(50vw - 534px));display:flex;gap:8px}
input{flex:1;font-size:16px;padding:10px 12px;border:1px solid var(--line);border-radius:10px;background:var(--card);color:var(--fg)}
button{padding:10px 16px;border:0;border-radius:10px;background:var(--acc);color:#fff;font-size:15px}
h2{font-size:17px;margin:22px 0 6px}details{margin:0 0 8px}summary{color:var(--mut);font-size:13px;cursor:pointer}
.rep{color:var(--mut);font-size:12px;white-space:pre-wrap}input{min-width:0}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(120px,1fr));gap:6px}
.grid div{position:relative}.grid img{width:100%;aspect-ratio:1;object-fit:cover;border-radius:6px;cursor:pointer;display:block}
.grid button{position:absolute;right:4px;top:4px;padding:2px 7px;font-size:12px;background:var(--bad);opacity:.85}
.grid button.more{right:auto;left:4px;top:auto;bottom:4px;background:rgba(0,0,0,.7)}
.grid div.in{outline:2px solid var(--acc);border-radius:6px}
#sheet img{max-width:100%;border:1px solid var(--line);border-radius:8px}
#big{position:fixed;inset:0;background:#000d;display:none;align-items:center;justify-content:center}#big img{max-width:96vw;max-height:94vh}
.status{color:var(--mut);font-size:13px}
</style></head><body><main>
<div id="chat"></div><div id="sheet"></div><div class="status" id="status"></div><div id="albums"></div>
<form id="f"><input id="q" placeholder="e.g. me heavier vs me fit in the past 6 months" autocomplete="off"><button>Send</button></form>
</main><div id="big" onclick="this.style.display='none'"><img id="bigimg"></div>
<script>
const K="__KEY__", u=p=>p+(p.includes('?')?'&':'?')+'key='+K;
const esc=s=>s.replace(/[&<>"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
let hidden=new Set(), opened={}, last="";
document.getElementById('f').onsubmit=async e=>{e.preventDefault();const q=document.getElementById('q');if(!q.value.trim())return;
 await fetch(u('/api/message'),{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({text:q.value})});q.value='';tick()};
async function wrong(id,el){hidden.add(id);el.parentNode.remove();
 await fetch(u('/api/wrong'),{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({item_id:id})})}
function tiles(a,ai){const vis=a.items.filter(i=>!hidden.has(i.id)),n={},seen={};
 vis.forEach(i=>n[i.g]=(n[i.g]||0)+1);
 return vis.map(i=>{const k=ai+':'+i.g,first=!seen[i.g];seen[i.g]=1;
  if(!first&&!opened[k])return '';
  const more=first&&n[i.g]>1?`<button class="more" title="near-identical shots of this moment" onclick="stack('${k}')">${opened[k]?'−':'+'+(n[i.g]-1)+' similar'}</button>`:'';
  return `<div${!first?' class="in"':''}><img loading="lazy" src="${u(i.thumb)}" onclick="big('${esc(i.id)}',${i.t==null?'null':i.t})"><button title="not right" onclick="wrong('${esc(i.id)}',this)">✕</button>${more}</div>`}).join('')}
function stack(k){opened[k]=!opened[k];last='';tick()}
function big(id,t){document.getElementById('bigimg').src=u('/photo/'+encodeURIComponent(id)+(t==null?'':'?t='+t));document.getElementById('big').style.display='flex'}
async function tick(){const s=await (await fetch(u('/api/state'))).json();
 document.getElementById('chat').innerHTML=s.chat.map(c=>`<div class="me">${esc(c.text)}</div><div class="re">${esc(c.reply.join('\n'))}${c.done?'':'\n…'}</div>`).join('');
 document.getElementById('status').textContent=s.busy?'Searching… results below update each round.':(s.finished?'Done.':'');
 document.getElementById('sheet').innerHTML=s.sheet?`<p>Faces I see most often (reply e.g. "Jay is 4"):</p><img src="${u(s.sheet)}">`:'';
 const html=s.albums.map((a,ai)=>`<h2>${esc(a.name)} — ${a.n}${a.items.length?` <small>(${new Set(a.items.map(i=>i.g)).size} moments)</small>`:""}</h2><details><summary>About this album</summary><div class="rep">${esc(a.report)}</div></details><div class="grid">`+
   tiles(a,ai)+'</div>').join('');
 if(html!==last){document.getElementById('albums').innerHTML=html;last=html}}
tick();setInterval(tick,2000);
</script></body></html>"""
