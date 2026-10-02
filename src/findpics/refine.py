"""Conversational refinement of an existing album: "remove the blurry ones", "only where I'm smiling", "get rid of ones
like these", "add more like this one", "also include videos".

An instruction becomes a list of edit ops (fixed JSON schema filled by the local LLM; ids come from the review page):
  keep_if(question)          judge every album item, keep the yes        (no new search; cheap: album-sized)
  remove_if(question)        judge every album item, drop the yes
  remove_ids(ids)            drop exactly these
  remove_like(ids, sim)      drop items whose image vector is close to any of these examples (near-duplicates/look-alikes)
  add_like(ids, k)           rank the rest of the library by similarity to these examples, judge the top k with the
                             album's own question, add the yes
Safety: removing only deletes album LINKS (albums.remove_album_link refuses anything else); originals never change.
Every op reports how many items it changed; the previous manifest is kept as manifest.v<N>.json (undo = copy back).
"""
from __future__ import annotations

import json
import os
import re
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

from .albums import remove_album_link


class Op(BaseModel):
    op: str                                   # keep_if | remove_if | remove_ids | remove_like | add_like
    question: str | None = None
    ids: list[str] = Field(default_factory=list)
    k: int = 200
    sim: float = 0.9


class EditPlan(BaseModel):
    album: str | None = None
    ops: list[Op]


TEMPLATE = """You turn a person's follow-up instruction about one of their photo albums into edit operations.
Albums: {albums}. Photos the person selected on the review page: {selected}.
Operations (JSON):
  {{"op":"keep_if","question":"<yes/no question about one photo>"}}    keep only photos where the answer is yes
  {{"op":"remove_if","question":"<yes/no question about one photo>"}}  remove photos where the answer is yes
  {{"op":"remove_ids","ids":[...]}}                                   remove exactly the selected photos
  {{"op":"remove_like","ids":[...]}}                                  remove the selected photos and ones that look like them
  {{"op":"add_like","ids":[...],"k":200}}                             find more photos like the selected ones
Use the selected ids only for ops that need ids. Return ONLY JSON: {{"album": str|null, "ops": [ ... ]}}
Instruction: {instruction}
JSON:"""


def plan_edits(instruction: str, llm, albums: list[str], selected: list[str]) -> EditPlan:
    out = llm(TEMPLATE.format(albums=albums, selected=selected or "none", instruction=instruction.strip()))
    m = re.search(r"\{.*\}", out, re.S)
    P = EditPlan.model_validate(json.loads(m.group(0)))
    sel = set(selected or [])
    for o in P.ops:      # code-enforced: ids may only come from what the person actually selected
        o.ids = [i for i in o.ids if i in sel]
    return P


def item_vectors(idx) -> np.ndarray:
    V = np.zeros((idx.n_items, idx.clip.shape[1]), np.float32)
    np.add.at(V, idx.units["item_row"].to_numpy(), idx.clip.astype(np.float32))
    return V / (np.linalg.norm(V, axis=1, keepdims=True) + 1e-9)


def apply_ops(idx, album: pd.DataFrame, P: EditPlan, judge, album_question: str | None, judge_rows=None,
              accept: float = 0.7) -> tuple[pd.DataFrame, list[str]]:
    """Pure function on (index, album rows) -> (new album rows, log). `judge_rows(idx, judge, rows, question)` -> P(yes)."""
    from .engine import _judge_rows
    jr = judge_rows or (lambda idx, J, rows, q: _judge_rows(idx, J, rows, np.full(len(rows), -1), q))
    row_of = {iid: i for i, iid in enumerate(idx.items.item_id.astype(str))}
    cur = album.copy(); log = []
    V = None
    for o in P.ops:
        n0 = len(cur)
        if o.op in ("keep_if", "remove_if") and o.question and len(cur):
            p = np.asarray(jr(idx, judge, cur.item_row.to_numpy(), o.question))
            yes = p >= accept
            cur = cur[yes] if o.op == "keep_if" else cur[~yes]
            log.append(f"{o.op} '{o.question}': {n0} -> {len(cur)} ({n0 - len(cur)} removed)")
        elif o.op == "remove_ids":
            cur = cur[~cur.item_id.astype(str).isin(set(o.ids))]
            log.append(f"remove_ids: removed {n0 - len(cur)}")
        elif o.op == "remove_like" and o.ids:
            V = item_vectors(idx) if V is None else V
            ex = [row_of[i] for i in o.ids if i in row_of]
            s = (V[cur.item_row.to_numpy()] @ V[ex].T).max(1) if ex else np.zeros(len(cur))
            drop = (s >= o.sim) | cur.item_id.astype(str).isin(set(o.ids)).to_numpy()
            cur = cur[~drop]
            log.append(f"remove_like ({len(ex)} examples, similarity >= {o.sim}): removed {n0 - len(cur)}")
        elif o.op == "add_like" and o.ids:
            V = item_vectors(idx) if V is None else V
            ex = [row_of[i] for i in o.ids if i in row_of]
            if ex:
                s = V @ V[ex].mean(0)
                s[cur.item_row.to_numpy()] = -np.inf; s[ex] = -np.inf
                cand = np.argsort(-s)[: o.k]
                if album_question:
                    keep = np.asarray(jr(idx, judge, cand, album_question)) >= accept
                    cand = cand[keep]
                add = pd.DataFrame(dict(item_row=cand, item_id=idx.items.item_id.to_numpy()[cand],
                                        path=idx.items.path.to_numpy()[cand], reason="added: like your examples"))
                cur = pd.concat([cur, add], ignore_index=True)
            log.append(f"add_like ({len(ex)} examples): judged top {o.k}, added {len(cur) - n0}")
        else:
            log.append(f"{o.op}: skipped (missing question or selection)")
    return cur, log


def write_edit(album_dir: Path, new_items: pd.DataFrame, log: list[str]) -> Path:
    """Version the manifest, then make the folder's links match the new item list (remove via the guarded remover)."""
    album_dir = Path(album_dir); man = album_dir / "manifest.json"
    m = json.loads(man.read_text())
    v = 1 + len(list(album_dir.glob("manifest.v*.json")))
    shutil.copy(man, album_dir / f"manifest.v{v}.json")
    keep_paths = set(new_items.path.astype(str))
    for link in album_dir.iterdir():
        if link.is_symlink() and os.readlink(link) not in keep_paths:
            remove_album_link(link)
    have = {os.readlink(l) for l in album_dir.iterdir() if l.is_symlink()}
    n = len(list(album_dir.iterdir()))
    for i, p in enumerate(new_items.path.astype(str)):
        if p not in have:
            os.symlink(p, album_dir / f"{n + i:05d}_{Path(p).name}")
    m["items"] = json.loads(new_items.to_json(orient="records", default_handler=str))
    m.setdefault("edits", []).append(log)
    man.write_text(json.dumps(m, indent=1, default=str))
    return album_dir
