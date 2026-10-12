"""Complex-query benchmark, step 2: owner-style requests + TRUTH PROGRAMS per library, written by a strong open LLM
(Qwen3.5-27B-FP8, text only, thinking on) from a digest of the library's per-photo facts and metadata.

The LLM never writes an answer set. It writes (1) the owner's words and (2) a truth program: yes/no questions about ONE
photo (answered later by the same 27B LOOKING at every photo of the library: oracle.py) plus exact metadata
constraints (dates, hours, place words, days, anchor photos). build.py turns programs + oracle answers into
candidate answer sets and drops degenerate/ambiguous queries; the eye audit measures how good those sets are.

Usage:
  python eval/complex_bench/gen.py digest            (CPU) -> eval/complex_bench/digests/<user>.txt
  python eval/complex_bench/gen.py gen --shard=k/K   (GPU, vLLM env) -> eval/complex_bench/programs/<user>.json
"""
import json
import os
import random
import re
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from common import ROOT, library_table, load_items  # noqa: E402

CATS = {
 "a": ("object/scene", 5,
       "Things or kinds of scene the owner remembers photographing. Mix COMMON things (\"photos of cows\", \"pictures "
       "inside churches\", \"the waterfall photos\") with SMALL or background things (\"the photo with the red umbrella\", "
       "\"pictures with a bicycle in them\", \"the one with the cat on the windowsill\"). Answer sets 1-150 photos."),
 "b": ("attribute/colour/style", 5,
       "Colour, light, style or photo quality: \"that picture with the cool magenta light\", \"my black and white "
       "photos\", \"the blurry concert pictures\", \"dark night shots of the city\", \"photos with that turquoise water\", "
       "\"close-ups of flowers\", \"photos with reflections in the water\". Usually a look PLUS a subject. Only looks that "
       "the digest shows exist."),
 "c": ("person + look", 4,
       "People's appearance or what they are doing: \"the photo of the guy in the gorilla suit\", \"pictures of kids "
       "with face paint\", \"photos of a woman in a red dress\", \"group photos where everyone wears hats\", \"people "
       "dancing\". If the PEOPLE section names someone, at least one request should be about that person by name with "
       "a look or activity (\"photos of Sam wearing sunglasses\", \"Sam at the beach\") using person + person_need."),
 "d": ("time/place/event", 5,
       "When/where: \"photos from our Venice trip\", \"the beach photos from summer 2011\", \"what did we eat in Tokyo\", "
       "\"photos from the morning of the marathon\", \"everything from the day of the wedding\", \"boats before the "
       "wedding\", \"photos between Christmas and New Year 2010\", \"the selfie I took on the plane last summer\". Use "
       "relative words (\"last summer\", \"two years ago\") relative to TODAY sometimes. Truth uses exact dates/place "
       "words/days from the digest plus (usually) a visual condition."),
 "e": ("multi-hop / cross-photo", 4,
       "The request identifies a moment, place or thing THROUGH ANOTHER PHOTO, never by its date: \"photos from the same "
       "day as the giant inflatable duck\", \"other pictures from the place where I photographed the red tram\", \"what "
       "we ate the day after the concert\", \"the building that is also in the drawing\", \"pictures taken right after "
       "the fireworks\", \"the street signs from the trip where we saw the volcano\". The anchor must be a distinctive "
       "photo listed in the digest (give its id in anchor.ids) and describable unambiguously."),
 "f": ("vague recollection of one photo", 4,
       "The owner half-remembers ONE specific photo and describes it vaguely, the way people do: \"that photo with the "
       "cool light in the bar\", \"the one where the dog is jumping into the lake, a few years back\", \"that weird "
       "statue with the hat, in some park\". Pick a DISTINCTIVE photo from the digest (ids). Program: ids = that photo "
       "(plus near-identical shots listed next to it if any), and need = 1-3 precise questions that the target "
       "photo passes and that other photos would rarely pass."),
 "g": ("comparison / two albums", 3,
       "Two groups at once, the way \"me heavier vs me fit\" is two albums: \"sunny vs rainy photos of the canal\", "
       "\"day vs night shots of the city\", \"cats vs dogs\", \"Sam with glasses vs without\" (named people only from "
       "PEOPLE), \"the beach in summer 2010 vs summer 2012\", \"indoor vs outdoor photos from the wedding\". Program: "
       "exactly 2 albums, in the order the request names them; the two should not overlap."),
 "h": ("conversation with follow-up edits", 3,
       "2 or 3 messages: a first request, then edits that narrow, widen, exclude or undo: \"photos of boats\" -> "
       "\"only the ones at sunset\" -> \"drop the ones with people\"; \"food photos from Japan\" -> \"also add the ones "
       "from Korea\"; \"church photos\" -> \"just the inside ones\" -> \"actually keep the outside ones too\". Give the "
       "full truth program AFTER EACH message (turn_programs, one per message)."),
}

SCHEMA = """TRUTH PROGRAM (per album; every field optional except name):
{"name": short album name,
 "need": [yes/no questions about ONE photo; ALL must be yes],
 "need_any": [yes/no questions; AT LEAST ONE must be yes] (use for "X or Y"),
 "not": [yes/no questions; ALL must be no],
 "person": a name from PEOPLE or null, "person_need": [questions about "the person in the red box"; all must be yes],
 "date_from": "YYYY-MM-DD"|null, "date_to": "YYYY-MM-DD"|null (EXCLUSIVE end),
 "hours": "HH:MM-HH:MM"|null (local clock, may wrap midnight), "place": words that must ALL appear in the photo's address, or null,
 "days": ["YYYY-MM-DD", ...]|null (photos taken on these days),
 "anchor": {"ids": [photo ids from the digest that identify the moment], "need": [questions that the anchor photos pass],
            "window": "same_day"|"same_event"|"same_week"|"same_place"|"days_before:N"|"days_after:N"|"minutes_before:N"|
                      "minutes_after:N"|"before_in_event"|"after_in_event"|"since"|"until"}|null,
 "ids": [photo ids] (category f only)}
Rules for questions: literal, about what is VISIBLE in one photo, answerable by someone who sees only that photo; never
"I/me/my/we/our"; never about dates, places by name (use date/place/days fields), or other photos. Say "real" when
statues/paintings/toys could also match ("Is there a real, live dog in this photo?"). Make them as precise as the
request: "the red umbrella" -> "Is there a red umbrella in this photo?". same_event = the burst of photos with no gap
longer than 3 hours around the anchor; same_place = photos whose address shares the anchor's city."""

PROMPT = """You are building a benchmark of realistic photo-search requests that the OWNER of this photo library would
type into the search box of a phone photo app. Below is a digest of the library, made from per-photo descriptions and
metadata (dates are the camera clock; addresses come from GPS). TODAY is {today}.

{digest}

Write {n} requests of category ({cat}) {cat_name}: {cat_desc}

Each request must:
- sound like the owner talking about their own photos (casual, "my/we/that time", no photo ids, no counts of photos,
  no exact timestamps unless people would say them, e.g. "summer 2011" is fine);
- have a definite answer in THIS library that the truth program captures exactly (1-150 photos; if the words could
  also match photos the program misses, rewrite the words or the program);
- be different from each other (different subjects, places, times).

{schema}

Return ONLY JSON: {{"requests": [{{"turns": ["owner's message", ...], "albums": [album program, ...],
  "turn_programs": [[album programs after message 1], [after message 2], ...] (category h only, else omit),
  "why": "one line: which photos of the digest this is about"}}]}}"""


def digest(user: str, it) -> tuple[str, str]:
    L = library_table(user, it)
    L = L[L.facts.notna()]
    lines = [f"LIBRARY: {len(L)} photos, {L.t.min():%Y-%m-%d} to {L.t.max():%Y-%m-%d}."]
    today = (L.t.max() + pd.Timedelta(days=75)).date()
    lib = json.loads((ROOT / "libraries" / f"{user}.json").read_text()) if (ROOT / "libraries" / f"{user}.json").exists() else {}
    names = ["Sam", "Alex"]
    G = [g for g in lib.get("face_groups", []) if g["n_items"] >= 25][:2]
    if G:
        lines.append("\nPEOPLE (face groups; you may use these names):")
        for nm, g in zip(names, G):
            sub = L[L.item_id.isin(set(g["item_ids"]))]
            ex = "; ".join(str((f.get("people") or [""])[0]) for f in sub.facts.head(40) if f and f.get("people"))[:300]
            lines.append(f"- {nm}: in ~{g['n_items']} photos, {sub.t.min():%Y-%m} to {sub.t.max():%Y-%m}. e.g.: {ex}")
    # days
    lines.append("\nDAYS (date | photos | place | most frequent things | 2 captions):")
    days = L.groupby("day")
    rows = []
    for d, g in days:
        c = Counter(t.split(":", 1)[1] for ts in g.terms for t in ts if t.split(":")[0] in ("objects", "animals", "scene", "event", "landmark"))
        caps = [f.get("caption", "")[:110] for f in g.facts.sample(min(2, len(g)), random_state=0)]
        rows.append((len(g), f"{d} | {len(g)} | {Counter(g.city[g.city != '']).most_common(1)[0][0] if (g.city != '').any() else '?'} | "
                             f"{', '.join(k for k, _ in c.most_common(8))} | {' / '.join(caps)}"))
    keep = sorted(sorted(rows, key=lambda r: -r[0])[:90], key=lambda r: r[1])
    lines += [r[1] for r in keep]
    if len(rows) > 90:
        lines.append(f"(+{len(rows) - 90} smaller days not listed)")
    # concepts
    cnt = Counter(t for ts in L.terms for t in ts)
    for field, title in [("objects", "THINGS"), ("animals", "ANIMALS"), ("scene", "SCENES"), ("depictions", "DEPICTIONS"),
                         ("activities", "ACTIVITIES"), ("event", "OCCASIONS"), ("landmark", "LANDMARKS"),
                         ("image_type", "IMAGE TYPES"), ("lighting", "LIGHTING"), ("quality", "QUALITY"), ("style", "STYLE"),
                         ("weather", "WEATHER"), ("colours", "COLOURS"), ("notable_colour", "NOTABLE COLOURS"),
                         ("people", "PEOPLE COUNT")]:
        items = [(t.split(":", 1)[1], n) for t, n in cnt.items() if t.startswith(field + ":")]
        items.sort(key=lambda x: -x[1])
        top = items[:60] if field in ("objects",) else items[:30]
        lines.append(f"\n{title} (photos): " + ", ".join(f"{k} ({n})" for k, n in top))
    # distinctive photos: rare terms
    rare = {t for t, n in cnt.items() if n <= 2 and t.split(":")[0] in ("objects", "animals", "depictions", "landmark",
                                                                         "notable_colour", "event", "activities")}
    cand = L[L.terms.map(lambda ts: len(ts & rare) >= 1)]
    cand = cand.sample(min(70, len(cand)), random_state=0).sort_values("t")
    lines.append("\nDISTINCTIVE PHOTOS (id | date time | place | caption | rare things):")
    for _, r in cand.iterrows():
        f = r.facts
        lines.append(f"{r.item_id} | {r.t:%Y-%m-%d %H:%M} | {r.city or '?'} | {f.get('caption', '')[:160]} | "
                     f"{', '.join(sorted(t.split(':', 1)[1] for t in r.terms & rare))[:80]}")
    return "\n".join(lines), str(today)


def main_digest():
    it = load_items()
    (ROOT / "digests").mkdir(exist_ok=True)
    for u in sorted(it.user.unique()):
        if not (ROOT / "facts" / f"{u}.jsonl").exists() or (ROOT / "digests" / f"{u}.txt").exists():
            continue
        from common import load_facts
        F = load_facts(u)
        if sum(v is not None for v in F.values()) < 0.98 * (it.user == u).sum():   # facts not finished yet
            continue
        d, today = digest(u, it)
        (ROOT / "digests" / f"{u}.txt").write_text(f"TODAY {today}\n{d}")
        print(u, len(d), "chars", flush=True)


def parse(txt: str):
    txt = txt.split("</think>")[-1]
    s, e = txt.find("{"), txt.rfind("}")
    try:
        return json.loads(txt[s:e + 1])
    except Exception:
        return None


def make_llm(max_len=40960, images=1):
    from vllm import LLM
    return LLM(model=os.environ.get("FP_GEN_MODEL", "Qwen/Qwen3.5-27B-FP8"), max_model_len=max_len,
               gpu_memory_utilization=0.90, trust_remote_code=True, limit_mm_per_prompt={"image": images},
               enable_prefix_caching=True, max_num_seqs=int(os.environ.get("FP_MAX_SEQS", "128")))


def gen_users(llm, users):
    """All categories of all given libraries in one batch (thinking on); -> programs/<user>.json"""
    from vllm import SamplingParams
    sp = SamplingParams(temperature=0.7, top_p=0.9, max_tokens=16000, seed=0)
    (ROOT / "programs").mkdir(exist_ok=True)
    users = [u for u in users if not (ROOT / "programs" / f"{u}.json").exists()]
    msgs, keys, todays = [], [], {}
    for u in users:
        raw = (ROOT / "digests" / f"{u}.txt").read_text()
        today, dig = raw.split("\n", 1)[0].split()[1], raw.split("\n", 1)[1]
        todays[u] = today
        for c, (name, n, desc) in CATS.items():
            p = PROMPT.format(today=today, digest=dig, n=n, cat=c, cat_name=name, cat_desc=desc, schema=SCHEMA)
            msgs.append([{"role": "user", "content": p}]); keys.append((u, c))
    if not msgs:
        return
    outs = llm.chat(msgs, sp, use_tqdm=False, chat_template_kwargs={"enable_thinking": True})
    res = {u: {"user": u, "today": todays[u], "requests": []} for u in users}
    js = [parse(o.outputs[0].text) for o in outs]
    bad = [k for k, j in enumerate(js) if not isinstance(j, dict)]
    if bad:   # one batched retry without thinking
        o2 = llm.chat([msgs[k] for k in bad], SamplingParams(temperature=0.3, max_tokens=6000), use_tqdm=False,
                      chat_template_kwargs={"enable_thinking": False})
        for k, o in zip(bad, o2):
            js[k] = parse(o.outputs[0].text)
    for (u, c), j, o in zip(keys, js, outs):
        for r in (j or {}).get("requests", []) if isinstance(j, dict) else []:
            if isinstance(r, dict):
                r["cat"] = c
                res[u]["requests"].append(r)
        print(u, c, "requests", len(j.get("requests", [])) if isinstance(j, dict) else 0,
              "| gen tokens", len(o.outputs[0].token_ids), flush=True)
    (ROOT / "programs_raw").mkdir(exist_ok=True)
    for u in users:
        raw = {c: o.outputs[0].text[-20000:] for (uu, c), o in zip(keys, outs) if uu == u}
        (ROOT / "programs_raw" / f"{u}.json").write_text(json.dumps(raw))
        (ROOT / "programs" / f"{u}.json").write_text(json.dumps(res[u], indent=1))


if __name__ == "__main__":
    main_digest()
