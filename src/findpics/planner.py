"""Natural-language request -> structured plan, via a fixed prompt template with the request injected.

The LLM never sees photos here; it only rewrites the request into fields the engine executes exactly
(dates, media type, person) plus short visual descriptions that the image-text model and the judge use.
"""
from __future__ import annotations

import json
import re
from datetime import date

from pydantic import BaseModel, Field, ValidationError


class AlbumSpec(BaseModel):
    name: str
    person: str | None = None                       # who must be in it ("me", "Dad", a name) or None
    looks: list[str] = Field(default_factory=list)  # short visual descriptions to rank by, e.g. "overweight man"
    avoid: list[str] = Field(default_factory=list)  # descriptions to push down
    judge_question: str                              # yes/no question the VLM answers per candidate
    date_from: str | None = None                     # ISO date, inclusive
    date_to: str | None = None                       # ISO date, exclusive
    media: str = "any"                               # "photo" | "video" | "any"
    want: str = "all"                                # "all" = find every match; "best" = top-ranked only
    max_items: int | None = None


class Plan(BaseModel):
    albums: list[AlbumSpec]
    notes: str = ""


TEMPLATE = """You convert a person's request about their own photo library into a JSON search plan.
Today's date is {today}. The library owner is {owner}. Known people in the library: {people}.

Return ONLY JSON matching this schema:
{{"albums": [{{"name": str, "person": str|null, "looks": [str], "avoid": [str], "judge_question": str,
  "date_from": "YYYY-MM-DD"|null, "date_to": "YYYY-MM-DD"|null, "media": "photo"|"video"|"any",
  "want": "all"|"best", "max_items": int|null}}], "notes": str}}

Rules:
- One album per group the person asks for. If they ask for two categories, make two albums.
- "person" must be one of the known people, "me" for the owner, or null if no specific person.
- "looks": 1-4 short, concrete VISUAL descriptions a camera could see (e.g. "a man with a heavy build and round face",
  "a slice of bread"). No judgments that need context the image lacks.
- "judge_question": a yes/no question about ONE image, mentioning the person as "the person in the red box" when a
  person is specified, e.g. "Does the person in the red box look overweight in this photo?"
- Convert relative times using today's date ("past 6 months" -> date_from = today minus 6 months). Leave dates null
  when no time is mentioned. Never invent a date range the person did not ask for.
- media: "video" only if they ask only for videos; "any" if they say photos and videos.
- want: "best" if they ask for the best/top items, else "all".

Request: {request}
JSON:"""


def build_prompt(request: str, owner: str = "me", people: list[str] | None = None, today: date | None = None) -> str:
    return TEMPLATE.format(today=(today or date.today()).isoformat(), owner=owner,
                           people=", ".join(people or []) or "unknown", request=request.strip())


def parse_plan(text: str) -> Plan:
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        raise ValueError(f"no JSON object in planner output: {text[:200]!r}")
    return Plan.model_validate(json.loads(m.group(0)))


def plan(request: str, llm, owner="me", people=None, today=None, retries: int = 2) -> Plan:
    """`llm(prompt) -> str`. Retries on malformed JSON, feeding the validation error back."""
    prompt = build_prompt(request, owner, people, today)
    last = None
    for _ in range(retries + 1):
        out = llm(prompt if last is None else prompt + f"\n(Previous output was invalid: {last}. Return valid JSON only.)\nJSON:")
        try:
            return parse_plan(out)
        except (ValueError, ValidationError, json.JSONDecodeError) as e:
            last = str(e)[:300]
    raise ValueError(f"planner failed: {last}")
