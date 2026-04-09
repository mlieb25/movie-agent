import json
import os

import pandas as pd
from openai import OpenAI
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# ---------------------------------------------------------------------------
# DO NOT EDIT: Data loading
# ---------------------------------------------------------------------------

DATA_PATH = os.path.join(os.path.dirname(__file__), "tmdb_top1000_movies.csv")
TOP_N = 40

df = pd.read_csv(DATA_PATH)
TOP_MOVIES = df.nlargest(TOP_N, "vote_count")

# ---------------------------------------------------------------------------
# DO NOT EDIT: FastAPI app and request/response schemas
#
# These define the API contract. Changing them will break the grader.
# ---------------------------------------------------------------------------

app = FastAPI(title="Movie Recommender")


class WatchHistoryItem(BaseModel):
    tmdb_id: int
    name: str


class RecommendRequest(BaseModel):
    user_id: int
    preferences: str
    history: list[WatchHistoryItem] = []


class RecommendResponse(BaseModel):
    tmdb_id: int
    user_id: int
    description: str


# ---------------------------------------------------------------------------
# TODO: Edit these to improve your recommendations
#
# build_prompt()  – controls what the LLM sees; this is the main thing to tune
# call_llm()      – controls how the LLM is called; you can adjust temperature,
#                   swap models, make multiple calls, etc.
# ---------------------------------------------------------------------------

MODEL = "gpt-5-nano"


def build_prompt(request: RecommendRequest) -> str:
    movie_list = "\n".join(
        f'- tmdb_id={row.tmdb_id} | "{row.title}" ({row.year}) | genres: {row.genres} | overview: {row.overview[:200]}'
        for row in TOP_MOVIES.itertuples()
    )
    history_text = (
        ", ".join(f'"{h.name}"' for h in request.history)
        if request.history
        else "none"
    )
    return f"""You are a movie recommendation assistant.

A user is looking for a movie to watch. Here are their preferences:
"{request.preferences}"

Movies they have already watched (do NOT recommend these):
{history_text}

Below is the list of candidate movies you may recommend. You MUST pick exactly one.

{movie_list}

Respond with ONLY a JSON object — no markdown, no extra text — in this exact format:
{{
  "tmdb_id": <integer>,
  "description": "<a compelling blurb ≤500 chars that tells the user why this movie matches their preferences>"
}}"""


def call_llm(prompt: str) -> dict:
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
    )
    return json.loads(response.choices[0].message.content)


# ---------------------------------------------------------------------------
# DO NOT EDIT: Endpoint
#
# Handles validation and enforces the output contract (valid tmdb_id,
# description ≤500 chars). Edit build_prompt / call_llm above instead.
# ---------------------------------------------------------------------------


@app.post("/recommend", response_model=RecommendResponse)
def recommend(request: RecommendRequest):
    if not os.environ.get("OPENAI_API_KEY"):
        raise HTTPException(status_code=500, detail="OPENAI_API_KEY not set")
    prompt = build_prompt(request)
    try:
        result = call_llm(prompt)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"LLM call failed: {e}")

    valid_ids = set(TOP_MOVIES["tmdb_id"].astype(int))
    tmdb_id = int(result.get("tmdb_id", -1))
    if tmdb_id not in valid_ids:
        raise HTTPException(
            status_code=502, detail=f"LLM returned invalid tmdb_id: {tmdb_id}"
        )

    description = str(result.get("description", ""))[:500]

    return RecommendResponse(
        tmdb_id=tmdb_id,
        user_id=request.user_id,
        description=description,
    )


@app.get("/")
def health():
    return {"status": "ok", "candidates": len(TOP_MOVIES)}
