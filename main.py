import csv
import json
import os

from openai import OpenAI
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# ---------------------------------------------------------------------------
# Data loading – runs once at startup
# ---------------------------------------------------------------------------

DATA_PATH = os.path.join(os.path.dirname(__file__), "tmdb_top1000_movies.csv")
TOP_N = 40


def load_top_movies(path: str, n: int) -> list[dict]:
    """Return the top-n movies by vote_count as a list of dicts."""
    movies = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                row["vote_count"] = int(row["vote_count"])
            except (ValueError, KeyError):
                row["vote_count"] = 0
            movies.append(row)
    movies.sort(key=lambda r: r["vote_count"], reverse=True)
    return movies[:n]


TOP_MOVIES = load_top_movies(DATA_PATH, TOP_N)

# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(title="Movie Recommender")

# ---------------------------------------------------------------------------
# Request / response schemas
# ---------------------------------------------------------------------------


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
# LLM helper
# ---------------------------------------------------------------------------

MODEL = "gpt-5-nano"


def call_llm(prompt: str) -> dict:
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
    )
    return json.loads(response.choices[0].message.content)


def build_prompt(request: RecommendRequest) -> str:
    movie_list = "\n".join(
        f'- tmdb_id={m["tmdb_id"]} | "{m["title"]}" ({m["year"]}) | genres: {m["genres"]} | overview: {m["overview"][:200]}'
        for m in TOP_MOVIES
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


# ---------------------------------------------------------------------------
# Endpoint
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

    # Validate tmdb_id is in our candidate list
    valid_ids = {int(m["tmdb_id"]) for m in TOP_MOVIES}
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
