"""
Agentic Movie Recommender — llm.py

get_recommendation() is called once per request with the user's input.
It returns a dict with keys "tmdb_id" and "description".

Architecture (Build-Measure-Learn Cycles 1–3):
  Stage 1 — Feature extraction (rule-based): parse preferences with keyword matching + negation
  Stage 2 — Pandas scoring: weighted genre/tone/keyword/quality/era scoring with dislike penalties
  Stage 3 — Final selection + description (single LLM call): pick best from scored candidates

Design decision: Rule-based extraction is used instead of an LLM call because:
  1. It's instant (<1ms vs ~2-5s for LLM)
  2. It handles negation reliably ("not comedy" → disliked_genres)
  3. The LLM struggled with feature extraction accuracy (e.g., extracting "thriller" for kids movies)
  4. A single LLM call keeps us well under the 20s budget

IMPORTANT: Do NOT hard-code your API key in this file. The grader will supply
its own OLLAMA_API_KEY environment variable when running your submission.
"""

import json
import os
import re
import time
import argparse

import ollama
import pandas as pd

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MODEL = "gemma4:31b-cloud"

DATA_PATH = os.path.join(os.path.dirname(__file__), "tmdb_top1000_movies.csv")
ALL_MOVIES = pd.read_csv(DATA_PATH)

# Canonical genre list (from the CSV's genre strings)
GENRE_KEYWORDS = [
    "action", "adventure", "animation", "comedy", "crime", "documentary",
    "drama", "family", "fantasy", "history", "horror", "music", "mystery",
    "romance", "science fiction", "thriller", "war", "western",
]

# For backwards compatibility, test.py imports TOP_MOVIES and reads VALID_IDS from it.
# We keep this as the full pool so all movie IDs are valid candidates.
TOP_MOVIES = ALL_MOVIES

# Ollama client — reused across calls
_client = None

def _get_client():
    """Lazy-init Ollama client."""
    global _client
    if _client is None:
        _client = ollama.Client(
            host="https://ollama.com",
            headers={"Authorization": f"Bearer {os.environ['OLLAMA_API_KEY']}"},
        )
    return _client


# ---------------------------------------------------------------------------
# Stage 1: Rule-Based Feature Extraction (instant, no LLM)
# ---------------------------------------------------------------------------

# Genre synonyms: map common words/phrases to canonical genre names
GENRE_SYNONYMS = {
    "sci-fi": "science fiction", "scifi": "science fiction", "science-fiction": "science fiction",
    "funny": "comedy", "hilarious": "comedy", "laugh": "comedy", "humorous": "comedy",
    "scary": "horror", "creepy": "horror", "terrifying": "horror", "frightening": "horror",
    "romantic": "romance", "love story": "romance", "love": "romance", "romcom": "romance",
    "animated": "animation", "cartoon": "animation", "pixar": "animation", "anime": "animation",
    "historical": "history", "period piece": "history", "period film": "history",
    "suspense": "thriller", "suspenseful": "thriller", "psychological": "thriller",
    "feel-good": "comedy", "feel good": "comedy", "lighthearted": "comedy", "heartwarming": "drama", "uplifting": "comedy", "happy": "comedy",
    "superhero": "action", "superheroes": "action", "action-packed": "action",
    "war film": "war", "military": "war", "battlefield": "war",
    "musical": "music", "singing": "music",
    "kids": "family", "children": "family", "kid-friendly": "family",
    "detective": "mystery", "whodunit": "mystery", "murder mystery": "mystery",
    "epic": "adventure", "quest": "adventure", "journey": "adventure",
    "magical": "fantasy", "magic": "fantasy", "fairy tale": "fantasy",
    "documentary": "documentary", "true story": "history",
    "gangster": "crime", "mafia": "crime", "heist": "crime",
}

# Negation patterns that indicate the user does NOT want something
NEGATION_PATTERNS = [
    r"(?:not|no|never|without|avoid|skip|hate|dislike|don'?t\s+(?:want|like|enjoy))\s+(?:any\s+)?(\w[\w\s-]*?)(?:\s+(?:movies?|films?|stuff|things?|genres?))?(?:[,.\s]|$)",
    r"(?:nothing\s+(?:too\s+)?)(scary|dark|violent|sad|slow|boring|cheesy|depressing|intense|heavy)",
    r"anything\s+but\s+(\w[\w\s-]*?)(?:\s+(?:movies?|films?))?(?:[,.\s]|$)",
]

# Tone inference patterns
TONE_HUMOR_MAP = {
    "comedic": ["funny", "hilarious", "laugh", "comedy", "humorous", "fun", "lighthearted",
                "feel-good", "feel good", "light", "silly", "goofy", "witty",
                "happy", "uplifting", "cheerful", "joyful"],
    "slapstick": ["slapstick", "physical comedy", "absurd"],
    "serious": ["serious", "intense", "gritty", "heavy", "dark", "bleak", "somber"],
    "dry": ["dry humor", "sarcastic", "deadpan", "dry wit", "subtle humor"],
}

TONE_INTENSITY_MAP = {
    "explosive": ["action-packed", "action packed", "explosive", "adrenaline", "intense action",
                  "thrilling", "high-octane", "blockbuster", "fast-paced"],
    "tense": ["tense", "scary", "creepy", "suspenseful", "edge of seat", "nail-biting", "gripping",
              "psychological", "slow-burn", "mystery", "thriller", "horror", "dark"],
    "calm": ["calm", "relaxing", "peaceful", "gentle", "soothing", "light", "lighthearted",
             "feel-good", "feel good", "cozy", "easy", "chill", "comfort", "heartwarming",
             "family", "kids", "children", "wholesome", "cute",
             "happy", "uplifting", "cheerful", "joyful"],
    "moderate": ["moderate", "balanced", "drama", "adventure"],
}

ERA_KEYWORDS = {
    "classic": ["classic", "old", "vintage", "retro", "80s", "90s", "70s", "60s", "50s", "golden age",
                "old-school", "oldschool", "old school", "nostalgia", "nostalgic"],
    "modern": ["modern", "2000s", "2010s", "contemporary"],
    "recent": ["recent", "new", "latest", "2020s", "2024", "2025", "brand new", "just released",
               "this year", "last year"],
}


def extract_features(preferences: str, history: list[str] = None) -> dict:
    """Extract structured features from free-text preferences using rule-based parsing.

    Returns a dict with: liked_genres, disliked_genres, tone_humor, tone_intensity,
    era_preference, and keywords.
    """
    text = preferences.lower()

    # --- Extract disliked genres (from negation patterns) ---
    disliked_genres = set()
    disliked_words = set()  # track raw words that are negated

    for pattern in NEGATION_PATTERNS:
        for match in re.finditer(pattern, text, re.IGNORECASE):
            neg_text = match.group(1).strip().lower()
            disliked_words.add(neg_text)
            # Map to canonical genre
            if neg_text in GENRE_SYNONYMS:
                disliked_genres.add(GENRE_SYNONYMS[neg_text])
            elif neg_text in GENRE_KEYWORDS:
                disliked_genres.add(neg_text)
            # Also check for tone words that imply genre dislikes
            if neg_text in ("dark", "violent", "gory"):
                disliked_genres.update(["horror", "thriller"])
            elif neg_text in ("sad", "depressing", "heavy"):
                disliked_genres.add("drama")
            elif neg_text in ("boring", "slow"):
                pass  # tone, not genre

    # --- Extract liked genres (excluding negated ones) ---
    liked_genres = set()

    # Check direct genre mentions
    for genre in GENRE_KEYWORDS:
        if genre in text:
            # Make sure this genre isn't in a negation context
            if genre not in disliked_words and genre not in disliked_genres:
                liked_genres.add(genre)

    # Check synonym matches
    for synonym, genre in GENRE_SYNONYMS.items():
        if synonym in text and synonym not in disliked_words:
            if genre not in disliked_genres:
                liked_genres.add(genre)

    # If "romcom" or "romantic comedy" is mentioned, add both
    if "romcom" in text or "romantic comedy" in text:
        liked_genres.update(["romance", "comedy"])

    # --- Extract tone (skip words that appear in negation context) ---
    tone_humor = None
    for tone, words in TONE_HUMOR_MAP.items():
        matching_words = [w for w in words if w in text and w not in disliked_words]
        if matching_words:
            # Don't set comedic tone if comedy is disliked
            if tone in ("comedic", "slapstick") and "comedy" in disliked_genres:
                continue
            tone_humor = tone
            break

    tone_intensity = None
    for tone, words in TONE_INTENSITY_MAP.items():
        matching_words = [w for w in words if w in text and w not in disliked_words]
        if matching_words:
            tone_intensity = tone
            break

    # --- Extract era preference ---
    era_preference = None
    for era, words in ERA_KEYWORDS.items():
        if any(w in text for w in words):
            era_preference = era
            break

    # --- Extract keywords (specific themes/topics) ---
    keyword_patterns = [
        "superhero", "superheroes", "zombie", "zombies", "alien", "aliens",
        "robot", "robots", "space", "time travel", "heist", "spy",
        "coming of age", "coming-of-age", "revenge", "survival",
        "friendship", "road trip", "treasure", "pirate", "pirates",
        "dragon", "dragons", "vampire", "vampires", "witch",
        "dystopian", "dystopia", "post-apocalyptic", "apocalypse",
        "sports", "racing", "boxing", "martial arts", "kung fu",
        "underdog", "prison", "courtroom", "political", "conspiracy",
    ]
    keywords = [kw for kw in keyword_patterns if kw in text]

    return {
        "liked_genres": sorted(liked_genres),
        "disliked_genres": sorted(disliked_genres),
        "tone_humor": tone_humor,
        "tone_intensity": tone_intensity,
        "era_preference": era_preference,
        "keywords": keywords,
    }


# ---------------------------------------------------------------------------
# Stage 2: Pandas Scoring Engine (no LLM)
# ---------------------------------------------------------------------------

def score_candidates(
    features: dict,
    history_ids: list[int],
    max_candidates: int = 15,
) -> pd.DataFrame:
    """Filter and score movies using extracted features. Returns ranked candidates."""
    df = ALL_MOVIES.copy()

    # 1. Exclude watched movies
    if history_ids:
        df = df[~df["tmdb_id"].isin(history_ids)]

    liked_genres = features.get("liked_genres", [])
    disliked_genres = features.get("disliked_genres", [])
    keywords = features.get("keywords", [])

    # 2. Genre match score (weight: 0.40)
    def calc_genre_score(genres_str):
        if pd.isna(genres_str):
            return 0.0
        genres_lower = genres_str.lower()
        if not liked_genres:
            return 0.5  # neutral if no preference
        matches = sum(1 for g in liked_genres if g in genres_lower)
        return matches / len(liked_genres)

    df["genre_score"] = df["genres"].apply(calc_genre_score)

    # 3. Dislike penalty (multiplier: 0.0–1.0)
    def calc_dislike_penalty(genres_str):
        if pd.isna(genres_str) or not disliked_genres:
            return 1.0  # no penalty
        genres_lower = genres_str.lower()
        matches = sum(1 for g in disliked_genres if g in genres_lower)
        if matches == 0:
            return 1.0
        # Each disliked genre match applies a 0.4 penalty
        return max(0.1, 1.0 - matches * 0.4)

    df["dislike_penalty"] = df["genres"].apply(calc_dislike_penalty)

    # 4. Keyword overlap score (weight: 0.15)
    def calc_keyword_score(row):
        if not keywords:
            return 0.0
        searchable = ""
        for col in ["overview", "keywords", "genres"]:
            val = row.get(col, "")
            if not pd.isna(val):
                searchable += " " + str(val).lower()
        matches = sum(1 for kw in keywords if kw in searchable)
        return min(matches / max(len(keywords), 1), 1.0)

    df["keyword_score"] = df.apply(calc_keyword_score, axis=1)

    # 5. Quality score (weight: 0.15) — vote_average and vote_count
    vote_avg_norm = df["vote_average"].fillna(0) / 10.0
    vote_cnt_norm = df["vote_count"].fillna(0).apply(lambda x: min(x / 10000, 1.0))
    df["quality_score"] = vote_avg_norm * 0.7 + vote_cnt_norm * 0.3

    # 6. Tone alignment score (weight: 0.20)
    def calc_tone_score(genres_str):
        if pd.isna(genres_str):
            return 0.0
        genres_lower = genres_str.lower()
        score = 0.0
        humor = features.get("tone_humor")
        intensity = features.get("tone_intensity")

        if humor in ("comedic", "slapstick"):
            if "comedy" in genres_lower:
                score += 0.5
        elif humor in ("serious", "dry"):
            if "drama" in genres_lower or "thriller" in genres_lower:
                score += 0.5

        if intensity == "explosive":
            if "action" in genres_lower or "adventure" in genres_lower:
                score += 0.5
        elif intensity == "tense":
            if any(g in genres_lower for g in ("thriller", "horror", "mystery", "crime")):
                score += 0.5
        elif intensity == "calm":
            if any(g in genres_lower for g in ("drama", "romance", "family", "animation", "comedy")):
                score += 0.5

        return score

    df["tone_score"] = df["genres"].apply(calc_tone_score)

    # 7. Era alignment score (weight: 0.10)
    def calc_era_score(year):
        if pd.isna(year):
            return 0.0
        era = features.get("era_preference")
        if era is None:
            return 0.5  # neutral
        year = int(year)
        if era == "recent" and year >= 2020:
            return 1.0
        elif era == "modern" and 2000 <= year < 2020:
            return 1.0
        elif era == "classic" and year < 2000:
            return 1.0
        return 0.2  # doesn't match era, but don't hard-exclude

    df["era_score"] = df["year"].apply(calc_era_score)

    # 8. Compute weighted total score
    df["total_score"] = (
        df["genre_score"] * 0.40
        + df["tone_score"] * 0.20
        + df["keyword_score"] * 0.15
        + df["quality_score"] * 0.15
        + df["era_score"] * 0.10
    ) * df["dislike_penalty"]

    # 9. Filter to movies with at least some genre relevance, then rank
    if liked_genres:
        relevant = df[df["genre_score"] > 0]
        if len(relevant) >= max_candidates:
            df = relevant

    df = df.nlargest(max_candidates, "total_score")

    return df


# ---------------------------------------------------------------------------
# Stage 3: Final Selection + Description (single LLM call)
# ---------------------------------------------------------------------------

def get_recommendation(preferences: str, history: list[str], history_ids: list[int] = []) -> dict:
    """Return a dict with keys 'tmdb_id' (int) and 'description' (str)."""

    # Stage 1: Extract features (instant, rule-based)
    features = extract_features(preferences, history)

    # Stage 2: Score and rank candidates using pandas
    candidates = score_candidates(features, history_ids)

    # Build movie list for the prompt
    movie_list = "\n".join(
        f'- tmdb_id={row.tmdb_id} | "{row.title}" ({row.year}) '
        f'| genres: {row.genres} | rating: {row.vote_average}/10'
        for row in candidates.itertuples()
    )

    # Build history text
    history_text = (
        ", ".join(f'"{name}"' for name in history) if history else "none"
    )

    # Stage 3: LLM call — select best movie and generate description
    prompt = f"""You are a movie recommendation assistant. Pick the BEST movie for this user from the candidates below.

User preferences: "{preferences}"
Already watched (do NOT recommend): {history_text}

Candidates:
{movie_list}

You MUST pick exactly one movie from the candidates above.
Respond with ONLY a JSON object, no other text:
{{
  "tmdb_id": <integer>,
  "description": "<a personalized pitch (max 500 chars) explaining why this movie matches their preferences>"
}}"""

    client = _get_client()
    response = client.chat(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        format="json",
    )

    # Parse response with fallback for malformed JSON
    content = response.message.content.strip()
    try:
        result = json.loads(content)
    except json.JSONDecodeError:
        # Try to extract JSON from response
        json_match = re.search(r'\{[^{}]*"tmdb_id"[^{}]*\}', content, re.DOTALL)
        if json_match:
            result = json.loads(json_match.group())
        else:
            # Last resort: pick top scored candidate
            top = candidates.iloc[0]
            result = {
                "tmdb_id": int(top["tmdb_id"]),
                "description": f"Based on your preferences, we recommend '{top['title']}' ({top['year']}).",
            }

    # Ensure tmdb_id is an int
    if "tmdb_id" in result and result["tmdb_id"] is not None:
        result["tmdb_id"] = int(result["tmdb_id"])

    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run a local movie recommendation test."
    )
    parser.add_argument(
        "--preferences",
        type=str,
        help="User preferences text. If omitted, you will be prompted.",
    )
    parser.add_argument(
        "--history",
        type=str,
        help='Comma-separated watch history titles. Example: "The Avengers, Up"',
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Print debug info (extracted features, candidate scores).",
    )
    args = parser.parse_args()

    print("Movie recommender – type your preferences and press Enter.")
    print(
        "For watch history, enter comma-separated movie titles (or leave blank)."
    )

    preferences = (
        args.preferences.strip()
        if args.preferences and args.preferences.strip()
        else input("Preferences: ").strip()
    )
    history_raw = (
        args.history.strip()
        if args.history and args.history.strip()
        else input("Watch history (optional): ").strip()
    )
    history = (
        [t.strip() for t in history_raw.split(",") if t.strip()]
        if history_raw
        else []
    )

    print("\nThinking...\n")
    start = time.perf_counter()

    if args.debug:
        # Show feature extraction results
        features = extract_features(preferences, history)
        print(f"📋 Extracted features: {json.dumps(features, indent=2)}")
        candidates = score_candidates(features, [])
        print(f"\n🎬 Top candidates (by score):")
        for _, row in candidates.head(5).iterrows():
            print(f"   {row['title']} ({row['year']}) — score: {row['total_score']:.3f}")
        print()

    result = get_recommendation(preferences, history)
    print(f"\n✅ Recommendation:")
    print(f"   tmdb_id    : {result.get('tmdb_id')}")
    print(f"   description: {result.get('description')}")
    elapsed = time.perf_counter() - start
    print(f"\n⏱  Served in {elapsed:.2f}s")
