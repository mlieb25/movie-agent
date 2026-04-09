# Movie Recommender – Starter

A minimal API that recommends a movie based on a user's stated preferences. It picks from the 40 most-voted movies in the TMDB dataset and uses an LLM to choose the best match and write a short pitch.

---

## Key libraries

### FastAPI

[FastAPI](https://fastapi.tiangolo.com/) is the web framework. It handles routing HTTP requests to Python functions.

The app is created with one line:

```python
app = FastAPI(title="Movie Recommender")
```

Routes are declared with decorators. The `@app.post("/recommend")` decorator means: when the server receives a `POST` request to `/recommend`, call the `recommend()` function.

```python
@app.post("/recommend", response_model=RecommendResponse)
def recommend(request: RecommendRequest):
    ...
```

FastAPI automatically reads the JSON body of the incoming request, validates it against `RecommendRequest`, and serializes the return value to JSON using `RecommendResponse`. You do not need to call `json.loads` or `json.dumps` yourself.

### Pydantic

[Pydantic](https://docs.pydantic.dev/) is what FastAPI uses under the hood to define and enforce data shapes. You declare a class that inherits from `BaseModel`, and Pydantic will automatically parse and validate incoming data against it.

There are three models in this project:

```python
class WatchHistoryItem(BaseModel):
    tmdb_id: int
    name: str

class RecommendRequest(BaseModel):
    user_id: int
    preferences: str
    history: list[WatchHistoryItem] = []   # optional, defaults to empty list

class RecommendResponse(BaseModel):
    tmdb_id: int
    user_id: int
    description: str
```

If a request arrives with a missing required field (e.g. no `user_id`), or the wrong type (e.g. `user_id` is a string that can't be cast to int), FastAPI will automatically return a `422 Unprocessable Entity` error — you never see that case inside `recommend()`.

### OpenAI (`openai`)

The [`openai`](https://pypi.org/project/openai/) package is the official Python SDK. It handles authentication and the HTTP call to the API.

```python
from openai import OpenAI

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
```

Making a call looks like this:

```python
response = client.chat.completions.create(
    model="gpt-5-nano",
    messages=[{"role": "user", "content": prompt}],
    temperature=0.7,
    response_format={"type": "json_object"},
)
result = json.loads(response.choices[0].message.content)
```

Setting `response_format` to `{"type": "json_object"}` is called **JSON mode** — it guarantees the model returns valid JSON, so you can call `json.loads` directly without any cleanup.

---

## Running locally

**1. Install dependencies**

```bash
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

**2. Set your API key**

```bash
export OPENAI_API_KEY=your_openai_api_key_here
```

**3. Start the server**

```bash
uvicorn main:app --reload
```

**4. Send a test request**

```bash
curl -X POST http://localhost:8000/recommend \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": 1,
    "preferences": "I love superheroes and feel-good buddy cop stories.",
    "history": [{"tmdb_id": 24428, "name": "The Avengers"}]
  }'
```

You can also open `http://localhost:8000/docs` in a browser for an interactive API explorer (provided automatically by FastAPI).

---

## Deploying to Leapcell

[Leapcell](https://leapcell.io/) can host a FastAPI app directly from a GitHub repo.

**1. Push your code to GitHub.**

**2. Create a new service on Leapcell.**

Connect your GitHub repo. Leapcell will detect `leapcell.yaml` and use it for build and run:

```yaml
build:
  buildCommand: pip install -r requirements.txt
run:
  runCommand: uvicorn main:app --host 0.0.0.0 --port 8080
```

**3. Set your API key secret in the Leapcell dashboard.**

Go to your service's **Environment Variables** settings and add `OPENAI_API_KEY` with your key as the value. Do not commit the key to your repo.

**4. Deploy.**

Leapcell will install dependencies, start the server, and give you a public URL. Submit that URL as your API endpoint.

---

## Improving the baseline

Some ideas to get you started:

- Expand the candidate pool beyond the top 40 (e.g. filter by genre first, then rank).
- Include genre, keywords, or cast in the prompt to give the LLM more signal.
- Use the watch history to steer away from movies too similar to ones already seen.
- Experiment with prompt phrasing — chain-of-thought or few-shot examples often improve output quality.
- Cache responses for identical inputs to stay safely under the 5-second deadline.
