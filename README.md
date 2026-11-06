# PromoPilot

Finds promo clips (20–60 s, hook in the first 3 s, cliffhanger ending) in microdrama screenplays.
Code owns timestamps and validation; the LLM only points at block IDs.

Status: Phase 4D — parser, LLM candidate generation, deterministic validation/repair, a blind quality
judge with Promo DNA and consistency checks, and ranking/reporting. Judge scores are model opinions.

## Setup

```
pip install -r requirements.txt
copy .env.example .env      # set PROMOPILOT_INPUT_DIR and PROMOPILOT_OUTPUT_DIR
```

The input pack is confidential: it is read in place and never copied into this repo.
Outputs contain screenplay text, so they go to `PROMOPILOT_OUTPUT_DIR` (keep it outside synced
folders); if unset they fall back to `./outputs`, which is gitignored.

## Run

```
python -m promopilot.pipeline parse                 # uses the two .env paths
python -m promopilot.pipeline parse --input DIR --out DIR
python -m pytest                                    # real-pack tests skip if no input folder is set
```

Outputs in `<PROMOPILOT_OUTPUT_DIR>/parsed/` (or `PROMOPILOT_PARSED_DIR`, or `--out`):

- `episodes/EPnnn.json` — scenes and blocks (`EPnnn_Bnnn` IDs in file order, times as original strings and integer ms, speaker, delivery notes, off-screen tag, verbatim text, source line) plus anomalies and timeline stats.
- `pack_summary.json` — counts, anomaly totals, speaker inventory per episode.
- `anomalies.csv` — every recorded source anomaly. Anomalies are reported, never corrected.
- `dataset.json` — the source file name and SHA-256 of every parsed episode, and a fingerprint of the set.

### New input packs

One folder holds one dataset. Parsing a different pack into a folder that already holds one is refused,
so episodes from two packs never mix. Either parse into a new folder, or pass `--replace` to move the old
dataset into `<folder>/history/<time>/` first (nothing is deleted):

```
python -m promopilot.pipeline parse --input D:\new_pack --out D:\kuku\parsed_new_pack
python -m promopilot.pipeline generate --episodes 1-10 --run new_pack --parsed D:\kuku\parsed_new_pack
python -m promopilot.pipeline validate --run new_pack        # later stages reuse the run's dataset
```

A run records the dataset it was generated from (`manifest.json` → `dataset`); validate, judge,
compare-judges and rank read that folder unless `--parsed` is given, and refuse a different dataset. Use a
new `--run` name for a new pack. Loading refuses files whose name and content disagree, duplicate episode
numbers, and episodes the dataset record does not list (stale leftovers). Before generating, the parsed
episodes are compared with the input pack (file names and hashes; `--input DIR` to name it,
`--no-input-check` to skip): a changed, added or removed screenplay stops the run until it is re-parsed.

## LLM providers

Both providers are free tiers reached through their OpenAI-compatible endpoints. Set in `.env`:

| Variable | Purpose |
|---|---|
| `LLM_PROVIDER` | `gemini` (candidate generation) or `groq` |
| `GEMINI_API_KEY`, `GEMINI_MODEL` | e.g. `gemini-3.5-flash` |
| `GROQ_API_KEY`, `GROQ_MODEL` | e.g. `openai/gpt-oss-120b` (reserved for independent evaluation) |

Other OpenAI-compatible endpoints (for example a local server) can be declared without code changes.
Nothing here is configured by default:

| Variable | Purpose |
|---|---|
| `PROMOPILOT_PROVIDERS` | comma-separated extra provider names, e.g. `local` (built-in names cannot be redefined) |
| `NAME_BASE_URL` | required, e.g. `LOCAL_BASE_URL=http://127.0.0.1:8000/v1` |
| `NAME_MODEL`, `NAME_JUDGE_MODEL` | generation model and optional judge model |
| `NAME_API_KEY` or `NAME_API_KEY_ENV` | the key, or the name of the variable that holds it |
| `NAME_JSON_MODE`, `NAME_SEND_TEMPERATURE` | `false` if the endpoint rejects `response_format` / `temperature` |
| `NAME_MAX_TOKENS_PARAM` | `max_tokens` (default), `max_completion_tokens` or `none` |

`NAME` is the provider name in capitals (`-` becomes `_`). The capability variables and `NAME_BASE_URL`
also apply to `gemini` and `groq`. A built-in provider on its default URL and parameters keeps its existing
cache keys and output folders; any other endpoint is part of the cache key (host and path only, never keys
or query strings), is recorded in `manifest.json`, `llm_calls.jsonl` and judge summaries, and adds
`__ep-<hash>` to the judge folder name, so two endpoints serving the same model name never share results.

Optional tuning (defaults in `promopilot/config.py`): `PROMOPILOT_MAX_REQUESTS_PER_RUN`,
`PROMOPILOT_MIN_REQUEST_INTERVAL_S`, `PROMOPILOT_RETRY_MAX_ATTEMPTS`, `PROMOPILOT_RETRY_BASE_DELAY_S`,
`PROMOPILOT_RETRY_MAX_DELAY_S`, `PROMOPILOT_REQUEST_TIMEOUT_S`, `PROMOPILOT_GEN_TEMPERATURE`,
`PROMOPILOT_GEN_MAX_TOKENS`, `PROMOPILOT_GEN_CANDIDATES`.

```
python -m promopilot.pipeline check-providers                      # one tiny request per provider
python -m promopilot.pipeline generate --episodes 1,4,6 --run pilot
python -m promopilot.pipeline generate --episodes 1-10 --run r1 --provider groq
python -m promopilot.pipeline generate --episodes 1,4 --run demo --max-requests 0   # cache only
```

Outputs in `<PROMOPILOT_OUTPUT_DIR>/runs/<run>/`: `candidates/EPnnn.json` (grounded and rejected
candidates with reasons, plus the calls made), `manifest.json` and `llm_calls.jsonl` (model, prompt
version, tokens including hidden reasoning tokens, latency, cache hits, retries, errors; never keys).
Raw responses are cached in `<PROMOPILOT_OUTPUT_DIR>/llm_cache/`, so re-runs and resumes cost no
requests. Runs stop cleanly on daily-quota errors and on the request budget, leaving unfinished
episodes `pending`; re-running the same command resumes them. There is no automatic provider fallback.
A run can be built in batches (`--episodes 1-20`, then `21-40`, ...): `manifest.json` → `requested` is
every episode asked for across invocations, `last_requested` the latest batch, and `episodes` keeps each
episode's latest status; completed episodes are skipped unless the model, endpoint or prompt changed.

## Validation (no API calls)

```
python -m promopilot.pipeline validate --run pilot                            # objective rules only
python -m promopilot.pipeline validate --run pilot --suspense-beats beats.json
python -m promopilot.pipeline validate --run pilot --preview-generator-beats  # provisional what-if
```

Writes `runs/<run>/validation/EPnnn.json` (or `validation_preview/`) and `summary.json`; candidate files
are never modified. Hard rules (limits in `.env`, inclusive): valid IDs in the episode, start before end,
20–60 s, hook within 3 s, no dialogue block cut by a boundary, no transition-only ending, and, once a
suspense-bearing beat is supplied, that beat inside the clip, not a transition, with at most 2 s after it.
The suspense beat is a semantic input (judge or human); without it those rules stay `pending`. Repairs
are one deterministic attempt (trim to the beat or past a transition tail, expand for cut dialogue,
extend the start backwards while the hook stays in its window), saved as `<id>_R1` with provenance.

## Blind judge

```
python -m promopilot.pipeline judge --run pilot [--episodes 1,4] [--max-requests 20]
```

Judge provider: `--provider`, else `PROMOPILOT_JUDGE_PROVIDER`, else `gemini`. Judge model, first match
wins: `--model`; `PROMOPILOT_JUDGE_MODEL` (only for the provider named by `PROMOPILOT_JUDGE_PROVIDER`, or
gemini when that is unset); `NAME_JUDGE_MODEL`; the provider's built-in judge default (gemini:
`gemini-3.5-flash-lite`; groq has none); `NAME_MODEL`. So `--provider groq` alone judges with `GROQ_MODEL`,
never a Gemini model. The judge sees only the clip's blocks under neutral IDs (K01…),
clip-relative times and masked text; never the generator's hook, cliffhanger, reasoning or the episode.
It picks the hook (first 3 s) and final suspense beat (last 8 s), scores six dimensions with block-ID
evidence, and returns Promo DNA and soft rejections. Python checks every cited ID, re-runs the hard rules
with the judge's beats (hard failures are final), applies the gates (hook, cliffhanger, context ≥ 3; hook
and beat valid; ending not resolved), and computes the score from hook 30 / cliffhanger 25 / context 30 /
emotion 15. A failing clip gets one repair and the repair is judged from scratch. Outputs:
`runs/<run>/judge/<provider>__<model>/{judgements,decisions}/`, `summary.json`, `llm_calls.jsonl`.
Re-running resumes; clips the provider failed to answer are retried.

Consistency (Phase 4C.1): Python flags contradictions between a score and its reasons (resolved ending
with a passing cliffhanger, listed confusions with a high context score, high emotion with thin evidence,
etc.). Each flagged judgement gets at most one targeted reassessment, stored in `reassessments/` beside
the untouched raw judgement. A revision counts only if it does not raise the flagged score or drop listed
confusions; otherwise the original stands and the dimension is marked for review. Options:
`--prompt-version judge-v1|judge-v2` (v1 is the default and the pilot benchmark; v2 = tightened anchors,
separate output folder and cache namespace), `--select IDS --out-subdir NAME` (judge only these existing
candidates, including stored repairs), `--stop-on-error`, `--new-judgements
none|IDS` (limit new judge calls), `--max-attempts N`, `--no-reassess`, `--baseline` (last ~30 s per
episode). A lock file refuses a second concurrent run on the same folder; replaced files go to `history/`.

Rubric comparison (no API calls; writes `runs/<run>/comparisons/<base>_vs_<new>/`):

```
python -m promopilot.pipeline compare-judges --run pilot --ids EP001_C03,EP004_C02 --base judge-v1 --new judge-v2
```

## Ranking (no API calls)

```
python -m promopilot.pipeline rank --run pilot --prompt-version judge-v1
```

Ranks only eligible clips (hard rules pass, valid judgement, all gates pass) by 30/25/30/15 (ties: higher
cliffhanger, hook, context, then ID). Two outputs:

- **Complete catalogue** (`catalogue.csv`): every eligible clip with a sequential `catalogue_rank`, with no
  per-episode or archetype limit. Clips overlapping a better one in the same episode by more than 50% (of
  the shorter clip) keep their rank, with `duplicate_of` and `duplicate_overlap`.
- **Editorial shortlist** (`shortlist.csv`): from that order, near-duplicates suppressed, at most one clip
  per episode and three per archetype; each row also carries its `catalogue_rank`.

Both CSVs include clip start/end, the hook and the suspense (cliffhanger) beat — block ID, episode
timestamp and the verbatim text of the parsed source block the judge selected — and a one-line
`explanation`. The explanation is model-written: the judge's Promo DNA appeal (`explanation_source =
judge_appeal`), or the generator's `why_it_works` when the judge gave none; nobody has reviewed it.
Missing or invalid evidence leaves a field empty and is listed in `evidence_issues`. The report also covers
best per episode ("no strong promo" when none qualifies), equal-weight sensitivity, Promo DNA, rejection
reasons, repairs, the baseline and API usage. Writes
`runs/<run>/ranking/<judge folder>/{ranking.json,catalogue.csv,candidates.csv,shortlist.csv,report.md}`.

Promos are single continuous spans of one episode; montages stitched from separate moments are out of
scope.

## Interface (local only, read-only)

```
pip install -r requirements-ui.txt
python -m promopilot_ui                 # http://localhost:8501
python -m promopilot_ui --server.port 8601
```

A Streamlit app over the persisted outputs in `PROMOPILOT_OUTPUT_DIR`: the Shortlist and a per-candidate
analysis next to the screenplay lines it covers. It only reads files; it never calls a model, re-ranks
or writes anywhere. Production pages always use the judge-v1 outputs (the rubric environment variable is
ignored), and experimental rubric folders are listed but never mixed in. The server binds to localhost
and usage telemetry is disabled (`.streamlit/config.toml`). The screenplay is confidential: do not
deploy the app or expose the port. Use "Reload data" after re-running a pipeline step.

Free-tier limits are tight: in testing the Gemini free tier allowed 20 requests per day for the
flash model, and failed 503 attempts appeared to count against it. A paid model (for example a larger
Gemini or GPT model) could be used for production by changing `*_MODEL`, or by adding one registry
entry in `promopilot/llm.py` for another OpenAI-compatible endpoint. Nothing in the pipeline depends on it.
