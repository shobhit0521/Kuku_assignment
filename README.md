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

Outputs in `<PROMOPILOT_OUTPUT_DIR>/parsed/`:

- `episodes/EPnnn.json` — scenes and blocks (`EPnnn_Bnnn` IDs in file order, times as original strings and integer ms, speaker, delivery notes, off-screen tag, verbatim text, source line) plus anomalies and timeline stats.
- `pack_summary.json` — counts, anomaly totals, speaker inventory per episode.
- `anomalies.csv` — every recorded source anomaly. Anomalies are reported, never corrected.

## LLM providers

Both providers are free tiers reached through their OpenAI-compatible endpoints. Set in `.env`:

| Variable | Purpose |
|---|---|
| `LLM_PROVIDER` | `gemini` (candidate generation) or `groq` |
| `GEMINI_API_KEY`, `GEMINI_MODEL` | e.g. `gemini-3.5-flash` |
| `GROQ_API_KEY`, `GROQ_MODEL` | e.g. `openai/gpt-oss-120b` (reserved for independent evaluation) |

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

Judge provider/model come from `PROMOPILOT_JUDGE_PROVIDER` / `PROMOPILOT_JUDGE_MODEL` (default
`gemini` / `gemini-3.5-flash-lite`). The judge sees only the clip's blocks under neutral IDs (K01…),
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

Ranks only eligible clips (hard rules pass, valid judgement, all gates pass) by 30/25/30/15, suppresses
clips overlapping a better one in the same episode by more than 50% (of the shorter clip), then selects at
most one clip per episode and three per archetype. Also reports best per episode ("no strong promo" when
none qualifies), equal-weight sensitivity, Promo DNA, rejection reasons, repairs, the baseline and API
usage. Writes `runs/<run>/ranking/<judge folder>/{ranking.json,candidates.csv,shortlist.csv,report.md}`.

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
