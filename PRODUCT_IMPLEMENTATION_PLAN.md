# Product Implementation Plan (MVP using Dolphin Llama 70B)

## Goal
Build a hosted MVP where a user:
1. Enters their X username once.
2. System ingests new tweets/replies since last sync (incremental).
3. Enters one draft tweet and objective.
4. System creates 10 variants, runs them in parallel using `cognitivecomputations/dolphin-llama-3-70b`, and returns the best option.

Authentication is intentionally excluded for now.

---

## Fixed model decision for MVP
- Primary generation model: `cognitivecomputations/dolphin-llama-3-70b` via OpenRouter.
- This is the only model used for variant generation + simulated replies in MVP.
- Add one fallback model key in config only for outage handling (not exposed to users).

Why this decision:
- Keeps behavior consistent across runs.
- Avoids per-experiment model drift.
- Simplifies debugging/calibration for design partners.

---

## 1) Frontend and Backend

## Simplest production stack
- Frontend: `Next.js` + `Tailwind` (Vercel).
- Backend API: `FastAPI` (Python).
- Job workers: `Redis + RQ` (or ARQ).
- Database: managed Postgres (Supabase Postgres is fine, auth disabled).
- Hosting:
  - `web` on Vercel.
  - `api` + `worker` on Railway/Render/Fly.

This keeps all simulation logic in Python and avoids rewriting your existing pipeline.

## Service boundaries
- `web`: UI and polling only.
- `api`: account setup, sync triggers, experiment creation, results APIs.
- `worker`: Apify ingestion, variant generation, simulation execution, scoring.

## Core backend modules
- `accounts`: track one-time username input and sync checkpoints.
- `ingestion`: trigger Apify actor, fetch results, normalize/upsert.
- `persona`: update account-level profile from latest data.
- `experiments`: create runs, track status, link variant jobs.
- `simulation`: call your existing simulation pipeline using OpenRouter model.
- `ranking`: compute objective score and winner.
- `costs`: capture tokens/latency per variant for operational control.

## Data model (minimum)
- `tracked_accounts`
  - `id`, `x_username`, `x_user_id`, `last_synced_tweet_id`, `last_synced_at`, `status`
- `tweets_raw`
  - `tweet_id`, `author_id`, `text`, `created_at`, `reply_to_id`, `metrics_json`, `source_run_id`
- `account_persona_snapshot`
  - `account_id`, `snapshot_at`, `political_dist_json`, `sentiment_dist_json`, `aggression_stats_json`
- `experiments`
  - `id`, `account_id`, `draft_text`, `objective`, `weights_json`, `status`, `created_at`
- `variants`
  - `id`, `experiment_id`, `variant_text`, `status`, `error`, `started_at`, `finished_at`
- `variant_scores`
  - `variant_id`, `pred_pos`, `pred_neu`, `pred_neg`, `pred_aggression`, `pred_left`, `pred_center`, `pred_right`, `composite_score`, `rank`
- `variant_runtime`
  - `variant_id`, `input_tokens`, `output_tokens`, `llm_latency_ms`, `total_runtime_ms`, `model_id`

## API endpoints (MVP)
- `POST /accounts`
  - input: `x_username`; stores tracked account.
- `POST /accounts/{id}/sync`
  - enqueue incremental Apify sync job.
- `GET /accounts/{id}`
  - sync state + persona summary.
- `POST /experiments`
  - input: `account_id`, `draft_text`, `objective`, `num_variants` (default 10).
  - creates experiment + enqueues 10 variant jobs.
- `GET /experiments/{id}`
  - status + progress counters + partial scores.
- `GET /experiments/{id}/results`
  - ranked variants + winner + runtime/cost summary.

---

## 2) APIs and Rate Limits

## External APIs
- Apify API (data ingestion).
- OpenRouter API (Dolphin Llama 70B inference).

## Rate-limit strategy
Keep hardcoded limits out of code. Read from env-config and adapt dynamically:
- `OPENROUTER_MAX_CONCURRENCY`
- `OPENROUTER_MAX_RPM`
- `OPENROUTER_MAX_TPM`
- `APIFY_MAX_RPS`

## OpenRouter execution policy
- One request queue for variant generation.
- One queue for simulation replies.
- Global concurrency cap (start at 3-5 concurrent LLM calls).
- Retry on `429/5xx` with exponential backoff + jitter.
- Timeout + retry budget per request.
- Log rate-limit headers and request IDs.

## Throughput/cost controls for 70B
- Parallel variants: 10 jobs submitted, worker executes under concurrency cap.
- Keep context window tight:
  - only last N thread messages (e.g. 8-10).
  - no unnecessary prompt text duplication.
- Add per-experiment token budget guardrail.
- If budget exceeded: stop remaining variants and return best-so-far.

---

## 3) Using Apify to Scrape

## One-time setup flow
1. User enters X username once.
2. Resolve and store stable account identity.
3. Initialize sync checkpoint (`last_synced_tweet_id`, `last_synced_at`).

## Incremental sync flow
1. Trigger sync on:
   - manual button,
   - scheduled job (every 1-3 hours),
   - pre-experiment stale check (if data older than threshold).
2. Run Apify actor with:
   - target username,
   - since checkpoint,
   - include replies and engagement metadata.
3. Pull dataset output and upsert by `tweet_id`.
4. Update checkpoint to newest tweet.
5. Rebuild persona snapshot.

## Apify ingestion requirements
- Idempotent ingestion (safe re-run).
- Raw payload archived for debugging.
- Data quality checks:
  - dedupe,
  - non-empty text,
  - valid timestamps.
- Failure handling:
  - mark sync failed reason,
  - automatic retries with cap.

---

## 4) Displaying data in the frontend

## Pages
- `/` Dashboard
  - tracked username
  - last sync status/time
  - manual sync button
- `/optimize`
  - draft input
  - objective selector
  - “Generate + simulate 10 variants”
- `/experiments/[id]`
  - live progress
  - ranked results
  - winner details

## Frontend run states
- `queued`
- `running`
- `partial_results`
- `complete`
- `failed`

## Results view (must-have)
- Winner card:
  - winning variant text
  - objective score
  - predicted sentiment/aggression/leaning summary
- Ranked table (10 variants):
  - rank
  - composite score
  - Neg/Neu/Pos split
  - aggression
  - audience lean split
  - runtime
- “Why this won” panel:
  - show objective weights and metric deltas vs runner-up.

---

## End-to-end experiment flow (MVP)
1. User submits draft + objective.
2. API validates account freshness; triggers sync if stale.
3. Worker generates 10 variants using the 70B model.
4. Worker runs 10 simulations in parallel (bounded concurrency).
5. Each variant is scored.
6. Ranking service computes winner.
7. Frontend shows winner + ranked alternatives.

---

## Build order (to full working MVP)

## Phase 1 (2-3 days): baseline platform
- Create FastAPI app + Postgres schema + Redis queue.
- Implement account endpoints and experiment skeleton.
- Deploy `web` and `api` with health checks.

## Phase 2 (2-3 days): Apify incremental sync
- Implement actor client, polling/webhook completion.
- Normalize/upsert tweets/replies.
- Store sync checkpoints and persona snapshots.

## Phase 3 (3-5 days): 70B model integration
- Add OpenRouter client in simulation code path.
- Force `model = cognitivecomputations/dolphin-llama-3-70b`.
- Add request retries, timeouts, and token/runtime logging.

## Phase 4 (3-5 days): parallel variants + ranking
- Variant generation worker.
- Parallel simulation workers for 10 variants.
- Objective-weighted ranking and winner selection.

## Phase 5 (2-4 days): frontend result UX
- Build dashboard, optimize page, and live results page.
- Polling/SSE for progress.
- Final ranked output and winner rationale.

## Phase 6 (2-3 days): hardening
- Rate-limit controls, queue backpressure, circuit breaker.
- Token budget guardrails.
- Error surfacing and rerun endpoints.

---

## MVP success criteria
- User enters username once; subsequent syncs are incremental.
- 10 variants run without manual ops.
- Results are visible in one page with winner and alternatives.
- Runtime and token cost are tracked per variant.
- Product is stable enough for live design-partner usage.

---

## References
- OpenRouter model catalog API: https://openrouter.ai/api/v1/models
- OpenRouter limits: https://openrouter.ai/docs/api-reference/limits
- OpenRouter quickstart: https://openrouter.ai/docs/quickstart
- Apify API v2: https://docs.apify.com/api/v2
- Apify actor runs endpoint: https://docs.apify.com/api/v2/act-runs-post
- Apify platform limits: https://docs.apify.com/platform/limits
