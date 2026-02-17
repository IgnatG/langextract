# Upstream Cherry-Pick Tracker

Tracking upstream PRs from [google/langextract](https://github.com/google/langextract/pulls) for cherry-picking into the managed fork (`ignatg/langextract`).

**Fork version:** 1.1.1 (based on upstream `3638fe4`)
**Branch:** `custom` (cherry-picks applied here)
**Last reviewed:** 2026-02-17

---

## Already Included in Fork

These fixes are already part of the fork's `main` branch:

| PR | Title | Commit |
|----|-------|--------|
| #284 | Multi-language tokenizer support (Unicode & Regex) | `0c1af87` |
| #261 | Enable `suppress_parse_errors` parameter in resolver_params | `6e36c37` |
| #306 | Cross-chunk context awareness for coreference resolution | `3638fe4` |
| #300 | Handle non-Gemini model output parsing edge cases | `4882369` |
| N/A | Resolver dict-in-text graceful skip (custom fix) | `cf80fba` |

---

## HIGH PRIORITY — Cherry-Pick Now

| Status | PR | Title | Impact | Notes |
|--------|----|-------|--------|-------|
| [x] | [#351](https://github.com/google/langextract/pull/351) | Fix: Matching built-in providers fails if specified by name (Fixes #335) | Ensures `providers.load_builtins_once()` runs before provider resolution. Affects our `langextract-mistral` plugin loading. | Size XS. Applied manually. |
| [x] | [#349](https://github.com/google/langextract/pull/349) | Fix: Correctly load built-in providers when using explicit 'provider' in ModelConfig | Related to #351 — same provider loading issue from a different angle. | Size S. Same fix as #351. |
| [x] | [#327](https://github.com/google/langextract/pull/327) | Add `require_grounding` parameter to filter ungrounded extractions | Filters extractions where `char_interval is None` (hallucinated from few-shot). Directly improves extraction quality for our 3-pass, 12-class setup. | Size M. Fixes #209. |
| [x] | [#375](https://github.com/google/langextract/pull/375) | Fix: continue extraction when chunk resolve/alignment fails | Adds opt-in `suppress_parse_errors` at annotation level to skip per-chunk failures. Complements our resolver dict-skip fix. | Applied as `c2bd1bb`. Merged with our `require_grounding` parameter from PR #327. |
| [x] | [#374](https://github.com/google/langextract/pull/374) | Fix: improve robustness to Unicode and malformed JSON | Normalizes CJK radicals, handles trailing commas, multiple fenced blocks in LLM output. Resilience for multilingual contracts. | Applied as `4028976`. Resolver conflict resolved manually (added unicode normalization to correct locations). Test indentation fixed for 4-space convention. |

## MEDIUM PRIORITY — Monitor / Evaluate

| Status | PR | Title | Impact | Notes |
|--------|----|-------|--------|-------|
| [x] | [#350](https://github.com/google/langextract/pull/350) | Fix incorrect `char_interval` for non-ASCII text (Fixes #334) | Fixes `RegexTokenizer` merging Latin + CJK characters. Uses regex V1 set subtraction to separate CJK scripts from Latin in token patterns. | Applied manually. Adds `_CJK_SCRIPTS`, `_CJK_PATTERN`, and modifies `_LETTERS_PATTERN` with V1 set subtraction. 142→421 tests pass (new retry tests included). |
| [x] | [#257](https://github.com/google/langextract/pull/257) | Add retry mechanism for transient API errors (503, 429, timeouts) | Exponential backoff with jitter for transient LLM failures. Chunk-level retry in annotation pipeline preserves successful chunks. | Applied via `git apply --reject` + manual conflict resolution. New files: `retry_utils.py` (278 lines), `retry_utils_test.py` (300 lines). Modified: `annotation.py`, `extraction.py`, `gemini.py`, `annotation_test.py`. Complementary to litellm's provider-level `num_retries`. |
| [ ] | [#356](https://github.com/google/langextract/pull/356) | Remove duplicate `model_id` assignment in `factory.create_model()` | Tiny cleanup (XS). Low risk, easy cherry-pick. | |
| [ ] | [#32](https://github.com/google/langextract/pull/32) | Multi-language tokenizer support | **Already in fork** as #284. | Skip — already included. |

## LOW PRIORITY — Not Relevant

| PR | Title | Why Skip |
|----|-------|----------|
| #369 | Fix format of `create_provider_plugin.py` | Script formatting only |
| #362, #317 | Edge case tests for prompt validation | Test-only, no functional impact |
| #360 | Fix: health check failure curl not found | Their Docker image, not ours |
| #359 | Fix: Gemini batch GCS cache hashing | Gemini-specific — we use Mistral |
| #354 | Added Langfuse Integration | We don't use Langfuse |
| #352 | Fix batch inference crash (GCS cache) | Gemini batch-specific |
| #347, #346, #321 | GitHub Actions upgrades | CI-only |
| #331 | Relax provider from model | Superseded by #351/#349 |
| #329 | Add Ollama model name regex patterns | We don't use Ollama |
| #326 | Add `thinking_config` for Gemini 2.0 | Gemini-specific |
| #325 | Use shields.io for DOI badge | Docs/badge only |
| #310 | Add Groq provider | We don't use Groq |
| #305 | Security fix for path traversal in validation script | We don't run this script |
| #294 | Migrate to uv | Build tooling — our Docker uses pip |
| #271, #46, #40, #18 | Misc (merge spam, conda badge, Ollama security) | Not relevant |
| #242, #241 | Model selection / GPT-5 reasoning | Not relevant to our Mistral setup |
| #267 | Ollama keep_alive | Ollama-specific |

---

## Cherry-Pick Log

| Date | PR | Commits | Result | Branch |
|------|----|---------|--------|--------|
| 2026-02-17 | #351/#349 | Manual apply | factory.py: moved `load_builtins_once()`/`load_plugins_once()` before provider conditional | main |
| 2026-02-17 | #327 | Manual apply | extraction.py: added `_filter_ungrounded_extractions()` + `require_grounding` param | main |
| 2026-02-18 | #375 | `c2bd1bb` | annotation.py: `suppress_parse_errors` at annotation level, merged with `require_grounding` from #327 | custom |
| 2026-02-18 | #374 | `4028976` | resolver.py: Unicode normalization (CJK radicals), trailing commas, multiple fenced blocks | custom |
| 2026-02-18 | #350 | Manual apply | tokenizer.py: CJK script separation via regex V1 set subtraction (`_CJK_SCRIPTS`, `_CJK_PATTERN`) | custom |
| 2026-02-18 | #257 | Manual apply | retry_utils.py (new), annotation.py, extraction.py, gemini.py: transient error retry with exponential backoff + jitter | custom |

---

## Upstream Sync Schedule

- **Frequency:** Monthly, or when Google merges a cherry-picked PR
- **Runbook:** See [UPSTREAM_SYNC.md](./UPSTREAM_SYNC.md)
- **Last sync:** N/A (initial setup)
