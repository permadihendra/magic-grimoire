# Magic Grimoire — Timeout Overhaul Plan

> **Status:** Proposed  
> Increase synthesis timeout from 60s → 120s to accommodate longer generations.

---

## Current Timeout Chain

| Layer | Location | Current | Used by |
|---|---|---|---|
| Gateway dispatch | `gateway.py` | 300s | All commands (outer boundary) |
| OllamaGuard (old query) | `engine.py:211` | **120s** | `generate_quiz`, `summarize_topic` (module-level functions) |
| OllamaGuard (query_with_sources) | `engine.py:413` | **120s** | `_handle_ask`, `_handle_quiz`, `_handle_summarize`, `_handle_qna` |
| **TreeSummarize** | `engine.py:427` | **60s ← the bottleneck** | All `query_with_sources()` calls |
| Fallback OllamaGuard | `engine.py:445` | 60s | Timeout fallback |
| Fallback generation | `engine.py:447` | 60s | Timeout fallback |
| OllamaGuard (chat tool) | `brain/handler.py` | 120s | `_tool_chat` |
| Parse timeout | `indexer.py` | 30s | `/index` only |
| `num_predict` | `models.py:48` | 2048 tokens | All LLM generation |

## Problem

At `num_gpu 99` + 80% power cap, generation speed is ~40 tok/s.

- 2048 tokens `num_predict` = **~50s** (fits in 60s timeout ✅)
- But complex queries with large context need more output tokens
- If generation spills past 60s, fallback runs with a simpler prompt → lower quality
- With `num_predict=2048`, the model stops at ~50s anyway, so 60s timeout is barely used

## Proposal: Bump `num_predict` + Timeouts

### Change 1: `num_predict` 2048 → 4096

At 40 tok/s: 4096 tokens = **~100s** generation. More headroom for long answers.

No VRAM change — `num_predict` only caps output length, doesn't affect KV cache.

### Change 2: OllamaGuard 120s → 180s (two places)

`engine.py:211` and `engine.py:413` — needs 60s margin over the 120s synthesis timeout.

### Change 3: TreeSummarize 60s → 120s

`engine.py:427` — allows up to 4096 tokens at 40 tok/s (100s) with 20s margin.

### Change 4: Fallback generation 60s → 90s

`engine.py:447` — fallback runs after synthesis timeout, 90s to generate with simpler prompt.

### Change 5: Fallback OllamaGuard 60s → 90s

`engine.py:445` — match the fallback generation timeout.

### No change: Gateway 300s

Still enough (120s synthesis + 90s fallback = 210s < 300s).

## Summary

| Setting | File:Line | Before | After |
|---|---|---|---|
| `num_predict` | `models.py:48` | 2048 | **4096** |
| OllamaGuard (old query) | `engine.py:211` | 120s | **180s** |
| OllamaGuard (query_with_sources) | `engine.py:413` | 120s | **180s** |
| TreeSummarize timeout | `engine.py:427` | 60s | **120s** |
| Fallback OllamaGuard | `engine.py:445` | 60s | **90s** |
| Fallback generation | `engine.py:447` | 60s | **90s** |

6 changes across 2 files. Total worst-case wait: 120s + 90s = 210s (under 300s gateway limit).

Ready to execute?
