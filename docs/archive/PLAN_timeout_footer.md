# Magic Grimoire — Timeout & Footer Improvement Plan

## Issue 1: Synthesis Timeout Too Short

**Current:** TreeSummarize has 30s hard cap at engine.py line 427.  
**Fix:** Increase to **60s**.

At 25 tok/s (num_gpu 12): 60s → 1,500 tokens → ~1,100 words.  
Handles "10 pivotal moments" (~40s needed) with comfortable margin.  
OllamaGuard outer timeout (120s) unchanged — still has 60s safety margin for fallback.

## Issue 2: Footer Diagnosis Not Sent

**Root cause:** `ask_query()` returns only `result["answer"]` — discards `follow_up`.  
Brain handler never sees it.

**Fix A:** Return dict from `ask_query()` instead of string.  
**Fix B:** In brain handler, extract `follow_up` and send as new message after main chunks.

## Files to Change

| File | Line | Change |
|---|---|---|
| `app/rag/engine.py` | 427 | 30.0 → 60.0 |
| `app/plugins/study/handler.py` | ~630 | Return dict from `ask_query()` |
| `app/plugins/brain/handler.py` | ~580 | Send `follow_up` as new message |
