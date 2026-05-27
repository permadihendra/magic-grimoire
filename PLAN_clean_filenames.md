# Magic Grimoire — PLAN: Clean Filenames via Display Names

## Problem

`shorten_filename()` is a 50-line regex monster that:
- Removes author names (fragile — depends on "Last, First - " format)
- Strips publisher/year/edition (fragile — depends on specific patterns)
- Truncates at subtitle words (guesses where title ends)
- Title-cases (messes up acronyms like "MVC", "API")

Every new file format breaks it. And it runs at QUERY time (every citation) 
instead of once at indexing time.

## Solution: Display Name — Clean Once at Index Time

### Approach: Simple Normalization + Truncation

No regex. No Gemini. No guessing. Just normalize what's already there:

```python
def clean_display_name(filename: str) -> str:
    """Convert raw filename to display name. Simple, predictable, no regex."""
    # 1. Remove extension
    name = filename.rsplit(".", 1)[0] if "." in filename else filename
    
    # 2. Replace common separators with spaces
    for sep in ["_", "-", "–", "—"]:
        name = name.replace(sep, " ")
    
    # 3. Collapse whitespace
    name = " ".join(name.split())
    
    # 4. Title case (preserves acronyms like "MVC", "API", "GPU")
    # Use str.title() which capitalizes first letter of each word
    name = name.title()
    
    return name

def display_name(filename: str, max_len: int = 50) -> str:
    """Get display name, truncating if needed."""
    name = clean_display_name(filename)
    if len(name) <= max_len:
        return name
    # Truncate at word boundary
    return name[:max_len - 3].rsplit(" ", 1)[0] + "..."
```

### Examples

| Raw Filename | Current shorten_filename | New display_name |
|---|---|---|
| `Statman, Meir - Finance for normal people _ how investors and markets behave-Oxford University Press (2017).pdf` | `Finance For Normal People` | `Statman, Meir   Finance For Normal People   How Investors And Markets Behave Oxford University Press (2017)` → truncated: `Statman, Meir   Finance For Normal Peo...` |
| `The_World_Economy_and_Financial_System_A_Paradigm_Change_Offering.epub` | `World Economy` (TOO SHORT, LOST INFO) | `The World Economy And Financial System A Par...` (accurate prefix) |

Wait, the simple approach is still ugly — "Statman, Meir" shouldn't be in the title.

### Better: Gemini Clean at Index Time (Recommended)

Ask Gemini ONCE at indexing to extract the book title. Store it in the DB.

```
Flow:
  1. User drops PDF in app/docs/
  2. /index reads filename: "Statman, Meir - Finance for normal people...pdf"
  3. Gemini: "Extract book title from this filename"
  4. Gemini returns: "Finance for Normal People"
  5. Store: display_name = "Finance for Normal People" in documents table
  6. All citations use display_name
```

### Database Change

```sql
ALTER TABLE documents ADD COLUMN display_name TEXT;
```

### Gemini Prompt (simple, one-shot, cached)

```python
_FILENAME_CLEAN_PROMPT = """Extract a clean, readable book title from this filename. 
Remove author names, publisher info, years, edition numbers, and file extensions.
Return ONLY the title, nothing else.

Examples:
"Statman, Meir - Finance for normal people _ how investors and markets behave-Oxford University Press (2017).pdf"
→ Finance for Normal People

"The_World_Economy_and_Financial_System_A_Paradigm_Change_Offering.epub"
→ The World Economy and Financial System: A Paradigm Change Offering

"python-crash-course-2nd-edition.pdf"  
→ Python Crash Course

"Deep Learning with Python, Second Edition"
→ Deep Learning with Python

Now clean this filename:
"""

async def gemini_clean_filename(raw_filename: str) -> str:
    """Ask Gemini to extract a clean book title from a messy filename."""
    try:
        response = await gemini_chat(
            system_prompt=_FILENAME_CLEAN_PROMPT,
            user_message=raw_filename,
            max_tokens=50,
            timeout=10.0,
        )
        cleaned = response.strip().strip('"').strip("'")
        if cleaned and len(cleaned) > 2:
            return cleaned
    except Exception:
        pass
    # Fallback: simple normalization
    return _simple_clean(raw_filename)
```

### Where Display Name Replaces shorten_filename

| Current | Replace With |
|---|---|
| `shorten_filename(node.metadata["file_name"])` | `node.metadata.get("display_name") or shorten_filename(...)` |
| `/files` output | Use `display_name` from DB |
| `list_docs()` tool | Return `display_name` |
| Source citations | Show `display_name` |
| Retrieve preview | Show `display_name` |

### Implementation Steps

1. Add `display_name` column to documents table
2. At indexing time: for each new file, call Gemini to get display_name
3. Store in DB: `INSERT INTO documents (..., display_name) VALUES (..., ?)`
4. `shorten_filename()` becomes a fallback for nodes without display_name
5. All display code uses display_name first, fallback to shorten_filename
6. `/files` and `list_docs()` show display_name

### Cost

- One Gemini call per NEW file at index time (not per query)
- Cache in DB — never re-requests
- Even with 100 docs, that's 100 calls (free tier: 1500/day)
- At 50ms per call, adds ~50ms to indexing per file

### Alternative: No Gemini, Better Regex

If Gemini is too slow or unreliable, use a focused regex approach:

```python
def clean_filename_simple(raw: str) -> str:
    """Simple filename cleaning without Gemini."""
    # Remove extension
    name = raw.rsplit(".", 1)[0]
    
    # Remove common patterns: "Author, Author - ", years in parens
    name = re.sub(r"^[A-Z][a-z]+,\s*[A-Z][a-z.]+\s*[-–]\s*", "", name)
    name = re.sub(r"\s*\(\d{4}\)\s*", "", name)
    name = re.sub(r"\s*[-–]\s*\d+(st|nd|rd|th)?\s*edition", "", name, flags=re.I)
    
    # Clean separators
    name = name.replace("_", " ").replace("  ", " ")
    name = " ".join(name.split())
    
    # Truncate if long
    if len(name) > 50:
        name = name[:47].rsplit(" ", 1)[0] + "..."
    
    return name
```

### Comparison

| Approach | Pro | Con |
|---|---|---|
| **Gemini at index** | Handles ANY format, learns patterns, "Finance for Normal People" | +50ms per new file, uses API quota |
| **Simpler regex** | No API calls, fast | Still fragile for exotic formats |
| **Simple normalize** | Zero fragility | "Statman, Meir - Finance..." still ugly |

### Recommendation

**Hybrid**: Gemini at index time (primary) with simple fallback (secondary).

- New files: ask Gemini → store in DB
- If Gemini fails: use simple regex → store in DB  
- At query time: read display_name from DB (no Gemini call, no regex)
- `shorten_filename()` becomes dead code
