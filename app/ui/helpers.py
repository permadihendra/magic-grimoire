"""Shared UI helpers for Magic Grimoire."""

import os
import re
from datetime import datetime
from typing import Literal

# ── Filename cleaner ─────────────────────────────────────

_STRIP_PATTERNS = [
    re.compile(r"^\d+[\s\-_\.]+"),       # leading number: "01 Introduction"
    re.compile(r"[\s\-_\.]+\d{4}[\s\-_\.]*$"),  # year suffix: "Book 2020"
    re.compile(r"[\s\-_\(]*2nd[\s\)_\.]*ed.*", re.IGNORECASE),
    re.compile(r"[\s\-_\(]*3rd[\s\)_\.]*ed.*", re.IGNORECASE),
    re.compile(r"[\s\-_]*(1st|2nd|3rd|4th|5th)[\s\-_]*edition.*", re.IGNORECASE),
    re.compile(r"[\s\-_]*revised.*", re.IGNORECASE),
    re.compile(r"[\s\-_]*penguin.*", re.IGNORECASE),
    re.compile(r"[\s\-_]*free.* ebook.*", re.IGNORECASE),
    re.compile(r"\.pdf$", re.IGNORECASE),
    re.compile(r"\.epub$", re.IGNORECASE),
    re.compile(r"\.txt$", re.IGNORECASE),
]

_EXT_EXCLUDE = {".pdf", ".epub", ".txt", ".md", ".docx"}


def _simple_clean_filename(raw: str) -> str:
    """Strip common cruft from filenames to get a readable title."""
    name = raw

    # Remove extension
    for ext in _EXT_EXCLUDE:
        if name.lower().endswith(ext):
            name = name[: -len(ext)]

    # Remove author-name patterns (simple heuristics)
    for pattern in _STRIP_PATTERNS:
        name = pattern.sub(" ", name)

    # Replace separators with spaces
    for sep in ["_", "-", "\u2013", "\u2014", "|", "/"]:
        name = name.replace(sep, " ")

    name = " ".join(name.split())

    # Truncate long titles
    if len(name) > 60:
        name = name[: 57].rsplit(" ", 1)[0] + "..."

    return name.strip() or raw


# ── Timestamp formatter ──────────────────────────────────

def _fmt_timestamp(ts: str | None) -> str:
    """Format a DB timestamp into a human-readable string."""
    if not ts:
        return "unknown"
    try:
        dt = datetime.strptime(ts, "%Y-%m-%d %H:%M:%S")
        return dt.strftime("%d %b %Y, %H:%M")
    except (ValueError, TypeError):
        return str(ts) if ts else "unknown"


# ── File card builder ────────────────────────────────────

def _build_file_card(
    row: dict,
    show_status: bool = True,
    show_display_name: bool = True,
    index_position: int | None = None,
) -> str:
    """Build a consistent, information-rich file card.

    Args:
        row: DB row dict with filename, file_size, word_count, chunk_count,
             parse_method, verified, indexed_at, display_name.
        show_status: Show indexed status + probe verified flag.
        show_display_name: Show display_name (clean title) alongside raw filename.
        index_position: If given, prepend [1], [2], etc.

    Example output:
        📄 **The Bhagavad Gita**
           └─ 📊 6.7 MB · 186 chunks · 46,677 words
           └─ ✅ ebooklib (EPUB) · probe verified · indexed 28 May 2026
    """
    fname = row.get("filename", "")
    display_name = row.get("display_name") or _simple_clean_filename(fname)

    # Sizes
    size_mb = (row.get("file_size") or 0) / (1024 * 1024)
    chunks = row.get("chunk_count") or 0
    words = row.get("word_count") or 0

    # Parsing
    parse_method = row.get("parse_method") or ""
    verified = bool(row.get("verified"))
    indexed_at = _fmt_timestamp(row.get("indexed_at"))

    # Build card
    id_prefix = f"[{index_position}] " if index_position is not None else ""

    lines = []
    if index_position is not None:
        lines.append(f"{id_prefix}📄 **{display_name}**")
    else:
        lines.append(f"📄 **{display_name}**")

    # Line 2: sizes
    size_str = f"{size_mb:.1f} MB" if size_mb > 0 else "?"
    words_str = f"{words:,} words" if words > 0 else "?"
    chunks_str = f"{chunks} chunks" if chunks > 0 else "?"

    if show_status and chunks > 0:
        lines.append(f"   └─ 📊 {size_str} · {chunks_str} · {words_str}")
    elif show_status and chunks == 0:
        lines.append(f"   └─ 📊 {size_str} · not yet indexed")
    else:
        lines.append(f"   └─ 📊 {size_str} · {words_str} words")

    # Line 3: parsing + status
    if show_status:
        if parse_method:
            status = "✅ probe verified" if verified else "⚠️ probe unknown"
            lines.append(f"   └─ {parse_method} · {status} · indexed {indexed_at}")
        else:
            lines.append(f"   └─ indexed {indexed_at}")
    else:
        if parse_method:
            lines.append(f"   └─ {parse_method}")
        if indexed_at != "unknown":
            lines.append(f"   └─ {indexed_at}")

    return "\n".join(lines)


def _build_files_card(
    fname: str,
    size_bytes: int,
    modified_ts: float,
    db_row: dict | None,
    position: int,
) -> str:
    """Build a /files list card with indexed status.

    Args:
        fname: Raw filename from disk.
        size_bytes: File size in bytes.
        modified_ts: os.path.getmtime() timestamp.
        db_row: Corresponding DB row (or None if not indexed).
        position: Line number [1], [2], etc.
    """
    size_mb = size_bytes / (1024 * 1024)
    modified = datetime.fromtimestamp(modified_ts)
    modified_str = modified.strftime("%d %b %Y")

    # Display name
    display_name = db_row.get("display_name") if db_row else None
    if display_name:
        label = f"💾 {display_name}"
    else:
        label = ""

    # Status + info
    if db_row:
        chunks = db_row.get("chunk_count") or 0
        words = db_row.get("word_count") or 0
        verified = bool(db_row.get("verified"))

        if verified and chunks > 0:
            status = "✅ indexed (probe verified)"
        elif chunks > 0:
            status = "✅ indexed"
        else:
            status = "⬜ not indexed"

        info_parts = [f"📊 {size_mb:.1f} MB"]
        if chunks > 0:
            info_parts.append(f"{chunks} chunks")
        if words > 0:
            info_parts.append(f"{words:,} words")
        info_parts.append(modified_str)
        info = " · ".join(info_parts)
    else:
        status = "⬜ not indexed"
        info = f"📊 {size_mb:.1f} MB · not indexed · {modified_str}"

    lines = [
        f"[{position}] {status} `{fname}`",
        f"   └─ {info}",
    ]
    if label:
        lines.append(f"   └─ {label}")
    elif not db_row:
        lines.append(f"   └─ ⏳ Run /index to include this file")

    return "\n".join(lines)


def _build_total_footer(
    total_docs: int,
    total_words: int | None = None,
    total_chunks: int | None = None,
    total_files: int | None = None,
    indexed_count: int | None = None,
) -> str:
    """Build a consistent summary footer for list commands."""
    parts = []
    if total_docs is not None:
        parts.append(f"{total_docs} document{'s' if total_docs != 1 else ''}")
    if total_files is not None:
        parts.append(f"{total_files} file{'s' if total_files != 1 else ''}")
    if indexed_count is not None and total_files is not None:
        pending = total_files - indexed_count
        if pending > 0:
            parts.append(f"({indexed_count} indexed · {pending} pending)")
        elif indexed_count > 0:
            parts.append(f"(all {indexed_count} indexed)")

    footer = f"_Total: {', '.join(parts)}_"
    if total_words and total_words > 0:
        footer += f"\n_~{total_words:,} words across all materials_"

    return footer


# ── Message splitter (Telegram 4096 char limit) ──────────


def _split_into_chunks(text: str, max_len: int = 4000) -> list[str]:
    """Split long text into chunks that fit Telegram's message limit.

    Each chunk is at most max_len characters. Non-first chunks get a
    continuation prefix, non-last chunks get a continuation suffix.
    Splits at paragraph boundaries where possible, otherwise at
    newlines, otherwise at spaces.
    """
    if not text:
        return [""]
    if len(text) <= max_len:
        return [text]

    chunks = []
    remaining = text

    while remaining:
        if len(remaining) <= max_len:
            chunks.append(remaining)
            break

        # Try to split at paragraph boundary (double newline)
        cut = remaining.rfind("\n\n", 0, max_len)
        if cut < max_len // 2:
            # Try single newline
            cut = remaining.rfind("\n", 0, max_len)
        if cut < max_len // 2:
            # Try space
            cut = remaining.rfind(" ", 0, max_len)
        if cut < max_len // 2:
            # Hard cut at max_len
            cut = max_len

        chunk = remaining[:cut].strip()
        remaining = remaining[cut:].strip()

        # Add continuation markers
        if chunks:
            chunk = "_continued from above_\n\n" + chunk
        if remaining:
            chunk += "\n\n_continued..._"

        chunks.append(chunk)

    return chunks