#!/bin/bash
set -euo pipefail

# ──────────────────────────────────────────────────────────
# stop-bot.sh — Kill ALL Magic Grimoire processes cleanly
#
# Kills (in order):
#   1. uvicorn
#   2. start-bot.sh
#   3. cloudflared tunnel
#   4. Any orphaned uv run python processes
#
# Usage:
#   bash app/stop-bot.sh
#   bash app/start-bot.sh   # to restart
# ──────────────────────────────────────────────────────────

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "${PROJECT_DIR}"

echo ""
echo "🛑 Stopping Magic Grimoire..."
echo ""

# ── 1. uvicorn (main bot process) ────────────────────────
UVICORN_PIDS=$(pgrep -f "uvicorn app.main" 2>/dev/null || true)
if [ -n "${UVICORN_PIDS}" ]; then
    echo "  Killing uvicorn (PIDs: $(echo ${UVICORN_PIDS} | tr '\n' ' '))..."
    kill ${UVICORN_PIDS} 2>/dev/null || true
    sleep 1
    # Force kill if still running
    UVICORN_PIDS=$(pgrep -f "uvicorn app.main" 2>/dev/null || true)
    if [ -n "${UVICORN_PIDS}" ]; then
        kill -9 ${UVICORN_PIDS} 2>/dev/null || true
    fi
    echo "  ✅ uvicorn stopped"
else
    echo "  ⏭  uvicorn not running"
fi

# ── 2. start-bot.sh ──────────────────────────────────────
BOT_PIDS=$(pgrep -f "start-bot.sh" 2>/dev/null || true)
if [ -n "${BOT_PIDS}" ]; then
    echo "  Killing start-bot.sh (PIDs: $(echo ${BOT_PIDS} | tr '\n' ' '))..."
    kill ${BOT_PIDS} 2>/dev/null || true
    sleep 1
    BOT_PIDS=$(pgrep -f "start-bot.sh" 2>/dev/null || true)
    if [ -n "${BOT_PIDS}" ]; then
        kill -9 ${BOT_PIDS} 2>/dev/null || true
    fi
    echo "  ✅ start-bot.sh stopped"
else
    echo "  ⏭  start-bot.sh not running"
fi

# ── 3. cloudflared tunnel ────────────────────────────────
CF_PIDS=$(pgrep -f "cloudflared tunnel" 2>/dev/null || true)
if [ -n "${CF_PIDS}" ]; then
    echo "  Killing cloudflared (PIDs: $(echo ${CF_PIDS} | tr '\n' ' '))..."
    kill ${CF_PIDS} 2>/dev/null || true
    sleep 1
    CF_PIDS=$(pgrep -f "cloudflared tunnel" 2>/dev/null || true)
    if [ -n "${CF_PIDS}" ]; then
        kill -9 ${CF_PIDS} 2>/dev/null || true
    fi
    echo "  ✅ cloudflared stopped"
else
    echo "  ⏭  cloudflared not running"
fi

# ── 4. Orphaned Python processes (uv run python, etc.) ───
PY_PIDS=$(pgrep -f "uv run python" 2>/dev/null || true)
if [ -n "${PY_PIDS}" ]; then
    echo "  Killing orphaned Python processes (PIDs: $(echo ${PY_PIDS} | tr '\n' ' '))..."
    kill ${PY_PIDS} 2>/dev/null || true
    sleep 1
    PY_PIDS=$(pgrep -f "uv run python" 2>/dev/null || true)
    if [ -n "${PY_PIDS}" ]; then
        kill -9 ${PY_PIDS} 2>/dev/null || true
    fi
    echo "  ✅ orphaned processes stopped"
else
    echo "  ⏭  no orphaned Python processes"
fi

# ── 5. Verify port is free ──────────────────────────────
sleep 1
PORT=8123
if ss -tlnp 2>/dev/null | grep -q ":${PORT} "; then
    echo "  ⚠️  Port ${PORT} still in use — force killing..."
    fuser -k "${PORT}/tcp" 2>/dev/null || true
    sleep 1
    if ss -tlnp 2>/dev/null | grep -q ":${PORT} "; then
        echo "  ❌ Port ${PORT} still in use. Manual cleanup needed."
    else
        echo "  ✅ Port ${PORT} freed"
    fi
else
    echo "  ✅ Port ${PORT} is free"
fi

echo ""
echo "✅ All processes stopped. Run \`bash app/start-bot.sh\` to restart."
echo ""