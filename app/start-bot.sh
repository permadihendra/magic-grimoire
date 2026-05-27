#!/bin/bash
set -euo pipefail

# ──────────────────────────────────────────────────────────
# start-bot.sh — Magic Grimoire Quick Tunnel Bot Launcher
#
# Automates:
#   1. Verify Ollama is running + models available
#   2. Start cloudflared quick tunnel (background)
#   3. Parse tunnel URL from logs
#   4. Update .env with new webhook URL
#   5. Register webhook with Telegram API
#   6. Send startup notification
#   7. Launch uvicorn (foreground)
#
# On shutdown (SIGTERM/SIGINT/EXIT):
#   → Kill cloudflared tunnel
#   → Clean exit
#
# Usage:
#   ./app/start-bot.sh
#
# For production with permanent tunnel:
#   Start Ollama as service: sudo systemctl enable --now ollama
#   Start bot directly:      uv run uvicorn app.main:app --port 8123
# ──────────────────────────────────────────────────────────

# ── Paths ────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
LOG_FILE="${PROJECT_DIR}/data/cloudflared.log"
TUNNEL_TIMEOUT=60          # seconds to wait for tunnel URL
PORT=8123

cd "${PROJECT_DIR}"

# Set TESSDATA_PREFIX for LiteParse OCR (if tessdata exists)
if [ -d "/usr/share/tesseract-ocr/5/tessdata" ]; then
    export TESSDATA_PREFIX="/usr/share/tesseract-ocr/5/tessdata"
elif [ -d "/usr/share/tesseract-ocr/4/tessdata" ]; then
    export TESSDATA_PREFIX="/usr/share/tesseract-ocr/4/tessdata"
fi

# ── Colors ────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; NC='\033[0m'
info()  { echo -e "${GREEN}[INFO]${NC}  $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC}  $*"; }
error() { echo -e "${RED}[ERROR]${NC} $*"; }

# ── Pre-flight checks ────────────────────────────────────

# 1. Check Ollama binary
command -v ollama >/dev/null 2>&1 || {
    error "ollama not found. Install it first:"
    echo ""
    echo "  curl -fsSL https://ollama.com/install.sh | sh"
    echo ""
    exit 1
}

# 2. Check cloudflared
command -v cloudflared >/dev/null 2>&1 || {
    error "cloudflared not found. Install it first:"
    echo ""
    echo "  # Raspberry Pi (ARM):"
    echo "  wget -q https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-arm"
    echo "  sudo mv cloudflared-linux-arm /usr/local/bin/cloudflared"
    echo "  sudo chmod +x /usr/local/bin/cloudflared"
    echo ""
    echo "  # Or via package manager:"
    echo "  sudo apt install cloudflared"
    echo ""
    exit 1
}

# 3. Check uv
command -v uv >/dev/null 2>&1 || {
    error "uv not found. Install: curl -LsSf https://astral.sh/uv/install.sh | sh"
    exit 1
}

# 4. Check .env
[ -f .env ] || {
    error ".env file not found."
    echo "  cp .env.example .env"
    echo "  # Then fill in your tokens"
    exit 1
}

# Warn if .env permissions are not 600
ENV_PERMS=$(stat -c "%a" .env 2>/dev/null || stat -f "%p" .env 2>/dev/null)
if [ "${ENV_PERMS}" != "600" ]; then
    warn ".env permissions are ${ENV_PERMS} (should be 600)"
    warn "  Run: chmod 600 .env"
fi

# ── Load environment ─────────────────────────────────────
# Safe .env parser — handles spaces, quotes, special chars
# Doesn't use `source` because unquoted values break bash
while IFS= read -r line; do
    # Skip comments and empty lines
    [[ -z "$line" || "$line" =~ ^# ]] && continue
    # Split on first =
    key="${line%%=*}"
    value="${line#*=}"
    # Strip surrounding quotes if present
    value="${value%\"}"
    value="${value#\"}"
    value="${value%\'}"
    value="${value#\'}"
    export "$key=$value"
done < .env

# ── Cleanup handler ──────────────────────────────────────
cleanup() {
    echo ""
    info "Shutting down..."

    # Kill cloudflared tunnel
    if [ -n "${CLOUDFLARED_PID:-}" ]; then
        kill "${CLOUDFLARED_PID}" 2>/dev/null || true
        wait "${CLOUDFLARED_PID}" 2>/dev/null || true
        info "cloudflared stopped"
    fi

    # Kill uvicorn if running in background (shouldn't happen, but safe)
    if [ -n "${UVICORN_PID:-}" ]; then
        kill "${UVICORN_PID}" 2>/dev/null || true
        wait "${UVICORN_PID}" 2>/dev/null || true
    fi

    info "Bye! 👋"
}
# TRAP: cleanup on SIGTERM/SIGINT, then exit to prevent EXIT trap re-fire
trap 'cleanup; exit' SIGTERM SIGINT
trap cleanup EXIT

# ── Ollama model checks (after .env loaded) ──────────────
# Now OLLAMA_LLM_MODEL and OLLAMA_EMBED_MODEL are available from .env

info "Checking Ollama models..."

# Check Ollama is running
if ! ollama list >/dev/null 2>&1; then
    error "Ollama is not running."
    echo ""
    echo "  Start it:  ollama serve &"
    echo "  Or enable as service:  sudo systemctl enable --now ollama"
    echo ""
    exit 1
fi

# Check required models (with .env vars now available!)
for _model in "${OLLAMA_LLM_MODEL:?OLLAMA_LLM_MODEL not set after .env load}" "${OLLAMA_EMBED_MODEL:-nomic-embed-text}"; do
    if ! ollama show "${_model}" >/dev/null 2>&1; then
        warn "Model '${_model}' not found locally or corrupted."
        info "Pulling ${_model} (this may take a while)..."
        ollama pull "${_model}"
    else
        info "  ✅ ${_model} ready"
    fi
done
info "✅ Ollama models ready"

# ── Step 1: Start cloudflared tunnel ─────────────────────
info "Step 1/7 — Starting cloudflared tunnel..."
mkdir -p "$(dirname "${LOG_FILE}")"
rm -f "${LOG_FILE}"

cloudflared tunnel --url "http://localhost:${PORT}" >"${LOG_FILE}" 2>&1 &
CLOUDFLARED_PID=$!
info "  cloudflared PID: ${CLOUDFLARED_PID}"

# ── Step 2: Wait for tunnel URL ──────────────────────────
info "Step 2/7 — Waiting for tunnel URL (timeout: ${TUNNEL_TIMEOUT}s)..."
TUNNEL_URL=""
for i in $(seq 1 "${TUNNEL_TIMEOUT}"); do
    sleep 1
    # Try grep with Perl regex first, fall back to basic grep
    # NOTE: || true is REQUIRED here — grep exits 1 when no match,
    # and with set -e + set -o pipefail the whole script would exit.
    TUNNEL_URL=$(grep -oP 'https://[a-zA-Z0-9\-]+\.trycloudflare\.com' "${LOG_FILE}" 2>/dev/null | head -1) || true
    if [ -z "${TUNNEL_URL}" ]; then
        TUNNEL_URL=$(grep -oE 'https://[a-zA-Z0-9\-]+\.trycloudflare\.com' "${LOG_FILE}" 2>/dev/null | head -1) || true
    fi
    if [ -n "${TUNNEL_URL}" ]; then
        break
    fi
    # Show progress every 5 seconds
    if [ $((i % 5)) -eq 0 ]; then
        info "  still waiting... (${i}s/${TUNNEL_TIMEOUT}s)"
    fi
done

if [ -z "${TUNNEL_URL}" ]; then
    error "Tunnel did not start within ${TUNNEL_TIMEOUT}s."
    echo ""
    error "📄 Last 20 lines of cloudflared log:"
    tail -20 "${LOG_FILE}" 2>/dev/null | while IFS= read -r line; do
        echo "    $line"
    done
    echo ""
    error "Common causes:"
    error "  • Slow internet — try increasing TUNNEL_TIMEOUT"
    error "  • cloudflared needs update"
    error "  • Port ${PORT} already in use"
    error "  • No internet connection"
    exit 1
fi

info "  ✅ Tunnel URL: ${TUNNEL_URL}"

# ── Step 3: Update .env ──────────────────────────────────
info "Step 3/7 — Updating TELEGRAM_WEBHOOK_URL in .env..."

if grep -q '^TELEGRAM_WEBHOOK_URL=' .env; then
    sed -i "s|^TELEGRAM_WEBHOOK_URL=.*|TELEGRAM_WEBHOOK_URL=${TUNNEL_URL%/}|" .env
else
    echo "TELEGRAM_WEBHOOK_URL=${TUNNEL_URL%/}" >> .env
fi

# Re-export: Python's pydantic-settings checks env vars BEFORE .env file.
# Without this re-export, the stale env var (loaded earlier) would override
# the updated .env and register the webhook with the OLD tunnel URL. ❌
export TELEGRAM_WEBHOOK_URL="${TUNNEL_URL%/}"

info "  ✅ .env updated"

# ── Step 4: Wait for DNS + register webhook ─────────────
info "Step 4/7 — Waiting for tunnel DNS propagation..."

# Extract hostname from tunnel URL (strip https:// and trailing path)
_tunnel_host="${TUNNEL_URL#https://}"
_tunnel_host="${_tunnel_host%%/*}"

_DNS_OK=false
for i in $(seq 1 15); do
    # Try host lookup (most common), fallback to getent, then ping -c
    if host "${_tunnel_host}" >/dev/null 2>&1 || \
       getent hosts "${_tunnel_host}" >/dev/null 2>&1; then
        _DNS_OK=true
        break
    fi
    sleep 2
    if [ $((i % 3)) -eq 0 ]; then
        info "  waiting for DNS... (${i} attempts)"
    fi
done

if [ "${_DNS_OK}" = false ]; then
    warn "  ⚠️ Tunnel hostname not resolving yet. Trying webhook registration anyway..."
fi

# ── Step 5: Register webhook ─────────────────────────────
info "Step 5/7 — Registering webhook with Telegram..."
if uv run python -m app.bot.setup_webhook; then
    info "  ✅ Webhook registered: ${TUNNEL_URL}/webhook"
else
    error "❌ Webhook registration failed."
    error "   Tunnel hostname may not be reachable from Telegram servers."
    error "   Try again in a few seconds, or use a permanent tunnel."
    exit 1
fi

# ── Step 6: Notify via Telegram ──────────────────────────
info "Step 6/7 — Sending startup notification..."
_start_time="$(date '+%Y-%m-%d %H:%M:%S')"

# Parse first chat_id from ALLOWED_CHAT_IDS (before first comma/space)
_first_chat="${ALLOWED_CHAT_IDS%%,*}"
_first_chat="${_first_chat%% *}"
if [ -n "${_first_chat}" ] && [ "${_first_chat}" != "0" ]; then
    _nl=$'\n'
    _notif_text="📖 *Magic Grimoire is LIVE!* 🚀${_nl}• Tunnel: ${TUNNEL_URL}${_nl}• Webhook: ${TUNNEL_URL}/webhook${_nl}• Model: ${OLLAMA_LLM_MODEL:-qwen3:4b}${_nl}• Started: ${_start_time}${_nl}${_nl}📚 Send /index after adding docs${_nl}📝 Try /ask or /quiz to study!${_nl}${_nl}⚠️ Quick Tunnel — URL berubah setiap restart."

    _tg_url="https://api.telegram.org/bot${TELEGRAM_TOKEN}/sendMessage"
    _http_code=$(curl -s -o /dev/null -w "%{http_code}" \
        -X POST "${_tg_url}" \
        --data-urlencode "chat_id=${_first_chat}" \
        --data-urlencode "text=${_notif_text}" \
        --data-urlencode "parse_mode=Markdown" 2>&1)

    if [ "${_http_code}" = "200" ]; then
        info "  ✅ Notification sent to chat ${_first_chat}"
    else
        warn "  ⚠️ Notif failed (HTTP ${_http_code}) — chat_id=${_first_chat} invalid?"
    fi
else
    warn "  ⚠️ No ALLOWED_CHAT_IDS set — skipping Telegram notification"
fi

# ── Step 7: Start bot ────────────────────────────────────
info "Step 7/7 — Starting uvicorn on port ${PORT}..."
echo ""
echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${GREEN}  Magic Grimoire is LIVE! 🚀${NC}"
echo -e "${GREEN}  📖 RAG Study Agent${NC}"
echo -e "${GREEN}  Tunnel: ${TUNNEL_URL}${NC}"
echo -e "${GREEN}  Webhook: ${TUNNEL_URL}/webhook${NC}"
echo -e "${GREEN}  Model: ${OLLAMA_LLM_MODEL:-qwen3:4b}${NC}"
echo -e "${GREEN}  Started: ${_start_time}${NC}"
echo -e "${GREEN}  Press Ctrl+C to stop.${NC}"
echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""

# Run uvicorn in foreground — trap will handle cleanup on exit
uv run uvicorn app.main:app \
    --host 127.0.0.1 \
    --port "${PORT}" \
    --workers 1 \
    --no-access-log
