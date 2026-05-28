#!/bin/bash
set -euo pipefail

# ──────────────────────────────────────────────────────────
# dependency-check.sh — Verify all Magic Grimoire dependencies
#
# Checks:
#   1. Python packages (via uv)
#   2. External tools (ollama, cloudflared)
#   3. Ollama models (magic-grimoire:3b, nomic-embed-text)
#   4. System packages (tesseract-ocr for LiteParse)
#   5. Email configuration
#
# Usage:
#   bash app/dependency-check.sh          # Check everything
#   bash app/dependency-check.sh --fix    # Auto-fix when possible
#   bash app/dependency-check.sh --quiet  # Only show failures
# ──────────────────────────────────────────────────────────

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_DIR}"

MODE="${1:-normal}"  # normal, fix, quiet
FIX=false
QUIET=false
HAS_ERROR=false
HAS_WARN=false

if [ "$MODE" = "--fix" ]; then
    FIX=true
elif [ "$MODE" = "--quiet" ]; then
    QUIET=true
fi

# ── Colors ────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; BLUE='\033[0;34m'; NC='\033[0m'
OK()   { echo -e "${GREEN}  ✅${NC} $1"; }
WARN() { echo -e "${YELLOW}  ⚠️  ${NC} $1"; HAS_WARN=true; }
ERR()  { echo -e "${RED}  ❌${NC} $1"; HAS_ERROR=true; }
INFO() { echo -e "${BLUE}  ℹ️ ${NC} $1"; }
HEAD() { echo -e "\n${BLUE}── $1 ──${NC}"; }

if [ "$QUIET" = false ]; then
    echo ""
    echo -e "${BLUE}═══════════════════════════════════════════════${NC}"
    echo -e "${BLUE}  Magic Grimoire — Dependency Check${NC}"
    echo -e "${BLUE}═══════════════════════════════════════════════${NC}"
fi

# ═══════════════════════════════════════════════════════════
# 1. Python Packages
# ═══════════════════════════════════════════════════════════
HEAD "Python Packages"

# Check uv is available
if command -v uv >/dev/null 2>&1; then
    # uv version format: uv 0.1.0 (doesn't have --version like other tools)
    UV_VER=$(uv --version 2>/dev/null | head -1 | grep -oP '\d+\.\d+' || echo "installed")
    OK "uv ($UV_VER)"
else
    ERR "uv not found — install: curl -LsSf https://astral.sh/uv/install.sh | sh"
fi

# Check virtual environment exists
if [ -d ".venv" ]; then
    OK "Virtual environment (.venv)"
else
    ERR "No virtual environment — run: uv sync"
fi

# Check core dependencies using pip list
# (faster than importing each one)
if [ -d ".venv" ]; then
    VENV_PYTHON=".venv/bin/python"
    
    # Core packages — check via pip list (faster and more reliable than import)
    # Some packages have different import names vs pip names:
    #   pip name              import name
    #   python-telegram-bot   telegram
    #   llama-index-llms-ollama  → subpackage
    # Core packages — check via uv pip list (uv-managed env has no pip)
    _pip_list=$(cd "$PROJECT_DIR" && uv pip list 2>/dev/null | tail -n +3 | awk '{print $1}')
    
    # Map expected pip names
    for pkg in fastapi uvicorn httpx python-telegram-bot pydantic-settings \
               llama-index llama-index-llms-ollama llama-index-embeddings-ollama \
               llama-index-embeddings-huggingface llama-index-readers-file \
               sentence-transformers torch numpy; do
        if echo "$_pip_list" | grep -qi "^${pkg}$" 2>/dev/null; then
             [ "$QUIET" = false ] && OK "$pkg"
        else
            ERR "$pkg — NOT installed"
        fi
    done

    # Optional: liteparse
    if echo "$_pip_list" | grep -qi "^liteparse$" 2>/dev/null; then
        [ "$QUIET" = false ] && OK "liteparse (PDF OCR)"
    else
        WARN "liteparse not installed — PDF scanning/OCR unavailable"
        if [ "$FIX" = true ]; then
            INFO "Installing liteparse..."
            uv sync --extra liteparse 2>/dev/null && OK "liteparse installed" || ERR "liteparse install failed"
        fi
    fi

    # Optional: ebooklib + html2text
    if echo "$_pip_list" | grep -qi "^ebooklib$" 2>/dev/null; then
        [ "$QUIET" = false ] && OK "ebooklib + html2text (EPUB)"
    else
        WARN "ebooklib/html2text not installed — EPUB support unavailable"
        if [ "$FIX" = true ]; then
            INFO "Installing EPUB deps..."
            uv sync --extra epub 2>/dev/null && OK "EPUB deps installed" || ERR "EPUB install failed"
        fi
    fi
fi

# ═══════════════════════════════════════════════════════════
# 2. External Tools
# ═══════════════════════════════════════════════════════════
HEAD "External Tools"

for tool in ollama cloudflared; do
    if command -v "$tool" >/dev/null 2>&1; then
        version=$($tool --version 2>/dev/null | head -1 || echo "installed")
        OK "$tool — $version"
    else
        ERR "$tool — NOT found"
    fi
done

# ═══════════════════════════════════════════════════════════
# 3. Ollama Models
# ═══════════════════════════════════════════════════════════
HEAD "Ollama Models"

if command -v ollama >/dev/null 2>&1; then
    # Check ollama is running
    if ollama list >/dev/null 2>&1; then
        OK "Ollama service running"

        # Get required models from .env
        LLM_MODEL="magic-grimoire:3b"
        EMBED_MODEL="nomic-embed-text:latest"

        if [ -f .env ]; then
            # Try to get from .env
            ENV_LLM=$(grep "^OLLAMA_LLM_MODEL=" .env | cut -d= -f2 | tr -d '"' || echo "")
            ENV_EMBED=$(grep "^OLLAMA_EMBED_MODEL=" .env | cut -d= -f2 | tr -d '"' || echo "")
            [ -n "$ENV_LLM" ] && LLM_MODEL="$ENV_LLM"
            [ -n "$ENV_EMBED" ] && EMBED_MODEL="$ENV_EMBED"
        fi

        # Check LLM model
        if ollama show "$LLM_MODEL" >/dev/null 2>&1; then
            # Get model size
            SIZE=$(ollama show "$LLM_MODEL" 2>/dev/null | grep -i "size" | head -1 | grep -oP '\d+\.?\d*[KMG]' || echo "")
            [ -n "$SIZE" ] && OK "$LLM_MODEL ($SIZE)" || OK "$LLM_MODEL"
        else
            ERR "$LLM_MODEL — not found (run: ollama create $LLM_MODEL -f Modelfile.3b)"
        fi

        # Check embed model
        if ollama show "$EMBED_MODEL" >/dev/null 2>&1; then
            OK "$EMBED_MODEL"
        else
            ERR "$EMBED_MODEL — not found (run: ollama pull nomic-embed-text)"
        fi
    else
        ERR "Ollama service not running — start: ollama serve"
    fi
fi

# ═══════════════════════════════════════════════════════════
# 4. System Packages (for LiteParse OCR)
# ═══════════════════════════════════════════════════════════
HEAD "System Packages (OCR)"

# Check tesseract for LiteParse OCR
if command -v tesseract >/dev/null 2>&1; then
    TESS_VER=$(tesseract --version 2>&1 | head -1 | grep -oP '\d+\.\d+\.\d+' || echo "installed")
    OK "tesseract-ocr ($TESS_VER)"

    # Check TESSDATA_PREFIX
    TESSDATA=$(tesseract --print-parameters 2>/dev/null | head -1 || echo "")
    if [ -n "$TESSDATA" ]; then
        [ "$QUIET" = false ] && OK "tesseract data accessible"
    else
        # Check common locations
        for dir in /usr/share/tesseract-ocr/5/tessdata /usr/share/tesseract-ocr/4/tessdata /usr/share/tessdata; do
            if [ -d "$dir" ]; then
                [ "$QUIET" = false ] && OK "tessdata at $dir"
                break
            fi
        done || WARN "tessdata not found — OCR may fail for some PDFs"
    fi
else
    WARN "tesseract-ocr not installed — LiteParse OCR may fail"
    INFO "Install: sudo apt install tesseract-ocr tesseract-ocr-eng"
fi

# Check Poppler (for pdf2image, used by some parsers)
if command -v pdftoppm >/dev/null 2>&1; then
    [ "$QUIET" = false ] && OK "poppler-utils (pdftoppm)"
else
    [ "$QUIET" = false ] && WARN "poppler-utils not installed — some PDF features may be limited"
fi

# ═══════════════════════════════════════════════════════════
# 5. Configuration
# ═══════════════════════════════════════════════════════════
HEAD "Configuration"

if [ -f .env ]; then
    OK ".env exists"
    
    # Check required env vars
    for var in TELEGRAM_TOKEN GEMINI_API_KEY; do
        VALUE=$(grep "^${var}=" .env | cut -d= -f2- | tr -d '"' || echo "")
        if [ -n "$VALUE" ] && [ "$VALUE" != "your-${var,,}-here" ]; then
            [ "$QUIET" = false ] && OK "${var}=...${VALUE: -4}"
        else
            ERR "${var} — missing or placeholder"
        fi
    done
else
    ERR ".env not found — copy .env.example and fill in tokens"
fi

# Check docs directory
if [ -d "app/docs" ]; then
    COUNT=$(find app/docs -maxdepth 1 -type f 2>/dev/null | wc -l)
    if [ "$COUNT" -gt 0 ]; then
        OK "app/docs/ — $COUNT file(s)"
    else
        WARN "app/docs/ — empty (add your study PDFs/EPUBs)"
    fi
else
    WARN "app/docs/ — does not exist (create it and add documents)"
fi

# Check data directory
for dir in data; do
    if [ -d "$dir" ]; then
        [ "$QUIET" = false ] && OK "$dir/"
    else
        WARN "$dir/ — will be created on first run"
    fi
done

# ═══════════════════════════════════════════════════════════
# Summary
# ═══════════════════════════════════════════════════════════
echo ""
if [ "$HAS_ERROR" = true ]; then
    echo -e "${RED}❌ Errors found — some features will not work.${NC}"
    echo -e "${RED}   Fix the ❌ items above and re-run this check.${NC}"
elif [ "$HAS_WARN" = true ]; then
    echo -e "${YELLOW}⚠️  Warnings found — bot will run but some features limited.${NC}"
else
    echo -e "${GREEN}✅ All dependencies OK!${NC}"
fi

# Exit with error code if there were errors
if [ "$HAS_ERROR" = true ]; then
    exit 1
fi
exit 0