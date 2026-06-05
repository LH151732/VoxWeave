.PHONY: install reinstall whisper uninstall dev test lint

# Install as a global uv tool (end-user mode): puts the voxweave command on PATH.
# The full local pipeline (separation / ASR / forced alignment / layout / song-skip), CJK
# line-break, and translation are baked into the core deps, so a bare `uv tool install voxweave`
# already works. Default EXTRAS=all additionally pulls the faster-whisper hybrid engine
# (--model large-v3*). Slim install without whisper: make install EXTRAS=qwen
#   (qwen is a no-op alias kept for back-compat; the core already contains it).
# torch wheel is pinned to the Blackwell sm_120 cu128 build (no GPU auto-detect) and
# installed into the same isolated tool venv (a bare `uv pip` cannot reach that venv).
EXTRAS ?= all

install:
	uv tool install --force --torch-backend=cu128 ".[$(EXTRAS)]"
	@voxweave --version
	@git diff --quiet 2>/dev/null && echo "installed (git $$(git rev-parse --short HEAD))" || echo "installed (git $$(git rev-parse --short HEAD), uncommitted changes present)"

# Force reinstall after pulling new code.
reinstall:
	uv tool install --force --reinstall --torch-backend=cu128 ".[$(EXTRAS)]"
	@voxweave --version
	@git diff --quiet 2>/dev/null && echo "reinstalled (git $$(git rev-parse --short HEAD))" || echo "reinstalled (git $$(git rev-parse --short HEAD), uncommitted changes present)"

# EXTRAS=all already includes whisper; this target is only needed if you used a slim
# EXTRAS (e.g. EXTRAS=qwen) and want to add [whisper] on top
# (pulls faster-whisper + ctranslate2 + PyAV; whisper weights auto-download on first use).
whisper:
	$(MAKE) install EXTRAS=$(EXTRAS),whisper

uninstall:
	uv tool uninstall voxweave

# Development environment (for code changes, matches CI).
dev:
	uv sync --all-extras --dev

# Unit tests (no network).
test:
	uv run pytest tests/ -v

# Lint / format (project-wide; repo has no ruff config but this is the canonical invocation).
lint:
	uv run --no-project --with ruff ruff check --fix .
	uv run --no-project --with ruff ruff format .
