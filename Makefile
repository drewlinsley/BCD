# BCD — one-liners for setup, ingest, API, iOS, and verification.
# Everything here runs on the Intel Mac in the repo's README (Python 3.12 + Swift CLT).

PY := .venv/bin/python
PIP := .venv/bin/pip
SOURCE ?= openbrewerydb
LIMIT ?= 50

.PHONY: help
help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}'

# ---- setup ----
.PHONY: venv
venv: ## create the Python 3.12 venv and install the project (editable + dev)
	python3.12 -m venv .venv || /usr/local/opt/python@3.12/bin/python3.12 -m venv .venv
	$(PIP) install -q --upgrade pip
	$(PIP) install -q -e ".[dev]"
	@echo "venv ready. Activate with: source .venv/bin/activate"

.PHONY: toolchain
toolchain: ## install Xcode 26.0 + brew deps (see docs). Xcode needs your Apple ID.
	brew install --cask xcodes-app || true
	brew install aria2 python@3.12 uv node postgresql@16 redis xcodegen jq gh git-lfs pgvector postgis
	@echo "Open Xcodes.app, sign in, install Xcode 26.0. Then: sudo xcode-select -s /Applications/Xcode.app"

# ---- data pipeline ----
.PHONY: ingest
ingest: ## run one connector end to end. SOURCE=demo|openbrewerydb|off|ttb LIMIT=50
	$(PY) -m bcd_ingest $(SOURCE) --limit $(LIMIT)

.PHONY: enrich
enrich: ## backfill chemistry-prior sensory vectors onto gold products (feeds vector search)
	$(PY) -m bcd_enrich

.PHONY: demo
demo: ## seed 4 recipe-complete demo products, then enrich → ready for scan + recommend
	$(PY) -m bcd_ingest demo
	$(PY) -m bcd_enrich
	@echo "Demo catalog ready. Set BCD_STORE_BACKEND=postgres to use Postgres. Then: make api"

.PHONY: validate-registry
validate-registry: ## check every data/registry/sources/*.yaml against schema.json
	$(PY) scripts/validate_registry.py

.PHONY: seed-registry
seed-registry: ## regenerate the source registry from scripts/seed_registry.py
	$(PY) scripts/seed_registry.py

# ---- api ----
.PHONY: api
api: ## serve the FastAPI backend on :8000 (reads the local ingest store)
	$(PY) -m uvicorn bcd_api.app:app --host 127.0.0.1 --port 8000 --reload

# ---- sentinels ----
.PHONY: sentinel-dryrun
sentinel-dryrun: ## validate sentinels/*.yaml; add LIVE=1 to issue one ~$$0.005 Parallel call
	@if [ "$(LIVE)" = "1" ]; then $(PY) -m bcd_sentinel dryrun --live; \
	else $(PY) -m bcd_sentinel dryrun; fi

# ---- codegen ----
.PHONY: codegen
codegen: ## regenerate Swift + Python telemetry bindings from telemetry/events.yaml
	$(PY) scripts/codegen_events.py

# ---- tests ----
.PHONY: test
test: test-py test-swift ## run all tests (python + swift)

.PHONY: test-py
test-py: ## run the python test suite
	$(PY) -m pytest tests/ -q

.PHONY: test-swift
test-swift: ## build + test BCDKit on the host (no Xcode needed)
	cd ios/BCDKit && swift build && swift test

.PHONY: lint
lint: ## ruff over the python code
	.venv/bin/ruff check packages services scripts tests

# ---- ios ----
.PHONY: ios-gen
ios-gen: ## generate BCDApp.xcodeproj from ios/project.yml (needs xcodegen)
	cd ios && xcodegen generate --spec project.yml
	@echo "Open ios/BCDApp.xcodeproj in Xcode 26 to build/run the app."

.PHONY: ios-build
ios-build: ios-gen ## build the app against the iOS 26 SDK (needs Xcode 26)
	cd ios && xcodebuild -project BCDApp.xcodeproj -scheme BCDApp \
		-destination 'generic/platform=iOS' build

# ---- verification (the plan's checklist) ----
.PHONY: verify
verify: validate-registry test-py test-swift ## run the laptop-verifiable checks
	@echo "✓ registry valid · python tests · swift tests all green"
