# ============================================================================
#  Makefile - every command this project needs, with a short name.
#
#  Type `make` or `make help` to see the menu.
#
#  WARNING: every indented line below must start with a REAL TAB, not spaces.
#  Spaces give you: "Makefile:NN: *** missing separator.  Stop."
# ============================================================================

# Running plain `make` with no arguments shows the help instead of erroring.
.DEFAULT_GOAL := help

# Read .env so targets below can use $(POSTGRES_USER) etc.
# The leading "-" means "don't fail if the file is missing".
-include .env
export

# Your shell has VIRTUAL_ENV set (pyenv). uv correctly ignores it in favour of
# this project's .venv, but warns loudly every time. Stop passing it down.
unexport VIRTUAL_ENV

COMPOSE := docker compose
PG_USER ?= $(or $(POSTGRES_USER),lodestar)
PG_DB   ?= $(or $(POSTGRES_DB),lodestar)


# ---------------------------------------------------------------------------
#  Help
#  Scans this file for lines with "## " and prints them as a menu.
#  That means the menu can never go out of date - it IS the file.
# ---------------------------------------------------------------------------
help:  ## Show this menu
	@echo ""
	@echo "  Lodestar - available commands"
	@echo ""
	@grep -hE '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'
	@echo ""


# ---------------------------------------------------------------------------
#  Setup
# ---------------------------------------------------------------------------
install:  ## Install/update all dependencies from uv.lock
	uv sync


# ---------------------------------------------------------------------------
#  Database (Docker)
# ---------------------------------------------------------------------------
up:  ## Start Postgres and wait until it is actually ready
	@$(COMPOSE) up -d
	@printf "waiting for postgres"
	@until $(COMPOSE) exec -T postgres pg_isready -U $(PG_USER) >/dev/null 2>&1; do \
		printf "."; sleep 1; \
	done
	@echo " ready"

down:  ## Stop containers (YOUR DATA IS KEPT)
	$(COMPOSE) down

ps:  ## Show what is running and whether it is healthy
	$(COMPOSE) ps

logs:  ## Tail the Postgres logs (ctrl-C to exit)
	$(COMPOSE) logs -f postgres

db:  ## Open a psql shell inside the database
	$(COMPOSE) exec postgres psql -U $(PG_USER) -d $(PG_DB)

db-reset:  ## DELETE ALL DATA and start a fresh database
	@echo "This will permanently delete the lodestar-pgdata volume."
	@read -p "Type 'yes' to continue: " ok; [ "$$ok" = "yes" ] || exit 1
	$(COMPOSE) down -v
	@$(MAKE) up


# ---------------------------------------------------------------------------
#  Database migrations (Alembic)
# ---------------------------------------------------------------------------
migrate:  ## Apply all pending migrations
	uv run alembic upgrade head

migration:  ## Generate a migration from model changes (make migration m="add x")
	@test -n "$(m)" || (echo 'usage: make migration m="what changed"'; exit 1)
	uv run alembic revision --autogenerate -m "$(m)"

rollback:  ## Undo the most recent migration
	uv run alembic downgrade -1

migrations:  ## Show which migration the database is on, and the history
	@uv run alembic current
	@uv run alembic history --indicate-current

sql:  ## Print the SQL for pending migrations WITHOUT running it
	uv run alembic upgrade head --sql

# ---------------------------------------------------------------------------
#  Ingestion
# ---------------------------------------------------------------------------
ingest:  ## Fetch from all five sources into the database
	uv run python -m lodestar.ingestion

enrich:  ## Fetch article bodies and transcripts (make enrich n=50)
	uv run python -m lodestar.enrichment $(or $(n),50)

ui:  ## Open a browser UI for the database (http://localhost:8080)
	@$(COMPOSE) --profile tools up -d adminer
	@echo ""
	@echo "  Open  http://localhost:8080"
	@echo ""
	@echo "    System    PostgreSQL"
	@echo "    Server    postgres"
	@echo "    Username  $(PG_USER)"
	@echo "    Password  (POSTGRES_PASSWORD from your .env)"
	@echo "    Database  $(PG_DB)"
	@echo ""
	@echo "  Then click the 'articles' table."
	@echo ""

ui-down:  ## Stop the database browser UI
	@$(COMPOSE) --profile tools stop adminer

peek:  ## Browse the corpus: counts, longest/shortest docs, a sample
	@uv run python -m lodestar.storage.peek

show:  ## Print one full document (make show q="transformer")
	@uv run python -m lodestar.storage.peek --show "$(q)"

dsn:  ## Print the connection string for a GUI client (TablePlus, DBeaver...)
	@uv run python -c "from lodestar.core.config import get_settings as g; print(g().database_url)"

articles:  ## Show what is in the database, by source
	@uv run python -c "from lodestar.storage.session import session_scope; from lodestar.storage.repositories import ArticleRepository; \
	  import contextlib; \
	  s=session_scope(); sess=s.__enter__(); r=ArticleRepository(sess); \
	  print('total:', r.count()); \
	  [print(f'  {k.value:<12} {v}') for k,v in sorted(r.count_by_source().items(), key=lambda x: -x[1])]; \
	  s.__exit__(None,None,None)"

# ---------------------------------------------------------------------------
#  Code quality
# ---------------------------------------------------------------------------
lint:  ## Check code style and find likely bugs
	uv run ruff check .

format:  ## Auto-fix formatting and safe lint errors
	uv run ruff format .
	uv run ruff check --fix .

typecheck:  ## Verify type hints are honest (mypy strict)
	uv run mypy lodestar

test:  ## Run the test suite
	uv run pytest

cov:  ## Run tests and report coverage
	uv run pytest --cov --cov-report=term-missing

check: lint typecheck test  ## Everything CI runs. Green here = green in CI.
	@echo ""
	@echo "  all checks passed"


# ---------------------------------------------------------------------------
#  Housekeeping
# ---------------------------------------------------------------------------
clean:  ## Delete caches and build junk (safe - nothing important)
	rm -rf .mypy_cache .ruff_cache .pytest_cache htmlcov .coverage
	find . -type d -name __pycache__ -not -path "./.venv/*" -exec rm -rf {} + 2>/dev/null || true
	@echo "cleaned"


# ---------------------------------------------------------------------------
#  These are COMMANDS, not files.
#  Without this line, a folder named "test" would make `make test` do nothing.
# ---------------------------------------------------------------------------
.PHONY: help install up down ps logs db db-reset ui ui-down ingest enrich peek show dsn articles migrate migration rollback migrations sql lint format typecheck test cov check clean
