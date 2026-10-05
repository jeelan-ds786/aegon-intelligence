.PHONY: bootstrap lint typecheck test check serve serve-web

UV := $(shell python3 -c 'import site; print(site.USER_BASE + "/bin/uv")')

bootstrap:
	python3 -m pip install --user "uv==0.8.22"
	corepack enable pnpm
	corepack install
	"$(UV)" sync --frozen --all-packages --dev
	pnpm install --frozen-lockfile
	.venv/bin/pre-commit install

lint:
	.venv/bin/ruff check .
	.venv/bin/ruff format --check .
	pnpm lint

typecheck:
	.venv/bin/pyright
	pnpm typecheck

test:
	.venv/bin/pytest
	pnpm test

check: lint typecheck test

serve:
	@echo "API service implementation is intentionally deferred."

serve-web:
	@echo "Web application implementation is intentionally deferred."