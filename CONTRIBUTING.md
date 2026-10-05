# Contributing to Aegon-RAG

## Prerequisites

- Python 3.12
- Node.js 22 or newer with Corepack
- Git

## Setup

Run `make bootstrap` from the repository root. This installs the pinned uv and pnpm
toolchains, creates the Python environment, installs both workspaces, and configures
pre-commit.

## Development

Keep each change focused and include tests. Run `make check` before opening a pull
request. Commits use Conventional Commits, such as `feat: add retrieval contract`.

Do not commit credentials, customer data, generated build output, or local environment
files. Copy `.env.example` to `.env` for local-only configuration.