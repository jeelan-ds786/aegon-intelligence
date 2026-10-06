# Continuous integration

Pull requests and pushes to `main` run two workflows without cloud credentials or repository
secrets. Fork pull requests receive only the read-only `GITHUB_TOKEN` supplied by GitHub. The
top-level workflow permission is `contents: read`; CodeQL alone receives the additional permissions
required to upload analysis.

## Required checks

Configure the `main` branch ruleset or branch protection rule with:

- Require a pull request before merging, with at least one approval.
- Dismiss stale approvals when new commits are pushed.
- Require review from Code Owners.
- Require conversation resolution before merging.
- Require branches to be up to date before merging.
- Require status checks `CI / CI Gate` and `Security / Security Gate`.
- Block force pushes and branch deletion.
- Do not allow bypasses for administrators or repository roles.

The aggregate gates fail when any required job fails, is cancelled, or is skipped. Keep the
individual checks visible for diagnosis, but use the two stable gate names for branch protection.

## Workflow coverage

`CI` runs Ruff, Pyright strict mode, pytest, changed-line coverage, ESLint, TypeScript strict
checking, Vitest, and the TypeScript project-reference build. Python changed lines must have at
least 90% coverage as measured by `diff-cover` against the pull request base branch.

`Security` runs `pip-audit`, production-only `pnpm audit`, Gitleaks across Git history, CodeQL for
Python and JavaScript/TypeScript, and a Syft-generated CycloneDX JSON SBOM. The SBOM is retained as
the `aegon-rag-cyclonedx-sbom` workflow artifact for 30 days.

All external actions use immutable full commit SHAs. Dependabot proposes weekly updates for GitHub
Actions, pnpm, Python, and Terraform dependencies; action updates must remain SHA-pinned.

## Secret-scanning acceptance evidence

Demonstrate the control in the implementation pull request with a synthetic Gitleaks test token,
never a real credential:

1. Create a temporary branch and add a file containing the documented Gitleaks test value
   `AKIAIOSFODNN7EXAMPLE`.
2. Push the commit and record the failing `Security / Secret Scan` run URL in the pull request.
3. Remove the file with a new commit, then squash or rebase the leaked test commit out of the
   branch history and force-push the repaired branch.
4. Record the succeeding `Security / Secret Scan` run URL and confirm `Security / Security Gate`
   is green.

Use this pull request description section:

```text
## Secret scanning acceptance

- Red run (synthetic test token present): <workflow-run-url>
- Green run (test token removed from Git history): <workflow-run-url>
- Test data only; no live secret was used: yes
```

Do not merge a branch that still contains the synthetic token in any reachable commit. The red and
green workflow URLs are the audit evidence; the test token does not belong in the final branch.