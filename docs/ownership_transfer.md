# Cairo Native — Ownership Transfer Document

> **From:** LambdaClass (lambdaclass)
> **To:** StarkWare
> **Repository:** `cairo_native`

---

## Table of Contents

1. [Overview](#1-overview)
2. [CI/CD Inventory](#2-cicd-inventory)
3. [Secrets & Credentials](#3-secrets--credentials)
4. [External Services](#4-external-services)
5. [LambdaClass-Owned External Repositories](#5-lambdaclass-owned-external-repositories)
6. [Pinned Toolchain & System Dependencies](#6-pinned-toolchain--system-dependencies)
7. [CODEOWNERS & Access Control](#7-codeowners--access-control)
8. [Release Process](#8-release-process)
9. [Hardcoded LambdaClass References](#9-hardcoded-lambdaclass-references)
10. [Action Items for StarkWare DevOps](#10-action-items-for-starkware-devops)
11. [Open Questions for LambdaClass](#11-open-questions-for-lambdaclass)

---

## 1. Overview

`cairo_native` is a compiler that converts Cairo's Sierra IR to MLIR and executes it natively (JIT or AOT). The CI is built entirely on **GitHub Actions** and consists of 8 workflow files covering linting, testing, benchmarking, block replay, coverage, documentation, release, and compatibility checking.

---

## 2. CI/CD Inventory

| Workflow File | Trigger | Purpose | Runner OS |
|---|---|---|---|
| `ci.yml` | Push to `main`, PRs, merge queue | Clippy, rustfmt, unused deps, tests (Linux + macOS), coverage (4 partitions), Dockerfile build, debug-utils builds | Ubuntu 24.04, macOS 14 |
| `bench-hyperfine.yml` | PRs, merge queue | Hyperfine benchmarks + base-vs-head comparison (posts PR comment) | Ubuntu 24.04 |
| `starknet-blocks.yml` | PRs, merge queue | Replays 5 specific Starknet mainnet blocks, compares Native vs VM state dumps | Ubuntu latest |
| `daily.yml` | Cron `0 0 * * *` (midnight UTC) | Replays ~50 block ranges (20 blocks each), compares Native vs VM; auto-creates GitHub issue on failure | Ubuntu latest |
| `cairo-release-check.yml` | Cron `0 8 * * 1` (Mondays 08:00 UTC) + manual dispatch | Checks if a new `cairo-lang` release exists, patches and runs tests; auto-creates issue on failure | Ubuntu latest |
| `release.yml` | Push tag `v[0-9]+.*` | Creates GitHub Release, builds & uploads binaries for Linux (x86_64) and macOS (aarch64) | Ubuntu 24.04, macOS 14 |
| `publish.yml` | Push tag `v*.*.*` | Publishes `sierra-emu`, `cairo-native`, `starknet-native-compile` to crates.io | Ubuntu 24.04 |
| `rustdoc.yml` | Push to `main` | Builds `cargo doc` and deploys to GitHub Pages (`gh-pages` branch) | Ubuntu 24.04 |

### Shared Composite Action

- **`.github/actions/install-linux-deps/action.yml`** — Frees disk space on the runner, adds the LLVM 19 apt repo, and installs all LLVM/MLIR/Clang 19 system packages. Used by `daily.yml`, `starknet-blocks.yml`, and `cairo-release-check.yml`.

### Concurrency Controls

- `ci.yml` and `bench-hyperfine.yml` use concurrency groups per workflow + branch ref, with cancel-in-progress enabled (except for `main` branch pushes).

---

## 3. Secrets & Credentials

The following GitHub Actions secrets **must be configured** in the new organization/repo:

| Secret Name | Used In | Purpose |
|---|---|---|
| `CODECOV_TOKEN` | `ci.yml` (upload-coverage) | Uploading coverage reports to Codecov |
| `CARGO_REGISTRY_TOKEN` | `publish.yml` | Publishing crates to crates.io |
| `RPC_ENDPOINT_TESTNET` | `daily.yml`, `starknet-blocks.yml` | Starknet testnet RPC endpoint for block replay |
| `RPC_ENDPOINT_MAINNET` | `daily.yml`, `starknet-blocks.yml` | Starknet mainnet RPC endpoint for block replay |
| `GITHUB_TOKEN` | Multiple (auto-provided) | Standard GitHub token — auto-generated, no migration needed |

### Actions Required

- **Codecov:** The Codecov integration needs to be set up under StarkWare's Codecov organization. The `CODECOV_TOKEN` must be regenerated for the new repo location.
- **Crates.io:** Determine who should own the crates (`cairo-native`, `sierra-emu`, `starknet-native-compile`) on crates.io. The existing token belongs to a LambdaClass account. StarkWare needs its own crates.io API token and must be added as an owner on these crates (or the crate names must be re-claimed if transferred).
- **RPC Endpoints:** These are Starknet RPC URLs (likely from an Alchemy/Infura-like provider, or self-hosted). StarkWare should either obtain the existing endpoints or provision its own. They are used by the block replay CI to fetch historical block data.

---

## 4. External Services

| Service | Usage | Migration Notes |
|---|---|---|
| **Codecov** | Coverage reporting with 90% threshold on `src/` | Re-register repo under StarkWare's Codecov org; `.github/codecov.yml` config is in-repo |
| **crates.io** | Publishing 3 crates | Transfer crate ownership or create new API token |
| **GitHub Pages** | Rustdoc hosted on `gh-pages` branch | Ensure Pages is enabled in repo settings; URL will change with the new org |
| **GitHub Issues (auto-created)** | `daily.yml` and `cairo-release-check.yml` auto-create issues from templates on failure | Works out of the box with `GITHUB_TOKEN` |

---

## 5. LambdaClass-Owned External Repositories

> ⚠️ **This is the most critical area for the transfer.** Several CI workflows depend on LambdaClass-owned forks/repos that are *not* part of this repository.

### 5.1 `lambdaclass/starknet-replay`

**Used in:** `daily.yml`, `starknet-blocks.yml`

This is a block-replay tool that re-executes historical Starknet blocks using either the Cairo VM or Cairo Native, then compares the resulting state dumps. Both the daily and per-PR block replay workflows check out this repo at either `main` or a pinned commit.

- `daily.yml` checks out `starknet-replay` at its latest `main`.
- `starknet-blocks.yml` checks out `starknet-replay` at a **pinned commit** (`40e8e5693168cdbc1c7f73aaee5206808989e329`).

Both workflows then **patch `starknet-replay`'s `Cargo.toml`** via `sed` to point its `cairo-native` and sequencer dependencies to local checkouts.

### 5.2 `lambdaclass/sequencer`

**Used in:** `daily.yml`, `starknet-blocks.yml`

This is a **fork of StarkWare's own sequencer** (`starkware-libs/sequencer`), maintained by LambdaClass with Cairo Native integration. The workflows check out specific branches/commits:

- `daily.yml` checks out branch `main-v0.14.1`.
- `starknet-blocks.yml` checks out a pinned commit (`995d850b31ff942058e400b69e853a437bbbedde`, noted as `main-v0.14.2`).

The sequencer's `Cargo.toml` is also patched to use the local `cairo_native`.

### 5.3 `lambdaclass/cairo-vm` (Git Dependency)

**Used in:** `Cargo.toml` (workspace dependency)

```toml
cairo-vm = { git = "https://github.com/lambdaclass/cairo-vm", rev = "a87e9e6a2cc4b759ae30a9c1a08efabf78ca4c17" }
```

This is a **pinned git dependency** on LambdaClass's `cairo-vm`. It's used in dev-dependencies for testing (comparing native execution against VM execution). If LambdaClass changes access to this repo, builds will break.

### 5.4 PR Template References

The PR template (`.github/pull_request_template.md`) instructs contributors to create companion PRs in:
- `https://github.com/lambdaclass/sequencer`
- `https://github.com/lambdaclass/starknet-replay`

This workflow/process needs to be redefined.

---

## 6. Pinned Toolchain & System Dependencies

| Dependency | Version | Where Pinned |
|---|---|---|
| **Rust** | `1.84.1` (CI workflows) / `1.89.0` (`rust-toolchain.toml` local) | CI `.yml` files use `dtolnay/rust-toolchain@1.84.1`; local dev uses `rust-toolchain.toml` with `1.89.0` |
| **LLVM/MLIR/Clang** | **19** | Installed from `apt.llvm.org` (Linux) or Homebrew (macOS) |
| **Cairo compiler** | `2.16.0` | `Makefile` (`CAIRO_2_VERSION`), downloaded from `starkware-libs/cairo` releases |
| **Scarb** | `2.16.0` | `Makefile` (`SCARB_VERSION`), installed via `docs.swmansion.com/scarb/install.sh` |
| **cairo-lang-* crates** | `~2.16.0` | `Cargo.toml` workspace dependencies |

### ⚠️ Note: Rust Version Discrepancy

There is a discrepancy: CI workflows pin Rust to `1.84.1`, but the local `rust-toolchain.toml` specifies `1.89.0`. The `cairo-release-check.yml` workflow also uses `1.89.0`. This should be reconciled.

---

## 7. CODEOWNERS & Access Control

The current `CODEOWNERS` file assigns all files to LambdaClass team members:

```
* @edg-l @igaray @azteca1998 @jrchatruc @entropidelic @fmoletta @Oppen @pefontana @gabrielbosio
```

**Action:** Replace with StarkWare team members and/or a GitHub team handle.

---

## 8. Release Process

From `docs/release.md`:

1. Update version in root `Cargo.toml`.
2. Open and merge a PR.
3. Run `git tag v<version>` and `git push --tags`.
4. This triggers `release.yml` (GitHub Release + binary uploads) and `publish.yml` (crates.io publish).
5. Edit the GitHub release to auto-generate release notes.

The release page URL in `docs/release.md` currently points to `https://github.com/lambdaclass/cairo_native/releases` — update after transfer.

### Published Binaries

The release builds and uploads these binaries for Linux and macOS:
- `cairo-native-compile`
- `cairo-native-dump`
- `cairo-native-run`
- `cairo-native-stress`
- `cairo-native-test`
- `starknet-native-compile`

### Published Crates

Published to crates.io in this order:
1. `sierra-emu`
2. `cairo-native`
3. `starknet-native-compile`

---

## 9. Hardcoded LambdaClass References

All references to `lambdaclass` that need updating:

| File | Reference | Type |
|---|---|---|
| `Cargo.toml` (line 23) | `repository = "https://github.com/lambdaclass/cairo_native"` | Metadata |
| `Cargo.toml` (line 100) | `cairo-vm = { git = "https://github.com/lambdaclass/cairo-vm", ... }` | Git dependency |
| `docs/release.md` (line 9) | `https://github.com/lambdaclass/cairo_native/releases` | Documentation |
| `.github/pull_request_template.md` (lines 24-25) | Links to `lambdaclass/sequencer` and `lambdaclass/starknet-replay` | Process |
| `daily.yml` (lines 89, 99) | `lambdaclass/starknet-replay`, `lambdaclass/sequencer` | CI checkout |
| `starknet-blocks.yml` (lines 33, 45) | `lambdaclass/starknet-replay`, `lambdaclass/sequencer` | CI checkout |

---

## 10. Action Items for StarkWare DevOps

### Immediate (Before/At Transfer)

- [ ] **Provision secrets:** `CODECOV_TOKEN`, `CARGO_REGISTRY_TOKEN`, `RPC_ENDPOINT_TESTNET`, `RPC_ENDPOINT_MAINNET`.
- [ ] **Set up Codecov** under StarkWare's Codecov organization for the new repo.
- [ ] **Claim crates.io ownership** of `cairo-native`, `sierra-emu`, `starknet-native-compile` (coordinate with LC for `cargo owner --add`).
- [ ] **Enable GitHub Pages** in repo settings (source: `gh-pages` branch).
- [ ] **Update `CODEOWNERS`** to StarkWare team members.
- [ ] **Ensure LLVM 19 availability:** CI installs LLVM 19 from `apt.llvm.org`. No self-hosted infra is needed, but be aware the jobs require ~15 GB of free disk space (runners aggressively free space).

### Short-Term (Post Transfer)

- [ ] **Update all `lambdaclass` references** listed in Section 9.
- [ ] **Reconcile Rust toolchain versions** (`1.84.1` in CI vs `1.89.0` in `rust-toolchain.toml`).
- [ ] **Decide on the block-replay strategy** (see Open Questions below).
- [ ] **Review merge queue / branch protection rules** — the CI workflows support `merge_group` triggers, meaning a merge queue is likely configured.
- [ ] **Review runner costs** — the `daily.yml` workflow runs 50 blocks × 2 runners = 100 jobs nightly, plus the per-PR `starknet-blocks.yml` runs 5 blocks × 2 runners = 10 jobs per PR.

### Ongoing

- [ ] **Monitor the weekly Cairo release check** — auto-creates issues when a new `cairo-lang` version is released and tests fail. This ensures the project stays up-to-date with Cairo compiler releases.

---

## 11. Open Questions for LambdaClass

### Block Replay Infrastructure

1. **`lambdaclass/starknet-replay`** — Will this repo be transferred to StarkWare as well, or should StarkWare fork/clone it? What is the maintenance status and who has been updating the block list?

2. **`lambdaclass/sequencer`** — This is a fork of `starkware-libs/sequencer` with Cairo Native integration. Key questions:
   - What modifications does this fork contain on top of upstream `starkware-libs/sequencer`?
   - Are branches `main-v0.14.1` and `main-v0.14.2` rebased on upstream, or are they diverged?
   - Is the plan for these changes to eventually land in the upstream StarkWare sequencer? If so, the block-replay workflows can be simplified to use the upstream repo directly.
   - How is this fork kept in sync with the upstream sequencer?

3. **RPC Endpoints** — What provider do the current `RPC_ENDPOINT_TESTNET` and `RPC_ENDPOINT_MAINNET` secrets point to? Are they rate-limited? The daily workflow runs up to 25 concurrent jobs, each making RPC calls.

4. **Block list curation** — How were the specific block numbers in `daily.yml` (740k–1.1M range) and `starknet-blocks.yml` (5 specific blocks) chosen? Are they meant to cover specific edge cases or contract types?

### Dependency Questions

5. **`lambdaclass/cairo-vm` git dependency** — The workspace `Cargo.toml` depends on a pinned commit of `lambdaclass/cairo-vm`. Is there a published crates.io version that could replace this? What modifications, if any, does this fork contain?

6. **Crate ownership on crates.io** — Who currently owns the `cairo-native`, `sierra-emu`, and `starknet-native-compile` crates? Can StarkWare team members be added as owners before the transfer?

### Process Questions

7. **Breaking change process** — The PR template includes a checklist for creating companion PRs in the sequencer fork and starknet-replay. How should this process work post-transfer?

8. **Daily failure triage** — When the daily block run fails and auto-creates an issue, who has been triaging these? What's the typical resolution workflow?
