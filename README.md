# Veystrix

**A local AI coding engine for Apple Silicon Macs. Free to use. Runs offline. No account.**

Veystrix runs a very large Mixture-of-Experts language model on a 16 GB MacBook
by streaming the model's experts from your SSD instead of holding them in memory.
It serves an OpenAI-compatible API on your own Mac, so coding agents such as
Hermes Agent, Qwen Code and the DeepSeek Harness can work with it for hours
without a cloud subscription. A native macOS app starts, stops and supervises the
engine.

> **Status: pre-release.** The first public build has not been published yet
> (see [Releases](#downloads)). Nothing on this page is a claim that a release has
> been validated on your hardware. Read [Verified performance](docs/performance.md)
> before you rely on any number.

Veystrix is **proprietary freeware**. It is free to download and use, and it is
**not open source**. See [LICENSE](LICENSE) and
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). Some components inside it are open
source and keep their own licenses.

## What it does

| | |
|---|---|
| **Runs locally** | Prompts, code and sessions stay on your Mac. Veystrix has no telemetry and needs no account or activation. |
| **Native macOS app** | SwiftUI app for Apple Silicon: dashboard, explicit Start / Stop / Restart, model folder validation, logs, sanitized diagnostics, settings. It never starts the model just because you opened it. |
| **Supervised engine** | The engine runs as a separate helper process with memory and swap guards, a watchdog, crash recovery and a persistent SSD prefix cache. A guard stop is a deliberate refusal, not a bug. |
| **OpenAI-compatible API** | `http://127.0.0.1:8080/v1` (loopback only by default). Chat completions with streaming, tool calls, reasoning content and per-request reasoning effort. |
| **Agent-ready** | Documented setups for [Hermes Agent, Qwen Code, the DeepSeek Harness and other OpenAI-compatible clients](docs/harness-integration.md). |
| **Offline** | After you install the app and place a model on disk, nothing needs the Internet. Update checks are optional. |

## Requirements

* A Mac with Apple Silicon (M-series). Developed and measured on a MacBook Air M4
  with 16 GB of unified memory. See [Supported hardware](docs/supported-hardware.md).
* macOS 26.2 or later. This is the minimum OS recorded in the Mach-O headers of the MLX 0.32.2 binaries that the
  engine uses (checked 2026-11-10); earlier macOS versions cannot run the engine. Not yet tested on any release
  other than the development machine's.
* Free disk space for the model checkpoint (about 113 GB for the primary
  checkpoint) plus the persistent cache (default quota 50 GB). The model is **not**
  bundled and is never downloaded without your action.
* AC power is recommended for long sessions. The engine also runs on battery, but it will not start
  when the battery is below 25%, and it stops if the charge falls below that during a run.

## Downloads

There is no public build yet. When the first release is published it will appear on the
[Releases page](https://github.com/joyboyreturns/Veystrix/releases) as:

* `Veystrix-<version>-arm64.dmg` and `Veystrix-<version>-arm64.zip`
* `SHA256SUMS.txt`
* release notes and the notices required by the license

Only builds from that page are official, and each is identified by its SHA-256 in
`SHA256SUMS.txt`. **Veystrix is a free app and is not (yet) notarized by Apple:** macOS will
warn the first time you open it and you approve it once in System Settings. See
[Installation](docs/installation.md) and [Verifying a download](docs/verifying-downloads.md).
Anything else, including copies on other sites, repackaged installers and "cracked" or
"activated" builds, is not Veystrix. Veystrix needs no activation, so any site that
offers one is not official.

## Documentation

* [Installation and first run](docs/installation.md)
* [Supported hardware](docs/supported-hardware.md)
* [Harness integration (Hermes Agent, Qwen Code, DeepSeek Harness, OpenAI-compatible clients)](docs/harness-integration.md)
* [Verified performance and what is not verified](docs/performance.md)
* [Privacy and security](docs/privacy.md)
* [Updates](docs/updates.md)
* [Verifying a download](docs/verifying-downloads.md)
* [FAQ](docs/faq.md)
* [Roadmap](ROADMAP.md) and [Changelog](CHANGELOG.md)

## What is in this repository

This repository is the public home of Veystrix: documentation, roadmap, release notes,
the license, the update feed and the official downloads. It does **not** contain the
Veystrix engine source code or its development history, which are private.

## Support and security

* Questions and bugs: [GitHub Issues](https://github.com/joyboyreturns/Veystrix/issues).
  Do not post prompts, private code, session files or API keys.
* Vulnerabilities: see [SECURITY.md](SECURITY.md).

## Lineage

Veystrix grew out of MIT-licensed work published earlier (including the FlashNext runtime).
That earlier code keeps its MIT license and attribution, which are reproduced in
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
