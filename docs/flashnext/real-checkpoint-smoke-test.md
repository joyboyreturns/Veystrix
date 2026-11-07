# Real checkpoint: preflight, launch command and the first safe smoke test

Status: written 2026-11-07 while the checkpoint was downloading. Nothing here has been run
against the real model. Tags: **[verified]** checked in this work from metadata or code;
**[recorded]** from earlier project results; **[estimate]** arithmetic, not a measurement;
**[not yet]** needs the finished download.

## 1. The checkpoint

| | |
|---|---|
| Repository | `TensorFold/Qwen3.8-Flash-Next-MLX-4bit-MTP`, revision `2b170fa6309d5d1ee380b35636075fac7945f286` |
| Same as | `Vontra/Qwen3.8-Flash-Next-MLX-4bit-MTP` **[verified]**: identical commit, the same 35 files, identical SHA-256 on all 23 LFS files and identical git blobs on the other 12 |
| Directory | `~/models/Qwen3.8-Flash-Next-MLX-4bit-MTP` (the name the alias `vontra-mtp` resolves to **[verified]**) |
| Size | 113.23 GB in 22 safetensors shards, 3,747 tensors **[verified]** |
| Architecture | `qwen4_exp`; hidden 2560, 48 layers (36 linear attention, 12 full attention), 512 routed experts, top-10, expert width 640, one shared expert, vocab 248,320, one MTP layer **[verified]** |
| Quantization | uniform 4-bit, group 32, affine, no per-module overrides **[verified]** |
| Routers | `mlp.gate` is a plain weight (no scales or biases) in all 48 layers and the MTP layer **[verified]**; this is what keeps routing exact |
| One expert | 3,072,000 bytes from the config; the slab layout stride is the same **[verified]** |

No compatibility problem was found in the metadata. Two observations, neither a blocker: the config says
`ple_layer_ids: [2]` while the PLE tensors sit under layer 1 (the runtime's loader handles this checkpoint; the
checker does not assert it), and 333 vision-tower tensors are present but unused (`keep_vision=False`).

**Resident-experts policy.** The runtime applies 8 resident experts only when the checkpoint's content identity
(a hash of the index, the config and every shard's name, size and header) equals `VONTRA_4BIT_MTP_IDENTITY`. A
bit-identical download matches. Any other copy silently runs with the generic default of 32, which changes memory
and speed. Check it once the download has finished (section 3).

## 2. The preflight checker

`python -m models.flashnext.preflight [PATH|ALIAS]` (module `models/flashnext/preflight.py`).

It reads `config.json`, the safetensors index, the tokenizer files and the directory listing, and calls `stat` on
the shards. **By default it never opens, reads or hashes a weight shard**, so it is safe while a download is
writing them. It does not import MLX and is not imported by the runtime.

What it checks: required files and chat template; model type; the exact Q4 settings (4-bit, group 32, affine,
uniform); the architecture profile above; layer types; the expert record size against the slab layout; shard
names; every backbone and expert tensor named in the index (2,941), every scales/biases tensor having its weight,
routers unquantized, the MTP block whole when present (and the loader's both-sidecar-and-indexed refusal);
shards present at the manifest's size, partial and oversize files, `.incomplete` files in
`.cache/huggingface/download` (mapped to their shard by the SHA-256 in the file name), and free disk against what
is left to download. The manifest is `models/flashnext/preflight_manifests/qwen3.8-flash-next-mlx-4bit-mtp.json`
(sizes and hashes from the Hub for the revision above).

Verdicts and exit status: `READY` 0, `INCOMPLETE` 1 (a download problem: missing, partial or still-downloading
files, including a missing tokenizer), `INCOMPATIBLE` 2 (wrong metadata, other quantization, a quantized router,
missing tensors), `MISSING` 2. `--json` prints the report as JSON; `--no-profile` checks structure only.

Two opt-in checks read **only shard headers** (a few hundred KB per shard) and refuse to run while the download is
incomplete:

* `--headers`: header size against file size, index against headers, uint32-packed 4-bit layout with group 32,
  router dtype, expert tensor shape.
* `--identity`: the content identity and whether the resident-experts policy will apply.

Run against the real metadata (22 shards still missing) every metadata check passed and the verdict was
`INCOMPLETE`, with all 22 shards named as missing **[verified]**.

## 3. After the download finishes

```
cd <repo root>
.venv/bin/python -m models.flashnext.preflight vontra-mtp --headers --identity
```

Expected: `VERDICT: READY`, `content identity ... the resident-experts policy (8) applies` **[not yet]**. If the
identity line says `WARN ... will NOT apply`, stop: the files are not the validated checkpoint. For a byte-level
check, compare each shard's SHA-256 with the `sha256` values in the manifest (this reads the whole 113 GB once, so
run it with nothing else active).

## 4. Launch command (interactive chat, not a benchmark)

```
./chat.sh --model flashnext --checkpoint vontra-mtp
```

(`README.md`; `vontra-mtp` resolves to `~/models/Qwen3.8-Flash-Next-MLX-4bit-MTP`. A full path to the directory
also works.) The chat defaults come from `models/flashnext/settings/launch.py` (`CHAT_ENV`), including the 6 GB
expert pool. The Stage 1 ring, scratch pool and cache cap are off unless their `FLASHNEXT_PREFILL_*` variables are
set; the production path is unchanged. Do not run it as a smoke test: it starts inference.

## 5. First safe real-model smoke test: F0 versus F2 at 1K, one round

Purpose: show the harness, the ring and the scratch pool run end to end on the real model and that the outputs
match. **It is not a speed measurement.** Each arm has n = 1, and the first arm reads the checkpoint from SSD
while the second finds part of it in the page cache, so the order alone biases the timing (a page-cache effect
inflated a smoke ratio to 1.55x in section 12).

### 5.1 Before you start

* Download complete; `preflight --headers --identity` is `READY` and the identity matches.
* **Plug the Mac in and turn Low Power Mode off** for numbers you will keep. On battery the run is allowed but
  every record is marked preliminary. Low Power Mode on always stops the run.
* Quit browsers and other large apps. The `--check-ready` step warns below 3 GB free or inactive.
* No other benchmark or download running.

### 5.2 Commands (in order; none of the first three loads the model)

```
.venv/bin/python -m models.flashnext.tests.bench.bench_prefill_memory --check-ready \
    --checkpoint vontra-mtp --sizes 1k --conditions F0 F2 --rounds 1
.venv/bin/python -m models.flashnext.tests.bench.bench_prefill_memory --dry-run \
    --sizes 1k --conditions F0 F2 --rounds 1
```

`--check-ready` verifies, without starting anything: the checkpoint preflight is `READY`; the child would resolve
the same directory; the production API the harness wraps still exists (a rename would otherwise fail after the
model had loaded); the real tokenizer gives prompts near the target size (measured: turn 1 prefills 1,025 / 2,049
/ 4,097 tokens and turn 2 1,024 / 2,048 / 4,096 for targets 1K / 2K / 4K **[verified]** with the checkpoint's
tokenizer; the harness constants were corrected from 64 and 16 to 36 and 23 to get there); the machine state
passes the watchdog rules; free memory; shell variables that would change the baseline. It exits 0 when `READY`.

Then, only with approval:

```
.venv/bin/python -m models.flashnext.tests.bench.bench_prefill_memory \
    --checkpoint vontra-mtp --sizes 1k --conditions F0 F2 --rounds 1 --final
```

Drop `--final` on battery. Two fresh child processes run: F0 (no pool exists during prefill, today's behaviour),
then F2 (`FLASHNEXT_PREFILL_RING_EXPERTS=512`, `FLASHNEXT_PREFILL_RING_SCRATCH=1`). The orchestrator samples power,
thermal, swap and memory pressure every 5 s and before each arm, kills the child on any stop rule, and writes the
lifecycle to `results/flashnext/<run>/records.jsonl` with one `arm-*.json` per child.

**[estimate]** About 5 to 10 minutes: load, a 1,025-token prefill (49.3 s was recorded for 1,007 tokens), 128
decode tokens, and a 30 s pause. MLX peak for F0 is about 6.1 GB **[recorded]**; F2 about 6.7 GB **[estimate]**.

### 5.3 What to read afterwards

| Check | Passing looks like |
|---|---|
| Exactness (G1) | `validation exactness-1k-first` is `pass`: same first token, KV-cache hash and 128-token digest in F0 and F2 |
| Ring health (G2) | F2 `ring_acquired` true, no `fallback_*` counts, no `last_error` |
| Memory (G3) | `memory_gate` pass, headroom at least 0.5 GB under the 12.713 GB Metal limit |
| Scratch cleanup (G4) | no pool at the end of F2's prefill, nothing retained, active memory near F0's |
| No stops | no `safety_stop` failure; no swap-in or swap-out |

**Cold-cache imbalance.** F0 runs first and reads the checkpoint cold from SSD; F2 then finds part of it in the
page cache. F2 can look faster purely from cache warming. The run record says so: `speed_evidence` is `false` for
fewer than 3 rounds, and `--check-ready` and the run itself print the warning. Treat any F0-versus-F2 timing from
this test as uninterpretable.

A `pass` here means the outputs match on one prompt and the mechanisms ran. It supports going on to 1K with 3
rounds in reversed order; it supports no speed claim.

### 5.4 If it stops

Exit status 3 with the partial record kept. The stop reasons are the watchdog's: battery below 25% on battery, Low
Power Mode, memory-pressure level 2 or higher, swap activity (swap-in or swap-out, or 64 MB of growth), thermal
throttling in two samples, exit 137, or a time limit. **Pressure level 2 stopped the earlier battery benchmark;
if the real model reaches it without swapping, report it and decide together before relaxing the rule.** Do not
re-run in a loop on fanless hardware.

## 6. Not yet verified

The finished download (shard hashes, headers, identity); anything on the real model: prefill and decode speed,
memory, the ring and scratch pool, exact outputs; whether pressure level 2 is reached in a normal run. The checker
and the readiness mode are tested with synthetic metadata and fakes only
(`models/flashnext/tests/unit/test_preflight.py`, `test_bench_prefill_memory.py`).
