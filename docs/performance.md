# Verified performance

This page lists only figures backed by recorded runs on the reference machine
(MacBook Air M4, 16 GB). It also lists what is **not** verified. Veystrix is
pre-release: none of this is a promise of speed on your Mac.

## Rules for this page

1. Figures measured with expert **pruned routing** (fewer than 10 routed experts per
   layer) are not shown here. They do not represent the model's full top-10 routing.
2. A figure from a run on battery power, or one run without repeats, is labelled
   **preliminary**.
3. Token-ID agreement between two runs is not the same as matching logits. This page
   says which one was checked.

## Full top-10 routing: measured

| Measurement | Result | Status |
|---|---|---|
| Routing confirmed at the model | Every one of 1,152 recorded layer calls (24 tokens × 48 layers) routed exactly 10 distinct experts. | Verified, one run. Does not by itself verify logits. |
| Decode, profiled diagnostic | 2.08 tokens/s median, about 439 ms/token, 2,239-token prompt. About 80% of the time is waiting on the SSD. | Diagnostic from a profiled run on battery, one arm. **Not** a speed benchmark. |
| Optional layer-major prefill, about 8K-token prompt | 1.76× faster than the default prefill by mean time ratio (three pairs, 1.70–1.84×), identical output token IDs in all six runs, about 90 GB instead of about 300 GB of SSD reads. | **Preliminary** (started on battery). One prompt, 8 output tokens, token-ID level. Opt-in, not the default. |
| Layer-major engine, 8K request after earlier turns | Completed; peak swap 1,353 MiB, below the 3,072 MiB guard. | **Preliminary**, single run. Does not cover cancellation, restart or long decodes. |
| Restoring a cached prefix in a new process | 2,107 of 2,732 tokens restored from SSD; remaining 645 tokens took 25.6 s versus 73.3 s cold (about 2.9× faster). | **Preliminary**, one pair, no resolution band. |

## Not verified yet

* Long-lived server safety of the optional layer-major engine (a swap-limit breach seen
  earlier remains open).
* Throughput and exactness at 16K tokens or longer, and the 262K-token context the
  model supports architecturally. A 262K prompt does not fit this Mac's admission
  limits; the largest completed real prefill is recorded in the engine's own validation
  reports.
* Logit-level equivalence to an independent reference at full routing.
* Any result on AC power under the final measurement standard.
* Any other Mac.

These are tracked in the [Roadmap](../ROADMAP.md).
