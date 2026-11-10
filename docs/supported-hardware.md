# Supported hardware

| Item | Status |
|---|---|
| MacBook Air M4, 16 GB unified memory, macOS (Apple Silicon) | **Development and measurement machine.** All figures in [performance.md](performance.md) come from it. |
| Other Apple Silicon Macs with 16 GB or more | Expected to work. **Not measured.** Treat as unverified until a result is published here. |
| Apple Silicon Macs with less than 16 GB | Not supported. The memory guards will refuse to start the engine. |
| Intel Macs | Not supported. The engine uses MLX and Metal, which require Apple Silicon. |
| Windows, Linux, iOS | Not supported. |

## Memory, disk and power

* **Memory:** the engine itself is about 7 GiB resident once loaded on the reference
  machine; the rest of a 16 GB Mac is shared with macOS and your other apps. Close
  memory-heavy apps before starting a long session. Veystrix measures available
  memory and swap and refuses a request it estimates it cannot serve safely.
* **Disk:** the primary checkpoint is about 113 GB. The persistent prefix cache has a
  quota (default 50 GB) that you can change. Keep at least 20 GB free beyond both.
* **SSD wear and I/O:** streaming experts reads a lot of data. Long prefills read
  hundreds of gigabytes in total. Reads are not writes and cause far less wear, but
  the persistent cache writes snapshots.
* **Power:** keep the Mac on AC power with Low Power Mode off for long sessions.
  Performance and stability measured on battery are marked preliminary.

## Model checkpoints

Veystrix does not include a model. The engine supports the Qwen3.8 Flash-Next family in
MLX 4-bit format (architecture `qwen4_exp`). Each checkpoint has its own license from
its publisher; read it before downloading or using it. The Veystrix app validates a
checkpoint folder (manifest and required files) without reading the weights.
