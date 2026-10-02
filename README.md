<p align="center"><img src="assets/toolslack-logo.png" width="520" alt="ToolSlack"></p>

# ToolSlack: Exploiting Tool Execution Windows for Efficient LLM Agent Serving

ToolSlack schedules native agent memory and exact-prefix KV preparation during
real tool execution. At tool return, the agent immediately uses a verified ready
candidate or falls back to its current context. This repository packages the
dense Qwen3-8B / LangGraph / LangMem implementation and its benchmark.

**Validation status:** 227 CPU tests passed on both Linux x86_64 and macOS arm64,
with no failures or skips. Real public-data preparation has also been executed.
Fresh installation of the bundled CUDA engine, GPU smoke, native DMA correctness
and paper performance reproduction remain unverified. The commands below
implement that workflow; their availability does not certify successful GPU
reproduction.

## One-command correctness check

```bash
./run.sh cpu
```

The entry point installs a pinned bootstrap tool, finds or downloads Python
3.12.14, creates an isolated `.venv`, installs the hash-locked CPU dependencies,
verifies the bundled engine source, and runs all CPU suites. Python 3.9 or newer
is needed to start the bootstrap; pip is not required. The installer verifies a
pinned official uv wheel before extracting its binary. Linux and macOS can run this check.
No model weights, API key, CUDA installation or GPU allocation is required.

Saved validation receipts are in [`validation/CPU_LINUX.json`](validation/CPU_LINUX.json),
[`validation/CPU_MACOS.json`](validation/CPU_MACOS.json) and
[`validation/VALIDATION_SUMMARY.json`](validation/VALIDATION_SUMMARY.json).
The Linux run started from a system Python without pip. Slow network downloads
required pre-filling the official uv wheel and managed-Python caches after
checksum verification; CPU packages were then installed by the original command.
This validates installation and execution with those two caches populated, not a
complete cold-network download. The macOS full-suite receipt predates a two-line
shell path-quoting fix; its additional receipt checks the fixed bootstrap under
a path containing spaces. Linux tests used the fixed final script.

An additional [agent API check](validation/AGENT_API_CPU.json) executed a compiled
LangGraph node and the installed official LangMem summarization algorithm using
the benchmark's pinned Qwen tokenizer. Its model response and exact-token
endpoint were local stubs, so it supplies API compatibility evidence rather
than model-quality or serving-performance evidence.

Each run writes a fresh directory under `results/`, with `status.json`,
`report.json`, test receipts, redacted logs, environment versions and
`SHA256SUMS`. CPU results explicitly carry `performance_evidence: false`.

## One-command GPU smoke and benchmark

On a Linux x86_64 machine with one independently reserved, empty GPU:

```bash
./run.sh smoke --gpu-indices 0 --exclusive-gpus
./run.sh full --gpu-indices 0 --exclusive-gpus
```

Replace `0` with the physical `nvidia-smi` index of your allocation. The runner
verifies the GPU is empty, binds startup to its UUID, and checks it again before
launch. It does not stop existing jobs. The default historical launch profile
uses a 128 GiB host HiCache and requires at least 150 GiB available host RAM.
The provisional device-memory preflight requires 45,000 MiB; this threshold has
not been validated across GPU models. Use a CUDA driver compatible with the
locked PyTorch 2.9.1 CUDA 12.8 runtime, and allow space for approximately 16.4 GB
of model shards plus the CUDA environment and data cache.
The default serving profile selects bfloat16 and targets GPUs with compute
capability 8.0 or newer.

The command automatically:

1. Installs the agent environment and a separate Python 3.11.13 native-engine
   environment, using the supplied version and wheel-hash locks.
2. Downloads pinned HotpotQA data, the official scorer and the Qwen tokenizer.
   Agent inputs contain no gold answers or supporting-fact labels.
3. Downloads the complete pinned Qwen3-8B checkpoint, or verifies a checkpoint
   provided with `--model-path /absolute/checkpoint`. All five model shards are
   checked against the public revision's LFS SHA256 values.
4. Starts only this run's engine and calibration proxy, collects cold prefix
   costs and the native KV lifecycle gate, then restarts its proxy with the
   newly measured cost profile. Previous cost estimates are not imported.
5. Calibrates the unchanged native memory operation on disjoint tasks, executes
   the requested benchmark and writes raw observations plus JSON/CSV reports.
6. Drains native work and stops only the process groups started by this run.

`smoke` evaluates OFF and FULL on four tasks. `full` runs the smoke gate and five
arms over two repetitions, in forward/reverse order:

| Arm | Behavior |
| --- | --- |
| `off` | Native memory runs serially after tool results. |
| `full` | ToolSlack selection, scheduling and exact-prefix KV. |
| `no_selector` | Largest legal scope, with feasibility and safety checks. |
| `fifo` | Arrival-order scheduling, with the same feasibility checks. |
| `no_kv` | Text maintenance only. |

Defaults are three calibration tasks, sixteen evaluation tasks, concurrency four,
8192 maximum memory tokens, a 4096-token summary trigger and 384 summary tokens.
The defaults are identical across arms. Example changes are explicit:

```bash
./run.sh full --gpu-indices 0 --exclusive-gpus --concurrency 8 --eval-tasks 64 --repeats 3
```

The default `--tool-mode model` executes real evidence-analysis calls on the
same serving GPU. It measures shared GPU contention and does not establish
external-tool GPU idle time. `--tool-mode cpu` performs actual paragraph ranking
and sentence extraction. Neither mode inserts sleeps, padding or a future
tool-duration oracle; short windows and L0 decisions are valid outcomes.

## Configuration and offline use

All normal environment variables are generated by the runner. Overrides can be
supplied as CLI arguments or corresponding `TOOLSLACK_*` variables. Run
`./run.sh --help` for the complete list.

| CLI argument | Environment variable | Default |
| --- | --- | --- |
| `--data-root` | `TOOLSLACK_DATASET_ROOT` | Automatically prepared pinned data. |
| `--subsets` | `TOOLSLACK_SUBSETS` | Deterministic disjoint public cohort. |
| `--tokenizer` | `TOOLSLACK_TOKENIZER` | Pinned Qwen3-8B tokenizer. |
| `--gpu-indices` | `TOOLSLACK_GPU_INDICES` | Must identify the reserved GPU. |
| `--engine-url` | `TOOLSLACK_ENGINE_URL` | Dedicated local tracked engine. |
| `--proxy-url` | `TOOLSLACK_PROXY_URL` | Dedicated local tracked proxy. |
| `--service-profile` | `TOOLSLACK_SERVICE_PROFILE` | Generated immutable binding. |
| `--cost-profile` | `TOOLSLACK_COST_PROFILE` | Fresh native measurements. |
| `--concurrency` | `TOOLSLACK_CONCURRENCY` | 4. |
| `--eval-tasks` | `TOOLSLACK_EVAL_TASKS` | 16. |
| `--repeats` | `TOOLSLACK_REPEATS` | 2. |

`--offline` requires resources and checkpoints already present in the cache.
It does not replace real inputs with synthetic data. Python/dependency downloads
must also have been cached or use an already installed environment with
`--no-bootstrap`. `prepare` is a mode, not a performance result:

```bash
./run.sh prepare --gpu-indices 0 --exclusive-gpus
```

It performs resource and dedicated-service preflight and writes the benchmark
configuration without running agent evaluation. Native serving-cost calibration
may still execute when the tracked service is constructed.

Advanced deployments can provide `--service-plan plan.json` with schema
`toolslack.artifact.service-plan.v1`, or a separately calibrated dedicated engine
and proxy with `--owned-service --exclusive-gpus`. Because comparison arms flush
native caches, shared production endpoints are unsuitable. Old project protocol
names have been consistently changed to the ToolSlack namespace in the bundled
controller, proxy, engine and fixtures; they must be used together.

## Reproduction scope and interpretation

This release implements one real public benchmark, not the entire paper's model
and framework matrix. Its default cohort is newly generated from sorted public
IDs with seed 20260930; it is **not the original paper sample**. Providing the
original frozen subset is required for exact sample reproduction. Public
resource revisions, licenses and hashes are recorded in
`resources/resource-lock.json`; the real preparation receipt is
`resources/VALIDATION.json`.

An optional archived V4 exploratory cohort is supplied with six calibration
and sixteen evaluation tasks. Its public data binding and exact reconstruction
are recorded in `resources/COHORT_V4_NOTES.json`. It has not been established as
the paper's final cohort. To repeat that exploratory sample, add
`--subsets resources/cohort-v4-experimental.json --cal-tasks 6 --eval-tasks 16`
to the GPU command.

QPS counts complete agent tasks over batch wall time including native drainage.
Reports preserve failed tasks, quality, latency, and prepared/consumed L1/L2.
Failed calibration, incomplete cells, failed tasks, invalid QPS or unconfirmed
drainage invalidate a performance report. Positive speedup is not enforced. A
run with zero new L1/L2 consumption cannot establish maintenance-overlap gains.

The native backend is restricted to dense MHA HiRadix, TP=DP=PP=1, direct
layer-first transfers and write-back HiCache. It does not implement SSD, PD
disaggregation or distributed KV sharing. OpenCode, Letta, hybrid models and
other model families are outside this benchmark's validated composition.

The exact engine source is bundled because a trustworthy public upstream base
commit for this fork was unavailable. `vendor/native_engine/SOURCE_MANIFEST.json`
verifies the distributed snapshot; installing the latest upstream SGLang is not
a substitute. CPU tests exercise mocked model/transfer endpoints and real local
HTTP, and do not validate CUDA payload equality, official summary quality or
GPU throughput.

## Anonymous review and licensing

This source tree omits original Git history, deployment logs, personal paths,
connection configuration and model/data caches. Generated logs redact host and
personal-path information. Inspect the final reviewer download independently;
repository hosting/account identity is separate from source anonymization.

ToolSlack-authored components use Apache License 2.0; see `LICENSE` and `NOTICE`.
The bundled engine preserves upstream notices. HotpotQA data is CC BY-SA 4.0,
while its evaluation code and Qwen3-8B are Apache-2.0. Downloaded resources retain
their respective notices. No third-party copyright was removed for anonymity.
