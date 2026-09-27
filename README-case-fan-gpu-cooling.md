# Case Airflow & GPU Cooling — Investigation and Fix

Date: 2026-09-05
Machine: `zmarty-aorus` (headless LLM inference box)

## Machine

| Component | Detail |
|---|---|
| CPU | AMD Ryzen 9 7950X3D (16c/32t) |
| RAM | 192 GB DDR5 |
| Motherboard | ASUS ROG Crosshair X670E Hero (NCT6799D super-I/O) |
| GPUs | 2× NVIDIA RTX PRO 6000 Blackwell 96 GB (600 W power cap each) |
| CPU cooler | Self-managed (AIO-style; no fan tach on motherboard headers) |
| Case fans | 7× 140 mm total (see wiring below) |

### Case fan wiring

| Motherboard header | Fans | Direction |
|---|---|---|
| CHA_FAN1P | 3× Noctua 140 mm (splitter), front | intake |
| CHA_FAN2P | 3× Noctua 140 mm (Nexus 2 hub + SATA power), top | exhaust |
| CHA_FAN3P | 1× Fractal 140 mm, rear-top | exhaust |
| CHA_FAN4 | 1× Fractal 140 mm, bottom | intake |

Linux pwm channel mapping (identified empirically by stopping each channel):
pwm1–pwm3 = the three Noctua groups (max ~1350/1530/1210 RPM),
pwm4/pwm5 = the two Fractals (max ~920/860 RPM), pwm6/pwm7 = unused pump
headers. All five case-fan channels get the same curve, so the exact
front/top assignment does not matter.

## The problem (before)

- BIOS Q-Fan (firmware) controlled all case fans, sourced from
  **CPU/motherboard temperatures only**. The motherboard has no visibility
  into GPU temperatures.
- Linux had **no fan visibility at all**: the `nct6775` driver was not
  loaded, so no `pwm*`/`fan*_input` entries existed in hwmon.
- GPU fans were (and still are) handled by `nvidia-fan-control.service`
  (`/opt/nvidia-fan-control/nvidia-fan-control.py`, `--mode quiet`,
  100 % at 65 °C). A separate `nvidia-power-limit.service` sets `-pl 600`.

### Measured "before" — vLLM serving benchmark (10-min soak)

`vllm bench serve`, 224 prompts × 8k in / 2k out (`--ignore-eos`),
concurrency 16 (`--max-num-seqs 16` cap), Qwen3.8-Flash-Next FP8 TP2:

- 793.7 tok/s output aggregate, TPOT ~19 ms, MTP acceptance 53 %, 0 failures

| Component | Idle | Peak (10-min inference) |
|---|---|---|
| GPU0 | 33 °C | **69 °C, GPU fan 100 %, 439 W** |
| GPU1 | 28 °C | **60 °C, GPU fan 85 %, 428 W** |
| CPU Tctl | ~50 °C | 74.5 °C |
| VRM | 34 °C | 53.9 °C |
| DDR5 avg | 39 °C | 47.5 °C |
| NVMe0 | 46 °C | 53.9 °C |

Key observation: GPU0's own fan saturated at 100 % while the case fans
(following CPU at only ~74 °C) barely ramped — the case fans had no idea
~860 W was being dumped into the box.

Note: LLM inference is memory-bandwidth-bound, so sustained power sits at
~65–70 % of the 600 W cap (~400–440 W). 600 W is only reachable with a
pure-compute workload (see GEMM burn below).

## The fix (after)

### 1. `nct6775` kernel module — fan headers visible to Linux

- Loads fine on this board, no `acpi_enforce_resources=lax` needed.
- Persisted via `/etc/modules-load.d/nct6775.conf`.
- Exposes 7 pwm channels + fan tachs as `nct6799` hwmon device.

### 2. `case-fan-control.service` — GPU-temp-driven case fans

- Script: `/opt/case-fan-control/case-fan-control.py`
- Unit: `/etc/systemd/system/case-fan-control.service` (enabled, `Restart=always`)
- Logic (every 2 s):
  - `duty = max(GPU_CURVE(max(GPU0,GPU1) temp), CPU_CURVE(CPU Tctl))`
  - GPU curve: 40 °C→30 %, 50→45, 55→60, 60→75, 65→85, 70→95, 75→100
  - CPU floor curve: 50 °C→30 %, 60→40, 70→55, 80→80, 85→100
  - Applies to pwm1–pwm5 (all case fans)
- Fail-safe: any exception → all case fans 100 %; clean stop (SIGTERM) →
  restores `pwmN_enable=5` (BIOS Q-Fan / Smart Fan IV)
- Logs: `journalctl -u case-fan-control -f`

The CPU cooler manages itself (verified: full CPU load with all case fans
stopped only cost ~4 °C), so no motherboard header needs CPU-fan treatment.

## Validation — GEMM power virus (worst case, after fix)

`/models/eval/gemm_burn.py` — 8192×8192 BF16 matmul loop on both GPUs,
10 min, run inside the vLLM image (`--entrypoint python3`). This is the
true thermal worst case: **600 W flat on both GPUs for 10 minutes**.

| | Peak temp | Power | GPU fan |
|---|---|---|---|
| GPU0 (top card, bus 01:00.0) | **91 °C** | 600 W sustained | 100 % |
| GPU1 (bus 03:00.0) | **75 °C** | 600 W sustained | 100 % |
| CPU / VRM / DDR5 / NVMe0 | 72 / 50 / 50 / 51 °C | — | — |

Case fans behaved exactly as designed: 30 % at idle → 100 % by 75 °C GPU
(~1450–1550 RPM Noctuas, ~920–1000 RPM Fractals), ramping back down as the
GPUs cooled.

## What we learned

1. **Case fans now track GPU load for the first time.** Before, they only
   responded to CPU heat; GPU-only workloads left them near idle.
2. **The 14–16 °C split between identical GPUs is card-local, not
   case-wide.** GPU0 (top card) is starved of intake air / recirculates its
   own exhaust; GPU1 runs cool under identical load. More case airflow
   cannot fix a delta this size — it needs slot spacing or airflow
   channeling to the top card.
3. **Real inference workloads are safe.** vLLM serving peaked at 69 °C on
   GPU0 (~430 W/card); the 91 °C GEMM number is a synthetic worst case that
   inference never reaches.
4. GPU0's own fan hits 100 % at 65 °C on the `quiet` curve — no headroom.
   Optional: switch `nvidia-fan-control.service` to `--mode aggressive`.
5. Useful extras now available in Linux: all fan RPMs, plus DDR5 (spd5118),
   NVMe, VRM and k10temp sensors.

## Files

| Path | Purpose |
|---|---|
| `/opt/case-fan-control/case-fan-control.py` | case fan daemon |
| `/etc/systemd/system/case-fan-control.service` | systemd unit (enabled) |
| `/etc/modules-load.d/nct6775.conf` | persist nct6775 module |
| `/opt/nvidia-fan-control/nvidia-fan-control.py` | pre-existing GPU fan daemon |
| `/models/eval/monitor_temps.sh` | 5 s temp/power logger (GPU/CPU/VRM/DDR/NVMe) |
| `/models/eval/gemm_burn.py` | GEMM power virus |
| `/models/eval/temps_*.log`, `bench_*.log`, `gemm_burn_*.log` | captured data |

## Operations cheatsheet

```bash
# live case-fan daemon log
journalctl -u case-fan-control -f

# live fan RPMs / pwm (H = nct6799 hwmon, e.g. hwmon12)
grep . /sys/class/hwmon/hwmon*/{name,fan?_input} 2>/dev/null

# hand case fans back to BIOS Q-Fan
sudo systemctl stop case-fan-control

# disable permanently
sudo systemctl disable --now case-fan-control

# temp logging during a run
/models/eval/monitor_temps.sh 5 &

# benchmark the server (inside the container)
docker exec qwen38-flash-next-fp8-vllm vllm bench serve \
  --model qwen3.8-flash-next \
  --tokenizer /models/fp8/Qwen-Qwen3.8-Flash-Next-FP8 \
  --dataset-name random --random-input-len 8192 --random-output-len 2048 \
  --ignore-eos --num-prompts 224 --max-concurrency 16
```
