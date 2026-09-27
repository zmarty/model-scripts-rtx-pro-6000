#!/usr/bin/env python3
"""GEMM power virus: saturates both GPUs with big BF16 matmuls.
Run inside the vLLM image (has Blackwell-capable torch):
  docker run --rm --gpus all -v /models:/models \
    vllm/vllm-openai:qwen38-flash-next python3 /models/eval/gemm_burn.py --minutes 10
"""
import argparse
import threading
import time

import torch

try:
    import pynvml
    pynvml.nvmlInit()
    NVML = True
except Exception:
    NVML = False


def burn(dev: int, stop: threading.Event, size: int):
    torch.cuda.set_device(dev)
    a = torch.randn(size, size, device=f"cuda:{dev}", dtype=torch.bfloat16)
    b = torch.randn(size, size, device=f"cuda:{dev}", dtype=torch.bfloat16)
    c = torch.empty(size, size, device=f"cuda:{dev}", dtype=torch.bfloat16)
    # warmup
    for _ in range(5):
        torch.mm(a, b, out=c)
    torch.cuda.synchronize()
    while not stop.is_set():
        for _ in range(50):
            torch.mm(a, b, out=c)
        torch.cuda.synchronize()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--minutes", type=float, default=10)
    ap.add_argument("--size", type=int, default=8192)
    args = ap.parse_args()

    ndev = torch.cuda.device_count()
    print(f"Burning {ndev} GPU(s), {args.size}x{args.size} BF16 GEMMs for {args.minutes} min")
    stop = threading.Event()
    threads = [
        threading.Thread(target=burn, args=(d, stop, args.size), daemon=True)
        for d in range(ndev)
    ]
    for t in threads:
        t.start()

    t0 = time.time()
    try:
        while time.time() - t0 < args.minutes * 60:
            time.sleep(10)
            if NVML:
                stats = []
                for i in range(ndev):
                    h = pynvml.nvmlDeviceGetHandleByIndex(i)
                    t_ = pynvml.nvmlDeviceGetTemperature(h, pynvml.NVML_TEMPERATURE_GPU)
                    p = pynvml.nvmlDeviceGetPowerUsage(h) / 1000
                    f = pynvml.nvmlDeviceGetFanSpeed(h)
                    stats.append(f"GPU{i}: {t_}C {p:.0f}W fan{f}%")
                el = time.time() - t0
                print(f"[{el:6.0f}s] " + " | ".join(stats), flush=True)
    except KeyboardInterrupt:
        pass
    finally:
        stop.set()
        for t in threads:
            t.join(timeout=5)
    print("Burn complete.")


if __name__ == "__main__":
    main()
