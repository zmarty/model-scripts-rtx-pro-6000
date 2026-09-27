#!/usr/bin/env python3
"""1 Hz GPU + 10 s host thermal logger for the zmarty-aorus eval runs."""
import subprocess, sys, time, glob, os, threading

OUT = sys.argv[1]
os.makedirs(OUT, exist_ok=True)
GPUCSV = os.path.join(OUT, "gpu_1hz.csv")
HOSTCSV = os.path.join(OUT, "host_10s.csv")

Q = ("timestamp,index,power.draw,power.limit,utilization.gpu,utilization.memory,"
     "temperature.gpu,fan.speed,clocks.current.sm,clocks.current.memory,memory.used,"
     "clocks_event_reasons.active,clocks_event_reasons.sw_power_cap,"
     "clocks_event_reasons.hw_slowdown,clocks_event_reasons.hw_thermal_slowdown,"
     "clocks_event_reasons.sw_thermal_slowdown")

if not os.path.exists(GPUCSV):
    open(GPUCSV, "w").write("epoch,gpu,power_W,power_limit_W,util_gpu,util_mem,gpu_tempC,"
                            "fan_pct,sm_MHz,mem_MHz,mem_used_MiB,throttle_mask,sw_power_cap,"
                            "hw_slowdown,hw_thermal,sw_thermal\n")
if not os.path.exists(HOSTCSV):
    open(HOSTCSV, "w").write("epoch,Tctl,Tccd1,Tccd2,vrm_C,nvme_Cs,dimm_Cs\n")


def hwmon():
    """name -> list of (label, degC)"""
    out = {}
    for h in glob.glob("/sys/class/hwmon/hw*"):
        try:
            name = open(os.path.join(h, "name")).read().strip()
        except OSError:
            continue
        rows = []
        for f in sorted(glob.glob(os.path.join(h, "temp*_input"))):
            try:
                val = int(open(f).read()) / 1000.0
            except (OSError, ValueError):
                continue
            lab = f.replace("_input", "_label")
            label = open(lab).read().strip() if os.path.exists(lab) else os.path.basename(f)
            rows.append((label, val))
        out.setdefault(name, []).extend(rows)
    return out


def pick(rows, *keys):
    for k in keys:
        for label, val in rows:
            if k.lower() in label.lower():
                return round(val, 1)
    return rows[0][1] if rows else ""


def gpu_loop():
    while True:
        t = int(time.time())
        try:
            r = subprocess.run(["nvidia-smi", f"--query-gpu={Q}",
                                "--format=csv,noheader,nounits"],
                               capture_output=True, text=True, timeout=20)
            for line in r.stdout.strip().splitlines():
                f = [x.strip() for x in line.split(",")]
                with open(GPUCSV, "a") as fh:
                    fh.write(",".join([str(t)] + f) + "\n")
        except Exception as e:
            with open(GPUCSV, "a") as fh:
                fh.write(f"{t},ERROR,{e}\n")
        time.sleep(1)


def host_loop():
    while True:
        t = int(time.time())
        hw = hwmon()
        k = hw.get("k10temp", [])
        vrm = pick(hw.get("asusec", []), "vrm")
        nvme = ";".join(str(round(v, 1)) for _, v in hw.get("nvme", []))
        dimm = ";".join(str(round(v, 1)) for _, v in hw.get("spd5118", []))
        with open(HOSTCSV, "a") as fh:
            fh.write(f"{t},{pick(k,'tctl')},{pick(k,'tccd1')},{pick(k,'tccd2')},"
                     f"{vrm},{nvme},{dimm}\n")
        time.sleep(10)


open(os.path.join(OUT, "logger.pids"), "w").write(str(os.getpid()))
print(f"logging -> {OUT}", flush=True)
threading.Thread(target=gpu_loop, daemon=True).start()
threading.Thread(target=host_loop, daemon=True).start()
while True:
    time.sleep(60)
