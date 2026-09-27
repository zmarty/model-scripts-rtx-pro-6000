#!/usr/bin/env bash
# Sample GPU + system temps/power/fans every ${1:-5}s to a CSV-ish log.
# Case fans are NOT visible to Linux on this board (no Nuvoton super-I/O
# hwmon driver loaded; ASUS Q-Fan runs them in firmware from CPU/mobo temps).
OUT="${2:-/models/eval/temps_$(date +%Y%m%d_%H%M%S).log}"
INT="${1:-5}"
echo "# ts | gpu0 tempC fan% powerW util% memMiB | gpu1 tempC fan% powerW util% memMiB | cpu_tctl ccd1 ccd2 | vrm | dimm_avg | nvme0 nvme1" > "$OUT"
while true; do
  ts=$(date +%H:%M:%S)
  gpu=$(nvidia-smi --query-gpu=temperature.gpu,fan.speed,power.draw,utilization.gpu,memory.used --format=csv,noheader,nounits | tr '\n' ';')
  g0=$(echo "$gpu" | cut -d';' -f1 | tr -d ' ')
  g1=$(echo "$gpu" | cut -d';' -f2 | tr -d ' ')
  cpu=$(sensors -j 2>/dev/null | python3 -c '
import json,sys
d=json.load(sys.stdin)
k=d.get("k10temp-pci-00c3",{})
tctl=k.get("Tctl",{}).get("temp1_input","")
c1=k.get("Tccd1",{}).get("temp3_input","")
c2=k.get("Tccd2",{}).get("temp4_input","")
vrm=d.get("asusec-isa-0000",{}).get("VRM",{}).get("temp4_input","")
dims=[v["temp1"]["temp1_input"] for n,v in d.items() if n.startswith("spd5118") and "temp1" in v]
davg=sum(dims)/len(dims) if dims else 0
nv0=d.get("nvme-pci-0200",{}).get("Composite",{}).get("temp1_input","")
nv1=d.get("nvme-pci-6c00",{}).get("Composite",{}).get("temp1_input","")
print(f"{tctl} {c1} {c2} | {vrm} | {davg:.1f} | {nv0} {nv1}")')
  echo "$ts | $g0 | $g1 | $cpu" | tee -a "$OUT"
  sleep "$INT"
done
