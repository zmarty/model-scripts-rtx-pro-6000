#!/usr/bin/env bash
# 1 Hz GPU + 10 s host thermals -> CSV. Usage: thermal_monitor.sh <outdir>
OUT="$1"; mkdir -p "$OUT"
GPUCSV="$OUT/gpu_1hz.csv"; HOSTCSV="$OUT/host_10s.csv"
if [ ! -f "$GPUCSV" ]; then
  echo "epoch,gpu,power_W,power_limit_W,util_gpu,util_mem,gpu_tempC,fan_pct,sm_MHz,mem_MHz,mem_used_MiB,throttle_mask,sw_power_cap,hw_slowdown,sw_thermal" > "$GPUCSV"
  echo "epoch,Tctl,Tccd1,Tccd2,vrm_C,nvme0_C,nvme1_C,dimmA0_C,dimmA1_C,dimmB0_C,dimmB1_C" > "$HOSTCSV"
fi
field () { sensors 2>/dev/null | awk -v k="$1" "\$0 ~ k {getline; gsub(/[+°C]/,\"\",\$2); print \$2; exit}"; }
dimms () { for h in /sys/class/hwmon/hw*; do [ "$(cat $h/name 2>/dev/null)" = spd5118 ] || continue; awk "{printf \"%.1f,\", \$1/1000}" $h/temp1_input 2>/dev/null; done; }
( while :; do
    nvidia-smi --query-gpu=timestamp,index,power.draw,power.limit,utilization.gpu,utilization.memory,temperature.gpu,fan.speed,clocks.current.sm,clocks.current.memory,memory.used,clocks_event_reasons.active,clocks_event_reasons.sw_power_capping,clocks_event_reasons.hw_slowdown,clocks_event_reasons.sw_thermal_slowdown --format=csv,noheader,nounits \
    | awk -F", " -v t="$(date +%s)" "{gsub(/ /,\"\",\$1); gsub(/:/,\"-\",\$1); print t\",\"\$0}" >> "$GPUCSV"
    sleep 1
  done ) &
GPUPID=$!
( while :; do
    printf "%s,%s,%s,%s," "$(date +%s)" "$(field k10temp)" "$(field Tccd1)" "$(field Tccd2)" >> "$HOSTCSV"
    sensors 2>/dev/null | awk "/^asusec/{f=1} f&&/VRM/{gsub(/[+°C]/,\"\",\$3); print \$3; exit}" | tr -d "\n" >> "$HOSTCSV"
    printf "," >> "$HOSTCSV"
    paste -sd, - <(for h in /sys/class/hwmon/hw*; do [ "$(cat $h/name)" = nvme ] && awk "{printf \"%.1f\n\", \$1/1000}" $h/temp1_input; done) | tr -d "\n" >> "$HOSTCSV"
    printf "," >> "$HOSTCSV"; dimms | sed "s/,\$"//" >> "$HOSTCSV"
    echo >> "$HOSTCSV"; sleep 10
  done ) &
HOSTPID=$!
echo "logger pids: gpu=$GPUPID host=$HOSTPID -> $OUT"
echo $GPUPID $HOSTPID > "$OUT/logger.pids"
wait
