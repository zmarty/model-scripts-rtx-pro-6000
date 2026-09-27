#!/usr/bin/env bash
T0=$(cat /models/eval/results/thermal-0829/bench_start_epoch)
D=/models/eval/results/thermal-0829/gpu_1hz.csv
H=/models/eval/results/thermal-0829/host_10s.csv
echo "--- GPU (since bench start) ---"
awk -F, -v t0=$T0 '
$1+0>=t0 { i=$3; p=$4+0; t=$8+0; f=$9+0; u=$6+0; m=$12+0;
  n[i]++; sp[i]+=p; if(p>mp[i])mp[i]=p; if(t>mt[i])mt[i]=t; if(f>mf[i])mf[i]=f; if(u>mu[i])mu[i]=u; if(m>mm[i])mm[i]=m;
  for(k=14;k<=17;k++) if($k!="Not Active") th[k]++ }
END{ for(i in n) printf "gpu%s  n=%d  Pwr mean=%.0fW max=%.0fW | Temp max=%.0fC | Fan max=%.0f%% | Util max=%.0f%% | Mem max=%.0fMiB\n", i, n[i], sp[i]/n[i], mp[i], mt[i], mf[i], mu[i], mm[i];
     printf "throttle: sw_power_cap=%d hw_slowdown=%d hw_thermal=%d sw_thermal=%d\n", th[14], th[15], th[16], th[17] }' $D
echo "--- host (since bench start) ---"
awk -F, -v t0=$T0 '
NR>1 && $1+0>=t0 { if($2>a)a=$2; if($5>b)b=$5; n=split($6,nv,";"); for(i=1;i<=n;i++) if(nv[i]+0>c)c=nv[i]; m=split($7,dv,";"); for(i=1;i<=m;i++) if(dv[i]+0>d)d=dv[i]; }
END{ printf "Tctl max=%.0fC | VRM max=%.0fC | NVMe max=%.0fC | DIMM max=%.0fC\n", a,b,c,d }' $H
