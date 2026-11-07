#!/bin/zsh
# Optional, needs sudo: hardware ANE/GPU/CPU power while one case runs. Not run by Claude.
# Usage:  sudo experiments/ane/verify_with_powermetrics.sh <python> <case>
#   <python>  the isolated venv python used by the experiments
#   <case>    ane | cpu | gpu    (router b256 on CPU_AND_NE, CPU_ONLY, or MLX GPU matmul)
# It samples every 500 ms for 10 s, starts the case 3 s in, and prints mean ANE / GPU / CPU mW
# for the idle window and the active window. ANE mW well above idle only in the 'ane' case is
# hardware confirmation that the model ran on the Neural Engine.
set -e
PY="$1"; CASE="$2"; HERE="${0:A:h}"; OUT="${HERE}/out/powermetrics-${CASE}.txt"
mkdir -p "${HERE}/out"
powermetrics --samplers cpu_power,gpu_power,ane_power -i 500 -n 20 > "$OUT" &
PM=$!
sleep 3
case "$CASE" in
  ane) "$PY" "${HERE}/e2_child.py" "${HERE}/out/router_b256.mlpackage" ane 5 256 2560 >/dev/null ;;
  cpu) "$PY" "${HERE}/e2_child.py" "${HERE}/out/router_b256.mlpackage" cpu 5 256 2560 >/dev/null ;;
  gpu) "$PY" "${HERE}/e2_child.py" mlx - 5 512 2048 >/dev/null ;;
esac
wait $PM
echo "samples: $OUT"
grep -E "ANE Power|GPU Power|CPU Power" "$OUT" | awk '{print $1, $2, $3, $4}' | paste - - - | head -24
