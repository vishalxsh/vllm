#!/bin/bash
# Run all 4 e2e combos, fresh process each, cooldown-gated (<45C) between runs.
set -u
SCRATCH=${OUTDIR:-/tmp/e2e-results}; mkdir -p $SCRATCH
GPU=3
MODEL=${MODEL:-Qwen/Qwen2.5-7B-Instruct}
PREFIX=${PREFIX:-e2e}
cd /home/vishal/Desktop/vllm

cooldown() {
  while true; do
    T=$(nvidia-smi -i $GPU --query-gpu=temperature.gpu --format=csv,noheader,nounits)
    if [ "$T" -lt 45 ]; then echo "COOLDOWN OK (${T}C)"; break; fi
    echo "cooling... ${T}C"; sleep 10
  done
}

for combo in "stock_cublas throughput" "fused_gate_up_silu throughput" \
             "stock_cublas latency" "fused_gate_up_silu latency"; do
  set -- $combo
  VARIANT=$1; MODE=$2
  OUT=$SCRATCH/${PREFIX}_${VARIANT}_${MODE}.json
  echo "=== RUN $VARIANT $MODE ==="
  cooldown
  BENCH_GPU=$GPU CUDA_VISIBLE_DEVICES=$GPU .venv/bin/python \
    $(dirname $0)/bench_e2e.py --variant $VARIANT --mode $MODE --out $OUT \
    --model "$MODEL" ${MAX_MODEL_LEN:+--max-model-len $MAX_MODEL_LEN} \
    > $SCRATCH/${PREFIX}_${VARIANT}_${MODE}.log 2>&1
  RC=$?
  if [ $RC -ne 0 ]; then
    echo "FAILED $VARIANT $MODE rc=$RC (see ${PREFIX}_${VARIANT}_${MODE}.log)"
  else
    grep "^RESULT" $SCRATCH/${PREFIX}_${VARIANT}_${MODE}.log
  fi
done
echo "ALL RUNS DONE"
