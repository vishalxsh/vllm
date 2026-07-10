#!/usr/bin/env bash
# Lock GPU clocks for stable benchmark numbers (reduces thermal/boost variance).
# Run BEFORE a benchmarking session; needs sudo. Reset with --reset when done.
#
#   sudo bash lock_gpu_clocks.sh            # lock GPU 3 (our benchmark GPU) to 1500 MHz
#   sudo bash lock_gpu_clocks.sh 1          # lock GPU 1
#   sudo bash lock_gpu_clocks.sh 3 1700     # lock GPU 3 to 1700 MHz
#   sudo bash lock_gpu_clocks.sh --reset    # restore default (auto-boost) on GPU 3
set -euo pipefail

GPU="${1:-3}"
if [ "$GPU" = "--reset" ]; then
  GPU="${2:-3}"
  nvidia-smi -i "$GPU" -rgc
  echo "GPU $GPU clocks reset to default (auto-boost)."
  exit 0
fi
CLK="${2:-1500}"      # RTX 3090 max is 2130; 1500 is a stable, cool, repeatable point

nvidia-smi -i "$GPU" -lgc "$CLK,$CLK"
echo "GPU $GPU locked to $CLK MHz. Verify:"
nvidia-smi -i "$GPU" --query-gpu=index,clocks.max.sm,clocks.sm --format=csv,noheader
echo "Reset afterwards with:  sudo bash lock_gpu_clocks.sh --reset $GPU"
