#!/bin/bash
# Regenera tudo (avaliações em cache são reaproveitadas). Requer modelos e dados: python3 common/baixar.py
set -e; cd "$(dirname "$0")"
python3 experiments/run_all.py a1 a6 a3 a2 a4 a5 a7
python3 experiments/fp32_check.py   # sensibilidade a precisão (Pythia e Qwen, fp32)
python3 experiments/slerp_bf16_check.py
python3 experiments/analyze.py
python3 experiments/an_taskvec.py
for s in art01_06_tables an_art02_03 an_art04_05_tables extra_art07_art10 an_art0809_extra an_art08_pure_selection_bias an_art09_random_tiebreak art01_rank_corr an_art07_fp32 an_art10_fp32 ci995_art10 an_art08_selection_vs_random; do python3 experiments/$s.py; done
python3 experiments/figures.py
