# Choosing the Weights of Small Open Language Models

Code, per-item evaluation caches and ten preprints on choosing and combining the weights of small open language models (Qwen2.5-0.5B, SmolLM2-360M, Pythia-160M): linear and spherical interpolation, task-vector extrapolation, three-model merges (TIES, DARE), layer-group search, checkpoint averaging, the winner's curse in validation-based selection, validation-set size, and the language-modeling cost of interpolation.

Every paper states a pre-registered question (`experiments/PREREG.md`), selects configurations on a validation split and reports them on disjoint test items, and uses paired bootstrap intervals. **Most results are null or negative and are reported as such.** One paper (7) reports a correction: an apparent gain from checkpoint averaging came from evaluating an fp16-trained model in bfloat16.

Collection DOI: [10.6084/m9.figshare.c.8840665](https://doi.org/10.6084/m9.figshare.c.8840665) · Author: Dheiver Francisco Santos, PhD ([ORCID 0000-0002-8599-9436](https://orcid.org/0000-0002-8599-9436)) · Papers: CC BY 4.0 · Code: MIT

## The ten papers

| # | Paper (DOI) | One-line result |
|---|---|---|
| 01 | [Validation-Selected Linear Interpolation Between Qwen2.5-0.5B Base and Instruct: Does It Beat Both Parents on ARC, HellaSwag and WinoGrande?](https://doi.org/10.6084/m9.figshare.35013979) | A validation-selected interpolation coefficient between Qwen2.5-0.5B base and instruct does not beat the better parent (+0.65 pp, 95% CI [-0.23, +1.52]). |
| 02 | [SLERP versus LERP for Merging Qwen2.5-0.5B Base and Instruct: A Paired Comparison on ARC, HellaSwag, and WinoGrande](https://doi.org/10.6084/m9.figshare.35013982) | SLERP and LERP differ by at most 0.35 pp on the same pair; the difference is at the level of bf16 rounding. |
| 03 | [Extrapolating the Instruct Task Vector of Qwen2.5-0.5B: Scaling λ Beyond 1 Under Validation-Selected Coefficients on ARC, HellaSwag, and WinoGrande](https://doi.org/10.6084/m9.figshare.35013988) | Extrapolating the instruct task vector (lambda > 1) does not improve over lambda = 1; lambda = 2 degrades. |
| 04 | [Merging Qwen2.5-0.5B-Instruct with Qwen2.5-Coder-0.5B-Instruct: Weighted Sums, TIES and DARE Fall Below Both Parents, and the Coder Task Vector Is 13 Times Larger](https://doi.org/10.6084/m9.figshare.35013991) | Weighted sums, TIES and DARE of instruct and Coder fall 10-18 pp below the best original; the Coder task vector is 13x larger than the instruct one. |
| 05 | [Layer-Group Coefficient Search for Qwen2.5-0.5B Merging: A Null Test of Layer-Group Search](https://doi.org/10.6084/m9.figshare.35013997) | A layer-group coefficient search gains +0.83 pp on validation but +0.39 pp on test over the global coefficient (not distinguishable from zero). |
| 06 | [Replicating Validation-Selected Weight Interpolation on SmolLM2-360M Base and Instruct: ARC, HellaSwag and WinoGrande](https://doi.org/10.6084/m9.figshare.35014000) | On SmolLM2-360M the validation-selected coefficient is not distinguishable from the base model; the instruct model is 5.6 pp worse. |
| 07 | [Averaging the Last Checkpoints of Pythia-160M: No Difference Larger Than 0.008 Nats From the Best Single Checkpoint, WikiText-2 NLL in fp32](https://doi.org/10.6084/m9.figshare.35014003) | Averaging the last Pythia-160M checkpoints ties the best single checkpoint in fp32; an apparent bf16 gain was an evaluation-precision artifact. |
| 08 | [Winner's Curse in Validation-Selected Model Merges: A Resampling Study on the Qwen2.5-0.5B Interpolation Grid (ARC, HellaSwag, WinoGrande)](https://doi.org/10.6084/m9.figshare.35014006) | The validation gain of the selected coefficient (+0.64 pp) is about 3.8 times its test gain (+0.17 pp): a winner's-curse measurement. |
| 09 | [How Many Validation Items Does Merge-Coefficient Selection Need? Selection Regret on Qwen2.5-0.5B Interpolations (ARC, HellaSwag, WinoGrande)](https://doi.org/10.6084/m9.figshare.35014009) | Selection regret does not fall with validation size between 30 and 200 items per task on this nearly flat grid. |
| 10 | [The Language-Modeling Cost Along the Base-to-Instruct Weight Interpolation Path of Qwen2.5-0.5B: WikiText-2 NLL versus ARC, HellaSwag and WinoGrande](https://doi.org/10.6084/m9.figshare.35014015) | WikiText-2 NLL rises along the base-to-instruct path (+0.0886 nat at t = 1) while the accuracy gain is not detectable. |

PDFs and LaTeX sources are in `artigos/`. A copy of each paper is also at https://dheiverphd.com/publications/.

## How to cite

Please cite the collection, and the specific paper when you use one result. `CITATION.cff` is read by GitHub ("Cite this repository"); `CITATIONS.bib` has BibTeX for the collection and all ten papers.

```bibtex
@misc{santos2026weightscollection,
  author    = {Santos, Dheiver Francisco},
  title     = {Choosing the Weights of Small Open Language Models: Ten Preprints},
  year      = {2026},
  publisher = {Figshare},
  doi       = {10.6084/m9.figshare.c.8840665}
}
```

## Reproducing the numbers

Each evaluation is stored as a 0/1 vector per item in `cache/` (`<config>__val.npz`, `<config>__test.npz`), so every table, interval and figure can be regenerated **without a GPU**:

```bash
pip install numpy scipy matplotlib datasets transformers torch safetensors
python3 experiments/analyze.py        # statistics from the caches
python3 experiments/figures.py        # all figures
```

To re-run the evaluations themselves, download the models and datasets (`python3 common/baixar.py`, exact revisions in `MANIFEST.json`) and run `./run_everything.sh`. The experiments used Apple Silicon (MPS), PyTorch 2.8.0, transformers 4.57.6, bfloat16 unless stated (fp32 checks are in `experiments/fp32_check.py` and the papers).

## Layout

- `experiments/` pre-registration, evaluation engine, analyses and figures
- `common/` evaluation kit (log-likelihood multiple choice), merge operations, statistics
- `cache/` per-item correctness vectors; `results/` summaries; `figs/` figures
- `artigos/` the ten papers (LaTeX and PDF)
- `refs/` verified reference metadata

## Limits

One hardware setup, models of at most 0.5B parameters, four multiple-choice benchmarks and one fixed split seed; bfloat16 shifts accuracy by up to 0.7 pp, comparable to the differences studied. See each paper's Discussion and Limitations.
