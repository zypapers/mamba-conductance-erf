# Effective Receptive Fields in Mamba: Conductance Products and Selectivity — code and results

Code and result files accompanying the NeurIPS 2026 paper *Effective Receptive Fields in Mamba: Conductance Products and Selectivity* (Zy Li, Seattle University). This repository lets a reader regenerate every number and figure in the paper from the released results, re-run the training and the ERF analyses from scratch, and trace each reported number to the script and result file that produced it.

The paper writes a Mamba block's sensitivity to past inputs through per-step conductances G_t = I − exp(A Δ_t). It gives an exact decomposition of that sensitivity, including differentiable selectivity. It then tests a closed-form block formula with selectivity held fixed against the full model's effective receptive field (ERF).

## What is here

- `src/model/`:
  - `single_layer_mamba.py`, the single-layer Mamba;
  - `scan_ref.py`, a pure-PyTorch reference scan that implements the trained recurrence (Ā_t = exp(A Δ_t) and input weight B̄_t = Δ_t B_t, as in the fused kernel) and exposes Δ_t, B_t and C_t.
- `src/curves/`: the block formula and the exact decomposition.
- `src/data/`: WikiText-103 with the GPT-2 tokenizer.
- `scripts/`: training, the ERF protocol, the analyses, and the scripts that write every number and figure in the paper.
- `results/`: the JSON files behind every number in the paper.
- Large files (checkpoints and per-sequence curves, 2.4 GB) are attached to the repository's v1.0 release as two archives; see `ARTIFACTS.md`.

## Requirements

Tested on Ubuntu (WSL2) with Python 3.12.3, PyTorch 2.11.0 for CUDA 13.0, mamba-ssm 2.3.1 and one NVIDIA RTX PRO 6000 (96 GB).

- **Numbers and figures from the released files:** CPU only, without mamba-ssm. `pip install -r requirements-analysis.txt`.
- **Training, ERF extraction and the other GPU steps:** a CUDA GPU and mamba-ssm 2.3.1.
  - Training uses mamba-ssm's fused selective-scan kernel. The ERF protocol computes the curves with the reference scan, and checks that scan against the kernel, so it imports mamba-ssm too, as do the retrieval decomposition and a few checks.
- **mamba-ssm 2.3.1 has no wheel on PyPI and no prebuilt wheel for PyTorch 2.11.** It builds from source with a CUDA 13.0 compiler, which pip supplies; in a fresh virtual environment:

```bash
pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu130
pip install packaging ninja wheel setuptools nvidia-cuda-nvcc==13.0.88 nvidia-cuda-cccl==13.0.85 \
    nvidia-cuda-crt==13.0.88 nvidia-nvvm==13.0.88
export CUDA_HOME=$(python3 -c "import sysconfig; print(sysconfig.get_paths()['purelib'])")/nvidia/cu13
ln -sfn lib "$CUDA_HOME/lib64"                          # PyTorch's extension build links from lib64
ln -sf libcudart.so.13 "$CUDA_HOME/lib/libcudart.so"    # and needs the unversioned -lcudart
MAMBA_FORCE_BUILD=TRUE pip install -r requirements.txt --no-build-isolation
```

## Reproduce the paper's numbers and figures (CPU, a few minutes)

```bash
pip install -r requirements-analysis.txt
bash scripts/fetch_artifacts.sh   # downloads and checks both archives; see ARTIFACTS.md
python3 scripts/task_sensitivity_v2_pooled.py
python3 scripts/section4_numbers.py
python3 scripts/abstract_numbers.py
python3 scripts/appendix_numbers.py
python3 scripts/c3_numbers.py
python3 scripts/test_overlap_sensitivity.py
python3 scripts/make_camera_ready_figures.py
git status --short results/    # no output: every released number was reproduced
```

## Reproduce from scratch (one GPU, about a day)

Times are measured on one RTX PRO 6000.

1. **Task models and the main language models:** `bash scripts/launch_retraining.sh`.
   - It trains the 18 retrieval-task models (multi-query associative recall and selective copying; seeds 42–44; about 24 minutes in all).
   - It trains the nine main language models (three A-initializations × seeds 1–3; about 9.2 hours, three at a time).
2. **ERF analysis of the main models** (about 2.3 hours): `python3 scripts/camera_ready_single_layer.py --cohort matched`, then `python3 scripts/analyze_camera_ready_matched.py`.
3. **Weight-decay control:** `bash scripts/launch_wd_control.sh nowd_timescale A_log dt_proj.bias`. It trains the nine control models (about 9.2 hours), then computes their ERFs (about 2.2 hours) and runs the statistical analyses.
4. **Statistics of the main models:** `python3 scripts/conductance_stats.py` and `python3 scripts/check_weight_decay_drift.py`.
5. **Retrieval decomposition** (minutes): `python3 scripts/task_sensitivity_v2.py`, then `python3 scripts/task_sensitivity_v2_pooled.py` and `python3 scripts/task_probe_control.py`.
6. **Pretrained Mamba-370M and the two-layer model:**
   - the two-layer model: `python3 scripts/two_layer_experiment.py` trains it (`checkpoints/model_2layer.pt`, also in the checkpoints archive);
   - `python3 scripts/scan_probe_large_sample.py --model mamba370m` (about 21 minutes) and `--model two_layer` (about 4 minutes);
   - `python3 scripts/last_layer_block_test.py`.
7. **Further checks:** `python3 scripts/griffiths_constant_input.py` (about 35 min on CPU) and `python3 scripts/check_selectivity_decomposition.py`. The latter runs on the three legacy single-layer checkpoints (`checkpoints/model_{slow,default,fast}.pt`, in the checkpoints archive).
8. **Numbers and figures:** the seven scripts in the section above.

The single-seed selective-copying run at sequence length 4096 (`results/selcopy/selcopy_intervention_seq4096.json`) was made with `scripts/selective_copy_intervention.py`; its configuration is recorded in that file.

## Where each result comes from

| Paper | Script | Result |
|---|---|---|
| §4.1, App. E: training, perplexity | `train_matched.py` | `results/matched*/training_*.json` |
| §4.2, App. D: identity checks | `camera_ready_single_layer.py`, `check_selectivity_decomposition.py`, `scan_probe_large_sample.py` | `results/camera_ready/matched*/single_layer_checks.json`, `results/selectivity_decomposition_check.json` |
| §4.3, Table 1, Figure 1 | `camera_ready_single_layer.py` → `analyze_camera_ready_matched.py` → `section4_numbers.py`, `abstract_numbers.py`, `make_camera_ready_figures.py` | `results/camera_ready/section4_numbers.json`, `abstract_numbers.json` |
| §4.4, App. H: retrieval decomposition | `task_sensitivity_v2.py` → `task_sensitivity_v2_pooled.py`, `task_probe_control.py` | `results/task_sensitivity_v2.json`, `results/task_sensitivity_v2_pooled.json`, `results/task_probe_control.json` |
| §4.5, App. F–G: task accuracy | `mqar_intervention.py`, `selective_copy_intervention.py` | `results/{mqar,selcopy}/rerun/*.json` |
| §4.5, App. E: weight-decay control | `launch_wd_control.sh` → `compare_wd_control.py` → `c3_numbers.py` | `results/camera_ready/c3_numbers.json` |
| §4.6, App. I–J: conductance regimes | `conductance_stats.py`, `scan_probe_large_sample.py` | `results/camera_ready/conductance_stats_*.json`, `results/pretrained/`, `results/two_layer/` |
| App. E: test-split sensitivity to WikiText-103's test/train overlap | `test_overlap_sensitivity.py` | `results/camera_ready/test_overlap_sensitivity.json` |
| App. C, D: Griffiths and constant input | `griffiths_constant_input.py` | `results/camera_ready/griffiths_constant_input.json` |
| Appendix tables | `appendix_numbers.py` | `results/camera_ready/appendix_numbers.json` |

## Names used in the code

- **`matched`:** the nine main language models, trained with one shared recipe (A-initializations slow, default and fast; seeds 1–3). **`matched_nowd_timescale`:** the nine weight-decay control models (the same runs with A_log and Δ's bias exempt from weight decay).
- **`legacy`:** three earlier single-layer models (`checkpoints/model_{slow,default,fast}.pt`), used by `check_selectivity_decomposition.py` (step 7), `check_scan_fix.py` and the diagnostic `check_discretization.py`.
- **`rerun`:** the 18 retrieval-task models the paper uses (seeds 42–44, paired training data).
- **`camera_ready`:** outputs of the paper's ERF protocol (`camera_ready_single_layer.py`) and of the scripts that read them. **`c3_numbers`:** the weight-decay control's numbers (§4.5, App. E); its key `claim2` holds the block-formula agreement statistics on the control models, computed as for the main models.
- **Curves:** E, the full model's ERF; X, the block formula (Eq. 17); S, the scan probe (Eq. 15); R, S with the readout; P, the block form without cross-channel terms; Su and Xu, S and X with each modal residue C_{T,i} B̄_{s,j,i} replaced by its absolute value, while the readout, convolution and input weights keep their signs (the sign ablation); Dall, the ERF with Δ, B and C detached (Ddt, DB and DC detach one at a time).
- **Legacy and diagnostic scripts**, whose own outputs are not released: `intervention_experiment.py` (supplies the training recipe), `two_layer_experiment.py` (trains the two-layer model), `pretrained_full_sweep.py` and `preview_pretrained_euler.py` (supply helpers to `scan_probe_large_sample.py`), `task_sensitivity_diagnostic.py` (supplies the task examples), `analyze_camera_ready_single_layer.py` and `check_discretization.py`.

## Definitions used throughout

- **ERF (language models):** the mean over sequences and targets T = 1024–2047 of ‖∂ℓ_T/∂e_{T−τ}‖², where e_{T−τ} is the block input (the token embedding, for these single-layer models), ℓ_T = ‖logits_T‖² and τ = 1–1023.
- **Comparisons:**
  - The full model and the block formula are compared by their raw ratio.
  - Comparisons with the scan probe are made after normalizing each curve at τ = 1.
- **Running products** are formed in log space.

## Citation

```bibtex
@inproceedings{li2026effective,
  title     = {Effective Receptive Fields in Mamba: Conductance Products and Selectivity},
  author    = {Li, Zy},
  booktitle = {Advances in Neural Information Processing Systems},
  year      = {2026}
}
```

## License

Code (`src/`, `scripts/` and the requirements files): MIT. Results (`results/`, and the archived checkpoints and per-sequence curves): Creative Commons Attribution 4.0 International (CC BY 4.0). See `LICENSE`, and please cite the paper (above) when using the results.
