# Attention-Graph Curvature for the Verification of Mathematical Reasoning

Code for our paper ([arXiv:2610.16709](https://arxiv.org/abs/2610.16709)).

We add directed Forman curvature of the attention graph to four step-level verifiers and compare it with two simpler attention features. The data are 1,994 ProcessBench solutions from seven open models on GSM8K, MATH, OlympiadBench and Omni-MATH.

## Results

| Verifier | Base | + Neg-log | + Stats | + Curvature |
|---|---:|---:|---:|---:|
| Spectral, logistic | 0.3261 | 0.3165 | 0.3298 | 0.3474 |
| Hidden states, linear | 0.1828 | 0.1872 | 0.1847 | 0.1907 |
| MLP probe | 0.2991 | 0.3033 | 0.3021 | 0.3037 |
| ReProbe | 0.2932 | 0.3068 | 0.3103 | 0.3472 |

A (the ProcessBench F1), averaged over the 22 generator-subset pairs weighted by number of solutions, with stratified 5-fold cross-validation grouped by problem and 10 seeds. Folds are in `experiments/fold_plan.json`, the method names of the rows in `GROUPS` in `graphcurv/report.py`.

## Tables from the saved results

```sh
pip install -e .
python -m graphcurv.report  # Tables 2-9 and Figure 2, written to outputs/paper/
python -m graphcurv.check_scores  # checks results/paper.json against the saved fold scores, ~10 min
```

`results/` has the fold scores but not the features, so refitting needs the GPU part below.

## Running everything

We used Python 3.12, PyTorch 2.5.1 with CUDA 12.1 and RTX 3090 GPUs. The 7B and 8B models need a 24 GB GPU with bfloat16, the weights of the seven models take 84 GB in the Hugging Face cache and the features about 10 GB in `data/`.

```sh
pip install -r requirements-cuda.txt
pip install -e . --no-deps
hf auth login  # Llama-3 and Llama-3.1 are gated, accept the license on both model pages first
python -m graphcurv.prepare --models  # ProcessBench and the seven models
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
python -m graphcurv.run  # or --cells gsm8k_Qwen2-1.5B-Instruct for one pair
python -m graphcurv.collect --runs outputs/runs --out outputs/reproduced.json
python -m graphcurv.report --results outputs/reproduced.json --out outputs/reproduced-tables
```

All 22 pairs take about 21 GPU hours, gsm8k_Qwen2-1.5B-Instruct alone about 15 minutes. `collect` takes the pairs that have finished and prints how far each one is from the paper. On Slurm, submit `experiments/run.sbatch` from the repo root after `prepare --models`, it runs one pair per array task (`python -m graphcurv.run --list` shows which).

A rerun on our GPUs gives `results/paper.json` exactly. Other GPUs give slightly different features, on a P100 the A values of gsm8k_Qwen2-1.5B-Instruct moved by up to 0.06. Blackwell needs a newer torch than 2.5.1.

## Curvature for another model

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from graphcurv import data
from graphcurv.extract import forward, features_from_forward

name, device = 'Qwen/Qwen2.5-1.5B-Instruct', 'cuda'
tokenizer = AutoTokenizer.from_pretrained(name)
model = AutoModelForCausalLM.from_pretrained(name, dtype=torch.bfloat16, attn_implementation='eager').to(device).eval()
text, char_spans = data.assemble(problem, steps)  # or any text and the character spans of its steps
enc = tokenizer(text, return_offsets_mapping=True, return_tensors='pt')
spans = data.token_spans(enc.pop('offset_mapping')[0].tolist(), char_spans)
f = features_from_forward(*forward(model, enc, device), enc, spans, device)
```

`f['q5_step']` is [steps, layers, 5], the min, quartiles and max of the curvature of the step's outgoing edges in the graph of the prefix up to the end of the step, and `f['q5_tok']` the same per token ([tokens, layers, 5], 0 for the problem). How each verifier uses them is in `graphcurv/conditions.py`. It takes one sequence at a time with eager attention, with the attention modules as `self_attn` under `model.model` (Llama, Qwen).

## Other data

Write `data/dataset.json` as `{"<subset>_<generator>": [{"sid", "problem", "steps", "label", "generator"}, ...]}`, with `label` the index of the first wrong step or -1, and add the generator to `experiments/models.json`. `python -m graphcurv.prepare --plan` then writes `experiments/manifest.json` and `experiments/fold_plan.json`. For ProcessBench this gives our files back, except the row order (and so the folds) of the two math Llama pairs. After that `run`, `collect` and `report` work as above (change `--array` in `run.sbatch`), except that Table 3 is only made for the seven generators we timed.

## Timing

`report` uses the timings from the paper (`results/grid_cost.json`). They were measured with a loop that kept the previous output in memory, so the memory use is too high and some steps that fit in memory were counted as out of memory (98% of the steps are within memory in Table 3, 99.7% with the fix). `results/timing-corrected/` has everything measured again with the loop fixed (with `CUBLAS_WORKSPACE_CONFIG=:4096:8` set, about 24 MiB more memory). Table 3 from them:

```sh
python -m graphcurv.report --timings results/timing-corrected/grid_cost.json --out outputs/corrected-tables
```

The timings in the appendix are in `results/forward_cost.json`, `curv_cost.json` and `single_step_cost.json`. `curv_cost.json` has an older format, the paper uses its `*_ms_per_step` and `*_mib` fields.

To time it on another GPU:

```sh
python -m graphcurv.benchmark --kind grid --out outputs/timing/grid_cost.json
```

`--kind forward`, `curvature` and `scoring` give the other three files.

## Tests

```sh
pip install -e '.[experiments]'
python -m unittest discover -s tests
```

## Citation

```bibtex
@inproceedings{
kim2026attentiongraph,
title={Attention-Graph Curvature for the Verification of Mathematical Reasoning},
author={Kanghyeon Kim and Johannes F. Lutzeyer and Yanzhu Guo},
booktitle={The Sixth Workshop on Mathematical Reasoning and AI},
year={2026},
url={https://openreview.net/forum?id=LY0W4Qcohd}
}
```

## Acknowledgements

`graphcurv/spectral.py` is ported from [spectral-trust](https://github.com/vcnoel/spectral-trust) 0.2.2 (AGPL-3.0), and `ReProbeClaim` in `graphcurv/probes.py` is adapted from `UncertaintyHeadClaim` in [llm-uncertainty-head](https://github.com/IINemo/llm-uncertainty-head) (MIT), which [ReProbe](https://github.com/ReProbe/ReProbe) uses. Their licenses are in `licenses/`. ProcessBench (Apache-2.0) and the model weights have their own licenses.

## License

AGPL-3.0, see [LICENSE](LICENSE).
