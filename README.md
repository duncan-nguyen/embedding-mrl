# Matryoshka Representation Learning: ACL Conference Experiments

Experimental code for an ACL submission comparing four ways to train Matryoshka
embedding models — representations that stay useful when truncated to
`[16, 32, 64, 128, 256, 512, 1024]` (or `768`), so inference cost can be traded
against quality at serving time.

Four methods × four backbones = sixteen experiments, all driven by one CLI and
one YAML file each.

## Layout

```
configs/            one YAML per experiment (method × backbone) + shared base.yaml
data/
  train/            final_data.csv — unsupervised SimCSE corpus
  test/             all evaluation CSVs (STS, classification, pair)
src/embedding_mrl/
  config.py         dataclass config, YAML inheritance, CLI overrides
  data.py           datasets, collators, task→CSV registry
  pooling.py        cls / mean / last-token pooling
  losses/           infonce.py, cka.py, ese.py, mipic.py, sdr.py
  geometry.py       semantic distortion-rate metrics (SDRA, kNN recall, P_nest)
  diagnostics.py    per-batch instrumentation of SDR-MRL's own quantities
  evaluation.py     the Matryoshka evaluation suite
  reporting.py      results.json / results.csv
  trainers/         base.py + one trainer per method
  cli.py            entry point
scripts/train.py    run without installing the package
scripts/run_all.sh  run every experiment in sequence
scripts/verify_sdr_math.py  check the SDR-MRL code against the paper's proofs
notebooks/colab/     run_experiment.ipynb — one experiment on Colab, GPU included
Dockerfile          self-contained training image (code + data + models)
docker/             build/push scripts, model baking, docker docs
tests/              173 tests, fully offline (no model downloads)
notebooks/          the original notebooks, kept for reference
```

## Setup

```bash
pip install -e .            # or: pip install -r requirements.txt
```

## Usage

### On Colab

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/duncan-nguyen/embedding-mrl/blob/main/notebooks/colab/run_experiment.ipynb)

`notebooks/colab/run_experiment.ipynb` clones this repo, installs what Colab is
missing, and runs one experiment end to end. Method, backbone and every training
and SDR-MRL setting are a form at the top of the notebook — pick them, then
*Runtime → Run all*. The training corpus and all evaluation CSVs ship in the
repo, so the only runtime download is the backbone itself. It finishes with the
results table, the quality- and distortion-rate curves, and the training
diagnostics.

BERT-base and TinyBERT-6L fit a free T4; BGE-M3 and Qwen3-0.6B need
`BATCH_SIZE` around 4–8.

### Locally

```bash
# train + evaluate
python scripts/train.py --config configs/mipic/bgem3.yaml

# same thing if the package is installed
embedding-mrl --config configs/mipic/bgem3.yaml

# override anything from the command line
python scripts/train.py --config configs/mrl/bert.yaml \
    --set train.epochs=3 --set train.batch_size=32 --set eval.split=validation

# inspect the fully resolved config without loading a model
python scripts/train.py --config configs/ese/bgem3.yaml --print-config

# evaluate an existing checkpoint
python scripts/train.py --config configs/mrl/bert.yaml --eval-only \
    --set model.name_or_path=outputs/mrl_bert/encoder
```

Training always finishes with an evaluation pass, and the scores are written to
disk. Artifacts land in `train.output_dir`:

| file | contents |
| --- | --- |
| `results.json` | **the final report** — run metadata, training losses, per-task scores at every dimension, family averages, and a flat per-dimension table |
| `results.csv` | the same table as CSV: one row per Matryoshka dimension, one column per task |
| `results_epoch{N}.json` | raw scores after epoch N (only when `eval.every_epoch` is true) |
| `history.json` | per-epoch train loss + summary metrics |
| `config.yaml` | the resolved config the run actually used |
| `encoder/` | trained weights + tokenizer |

`results.json` looks like this:

```json
{
  "experiment": {
    "name": "mipic_bgem3", "method": "mipic", "model": "BAAI/bge-m3",
    "hidden_dim": 1024, "pooling": "cls",
    "matryoshka_dims": [16, 32, 64, 128, 256, 512, 1024],
    "eval_split": "test", "epochs": 8, "finished_at": "2026-08-27T15:49:22+07:00"
  },
  "training": { "final_loss": 2.41, "loss_per_epoch": [...], "duration_seconds": 9821.4 },

  "classification": { "banking77": { "dim_16": {"accuracy": ..., "f1": ...}, ... }, ... },
  "sts":            { "stsb":      { "dim_16": 0.71, ... }, ... },
  "pair":           { "mrpc":      { "dim_16": {"accuracy": ..., "f1": ..., "average_precision": ...}, ... }, ... },

  "summary": { "classification": {"dim_16": ...}, "sts": {...}, "pair": {...} },
  "table":   { "dim_16": {"classification/banking77": ..., "mean/sts": ..., ...}, ... }
}
```

The `classification` / `sts` / `pair` keys keep the layout this project has
always used; `experiment`, `training`, `summary` and `table` are additions.

The same table is printed at the end of the run:

```
   dim | classification |            sts |           pair
---------------------------------------------------------
    16 |         0.6421 |         0.7103 |         0.6890
    32 |         0.7015 |         0.7488 |         0.7102
   ...
```

To evaluate only at the end instead of after every epoch (much faster), set
`eval.every_epoch: false`. To score an already-trained checkpoint, use
`--eval-only`; it writes the same `results.json` / `results.csv`.

## The sixteen configs

| | BERT (768) | TinyBERT-6L (768) | BGE-M3 (1024) | Qwen3-0.6B (1024) |
| --- | --- | --- | --- | --- |
| **MRL** | `configs/mrl/bert.yaml` | `configs/mrl/tinybert_6l.yaml` | `configs/mrl/bgem3.yaml` | `configs/mrl/qwen3_0.6b.yaml` |
| **ESE** | `configs/ese/bert.yaml` | `configs/ese/tinybert_6l.yaml` | `configs/ese/bgem3.yaml` | `configs/ese/qwen3_0.6b.yaml` |
| **MIPIC** | `configs/mipic/bert.yaml` | `configs/mipic/tinybert_6l.yaml` | `configs/mipic/bgem3.yaml` | `configs/mipic/qwen3_0.6b.yaml` |
| **SDR-MRL** | `configs/sdr/bert.yaml` | `configs/sdr/tinybert_6l.yaml` | `configs/sdr/bgem3.yaml` | `configs/sdr/qwen3_0.6b.yaml` |

Each inherits `configs/base.yaml` via `_base_` and overrides only what differs.

## Methods

All four train a bi-encoder with unsupervised SimCSE: the same sentence is
encoded twice, and dropout makes the two views differ.

### MRL (baseline) — `losses/infonce.py`

InfoNCE applied independently at every nested prefix, summed:

```
L_MRL = Σ_d InfoNCE(e1[:d], e2[:d])
```

### ESE / EPRESSO (baseline) — `losses/ese.py`

Adds a second axis: the same nested-dimension loss is also applied to sampled
*intermediate layers*, with log-based weights that favour the smaller dims and
the deeper layers.

```
L_ESE = Σ_d w_d · InfoNCE(e1[:d], e2[:d])  +  Σ_layers w_l · (same, on that layer)
w_d = 1 / (1 + log(i+1)),  w_l = 1 / (1 + log(L - l))
```

### MIPIC (our method) — `losses/mipic.py`

Adds cross-dimensional and depth-wise alignment on top of the MRL objective. The
full hidden width is its own teacher — no second model is loaded. Equation
numbers below refer to [docs/MIPIC.pdf](docs/MIPIC.pdf).

**SIA — Self-Distilled Intra-Relational Alignment** (Sec 3.2), applied at every
layer in `mipic.layers` and every truncated prefix `d_i`:

1. *Attention distribution matching* (Eq 4–6) — the truncated prefix must rank
   tokens the way the full width does. The full-dimensional `h_CLS` is the query
   for both; a learnable `P_i ∈ R^{d_i×D}` lifts the truncated token back to `D`.

   ```
   s_j^(D) = h_CLS · h_j / √D            a_D = softmax(s^(D)/τ)
   s_j^(i) = h_CLS · P_iᵀ h_j^(i) / √D   a_i = softmax(s^(i)/τ)
   L_att^(i) = KL(a_i ‖ a_D)
   ```

2. *Top-k CKA alignment* (Eq 7–12) — the geometry of the `k_i` most important
   tokens must agree. Tokens are ranked once by the **teacher** `a_D`, so the
   selected sets are nested (`S_k₁ ⊂ S_k₂ ⊂ …`), with
   `k_i = max(8, ⌈γ_i·m⌉)` and `γ = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7]` (Appendix A.5).

   ```
   L_CKA^(i) = 1 − CKA(h_i, H_i)
   ```

**PIC — Progressive Information Chaining** (Sec 3.3) — each `(dim, layer)`
checkpoint in `mipic.checkpoints` must predict the next, deeper and wider one,
e.g. `(16, layer 2) → (32, layer 4) → … → (768, layer 12)` for BERT-base:

```
L_chain^(i) = InfoNCE(φ_i(z_i), z_{i+1})        (Eq 16)
```

**Aggregation** (Eq 13, 14, 17, 18) — unweighted sums, and a single
hyperparameter `α` trades off MRL against the alignment terms:

```
L_SIA = Σ_{k∈L} Σ_i (L_att^(i) + L_CKA^(i))
L_PIC = Σ_i L_chain^(i)
L_MIPIC = α·L_MRL + (1 − α)·(L_SIA + L_PIC)
```

### SDR-MRL (proposal) — `losses/sdr.py`

[docs/latex/main.pdf](docs/latex/main.pdf). The other three methods ask *how similar
is the truncated representation to the full one*. SDR-MRL asks a different
question: **at a given storage rate, how much of the teacher's semantic
neighborhood is still recoverable?**

The full-dimensional embedding defines a distribution over which other item in
the batch is the anchor's semantic neighbor, and each prefix has to reproduce
it with nothing but its own cosine geometry:

```
p_T(S = j | x_i) = softmax_j cos(z_i^(K), z_j^(K)) / τ_T        (Eq 23, stop-grad)
q_k(S = j | z_i) = softmax_j cos(z_i^(k), z_j^(k)) / τ_S        (Eq 25)
D_k              = E_X KL(p_T ‖ q_k)                            (Eq 26)
```

The decoder has no parameters — the prefix's own geometry *is* the decoder — so
nothing is added at inference time. What makes `D_k` more than another alignment
loss is Proposition 1: it decomposes as `D_k = I_T(S; X | Z_k) + ε_k` with
`ε_k ≥ 0`, so it is a variational **upper bound on the semantic information the
prefix has lost**. Because the prefixes are nested, the drop between adjacent
rates is the new block's conditional information, `D*_{k−1} − D*_k =
I_T(S; B_k | Z_{k−1})` (Eq 45).

Training minimises that distortion across a distribution of deployment rates:

```
L_SDR = L_task + λ_sem · Σ_k π_k D_k + λ_mono · Σ_k [D_k − D_{k−1}]_+   (Eq 55)
```

`L_task` is the ordinary Matryoshka InfoNCE and is not decoration: the semantic
term alone is minimised by a collapsed teacher, whose neighborhood every prefix
reproduces trivially. `π` is the deployment-rate prior (uniform by default,
`inverse_dim` to bias toward small prefixes). `λ_mono = 0` in the shipped
configs — Eq 56's minimal model — because Prop 2 guarantees monotonicity only
for the optimal decoder, and whether the practical one needs the regulariser is
an empirical question.

Setting `sdr.stochastic_rate: true` draws one rate per step instead of walking
every prefix; it is unbiased for the same objective (Eq 68) at a fraction of the
cost. Every axis the proposal proposes to vary is a config field:

| field | Eq / §  | choices |
| --- | --- | --- |
| `divergence` | Eq 27, §8.2 | `forward_kl` (mass-covering), `reverse_kl`, `js` |
| `geometry` | §8.2 | `snd`, `gram_mse`, `cka`, `hard_neighbor` |
| `candidates` | §8.6 | `all`, `teacher_topm`, `teacher_topm_student_hard` |
| `rate_prior` | Eq 49-50 | `uniform`, `inverse_dim`, `custom` + `rate_weights` |
| `teacher` | §4.15 | `online` (stop-grad self), `ema`, `frozen` |
| `teacher_temperature` / `student_temperature` | Eq 125-126 | τ_T, τ_S |

A frozen teacher may have any hidden width — only its neighborhood distribution
is ever read — but it must accept the student's tokenizer.

### Debugging SDR-MRL

The loss is one scalar and hides everything the propositions are stated in
terms of. `sdr.diagnostics_every: N` recomputes those quantities every `N`
steps, logs them and appends them to `diagnostics.jsonl`:

```
teacher: H=2.175 nats (perplexity 8.8 of 15 candidates, max p=0.275, mean cos=+0.799)
D_0=0.533  D_full=-9.66e-09  V_mono=0.00

   dim |      D_k |  D_k/D_0 |     gain |  eta*1e3 |    kNN@3 |  H(q_k) |  |dbary|
----------------------------------------------------------------------------------
     4 |   0.5672 |    1.065 |        - |        - |    0.479 |   2.073 |   0.0912
     8 |   0.3446 |    0.647 |   0.2226 |   55.659 |    0.458 |   2.236 |   0.0826
    16 |   0.1193 |    0.224 |   0.2253 |   28.160 |    0.708 |   2.184 |   0.0591
    32 |  -0.0000 |   -0.000 |   0.1193 |    7.455 |    1.000 |   2.175 |   0.0000
```

Reading it:

- **`H` (teacher entropy, Eq 112)** — the τ_T knob made legible. `perplexity` is
  the effective number of semantic neighbors: near 1 the teacher has degenerated
  to hard-neighbor supervision, near the candidate count it is uniform and `D_k`
  carries no signal for any prefix.
- **`mean cos`** — collapse detector for the §4.9 failure mode. Climbing toward
  1 means the semantic term is being satisfied by collapse, not by ordering.
- **`D_k/D_0`** — distortion against the zero-rate uniform decoder (Eq 88-89).
  **Above 1 the prefix is worse than knowing nothing**, which the raw `D_k`
  never makes obvious. Above it is exactly what happens at `d = 4`.
- **`gain` / `eta`** — `D_{k−1} − D_k`, the block's conditional semantic
  information (Eq 45), and the same per added coordinate (Eq 122). This is where
  a model reveals how it allocates capacity, and where MRL, ESE, MIPIC and
  SDR-MRL are expected to differ even at equal accuracy.
- **`kNN`** — rank-based counterpart to `D_k`. Distortion can fall by matching
  the teacher's similarity *scale*; recall only moves when the neighbor ordering
  genuinely improves, so the two disagreeing is itself a finding.
- **`|dbary|`** — `‖Σ_j (q_ij − p_ij) z_j‖`, which by Eq 61 *is* the gradient up
  to `1/τ`: the distance from the student's local semantic barycenter to the
  teacher's. It should shrink as training proceeds.
- **`D_full`** — must be ~0 when `τ_S == τ_T`, because student and teacher are
  then the same distribution. Anything else is an irreducible floor under the
  objective and almost always a mismatched-temperature config slip. The trainer
  warns about this at startup.
- **`V_mono`** (Eq 108) — the fraction of adjacent prefixes where adding
  coordinates made the neighborhood *harder* to recover. This is the evidence
  for or against needing `λ_mono` at all.

`scripts/verify_sdr_math.py` checks the implementation against the derivations
themselves — 35 assertions over independently constructed worlds:

```
Eq 59/61/65   the gradient identities, by autograd vs closed form
Eq 30-32      D_k = I_T(S;X|Z_k) + ε_k on an exact discrete S–X–Z chain
Eq 39/42/45   the optimal decoder, monotonicity, and the refinement identity
Eq 68         unbiasedness of stochastic rate sampling
Eq 75-84      Theorem 1's linear-Gaussian nested optimum, incl. PCA's suboptimality
Eq 114-118    rotation leaves ZZ^T untouched but wrecks the prefixes
```

The last one is the motivation for the whole method, and it is worth running
once for the intuition: an orthogonal rotation preserves every full-dimensional
retrieval metric exactly while dropping prefix kNN recall by ~0.5.

## Evaluation

Every task is scored at all seven Matryoshka dimensions.

| family | tasks | metric |
| --- | --- | --- |
| Classification | Banking77, Emotion, Tweet | logistic-regression probe on frozen embeddings → accuracy, macro-F1 |
| STS | SICK, STS12–16, STSB, SICK-R | Spearman correlation on cosine similarity |
| Pair classification | MRPC, SciTail, WiC | threshold-tuned accuracy, macro-F1/precision/recall, average precision |

Switch `eval.split` between `test` and `validation`. `data.py` also registers
RTE and QNLI, which ship in `data/test/` but were not part of the notebooks'
default suite — add them to `eval.pair_tasks` to use them.

### Semantic distortion-rate protocol

`eval.semantic_distortion: true` adds the SDR-MRL measurement protocol (§6). It
runs for **every** method, which is the point — the proposal's claim is about
where in the code each method places semantic information, not only about final
accuracy. Results land under `results.json → semantic`:

| key | Eq | meaning |
| --- | --- | --- |
| `distortion` | 86 | `D_k^ref` per dimension, against a fixed teacher |
| `normalized_distortion` | 89 | rescaled so zero-rate = 1 and full width = 0 |
| `sdra` | 91 | area under that curve; **lower means useful structure appears earlier** |
| `distortion_reduction`, `refinement_gain` | 119, 122 | per-block and per-coordinate semantic gain |
| `monotonicity_violation_rate` | 108 | fraction of prefixes where more coordinates hurt |
| `preservation` | 96, §6.4 | kNN recall, similarity Spearman, trustworthiness, continuity |
| `rotation_stress_test` | 114-118 | set `eval.rotation_trials > 0` |

Set `eval.reference_model` to hold the teacher **fixed across every compared
model** — Eq 86 requires it. Without it each model is scored against its own
full width, which measures self-consistency rather than semantic quality; the
report records `"reference": "self"` when that happens.

`geometry.price_of_nestedness` (Eq 92-93) compares a Matryoshka prefix against a
model trained at that width alone. Produce the independent baseline with e.g.
`--set matryoshka.dims='[64]' --set model.hidden_dim=768`, then feed both
`results.json` tables to the function.

## Training configuration

Values follow Table 7 and Appendix A.5 of the paper.

| | value |
| --- | --- |
| Task | unsupervised pair classification (bi-encoder, SimCSE) |
| Max sequence length | 256 |
| Batch size | 16 |
| Epochs | 5 |
| Learning rate | 2e-5 |
| Optimiser | AdamW, weight decay 0.01 |
| LR scheduler | cosine |
| Temperature τ | 0.05 |
| Matryoshka dims | 16, 32, 64, 128, 256, 512, 768 / 1024 |
| Precision | FP16 autocast |
| Pooling | CLS for MRL/MIPIC, mean for ESE |

MIPIC's per-backbone settings (Table 7 for `α`, Appendix A.5 for `L` and `C`):

| backbone | α | `layers` (L) | `checkpoints` (C, as `(dim, layer)`) |
| --- | --- | --- | --- |
| TinyBERT-6L | 0.4 | 1,2,3,4,5,6 | (16,1) (32,2) (64,3) (256,4) (512,5) (768,6) |
| BERT-base | 0.4 | 2,4,6,8,9,10,12 | (16,2) (32,4) (64,6) (128,8) (256,9) (512,10) (768,12) |
| BGE-M3 | 0.5 | 1,4,7,11,15,19,24 | (16,1) (32,4) (64,7) (128,11) (256,15) (512,19) (1024,24) |
| Qwen3-0.6B | 0.5 | 2,6,12,16,20,24,28 | (16,2) (32,6) (64,12) (128,16) (256,20) (512,24) (1024,28) |

Top-k schedule: `k_i = max(8, ⌈γ_i·m⌉)` with `γ = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7]`
over the six truncated prefixes, `m` the sequence length.

### Where the paper is silent

A few settings are not stated and are carried over from the original notebooks:

- **Warmup and minimum LR** — the paper says only "Cosine". The configs use
  `cosine_with_min_lr` with 10% warmup and `min_lr = 2e-6`.
- **Gradient clipping** — not mentioned; MIPIC keeps `max_grad_norm: 1.0`.
- **Baseline hyperparameters** — Table 7 covers MIPIC only. MRL and ESE use the
  same epochs / LR / batch size / τ, which is what "fair comparison" (Appendix
  A.2) implies.
- **Training corpus** — Appendix A.2 describes ~24,000 sentences;
  `data/train/final_data.csv` ships 20,240 (Banking77 3,000, TweetEval 3,000,
  WiC 3,000, MRPC 3,000, SciTail 3,000, STS-B 2,893, SICK 2,347). The shipped
  file is used as-is.

### Two things worth knowing about τ = 0.05

Table 7 lists a single temperature and the paper reuses the symbol `τ` in the
attention softmax (Eq 4–5), the chain InfoNCE (Eq 16) and SimCSE (Eq 19), so the
configs apply 0.05 everywhere. Measured on BERT-shaped activations:

- **The attention distribution collapses to one-hot.** `h_CLS·h_j/√D` already
  spans roughly `[-1, 7]`, so dividing by 0.05 gives a hard argmax (entropy 0).
  `L_att` still trains — it pushes `P_i` to match the teacher's argmax — but it
  no longer matches a soft ranking. Decouple it with
  `--set mipic.attention_temperature=1.0` if that is not the intent. Top-k
  selection is unaffected: it ranks the raw scores, not the softmax.
- **`L_att` dominates the objective at initialisation.** With BERT's 7 layers ×
  6 prefixes, Eq 14 sums 42 KL terms: `L_att ≈ 679` versus `L_CKA ≈ 5`,
  `L_PIC ≈ 24` and `L_MRL ≈ 20`. That is what Eq 13/14 specify; use
  `mipic.w_att`, `mipic.w_cka`, `mipic.w_pic` or `mipic.aggregate: mean` to
  rebalance for an ablation.

## Docker

For the internal GPU server there is a self-contained image — **code + data +
model weights baked in**, so it runs with no network access:

```bash
# build and push (see docker/README.md for choosing the CUDA base image)
DOCKERHUB_USER=yourname ./docker/build_and_push.sh

# on the server
docker run --rm --gpus all --shm-size=8g \
    -v "$PWD/outputs:/workspace/outputs" \
    yourname/embedding-mrl:latest --config configs/mipic/bgem3.yaml
```

`make build` / `make push` / `make run` wrap the same commands. Full details,
including image size and how to bake fewer models, are in
[docker/README.md](docker/README.md).

## Tests

```bash
pytest
```

173 tests covering config inheritance and validation, the task registry against
the real `data/` files, loss maths (gradient flow, masking, CKA invariances,
top-`k` selection), pooling, evaluation metrics, the semantic distortion-rate
protocol, and one-epoch training runs for all four methods. They use a locally
built stub encoder, so nothing is downloaded and the suite finishes in seconds.

The SDR-MRL derivations have their own harness, which the suite also runs:

```bash
python scripts/verify_sdr_math.py
```

## Notes on the refactor

The `notebooks/` directory holds the original twelve notebooks. Behaviour is
preserved, with these deliberate changes:

- **Data paths** — `/kaggle/input/...` → the local `data/` directory.
- **Dead teacher model removed** — every notebook loaded a second copy of the
  backbone as a "teacher"; no loss ever used it. ESE additionally ran a teacher
  forward pass each step and discarded the result.
- **Duplicate forward passes removed** — the ESE step encoded each batch twice
  (once in the loop, once inside the loss). The loss now takes pre-computed
  hidden states. Identical objective, ~2× faster.
- **Evaluation encodes once, not once per dimension** — embeddings are extracted
  at full width and truncated per dimension. Mathematically identical, 7× fewer
  forward passes.
- **Pair-metric crash fixed** — `get_metric_pair_classification` compared a
  Python list against a float (`scores >= thr`), which raises `TypeError`.
  Inputs are now coerced with `np.asarray`.
- **`results.json` is actually written** — the old README documented it, but no
  notebook produced it. Training now always ends with an evaluation pass that
  writes `results.json` (plus a flat `results.csv`) with run metadata, per-task
  scores at every dimension, and family averages.
- **Per-model constants are config, not code** — layer indices, checkpoints,
  `hidden_dim` and the dimension lists were hard-coded and had to be edited in
  several places per notebook.
- **MIPIC now follows the paper, not the notebooks.** The released notebooks
  diverged from `docs/MIPIC.pdf` in five ways, all corrected here:

  | | notebooks | paper (current) |
  | --- | --- | --- |
  | Attention KL | zeroed before returning — no gradient | Eq 6, active |
  | Attention scores | shared 64-d space, learned `W_Q`/`W_K` | Eq 4–5, full-dim `h_CLS` query + `P_i` lift |
  | Top-k ranking | per-dimension scores (sets not nested) | teacher `a_D` (Sec 3.2.2, nested) |
  | Top-k size | `max(8, dim/D · 64)`, fixed | `max(8, ⌈γ_i·m⌉)`, sequence-dependent (App. A.5) |
  | Aggregation | weighted mean, `α=β=0.4, γ=0.2` | Eq 13/14/17 sums, single `α` (Eq 18) |

  Layer/checkpoint sets, `α`, τ and the epoch count also differ — see
  [Training configuration](#training-configuration). The notebook variants are
  reachable through config (`mipic.aggregate: mean`, `mipic.w_att`,
  `mipic.attention_temperature`), and the original code is in `notebooks/`.
- **Epochs and temperature** — the notebooks ran MIPIC for 8 epochs at τ=0.07;
  Table 7 says 5 epochs at τ=0.05 for every backbone.
- **`torch.cuda.amp` → `torch.amp`**, with a fallback for older torch.

## Citation

```bibtex
@misc{huy2026mipicmatryoshkarepresentationlearning,
      title={MIPIC: Matryoshka Representation Learning via Self-Distilled Intra-Relational and Progressive Information Chaining},
      author={Phung Gia Huy and Hai An Vu and Minh-Phuc Truong and Thang Duc Tran and Linh Ngo Van and Thanh Hong Nguyen and Trung Le},
      year={2026},
      eprint={2604.24374},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2604.24374}
}
```
