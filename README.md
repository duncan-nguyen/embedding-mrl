# Matryoshka Representation Learning: ACL Conference Experiments

Experimental code for an ACL submission comparing three ways to train Matryoshka
embedding models — representations that stay useful when truncated to
`[16, 32, 64, 128, 256, 512, 1024]` (or `768`), so inference cost can be traded
against quality at serving time.

Three methods × four backbones = twelve experiments, all driven by one CLI and
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
  losses/           infonce.py, cka.py, ese.py, mipic.py
  evaluation.py     the Matryoshka evaluation suite
  trainers/         base.py + one trainer per method
  cli.py            entry point
scripts/train.py    run without installing the package
tests/              68 tests, fully offline (no model downloads)
notebooks/          the original notebooks, kept for reference
```

## Setup

```bash
pip install -e .            # or: pip install -r requirements.txt
```

## Usage

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

Artifacts land in `train.output_dir`:

| file | contents |
| --- | --- |
| `results.json` | final scores: `{classification, sts, pair, summary}`, each broken down per dimension |
| `results_epoch{N}.json` | the same, per epoch |
| `history.json` | per-epoch train loss + summary metrics |
| `config.yaml` | the resolved config the run actually used |
| `encoder/` | trained weights + tokenizer |

## The twelve configs

| | BERT (768) | TinyBERT-6L (768) | BGE-M3 (1024) | Qwen3-0.6B (1024) |
| --- | --- | --- | --- | --- |
| **MRL** | `configs/mrl/bert.yaml` | `configs/mrl/tinybert_6l.yaml` | `configs/mrl/bgem3.yaml` | `configs/mrl/qwen3_0.6b.yaml` |
| **ESE** | `configs/ese/bert.yaml` | `configs/ese/tinybert_6l.yaml` | `configs/ese/bgem3.yaml` | `configs/ese/qwen3_0.6b.yaml` |
| **MIPIC** | `configs/mipic/bert.yaml` | `configs/mipic/tinybert_6l.yaml` | `configs/mipic/bgem3.yaml` | `configs/mipic/qwen3_0.6b.yaml` |

Each inherits `configs/base.yaml` via `_base_` and overrides only what differs.

## Methods

All three train a bi-encoder with unsupervised SimCSE: the same sentence is
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

Adds horizontal and vertical alignment on top of the MRL objective. The full
hidden width is its own teacher — no second model is loaded.

1. **Horizontal attention alignment (SIA)** — token-importance ordering should
   agree between a truncated prefix and the full width (KL divergence).
2. **Submatrix CKA** — the geometry of the top-`k` important tokens should
   agree, measured with per-example CKA.
3. **Pipeline InfoNCE (PIC)** — a shallow/narrow stage should predict the next
   deeper/wider one, e.g. `(layer 3, dim 16) → (layer 7, dim 128) → (layer 11, dim 768)`.

```
L_align = α·L_SIA + β·L_CKA + γ·L_PIC          (α=0.4, β=0.4, γ=0.2)
L_MIPIC = w_align·L_align + w_matryoshka·L_MRL  (0.4 / 0.6)
```

> **`use_attention_kl` is `false` by default.** The released notebooks zeroed the
> SIA KL term before returning it, so `α·L_SIA` contributed nothing to the
> gradient; only the attention *scores* were used, to pick the top-`k` tokens for
> the CKA term. The default reproduces the published runs. Set
> `--set mipic.use_attention_kl=true` to enable the term as the paper describes —
> that is a different experiment and will not reproduce the notebook numbers.

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

## Training configuration

| | value |
| --- | --- |
| Task | unsupervised pair classification (bi-encoder, SimCSE) |
| Max sequence length | 256 |
| Batch size | 16 |
| Epochs | 5 (MRL, ESE) / 8 (MIPIC) |
| Optimiser | AdamW, lr 2e-5, weight decay 0.01 |
| Schedule | cosine with min lr 2e-6, 10% warmup |
| Temperature | 0.07 |
| Precision | FP16 autocast |
| Pooling | CLS for MRL/MIPIC, mean for ESE |

## Tests

```bash
pytest
```

68 tests covering config inheritance and validation, the task registry against
the real `data/` files, loss maths (gradient flow, masking, CKA invariances,
top-`k` selection), pooling, evaluation metrics, and one-epoch training runs for
all three methods. They use a locally built stub encoder, so nothing is
downloaded and the suite finishes in seconds.

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
  notebook produced it.
- **Per-model constants are config, not code** — `align_layers`,
  `pipeline_pairs`, `hidden_dim` and the dimension lists were hard-coded and had
  to be edited in several places per notebook.
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
