# GSR Implementation Plan

This plan integrates [method.md](docs/method.md) into the repository's existing
`config -> trainer -> epoch evaluation -> report` protocol. It deliberately
keeps MRL, ESE, and MIPIC behavior unchanged.

Implementation status: phases A-D and the offline Phase-E gates are complete.
All four production configs resolve and the CPU stub smoke suite exercises an
active teacher refresh. A real-backbone/GPU smoke run remains intentionally
unexecuted until training infrastructure and model weights are selected.

## 1. Locked implementation decisions

- Add `gsr` as a fourth method. GSR retains the existing Matryoshka InfoNCE
  objective and adds the residual distance-shell loss.
- Do not keep a second model in memory. At each refresh boundary, run the
  current encoder in `eval()` and `no_grad()`, cache its deterministic
  full-normalized corpus embeddings, compute the spectral teacher, then freeze
  those tensors while the encoder trains.
- Identify corpus examples by a stable integer `sample_id`; never associate a
  shuffled batch with teacher targets by batch position.
- Compute teacher covariance over the full configured training corpus, not an
  optimization mini-batch.
- Cache teacher spectral coordinates on CPU in FP32. For the current corpus
  (20,243 examples), a full `N x 1024` cache is about 79 MiB.
- Compute eigendecomposition and all geometry losses in FP32 or FP64, outside
  mixed precision. Encoder forward passes may remain under AMP.
- Use pairwise squared-distance shells and all unordered within-batch pairs.
  Batch size affects variance only; it must never alter shell definitions.
- Use a common, teacher-fixed normalization constant `c_teacher`.
- Keep task prefixes at every configured Matryoshka dimension. Merge geometry
  shells only when an exact/numerical eigenvalue tie crosses a boundary.
- Refresh the teacher only between epochs/outer blocks. Targets must not move
  inside an optimization block.
- Make non-finite geometry, incomplete teacher coverage, invalid cache indices,
  and degenerate `c_teacher` hard errors with a diagnostic artifact.

## 2. Configuration surface

Add `GSRConfig` to `config.py` and `gsr: GSRConfig` to `ExperimentConfig`.
Extend `METHODS` with `gsr`.

Proposed fields:

```yaml
gsr:
  weight: 1.0
  warmup_epochs: 1
  refresh_every_epochs: 1
  teacher_batch_size: 64
  geometry_dims: null          # null -> matryoshka.dims
  eigengap_tolerance: 1.0e-6   # numerical tie, not a tuned gap threshold
  merge_tied_shells: true
  eps: 1.0e-8
  cache_dtype: float32
  save_teacher_tensors: true
  diagnostics_every_steps: 100
  diagnostic_samples: 512
  diagnostic_pairs: 8192
  fail_on_nonfinite: true
```

Validation must reject:

- non-positive weights, batch sizes, refresh intervals, or epsilons;
- `warmup_epochs >= train.epochs` unless GSR is intentionally disabled;
- unsorted, repeated, non-positive, or over-width geometry dimensions;
- geometry dimensions not contained in `matryoshka.dims`;
- unsupported cache dtypes;
- a largest Matryoshka dimension different from `model.hidden_dim`.

Add four shipped configs:

```text
configs/gsr/bert.yaml
configs/gsr/tinybert_6l.yaml
configs/gsr/bgem3.yaml
configs/gsr/qwen3_0.6b.yaml
```

They inherit `configs/base.yaml`, use the same backbone/pooling settings as MRL,
and change only `name`, `method`, `output_dir`, and GSR-specific values.

Update the public integration points at the same time:

- export GSR primitives from `losses/__init__.py`;
- register `GSRTrainer` in `trainers/__init__.py`;
- update the README method/config tables from 12 to 16 experiments;
- update CLI descriptions and examples;
- keep `scripts/run_all.sh` unchanged because it discovers configs dynamically;
- verify the Makefile's existing “all 16 experiments” text now matches reality.

## 3. Stable sample identity and teacher data pass

### `data.py`

Change `SimCSEPairDataset` to return a record containing:

```python
{"sample_id": idx, "view1": text, "view2": text}
```

Update `SimCSECollator` to emit `sample_ids: LongTensor[B]` alongside the current
token tensors. Existing trainers ignore this extra field.

Add `build_teacher_loader(...)`:

- same dataset and tokenizer;
- `shuffle=False`, `drop_last=False`;
- one tokenized view only;
- configurable `teacher_batch_size`;
- always returns `sample_ids`;
- deterministic order, but the cache writer must still place rows by ID rather
  than assuming order.

Keep `sample_ids` on CPU in `BaseTrainer._to_device`; GSR uses them to index the
CPU cache before transferring only the selected teacher scores to the GPU.

Teacher refresh must track a boolean `seen[N]` vector and fail if it observes a
duplicate, missing, negative, or out-of-range ID. This prevents the most
dangerous silent failure: correct tensor shapes paired with the wrong samples.

## 4. Mathematical core

### New `src/embedding_mrl/losses/gsr.py`

Implement small, independently testable functions:

- `full_normalize(z, eps)`;
- `condensed_squared_distances(x)` using unordered pairs only;
- `build_shell_slices(dims)`;
- `merge_tied_shells(dims, eigenvalues, tolerance)`;
- `gsr_shell_loss(student, teacher_scores, shell_slices, c_teacher)`.

`gsr_shell_loss` receives:

- `student: [B, D]`, already full-normalized;
- `teacher_scores: [B, D]`, cached coordinates `(q_teacher - mean) @ V`;
- shell intervals `(start, end)`;
- scalar `c_teacher`.

For each shell it computes:

```text
s_k = pdist(student[:, start:end])^2
r_k = pdist(teacher_scores[:, start:end])^2
loss_k = mean((s_k - r_k)^2) / (c_teacher + eps)
```

The returned object should retain tensor-valued component losses for gradient
debugging and detached statistics for logs. Require `B >= 2`.

### New `src/embedding_mrl/gsr_teacher.py`

Add a `SpectralTeacherCache` dataclass containing:

- normalized corpus mean;
- descending eigenvalues and eigenvectors;
- cached spectral coordinates by `sample_id`;
- original and merged shell boundaries;
- `c_teacher`;
- teacher refresh number and source epoch;
- numerical diagnostics and timings.

Teacher construction:

1. Run deterministic pooled embeddings for every corpus row.
2. Full-normalize in FP32 and fill `Q[N, D]` by stable ID.
3. Compute `mean = Q.mean(0)` and centered covariance
   `(Q - mean).T @ (Q - mean) / N`.
4. Symmetrize covariance before `torch.linalg.eigh`.
5. Sort eigenpairs descending and compute `(Q - mean) @ V`.
6. Merge only boundaries whose relative eigengap is below numerical tolerance.
7. Compute `c_teacher` exactly without an `N x N` Gram matrix. For unit vectors,
   use the corpus sums of pairwise dot products and squared dot products:

   ```text
   distance^4 = 4 * (1 - 2 * dot + dot^2)
   ```

   with off-diagonal means derived from `sum(Q)` and `Q.T @ Q`.
8. Release `Q` after the spectral-coordinate cache is built.

Hard checks:

- every embedding and cache tensor is finite;
- normalized row norms are within tolerance of one;
- covariance symmetry residual is small;
- negative eigenvalue mass is within numerical tolerance;
- eigenpairs satisfy residual and orthogonality tolerances;
- `c_teacher > eps`;
- cached row count equals dataset length.

## 5. Trainer lifecycle

### Minimal hooks in `BaseTrainer`

Add no-op hooks instead of copying the whole training loop into GSR:

```python
on_train_start()
on_epoch_start(epoch_index)
on_after_backward(epoch_index, step, loss, logs)
on_epoch_end(epoch_index, record)
method_report_metadata()
```

Also make these backward-compatible loop improvements:

- expose `self.epoch_index`, `self.step_in_epoch`, and `self.global_step`;
- aggregate every scalar returned by `compute_loss`, weighted by batch size;
- persist epoch averages under `history[*]["train_metrics"]`;
- always unscale gradients before diagnostic hooks and clipping;
- record learning rate, AMP scale, total gradient norm, maximum gradient value,
  non-finite gradient count, step time, and GPU memory;
- keep current progress-bar behavior for all existing methods.

### New `src/embedding_mrl/trainers/gsr.py`

`GSRTrainer` extends `BaseTrainer`.

At epoch start:

- if still in warmup, disable geometry and log the effective weight as zero;
- otherwise refresh when no cache exists or the refresh interval is due;
- save teacher summary/tensors before taking an optimizer step;
- restore model train mode after the deterministic teacher pass.

For each training batch:

1. Run the current two student forwards and pool exactly as `MRLTrainer`.
2. Compute ordinary Matryoshka InfoNCE.
3. Full-normalize both pooled embeddings in FP32.
4. Gather cached teacher scores by CPU `sample_ids` and transfer that slice.
5. Compute GSR for both student views against the same deterministic teacher.
6. Return `MRL + effective_weight * mean(GSR_view1, GSR_view2)`.
7. Fail before backward if any component is non-finite.

Do not add trainable loss modules or optimizer parameter groups.

## 6. Diagnostics designed for post-training debugging

### Persistent artifacts

Every run should contain:

```text
output_dir/
  train.log
  config.yaml
  history.json
  diagnostics/
    steps.jsonl
    teacher_epoch{E}.json
    teacher_epoch{E}.pt
    geometry_epoch{E}.json
    failure_step{S}.pt          # only on hard failure
  results_epoch{E}.json
  results.json
  results.csv
  encoder/
```

`steps.jsonl` is append-only and contains one compact record every configured
diagnostic interval. Large matrices belong in `.pt`, never JSON.

### Teacher refresh metrics

Log and save:

- corpus size, hidden dimension, pooling, cache dtype, cache MiB;
- encode, covariance, eigendecomposition, projection, and total refresh time;
- ID coverage, duplicates, missing IDs, and non-finite rows;
- raw embedding norm and normalized row-norm min/mean/max;
- teacher mean norm;
- covariance trace, Frobenius norm, symmetry residual;
- minimum eigenvalue, negative eigenvalue count/mass;
- eigensolver residual `||Sigma V - V Lambda|| / ||Sigma||`;
- eigenvector orthogonality residual `||V.T V - I|| / D`;
- effective rank and stable rank;
- explained variance and squared spectral energy at every requested dimension;
- absolute/relative eigengap at every boundary;
- original boundaries, merged boundaries, and merge reasons;
- spectral energy assigned to every shell;
- `c_teacher`;
- drift from the preceding teacher: mean shift, relative spectrum change, and
  cumulative-subspace projector distance at every stable boundary.

These metrics distinguish bad corpus coverage, numerical PCA failure, spectrum
collapse, unstable boundaries, and an excessively moving teacher.

### Step and epoch metrics

Always aggregate per epoch; write step-level values at the diagnostic interval:

- total, MRL, full-task, GSR, and effective weighted GSR losses;
- per-dimension InfoNCE loss and in-batch retrieval accuracy;
- per-shell GSR loss;
- per-shell teacher energy, student energy, energy ratio, signed bias, normalized
  RMSE, and Pearson/cosine alignment of pair-distance vectors;
- cumulative prefix geometry RMSE at every stable geometry boundary;
- shell alignment matrix between every student band and teacher spectral shell,
  plus diagonal alignment and off-diagonal leakage summaries;
- pre-normalization full embedding norm;
- prefix and band energy fractions, coordinate variance range, and dead-coordinate
  fraction;
- task-gradient norm, geometry-gradient norm, and their cosine measured at the
  pooled student embeddings every `diagnostics_every_steps`;
- combined parameter-gradient norm/max/non-finite count after AMP unscale;
- learning rate, AMP scale, step time, and CUDA allocated/reserved memory;
- batch size, unordered pair count, epoch, step, global step, and sample-ID range.

Gradient conflict should be measured with `torch.autograd.grad` on the two pooled
embedding tensors, not all model parameters. This gives a useful task-vs-geometry
signal without doubling a full backward pass.

### Fixed diagnostic panel

Choose `diagnostic_samples` stable sample IDs once from the training corpus using
the experiment seed, and choose `diagnostic_pairs` stable unordered pairs from
that panel. At every teacher refresh and epoch end, compute on exactly this panel:

- shell alignment matrix;
- shell and cumulative-prefix normalized RMSE;
- student/teacher energy ratios;
- boundary leakage;
- student drift and teacher drift.

Using fixed examples prevents changing mini-batch composition from masquerading
as training progress.

### Failure dump

On the first non-finite loss/gradient or cache integrity failure, save:

- epoch/step/global step and sample IDs;
- current loss components and optimizer/AMP state metadata;
- selected student embeddings and teacher score rows;
- shell slices, eigenvalue neighborhoods, `c_teacher`, and recent step metrics;
- RNG states needed to reproduce the failing batch.

Then raise immediately. Do not continue training through NaNs.

## 7. Reporting changes

Extend `build_report` without changing historical task keys:

```json
{
  "experiment": {"method": "gsr", "...": "..."},
  "training": {"...": "..."},
  "geometry": {
    "final_teacher_refresh": 4,
    "shell_boundaries": [0, 16, 32, 64, 128, 256, 512, 768],
    "merged_boundaries": [],
    "effective_rank": 0.0,
    "per_shell": {},
    "per_prefix": {},
    "teacher_drift": {}
  },
  "classification": {},
  "sts": {},
  "pair": {},
  "summary": {},
  "table": {}
}
```

The final report should include paths to detailed diagnostic artifacts rather
than embedding full spectra or alignment matrices into `results.json`.

Add a file handler after config resolution so the same INFO/DEBUG stream shown
on the terminal is persisted to `output_dir/train.log`.

## 8. Test plan

### Loss mathematics

- full normalization preserves normalized prefix cosine;
- teacher and student shell distances sum exactly to their cumulative distances;
- PCA-rotated teacher embeddings yield zero GSR loss;
- within-shell orthogonal rotation leaves GSR unchanged;
- shell corruption increases loss;
- only student tensors receive gradients;
- finite gradients with `B=2` and with shell widths larger than `B`;
- numerical verification of the shell-to-prefix error bound;
- exhaustive small-corpus verification that the batch U-statistic is unbiased.

### Spectral teacher

- streaming covariance equals a direct centered covariance;
- descending eigenpairs reconstruct covariance within tolerance;
- analytic `c_teacher` equals explicit enumeration of all corpus pairs;
- cached scores match direct `(Q - mean) @ V`;
- shuffled cache construction still maps every row to the correct sample ID;
- duplicate/missing/out-of-range IDs fail loudly;
- tied boundary eigenvalues merge deterministically;
- non-finite embeddings and collapsed teacher geometry fail loudly;
- refresh diagnostics contain all required fields.

### Trainer integration

- `gsr` is registered and its shipped configs load;
- warmup epochs use only MRL and later epochs activate GSR;
- refresh occurs at the configured epochs only;
- teacher construction leaves the encoder in the correct train/eval mode;
- one CPU smoke epoch updates encoder parameters and writes all artifacts;
- geometry loss is finite for the existing tiny `N < D` fixture;
- wrong sample IDs cannot silently select a teacher row;
- epoch histories aggregate all component metrics;
- final report preserves existing classification/STS/pair schema;
- MRL, ESE, and MIPIC regression tests remain unchanged and pass.

## 9. Implementation sequence and gates

### Phase A — infrastructure without behavior change

1. Add sample IDs and teacher loader.
2. Add BaseTrainer lifecycle hooks and scalar aggregation.
3. Add persistent `train.log`.
4. Run the complete existing test suite.

Gate: all current tests pass; MRL smoke history and reports remain schema-compatible.

### Phase B — mathematical primitives

1. Implement GSR loss functions.
2. Implement spectral cache construction and integrity checks.
3. Implement unit tests, including exhaustive unbiasedness and PCA-zero-loss tests.

Gate: mathematical tests pass in CPU FP32/FP64 and gradients are finite.

### Phase C — trainer and configuration

1. Add `GSRConfig`, registry entry, trainer, and four configs.
2. Implement warmup, refresh, cache lookup, and combined objective.
3. Add CPU end-to-end trainer tests.

Gate: a tiny two-epoch run visibly transitions from MRL-only to MRL+GSR and
produces a finite, decreasing shell loss on the fixed diagnostic panel.

### Phase D — diagnostics and reports

1. Add teacher, step, gradient-conflict, fixed-panel, and failure diagnostics.
2. Extend histories and final report.
3. Test JSON/JSONL serialization and failure dumps.

Gate: from artifacts alone, one can determine whether a failed run was caused
by sample-cache mismatch, PCA numerics, eigengap instability, scale imbalance,
task/geometry gradient conflict, coordinate collapse, or downstream mismatch.

### Phase E — real-model smoke checks before full training

1. `--print-config` for all four GSR configs.
2. Run a small corpus cap with BERT/TinyBERT and one teacher refresh.
3. Inspect teacher integrity metrics and the fixed diagnostic panel.
4. Run one short GPU training job with evaluation disabled, then one with the
   normal epoch evaluation path.

Gate: no non-finite values, cache coverage is exact, eigensolver residuals are
small, shell targets are non-zero beyond the optimization batch rank, and all
expected artifacts are present.

## 10. Definition of done

Implementation is complete when:

- all existing and new tests pass;
- every backbone has a loadable GSR config;
- GSR trains with batch size 16 without zeroing shells wider than the batch;
- teacher statistics are deterministic for a fixed checkpoint and seed;
- task and geometry metrics are independently recoverable from artifacts;
- a non-finite or sample-alignment failure stops immediately with a reproducible
  dump;
- `results.json` remains comparable with MRL/ESE/MIPIC while linking the complete
  geometry diagnostics;
- inference uses the saved encoder exactly like ordinary MRL, with no GSR module
  or teacher cache required.
