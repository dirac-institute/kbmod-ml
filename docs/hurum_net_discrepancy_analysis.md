# HurumNet / Notebook Output Discrepancy Analysis

This document summarizes likely reasons why the self-contained notebook model in
`src/kbmod_ml/models/kbmod.ipynb` and the Hyrax-converted model in
`src/kbmod_ml/models/hurum_net.py` can produce different output values.

The analysis is based on static inspection of the checked-in model, dataset, and
configuration code. The original runtime data and model checkpoints were not
available in this environment.

## Executive Summary

The most likely causes of divergent outputs are:

1. **Input-channel mismatch**: the Hyrax model hard-codes one input channel,
   while the notebook model is defined and instantiated with three input
   channels.
2. **Preprocessing mismatch**: the notebook applies per-stamp z-score
   normalization, while the Hyrax dataset applies percentile clipping,
   min-subtraction, and sum normalization to selected coadd channels.
3. **Output interpretation mismatch**: the notebook compares class-1 softmax
   probabilities, while the converted Hyrax model returns raw logits from
   `forward()`.
4. **Training-loop mismatch** if the Hyrax model was retrained: the notebook
   uses a balanced sampler, gradient clipping, AdamW, cosine scheduling, and
   validation-AUC checkpointing; the active converted Hyrax class delegates much
   of that behavior to Hyrax configuration and does not implement all of the
   notebook behavior directly.

Any one of the first three issues can change numerical outputs. Together, they
are sufficient to explain substantially different predictions.

## Detailed Findings

### 1. The Hyrax model uses one input channel, while the notebook model uses three

In `hurum_net.py`, `KBModNet.__init__()` hard-codes:

```python
in_channels = 1
```

and uses that value to construct the first convolution:

```python
nn.Conv2d(in_channels, base, 3, padding=1, bias=False)
```

In the notebook, `KBModNet` defaults to `in_channels=3`, and the training block
explicitly instantiates:

```python
model = KBModNet(in_channels=3, base=32, dropout=0.3)
```

This means the two models are not architecturally identical. If the notebook
checkpoint was trained with three channels, then a one-channel Hyrax model cannot
be expected to reproduce the same logits or probabilities.

#### Recommended checks

- Confirm the shape of the tensor fed to the notebook model during inference.
- Confirm the shape of the tensor fed through Hyrax immediately before
  `KBModNet.forward()`.
- If the notebook checkpoint is authoritative, make the Hyrax model use the same
  channel count as the checkpoint.

### 2. The Hyrax dataset preprocessing does not match the notebook preprocessing

The notebook normalization function does the following per sample:

1. Casts stamps to `float32`.
2. Flattens each sample.
3. Computes the per-sample mean and standard deviation.
4. Replaces zero standard deviations with `1.0`.
5. Returns `(stamp - mean) / std`, reshaped to the original sample shape.

The notebook applies that same normalization during training dataset creation and
during standalone inference.

The Hyrax dataset in `src/kbmod_ml/data_sets/kbmod_stamps.py` uses a different
normalization pipeline:

1. Selects active coadd columns.
2. Replaces non-finite values with the stamp mean or zero.
3. Computes percentiles and a robust `sigmaG` estimate.
4. Clips low pixels.
5. Subtracts the minimum.
6. Divides by the sum of the stamp.
7. Replaces non-finite values again.

That is not numerically equivalent to the notebook's z-score normalization. Even
with the same weights and architecture, feeding differently normalized inputs
will generally produce different logits and probabilities.

#### Recommended checks

- Choose a small fixed set of raw stamps.
- Run the notebook normalization on those stamps.
- Run the Hyrax dataset normalization on the same stamps.
- Compare min, max, mean, standard deviation, and a few pixel values.
- If the notebook behavior is desired, port the notebook `normalize_stamps()`
  logic into the Hyrax data path or a model input-preparation hook.

### 3. The Hyrax data path appears to select only the `mean` coadd column

`KbmodStamps` defines a mapping for four coadd types:

- `median`
- `mean`
- `sum`
- `var_weighted`

but the active-column loop currently only includes:

```python
for c in ["mean"]:
```

Therefore the Hyrax data path produces one selected channel. This matches the
one-channel hard-coded Hyrax model, but it does not match the notebook model
instantiation with `in_channels=3`.

If the notebook model was trained on three channels, the Hyrax data path is
throwing away information that the notebook model used.

#### Recommended checks

- Determine which coadd channels were actually present in `tp` and `fp` when the
  notebook was trained.
- Determine whether the intended model should consume one, three, or four
  channels.
- Make `KbmodStamps.active_columns` and `KBModNet.in_channels` agree with the
  trained checkpoint.

### 4. The notebook reports class-1 softmax probabilities; the Hyrax model returns raw logits

The notebook inference path computes:

```python
F.softmax(model(x.to(device)), dim=1)[:, 1]
```

and treats that value as `P(real)`.

The converted Hyrax model's `forward()` returns:

```python
return self.head(x)
```

That is a two-logit tensor, not a scalar class-1 probability. If the Hyrax output
being compared is the direct model output, it will not match the notebook's
reported probability.

#### Recommended checks

- Compare raw logits from both models first.
- Then compare `softmax(logits, dim=1)[:, 1]` from both models.
- Avoid comparing notebook probabilities to Hyrax logits.

### 5. The converted `train_step()` uses one-hot targets instead of integer labels

The notebook focal loss passes integer labels directly to `F.cross_entropy()`:

```python
loss = focal_loss(model(x), y)
```

The converted Hyrax `train_step()` constructs one-hot targets first:

```python
targets = torch.zeros((len(labels), 2))
targets[labels == 0, 0] = 1
targets[labels == 1, 1] = 1
loss = focal_loss(outputs, targets)
```

Recent PyTorch versions support probability targets for `cross_entropy()`, so
one-hot targets can be mathematically close to integer class labels. However,
this is still a behavioral difference and can become problematic if label shapes,
dtypes, devices, or PyTorch versions differ.

There is also a device-risk in the converted implementation: `targets` is created
on the default device. If `outputs` are on GPU and `targets` remain on CPU,
training will fail or require implicit handling elsewhere.

#### Recommended checks

- Keep targets as integer class labels if exact notebook behavior is desired.
- Ensure labels and any constructed targets are on the same device as `outputs`.
- Confirm that Hyrax collation returns labels in the expected shape and dtype.

### 6. The notebook clips gradients; the converted Hyrax `train_step()` does not

The notebook training loop calls:

```python
nn.utils.clip_grad_norm_(model.parameters(), 1.0)
```

before `optimizer.step()`.

The active Hyrax `train_step()` calls backward and optimizer step without
gradient clipping.

This only matters if the Hyrax model is being retrained. If it is, the resulting
weights can diverge from the notebook-trained weights.

#### Recommended check

- Add equivalent gradient clipping to the Hyrax training path if retraining is
  intended to reproduce the notebook.

### 7. The optimizer and scheduler are defined in the notebook but not in the active Hyrax model class

The notebook uses:

- `torch.optim.AdamW`
- `lr=3e-4`
- `weight_decay=1e-4`
- `torch.optim.lr_scheduler.CosineAnnealingLR`
- `eta_min=1e-6`
- 50 epochs in the shown training call

The converted Hyrax class relies on `self.optimizer` in `train_step()`, but the
active class does not define optimizer or scheduler setup. A commented notebook
training loop remains in `hurum_net.py`, but commented code is not used by Hyrax.

If Hyrax retraining uses a different optimizer, learning rate, scheduler,
weight-decay value, epoch count, or checkpoint-selection metric, the model will
not reproduce the notebook checkpoint.

#### Recommended checks

- Inspect the Hyrax training configuration used for this run.
- Confirm optimizer type, learning rate, weight decay, scheduler, number of
  epochs, batch size, and checkpoint selection criteria.
- If reproducibility is required, encode the notebook's optimizer/scheduler setup
  in the Hyrax config or model integration.

### 8. The notebook uses class-balanced sampling

The notebook's `make_loader()` uses `WeightedRandomSampler` when labels are
provided and balancing is enabled. It assigns inverse-frequency weights to real
and bogus examples.

That is important because the dataset is highly imbalanced. If Hyrax training
uses normal shuffling without the same sampling strategy, the learned weights and
output calibration can differ significantly.

#### Recommended checks

- Confirm whether the Hyrax DataLoader uses class-balanced sampling.
- If not, add an equivalent sampler or other balancing mechanism.

### 9. The notebook uses a bespoke false-positive subsample and split procedure

The notebook constructs a specific false-positive subset:

```python
N_FP_TARGET = 300_000
N_FP_7S = 157_900
rng = np.random.default_rng(42)
idx_5s = rng.choice(...)
fp_sub = np.concatenate([fp[:N_FP_7S], fp[idx_5s]], axis=0)
```

It then separately permutes true-positive and false-positive arrays before
combining them into train, validation, and test sets.

A generic Hyrax split will not match this unless it explicitly reproduces the
same subsampling, random seed, ordering, and class-wise split strategy.

#### Recommended checks

- Verify the exact files and sample indices used by the Hyrax run.
- Verify whether Hyrax uses the same 7-sigma and 5-sigma false-positive mix.
- Verify the train/validation/test indices match the notebook when reproducing
  notebook results.

### 10. The checked-in default config points to different data files than the notebook

The checked-in default config references:

```toml
true_positive_file_name = 'true_positive_stamps_full.npy'
false_positive_file_name = 'false_positive_stamps_trimmed.npy'
```

The notebook creates and trains/evaluates on:

```python
tp_combined_5_7sigma.npy
fp_combined_5_7sigma.npy
```

If the Hyrax run used the default config, it may not be using the same data as
the notebook.

#### Recommended checks

- Confirm the actual Hyrax config used for the converted model run.
- Confirm that it points to the same combined 5-sigma/7-sigma arrays if that is
  the intended comparison.
- Confirm any trimming/subsampling is identical.

## Suggested Minimal Reproduction Procedure

To isolate whether the difference comes from architecture/weights or from the
Hyrax input pipeline:

1. Load a tiny batch of raw stamps.
2. Apply the notebook `normalize_stamps()` function manually.
3. Convert the result to a `torch.float32` tensor with the exact channel shape the
   notebook expects.
4. Run the notebook model and collect raw logits.
5. Instantiate the Hyrax model with the same architecture and load the same
   checkpoint.
6. Feed the exact same tensor directly into `KBModNet.forward()`, bypassing the
   Hyrax dataset.
7. Compare raw logits.
8. Compare `F.softmax(logits, dim=1)[:, 1]`.
9. Only after raw tensor inference matches should the Hyrax dataset/pipeline be
   reintroduced.

This procedure separates model equivalence from data-pipeline equivalence.

## Recommended Fix Order

1. **Decide the authoritative input shape** from the trained notebook checkpoint.
2. **Make `hurum_net.py` match that input shape** instead of hard-coding a
   different number of channels.
3. **Make Hyrax preprocessing match the notebook normalization** if the notebook
   output is the expected reference.
4. **Compare the same output type**: logits to logits, or softmax probabilities
   to softmax probabilities.
5. **If retraining in Hyrax**, reproduce the notebook's sampler, optimizer,
   scheduler, gradient clipping, split/subsample procedure, and checkpoint metric.

## Most Likely Root Cause

The strongest explanation is a combination of:

- the Hyrax model using one input channel while the notebook model uses three,
- the Hyrax dataset using a different normalization strategy from the notebook,
- and the notebook comparing class-1 softmax probabilities while the Hyrax model
  directly returns logits.

Those differences should be resolved before investigating smaller numerical or
framework-level causes.
