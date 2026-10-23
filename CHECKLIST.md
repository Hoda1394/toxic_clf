# Improvement Checklist

Work through these in order and change one thing at a time, so each change's
effect on the metrics is visible. After each step, record the val/test PR-AUC
and F1 in the results log at the bottom.

## 1. Establish a reference point

- [x] **Add a TF-IDF + logistic regression baseline** (`scripts/baseline.py`).
  Test: ROC-AUC 0.939, PR-AUC 0.694, F1 0.630 (tuned threshold). This is the
  number the transformer has to beat.
- [x] **Overfit a tiny subset as a sanity check.** Train on ~1,000 examples with
  dropout 0, no clamps, and no early stopping. Training loss should go to ~0.
  If it doesn't, there is a bug in the model or training loop.

## 2. Stop the model from being held back (underfitting)

- [x] **Lower dropout** from 0.5 to 0.1 (`src/config.py`, `ModelConfig.dropout`).
- [x] **Loosen gradient clipping** from `max_norm=0.25` to `1.0` (`src/trainer.py`,
  both the AMP and non-AMP branches).
- [x] **Remove the `torch.clamp` calls** on embeddings, normalised activations,
  pooled output and logits (`src/model.py`). Clamps pass zero gradient once a
  value hits the limit, and the ±10 logit clamp puts a floor under the loss.
- [x] **Remove the NaN checks inside the model** (`src/model.py`): the "skip the
  residual" branches and the reset of hidden states to zeros. They hide bugs
  and force a GPU sync on every layer. If NaNs appear afterwards, find the
  cause (usually the learning rate) instead of masking it.
- [x] **Simplify the NaN handling in the trainer** (`src/trainer.py`). Keep at
  most a single loss check, and make it fail loudly rather than skipping batches.

## 3. Learning-rate schedule

- [x] **Replace `ReduceLROnPlateau`** with linear warmup (~5% of total steps)
  followed by cosine or linear decay, stepped every batch
  (e.g. `transformers.get_cosine_schedule_with_warmup`, or `LambdaLR`).
- [x] **Remove the manual warmup code** in `Trainer.train_epoch` once the
  scheduler handles warmup.
- [x] **Try a few peak learning rates** (e.g. 3e-4, 5e-4, 1e-3 with AdamW) on a
  fraction of an epoch and keep the best.

## 4. Evaluation and model selection

- [x] **Add ROC-AUC and PR-AUC** to `Trainer.evaluate` (`src/trainer.py`), using
  `sklearn.metrics.roc_auc_score` and `average_precision_score`. Collect the
  probabilities, not just the 0/1 predictions.
- [x] **Select checkpoints and early-stop on PR-AUC** instead of F1 at 0.5. It is
  much less noisy from epoch to epoch.
- [x] **Tune the decision threshold on validation** after training, and apply
  that threshold to the test set (`scripts/train.py` and `scripts/evaluate.py`).
- [ ] **Use the official labelled hold-out set** (`data/test_public_expanded.csv`
  or `data/test_private_expanded.csv`) as the final test set instead of a
  random split of `train.csv`. Check its column names first.
- [ ] **Remove texts duplicated between the splits.** About 1.7% of validation
  texts also appear in the training set.

## 5. Tokenizer and input pipeline

- [x] **Pad each batch to its own longest sequence** instead of always to 256.
  The median comment is 36 tokens (99th percentile 181), so most of the
  compute is currently spent on padding. Do the padding in `collate_fn`.
- [x] **Group batches by length** (`LengthGroupedBatchSampler` in `src/data.py`).
  Random batches still pad to ~175 tokens; grouped batches pad to ~1% above
  the real length. Word-tokenizer epoch: 38 min → ~18 min.
- [x] **Implement a subword tokenizer:** byte-level BPE (`BPETokenizer` in
  `src/data.py`), trained on the training split, selected with
  `--tokenizer bpe` and saved as `tokenizer.json` in the run folder.
- [ ] **Run the BPE experiment** on the full data with the same settings as the
  step 3 run (lr 5e-4, 4 epochs, vocab 10,000) and compare.
- [x] **`max_seq_len`: keep 256.** With word tokens 192 would cover 99%, but BPE
  tokens are ~35% longer (99th percentile 248), so 192 would truncate 7.8% of
  comments. 256 covers 99.5% with BPE.

## 6. Labels and loss

- [ ] **Compare with and without `pos_weight`.** With threshold tuning in place,
  the class weighting may no longer be needed, and without it the predicted
  probabilities are better calibrated.
- [ ] **Train on soft labels:** use the continuous `target` score (0–1) as the
  BCE target instead of the 0/1 label, and keep the 0/1 label for metrics.
  About a third of the "toxic" examples have a target below 0.6.

## 7. Scale up

- [ ] **Train a larger model** once steps 2–5 are done: e.g. 4 layers,
  `embedding_dim` 256, `dim_feedforward` 1024 (the `medium` preset), roughly
  5–10M parameters. 1.26M training examples is plenty for this size.
- [ ] **Fine-tune a small pretrained model** (DistilBERT or MiniLM) on the same
  split as a reference for what is achievable (expect roughly 0.70 F1). The gap
  to your from-scratch model is a lesson in itself.

## 8. Clean-up

- [ ] **Fix AMP on Apple Silicon:** `torch.cuda.amp` doesn't work on MPS. Use
  `torch.autocast(device_type=...)`, or disable `use_amp` on MPS.
- [ ] **Remove the unused `compute_class_weights`** call in `scripts/train.py`
  (and the function, if nothing else uses it). Its test,
  `tests/test_class_weights.py`, already fails: it asserts `neg / (total / 2)`
  instead of `total / (2 * neg)`. Remove it together with the function.
- [ ] **Keep the padding embedding at zero:** the custom init in
  `ToxicCommentTransformer._init_weights` overwrites the `padding_idx` row.
- [ ] **Fix the README:** the learning rate is listed as `1e-7`, and the
  documented default target column is `toxicity` but the data uses `target`.

## Results log

| Step | Change | Val PR-AUC | Val F1 (tuned) | Test PR-AUC | Test F1 (tuned) | Notes |
|---|---|---|---|---|---|---|
| 1 | TF-IDF + LR baseline (400k rows) | 0.698 | 0.633 | 0.694 | 0.630 | threshold 0.26 |
| – | Current transformer (best run) | – | – | – | 0.623 (at 0.5) | before any changes; 759k params (64-dim) |
| 2 | Dropout 0.1, clip 1.0, no clamps/NaN checks; lr 1e-3 constant | – | – | – | 0.573 (at 0.5) | run `2026-10-13_12-06-52`; 1.9M params (128-dim); train loss rose every epoch → lr too high; stopped at epoch 4, best epoch 1 |
| 3 | Warmup (5%) + cosine decay, peak lr 5e-4, 4 epochs | 0.694 | 0.632 | 0.696 | 0.634 | run `2026-10-16_13-41-27`; best epoch 4; F1 at 0.5: 0.617 test; tuned threshold 0.80; ROC-AUC 0.936 test; train loss 0.78 → 0.61, val loss rose after epoch 2 |
