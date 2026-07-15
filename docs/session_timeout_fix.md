# Surviving the Kaggle 12-Hour Session Timeout

## Problem

Full training runs (notably ByteNet) exceeded Kaggle's 12-hour session limit.
The session was killed **mid-epoch**, so that epoch's work was lost and there
was no clean way to continue — effectively meaning long runs could never finish.

## What was already there — and what was missing

`Trainer` already saved `checkpoint_last.pt` every epoch and had a
`resume_from()` method, so this looked solved. It wasn't. `resume_from()`
restored only **model + optimizer weights**. Everything else silently reset:

| State | Consequence of losing it |
|---|---|
| LR scheduler position | The schedule restarts. A cosine/plateau schedule that had decayed to 1e-5 jumps back to 3.75e-4 — the model gets kicked out of its minimum. |
| Best-metric tracker | The resumed run thinks *nothing* is the best yet, so the **first** epoch it runs overwrites `checkpoint_best.pt` — potentially replacing a good model with a worse one. |
| Early-stopping counters | `epochs_no_improve` resets to 0, so the patience window starts over and training can run far past where it should have stopped. |
| Training history | Loss/accuracy curves lose everything before the resume. |
| AMP grad scaler | Scale factor resets, causing avoidable gradient over/underflow after the resume. |

So a "resume" would have quietly produced a *worse* result than an uninterrupted
run, in ways that are hard to notice from the final numbers.

## Fix

### 1. Full resumable state

`checkpoint_last.pt` now carries the scheduler state, AMP scaler state,
best-metric + best-epoch, early-stopping counters, and the full history.
`Trainer.resume_from()` restores all of it.

`EarlyStopping` gained `state_dict()` / `load_state_dict()`.

### 2. `Trainer.fit_or_resume()`

One idempotent call:

* resumes from `checkpoint_last.pt` if present,
* starts fresh if not,
* **no-ops** if training already reached `epochs`.

Wired into the CarveFormer script, the DepthwiseCNN script, and the unified
runner. Re-running the training cell after a timeout Just Works.

### 3. Wall-clock budget — `max_hours`

The real fix for the timeout. Before starting each epoch, the trainer estimates
whether the next epoch will fit in the remaining budget (using the mean epoch
time so far). If not, it stops **at an epoch boundary**, where a checkpoint has
just been written.

```yaml
training:
  max_hours: 11.0   # on a 12h Kaggle session; leaves ~1h for eval + upload
```

This converts *"the session was killed mid-epoch and we lost that work"* into
*"the session ended cleanly; re-run the cell to continue."*

Log on hitting the budget:

```
WARNING Time budget reached (10.94 h of 11.00 h used; next epoch would need
~0.62 h). Stopping cleanly at epoch 34/50. 16 epoch(s) remain — rerun to
resume from checkpoint_last.pt in a fresh session.
```

### 4. Throughput

Both notebooks now set:

* `num_workers=4` — matters far more now that data is memory-mapped off disk
  rather than sitting in RAM; workers overlap I/O with compute.
* `amp=True` — mixed precision, typically ~1.5–2× on a T4.
* `pin_memory` / `prefetch_factor` — already supported by `build_dataloader`,
  now actually used.

## How to run a long job

1. Run the notebook. It trains until `MAX_HOURS`, then stops cleanly.
2. If it didn't finish, start a **fresh session**, run the setup cells, and
   **re-run the training cell**.
3. Repeat until it reports training is complete. Checkpoints live in
   `/kaggle/working`, which persists as notebook output.

## Verified

* Resume across **separate processes**: session 1 ran 3 epochs; a fresh process
  resumed at epoch 4, and the final history contained all 6 epochs — with the
  best-metric and decayed LR from session 1 preserved.
* Idempotency: re-running a completed job trains 0 extra epochs.
* Time budget: with a deliberately tiny budget, training stopped early, set
  `stopped_on_time_budget`, and left a resumable checkpoint.
* 36/36 tests pass; ruff clean; mypy clean.

## Note on what this does *not* solve

If the run is **GPU-bound** rather than data-bound, this makes long runs
*finishable*, not *faster*. Worth checking `nvidia-smi` GPU utilisation during
training:

* **Low utilisation (~30–50%)** → the GPU is starved; the `num_workers` / AMP
  changes above should give a real speedup.
* **Pegged (~95%+)** → genuinely compute-bound; the checkpoint/resume route is
  the fix, plus considering fewer epochs with early stopping (the accuracy curve
  usually flattens well before epoch 50).
