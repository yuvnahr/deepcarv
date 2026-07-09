# CarveFormer Architecture

Implementation of **CarveFormer** from *Transformer-Based File Fragment Type
Classification for File Carving in Digital Forensics* (Guzhov & Wirth, DFKI,
ECCWS 2025), integrated into the DeepCarv benchmark framework.

## Core idea

CarveFormer adapts **Swin Transformer V2 Tiny** to raw byte fragments. The
paper's single architectural change is to replace Swin's image convolutional
patch-embed front-end with a **96-dimensional byte embedding**, then reshape
the embedded byte sequence into a 2D token grid so the Swin backbone can
process it. Everything after the front-end (the Swin V2 stages, norm, and
classification head) is unchanged.

## Data flow

```
bytes  x : [B, L]                        (L = 512 or 4096, values 0..255)
   |
   |  nn.Embedding(256, 96)              ByteEmbeddingPatchifier
   v
   [B, L, 96]
   |
   |  LayerNorm(96)                      (mirrors Swin's post-patch-embed norm)
   |  reshape -> [B, H, W, 96]           H*W == L  (grid from fragment size)
   v
   [B, H, W, 96]                         == the token grid Swin's stages expect
   |
   |  SwinV2-Tiny stages (unchanged)     timm swinv2_tiny_window8_256
   v
   [B, H', W', 768]
   |
   |  head: global avg pool + Linear     -> [B, num_classes]
   |  log_softmax
   v
   log-probabilities [B, num_classes]
```

## How this maps onto timm's SwinV2-Tiny

Stock `swinv2_tiny_window8_256` does `Conv2d(3, 96, kernel=4, stride=4)` on a
256×256 image, producing a **64×64 grid of 96-d patch tokens**, then runs the
Swin stages, ending at `[B, 8, 8, 768]` and a pooled classifier head.

CarveFormer keeps the stages, norm, and head untouched and swaps only the
`patch_embed` for `ByteEmbeddingPatchifier`, which produces the *same*
`[B, H, W, 96]` token layout directly from bytes. Because the Swin V2 stages
bake resolution-dependent window-attention masks at construction time, the
backbone is built with an `img_size` chosen so its native patch grid equals the
byte-token grid (`img_size = (H*4, W*4)`, since the native patch size is 4).
timm interpolates the pretrained relative-position tables to that resolution
automatically.

## Reshape grid

| Fragment size | Grid (H×W) | Notes |
|---|---|---|
| 4096 | 64×64 | Matches SwinV2-Tiny's native token grid exactly. |
| 512 | 16×32 | Paper does not specify the 512 grid; this is a documented default. |

Both dimensions are multiples of 8 so the four Swin downsampling stages stay
well-defined. The reshape lives in one isolated helper
(`default_grid_for_length` / `ByteEmbeddingPatchifier`) so the grid can be
changed later without touching the rest of the model. An explicit `grid_hw`
config override is supported.

## Framework integration

- `src/model.py` — framework-independent model + `build_carveformer` factory.
  Returns `log_softmax` to match `FragmentClassifier` (default loss `nll_loss`).
- `src/adapter.py` — thin `CarveFormerAdapter(FragmentClassifier)` wrapping the
  model; config-driven fragment size / class count.
- `src/models/adapters/carveformer_adapter.py` (framework) — registry hook;
  loads the real adapter from this benchmark dir, with graceful fallback to the
  `NotAvailableModel` stub if the code or `timm` is unavailable. Registry key
  `"carveformer"` unchanged.
- `scripts/train.py` / `scripts/evaluate.py` — thin entrypoints using the shared
  `Trainer` / `Evaluator`. No duplicated training logic.

## Paper hyperparameters (in `configs/benchmark.yaml`)

AdamW, learning rate 3.75e-4, weight decay 0.05, 50 epochs, effective batch
size 1024, ImageNet1k-pretrained backbone, best checkpoint by validation
accuracy.

## Paper-grounded sanity targets (FFT-75)

| Scenario | 512 B | 4096 B |
|---|---|---|
| #1 | 72.10% | 82.99% |
| #2 | 90.62% | 93.96% |
| #3 | 93.44% | 96.87% |

The goal is a faithful, runnable benchmark landing in the same general range
and preserving the paper's trends — not bit-exact reproduction.

## Assumptions and deviations (documented, not silent)

1. **512-byte reshape grid (16×32).** The paper specifies the 2D reshape only
   implicitly; 4096→64×64 is natural and matches Swin's native grid, but the
   512 grid is a choice. Isolated and overridable via `grid_hw`.
2. **Effective batch size 1024.** The shared `Trainer` has no gradient
   accumulation, so `batch_size` in the config is the *real* batch the GPU must
   hold. Reaching the paper's effective 1024 needs a large-memory GPU or a
   future accumulation feature in the shared trainer. `grad_accum` is present in
   the config as documentation of intent.
3. **Scheduler.** The paper warms up then decays; the closest built-in shared
   scheduler is `cosine`, used here. Exact warmup schedule is not reproduced.
4. **Pretrained-weight fallback.** If ImageNet1k weights cannot be downloaded
   (offline/CI), the backbone falls back to random init with a warning — the
   architecture is unchanged, only initialization differs.

## Known external dependency

The Swin V2 backbone requires `timm` (`pip install "timm>=1.0.0"`), already in
the project requirements. Without it the registry hook falls back to the stub.

## Related framework note (out of scope for this branch)

`src/data/dataset.py` currently validates NPZ keys against lowercase
`{'x','y'}` and reads `data['x']`, while the documented format (and its own
docstring) is uppercase `X`/`y`. This is a pre-existing framework bug unrelated
to CarveFormer; the Kaggle notebook's verification cell accepts either case
defensively, but the loader should be fixed separately so real uppercase-`X`
FFT-75 files load.
