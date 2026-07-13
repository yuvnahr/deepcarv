# CarveFormer — Paper Notes

**Paper:** Guzhov & Wirth, *Transformer-Based File Fragment Type Classification
for File Carving in Digital Forensics*, ECCWS 2025 (DFKI). Funded by BMBF
project Carve-DL.

## What the paper does

Adapts **Swin Transformer V2 Tiny** (ImageNet1k-pretrained) to raw byte
fragments for file-fragment type classification, evaluated on **FFT-75**. The
one architectural change: replace the image conv/norm patch-embed with a 96-d
byte **embedding** layer, then reshape the embedded sequence into a pseudo-2D
tensor for the Swin backbone.

## Architecture

- Backbone: Swin Transformer V2 Tiny (~28M params), 4 stages, channel dims
  [96, 192, 384, 768], window-based shifted self-attention with scaled-cosine
  attention and log-spaced continuous position bias.
- Front-end: `Embedding(256, 96)` replacing the conv patch-embed.
- Reshape: embedded `[L, 96]` → 2D grid (`H*W = L`). 4096 → 64×64 (native Swin
  grid). 512 grid unspecified in the paper.
- Head: global average pool + linear → num_classes; dropout before the head.

## Training

- AdamW, lr 3.75e-4, weight decay 0.05.
- Effective batch size 1024.
- 50 epochs; best model by validation accuracy.
- Cross-entropy loss. ImageNet1k-pretrained initialization.

## Reported accuracy (FFT-75)

| Scenario | 512 B | 4096 B |
|---|---|---|
| #1 (75 classes) | 72.10% | 82.99% |
| #2 (11 classes) | 90.62% | 93.96% |
| #3 (25 classes) | 93.44% | **96.87%** (SoTA) |

Best at 512 B on the hardest scenarios (#1, #2); competitive at 4096 B.

## Stated limitations (preserved in our implementation notes)

- Transformer quadratic cost limits scaling as fragment size grows; 4096-byte
  handling is memory-intensive.
- FFT-75 uses randomly sampled fragments (no realistic fragmentation patterns),
  and has intrinsic label ambiguity for container formats (e.g. data embedded
  in PDF), capping achievable accuracy (~73% @512, ~84% @4096 on Scenario #1).

## FFT-75 scenarios (class counts)

#1: 75 · #2: 11 · #3: 25 · #4: 5 · #5: 2 · #6: 2.
