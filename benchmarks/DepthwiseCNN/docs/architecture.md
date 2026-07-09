# DepthwiseCNN Architecture Documentation

## Overview

The `DepthwiseCNN` model family is designed for lightweight, high-performance file fragment type classification, primarily targeting scenarios with 512-byte or 4096-byte sequences. The model reduces parameters and computational complexity by decomposing standard 1D convolutions into depthwise and pointwise convolutions, and utilizes an inception-like block structure to capture features at multiple scales.

## Architecture Variants

The benchmark supports three distinct architectural variants, as described in the reference paper: **DSC**, **DSC-SE**, and **M-DSC**.

### 1. Depthwise Separable Convolution (DSC)
- **Input & Embedding**: The raw byte values (0-255) are embedded into a dense continuous space using an `nn.Embedding(256, 32)` layer.
- **Initial Convolution**: A standard 1D Convolution with kernel size 19, padding 9, and stride 2 processes the embedded sequence. This layer uses Batch Normalization and a Hardswish activation.
- **Inception Blocks**:
  - The network contains three Inception Blocks.
  - Each block splits the input into three parallel branches, applying depthwise separable convolutions with kernel sizes 11, 19, and 27, respectively.
  - The branches are normalized (BatchNorm1d) and their outputs are summed.
  - The summed output undergoes max pooling (size 4, stride 4).
  - A shortcut connection (using 1x1 convolution if channels change) is also max-pooled and added as a residual before the activation function (Hardswish).
- **Global Pooling and Classification**: 
  - After the 3 Inception blocks, the sequence spatial dimension is collapsed using Global Average Pooling.
  - A final 1x1 1D Convolution maps the features to the desired `num_classes` (e.g., 75).
  - Outputs are converted to log-probabilities via `LogSoftmax`.

### 2. DSC-SE (DSC + Squeeze-and-Excitation)
- Built on top of the standard DSC architecture.
- A **Squeeze-and-Excitation (SE) Block** is inserted immediately after each Inception Block.
- **SE Block Mechanism**: 
  - Squeezes spatial information into a channel descriptor via Global Average Pooling.
  - Learns channel-wise weights using a two-layer bottleneck structure (Linear -> ReLU -> Linear -> Sigmoid).
  - Multiplies the original feature map by these channel weights to emphasize informative features and suppress less useful ones.

### 3. Modified DSC (M-DSC)
- Designed to further optimize inference performance and stability.
- **Initial Convolution**: The first 1D convolution is replaced with a purely *depthwise* convolution.
- **Normalization**: Batch Normalization is replaced by **GroupNorm** (using 8 groups) across all layers, improving stability over small batch sizes.
- **Activation**: Hardswish activations are replaced by simpler **ReLU** activations.
- **Regularization**: A **Dropout layer (p=0.2)** is injected immediately before the final classifier to prevent overfitting.

## Framework Integration

The model operates fully within the standard DeepCarv `FragmentClassifier` protocol. 
It accepts a batch of byte values of shape `[B, L]` and yields log-probabilities of shape `[B, num_classes]`. The spatial dimension (length `L`) is fully collapsed before classification, allowing the same initialized model to technically process varying fragment sizes, although the benchmark typically initializes distinct instances for 512-byte and 4096-byte datasets.
