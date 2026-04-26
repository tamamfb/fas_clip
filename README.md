# Fine-Tuning CLIP for Face Anti-Spoofing: Multi-Level Features, Progressive Training, and Ensemble Inference

> **Implementation of [I-FAS (AAAI 2025)](https://arxiv.org/abs/2501.01720)** adapted for a 6-class Kaggle competition.  
> Backbone: CLIP ViT-L/14@336px · Connector: GAC (depth=4, queries=96) · GPU: NVIDIA H200 141 GB

> Datasets: https://drive.google.com/drive/folders/1S2KaTlW31bHUQS8_i56MuMM9aQypNj1X?usp=sharing

---

## Table of Contents

1. [Problem Overview](#1-problem-overview)
2. [Why CLIP as a Backbone](#2-why-clip-as-a-backbone)
3. [Architecture: The Globally Aware Connector (GAC)](#3-architecture-the-globally-aware-connector-gac)
4. [Loss Functions](#4-loss-functions)
5. [Data Augmentation](#5-data-augmentation)
6. [Progressive Fine-Tuning Strategy](#6-progressive-fine-tuning-strategy)
7. [Pseudo-Labeling (3 Rounds)](#7-pseudo-labeling-3-rounds)
8. [Inference: TTA + 3-Branch Ensemble](#8-inference-tta--3-branch-ensemble)
9. [Key Implementation Notes](#9-key-implementation-notes)
10. [Results](#10-results)
11. [What I Would Try Next](#11-what-i-would-try-next)

---

## 1. Problem Overview

Face Anti-Spoofing (FAS) detects whether a face presented to a camera is real or a presentation attack. Attacks range from a printed photo to a 3D silicone mask — each requiring a different detection strategy.

**6 classes in this competition:**

| Label | Description |
|---|---|
| `realperson` | Genuine live face (including faces wearing cloth/medical masks) |
| `fake_printed` | Photo of a face **held in front** of the camera |
| `fake_screen` | Face replayed on a digital display |
| `fake_mask` | 3D mask or photo **physically attached** to a face |
| `fake_mannequin` | Plastic dummy or mannequin head |
| `fake_unknown` | Unknown or ambiguous attack type |

> **Competition-specific finding (validated empirically):** `fake_mask` covers photos physically *attached* to a face, while `fake_printed` covers photos *held* in front of a face. This distinction was discovered through Kaggle submission feedback — not from the paper — and corrects the intuitive but incorrect categorization.

---

## 2. Why CLIP as a Backbone?

CLIP (`ViT-L/14@336px`) was pre-trained on 400 million image-text pairs, giving it two properties particularly valuable for FAS:

1. **Generalizable representations** — it understands semantic content, not just low-level texture patterns, which helps across attack types it has never seen.
2. **Multi-scale feature richness** — different transformer depths encode different information: early layers capture texture/edges (moiré patterns, paper blur), deep layers capture semantic structure (mask shape, mannequin face geometry). FAS benefits from *both*.

The visual encoder is a Vision Transformer with 24 transformer blocks, 1024-dim hidden states, and 336×336 input. Each image produces 576 patch tokens + 1 CLS token per block.

---

## 3. Architecture: The Globally Aware Connector (GAC)

The standard approach takes only the final CLS token from the last block. The I-FAS paper's **Globally Aware Connector** instead uses CLS tokens from **all 24 layers** simultaneously.

### Why Multi-Level Features?

> "Different layers of the visual encoder exhibit distinct biases: shallow layers capture low-level information, while deep layers are proficient in semantic comprehension." — I-FAS paper

Print and screen attacks are primarily characterized by low-level texture cues (moiré, blur). Mask and mannequin attacks have structural/semantic differences. Using all 24 layers lets the model attend to both simultaneously.

### Capturing CLS Tokens via Forward Hooks

```python
class CLIPExtractor(nn.Module):
    def __init__(self, model_name='ViT-L/14@336px'):
        super().__init__()
        self.clip, _ = clip.load(model_name, device=DEVICE, jit=False)
        self.clip = self.clip.float()
        for p in self.clip.parameters():
            p.requires_grad = False
        self._cls_buf = []
        for block in self.clip.visual.transformer.resblocks:
            block.register_forward_hook(self._hook_fn)
        self.embed_dim  = self.clip.visual.transformer.width   # 1024
        self.num_layers = len(self.clip.visual.transformer.resblocks)  # 24

    def _hook_fn(self, module, inp, out):
        x = out[0] if isinstance(out, tuple) else out
        self._cls_buf.append(x[0])   # x[0] = CLS token → [B, D]
```

After one forward pass, `torch.stack(self._cls_buf, dim=1)` gives shape `[B, 24, 1024]`.

### GACLayer

Each GAC layer applies three sequential operations with residual connections and **DropPath** (stochastic depth):

```
Q [B, num_queries+24, D]
    │
Self-Attention  (queries attend to each other)
    │
Cross-Attention (queries attend to local patch features [B, 576, D])
    │
Feed-Forward    (MLP with GELU)
    │
Q' [B, num_queries+24, D]
```

DropPath probability increases linearly from 0 at layer 0 to `drop_path=0.1` at layer 3. This forces the model not to rely on any single residual path.

```python
class GACLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_ratio=4.0, dropout=0.1, drop_path=0.0):
        super().__init__()
        self.ln_sa = nn.LayerNorm(embed_dim)
        self.sa    = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.ln_ca = nn.LayerNorm(embed_dim)
        self.ln_xv = nn.LayerNorm(embed_dim)
        self.ca    = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        mlp_d = int(embed_dim * mlp_ratio)
        self.ln_ff = nn.LayerNorm(embed_dim)
        self.ff    = nn.Sequential(
            nn.Linear(embed_dim, mlp_d), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(mlp_d, embed_dim), nn.Dropout(dropout))
        self.dp = DropPath(drop_path)

    def forward(self, q, xv):
        q_n = self.ln_sa(q); sa_o, _ = self.sa(q_n, q_n, q_n); q = q + self.dp(sa_o)
        ca_o, _ = self.ca(self.ln_ca(q), self.ln_xv(xv), self.ln_xv(xv)); q = q + self.dp(ca_o)
        return q + self.dp(self.ff(self.ln_ff(q)))
```

### Full Architecture Flow

```
Input [B, 3, 336, 336]
       │
  CLIPExtractor (ViT-L/14@336px, hooks on all 24 blocks)
  ├── cls_tokens   [B, 24, 1024]   ← CLS from each of 24 layers
  ├── local_feats  [B, 576, 1024]  ← patch tokens from final layer
  └── final_cls    [B, 1024]       ← ln_post(CLS of last layer)
       │
  EnhancedGAC (depth=4, num_queries=96, heads=8, drop_path=0.1)
  ├── learnable queries [B, 96, 1024]    (nn.Parameter × 0.02 init)
  ├── cls_proj(cls_tokens) [B, 24, 1024]
  ├── concat → Q [B, 120, 1024]
  └── 4 × GACLayer (SA → CA with local_feats → FF) → Q' [B, 120, 1024]
       │
  FASClassifier
  ├── AdaptiveAvgPool1d(1) → pooled [B, 1024]
  ├── LayerNorm(pooled + final_cls)
  ├── MLP: 1024 → 512 → 256 → 6   (GELU, Dropout=0.4)
  └── Shortcut: Linear(1024, 6)
       │
  logits [B, 6]  +  proj_head [B, 256] (for SupCon)
```

The **shortcut** (`Linear(1024, 6)` added directly to MLP output) ensures gradient flow even when MLP layers are still poorly initialized early in training.

The **proj_head** (`Linear → GELU → Linear → L2-normalize`) produces 256-dim embeddings used only for Supervised Contrastive Loss and is discarded at inference.

---

## 4. Loss Functions

### Primary: Combined Classification Loss

```python
loss_cls = 0.6 × FocalLoss(γ=2) + 0.4 × WeightedCrossEntropy
```

**Focal Loss** reweights by prediction confidence: `(1 - p_t)^γ × CE`. With γ=2, hard samples receive up to 4× more gradient signal. This is critical for `fake_unknown` and `fake_mannequin` which are visually ambiguous and scarce.

**Weighted CE** uses per-class weights from `sklearn.compute_class_weight('balanced')`, computed on the training set. This directly counters the class imbalance (realperson: 424 samples, fake_printed: 142).

Both use `label_smoothing=0.08` to prevent overconfident predictions.

### Secondary: Supervised Contrastive Loss (SupCon)

```python
L_supcon = -1/|P(i)| × Σ_{p∈P(i)} log [ exp(z_i·z_p / τ) / Σ_{a≠i} exp(z_i·z_a / τ) ]
```

Temperature `τ=0.07` makes the loss sensitive to small cosine similarity differences, producing tightly clustered and well-separated class representations in the 256-dim embedding space.

**Total training loss:**

```python
loss = CombinedLoss(logits, labels) + 0.2 × SupConLoss(proj, labels)
```

> **Note:** SupCon is disabled when MixUp is active in that batch. Mixed-label samples cannot form valid positive pairs for contrastive learning.

---

## 5. Data Augmentation

### Training Pipeline

```python
A.RandomResizedCrop(size=(336, 336), scale=(0.5, 1.0))   # wide scale forces partial-face detection
A.HorizontalFlip(p=0.5)
A.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.3, hue=0.1, p=0.8)
A.GaussianBlur(blur_limit=(3, 7), p=0.3)
A.ToGray(p=0.1)
A.GridDistortion(p=0.2)
A.RandomRotate90(p=0.1)
A.GaussNoise(p=0.2)
A.CoarseDropout(num_holes_range=(1,8), hole_height_range=(8,48), hole_width_range=(8,48), p=0.3)
A.Normalize(mean=CLIP_MEAN, std=CLIP_STD)
```

The wide crop scale `(0.5, 1.0)` is intentional — it forces the model to detect spoofing cues from partial face views rather than relying on whole-face holistic patterns that may not generalize.

### MixUp

Applied with probability 0.5 per batch:

```python
λ ~ Beta(0.2, 0.2)
mixed_x = λ × x_i + (1-λ) × x_j
loss    = λ × CE(pred, y_i) + (1-λ) × CE(pred, y_j)
```

MixUp encourages smooth decision boundaries, preventing the model from memorizing hard visual shortcuts at training time.

### Class Balancing: WeightedRandomSampler

Training uses `WeightedRandomSampler` so each batch has balanced class distribution, independent of raw class frequencies. This is stronger than simply adjusting loss weights because the model sees balanced *data*, not just balanced *gradients*.

---

## 6. Progressive Fine-Tuning Strategy

Directly applying large LRs to the CLIP backbone causes **catastrophic forgetting** — destroying CLIP's pretrained representations before the connector has learned anything useful. Three phases mitigate this:

```
Phase 1: CLIP frozen      → GAC/Classifier learn from scratch
Phase 2: Last 12 blocks   → Careful backbone adaptation
Phase 3: All 24 blocks    → Fine calibration at very low LR
```

### Phase 1 — Warm-Up (Frozen Backbone)

| Component | LR |
|---|---|
| GAC | `3e-4` |
| Classifier + Proj Head | `1e-3` |
| CLIP visual encoder | **frozen** |

Max epochs: 35, patience: 12.

The connector and classifier learn to use CLIP's features before any backbone weights are modified.

### Phase 2 — Partial Unfreeze (Last 12 Blocks)

| Component | LR |
|---|---|
| CLIP (last 12 blocks) | `5e-7` |
| GAC | `5e-6` |
| Classifier | `5e-5` |

Max epochs: 30, patience: 10. LR ratio ≈ 1 : 10 : 100.

The discriminative LR ratio ensures that layers closest to the task output adapt more, while pretrained deep layers change only slightly. `LayerNorm` layers inside the visual encoder are kept in `eval()` mode during Phase 2 and 3 to preserve their running statistics.

### Phase 3 — Full Fine-Tuning (All 24 Blocks)

| Component | LR |
|---|---|
| CLIP (all 24 blocks) | `3e-8` |
| GAC | `5e-7` |
| Classifier | `5e-6` |

Max epochs: 20, patience: 8.

`lr_clip = 3e-8` is ~10,000× smaller than Phase 1 connector LR. The goal is subtle calibration of early CLIP blocks, not large weight changes.

**All phases** use `OneCycleLR` (10% linear warmup → cosine decay) and gradient clipping (`max_norm=1.0`). Optimizer: AdamW with `weight_decay=1e-2`.

---

## 7. Pseudo-Labeling (3 Rounds)

After Phase 3, high-confidence test predictions are added to the training set for semi-supervised refinement.

```
Phase 3 → Predict test set with TTA → Filter by confidence ≥ threshold
       → Add to train_df → Retrain (abbreviated) → Repeat
```

| Round | Threshold | LR multiplier | Added samples (approx) |
|---|---|---|---|
| R1 | 0.998 | 0.5× Phase 3 | Smallest, near-certain only |
| R2 | 0.970 | 0.3× Phase 3 | More samples, model more reliable |
| R3 | 0.920 | 0.2× Phase 3 | Broadest coverage, minimal LR |

Going below ~0.92 risks **confirmation bias** — mislabeled pseudo-samples reinforce the model's own errors in subsequent rounds. The decreasing LR multiplier further limits drift.

**Observed result:** Best checkpoint improved from F1=0.9888 (post Phase 3) to F1=0.9946 (post Round 3).

---

## 8. Inference: TTA + 3-Branch Ensemble

### Test-Time Augmentation (20 steps)

20 different augmentation pipelines are applied to each test image, and softmax probabilities are averaged:

| Group | Description |
|---|---|
| Multi-scale center crop | Resize to 1.05×, 1.143×, 1.25× then center crop |
| Flips | Horizontal flip of each scale variant |
| Random crops | Scale (0.75–0.95), (0.85–1.0), (0.95–1.0) |
| Color variants | ColorJitter on top of center crops |
| Blur variant | GaussianBlur on the 1.25× scale |

This averages out stochastic variance and makes the prediction less sensitive to any single spatial or color framing.

### 3-Branch Ensemble

```python
final_prob = 0.55 × NN_ensemble + 0.25 × ZeroShot + 0.20 × KNN
```

**Branch 1 — NN Ensemble (4 models, seeds 42/123/456/999):**  
Each model is independently trained through the full Phase 1→2→3→pseudo pipeline. Averaging 4 diverse models reduces prediction variance significantly. Each model contributes its TTA-averaged softmax probabilities.

**Branch 2 — CLIP Zero-Shot:**  
4 text prompts per class are encoded by CLIP's text encoder and compared to image embeddings via cosine similarity. No fine-tuning; pure semantic reasoning from CLIP pre-training.

```python
TEXT_PROMPTS = {
    'realperson':     ['a real human face', 'a live genuine person face photo', ...],
    'fake_printed':   ['a photo of a printed face', 'a paper face attack photo', ...],
    'fake_screen':    ['a face shown on a screen', 'face replay attack on monitor', ...],
    'fake_mask':      ['a person wearing a face mask', 'a 3D silicone face mask', ...],
    'fake_mannequin': ['a mannequin head face', 'a dummy plastic face', ...],
    'fake_unknown':   ['an unknown face spoofing attack', 'a synthetic fake face', ...],
}
```

**Branch 3 — K-NN (k=15):**  
Cosine similarity search in CLIP feature space (from the frozen `encode_image` output). Temperature-weighted voting `exp(sim × 15)` gives higher weight to closer neighbors. This is an instance-based reasoner that complements the parametric NN predictions, especially useful when the NN ensemble is uncertain.

---

## 9. Key Implementation Notes

### Hook Memory Leak Fix

`encode_image_for_zs()` calls `clip.encode_image()`, which triggers all 24 registered hooks and fills `_cls_buf` with 24 tensors per call. Without clearing, N inference batches accumulate 24N tensors → OOM.

```python
def encode_image_for_zs(self, x):
    self._cls_buf.clear()                                     # ← clear before
    result = F.normalize(self.clip.encode_image(x).float(), dim=-1)
    self._cls_buf.clear()                                     # ← clear after
    return result
```

The same `_cls_buf.clear()` pattern is used at the start of `_visual_forward()` for the primary forward pass.

### LayerNorm in eval() During Fine-Tuning

```python
for m in model.extractor.clip.visual.modules():
    if isinstance(m, nn.LayerNorm): m.eval()
```

During Phases 2 and 3, CLIP's visual `LayerNorm` layers are kept in `eval()` mode even during training. This prevents the running statistics learned during CLIP pre-training from being corrupted by the small fine-tuning dataset (only ~1483 training images).

### `num_workers=0` for Inference DataLoaders

Jupyter Notebook multiprocessing DataLoaders can crash with `AssertionError: can only test a child process` when CUDA is already initialized in the main process. All inference DataLoaders use `num_workers=0` to avoid this.

### Inference Batch Size

After Phase 3 with all 24 CLIP blocks unfrozen, model weights alone occupy ~6 GB VRAM. With AMP, optimizer states, and activations during training, VRAM usage is high. Inference uses `batch_size=16` with `torch.cuda.empty_cache()` between KNN/ZeroShot/NN passes.

### `autocast` and `GradScaler` Import

```python
from torch.amp import autocast, GradScaler   # not torch.cuda.amp (deprecated)
```

The old `torch.cuda.amp` API was deprecated in PyTorch 2.x. Use `torch.amp` with the device string argument: `GradScaler('cuda')` and `autocast('cuda')`.

### Label Mapping Detail

The competition's folder name for printed attacks is `fake_printed`, but the internal class label used during training (matching `IDX2CLASS`) is `fake_print`. A relabeling step at submission time converts `fake_print` back to `fake_printed`.

```python
sub['label'] = sub['label'].replace('fake_print', 'fake_printed')
```

---

## 10. Results

| Stage | Val F1 (macro) |
|---|---|
| Phase 1 best (ep 16) | 0.9888 |
| Phase 2 best | 0.9888 |
| Phase 3 best | 0.9888 |
| Pseudo R1 | 0.9892 |
| Pseudo R2 | 0.9892 |
| **Pseudo R3** | **0.9946** |

Training data: ~1,483 images (90% of 1,648 total). Val set: 165 images. Test set: 404 images.

Best checkpoint: `ps_r3_f10.9946_ep1.pth` — reached F1=0.9946 on the 165-sample val set.

---

## 11. What I Would Try Next

- **Cross-architecture ensemble** — combine ViT-L CLIP with DINOv2-ViT-L or ConvNeXt-XXL. More architectural diversity → less correlated prediction errors
- **Frequency domain branch** — print and screen attacks produce moiré patterns detectable via FFT; a separate frequency branch fused before the classifier could help
- **Test-time adaptation** — brief entropy minimization (TENT-style) on each test batch before final prediction, adapting BN/LN statistics to the test distribution
- **448px resolution** — CLIP supports this via positional embedding interpolation; higher resolution gives richer patch tokens for the GAC cross-attention
- **Larger pseudo-label pool** — if the competition allows using external public FAS datasets as pseudo-labeled support, KNN-based label assignment could significantly expand training data

---

## References

- Zhang et al., *Interpretable Face Anti-Spoofing: Enhancing Generalization with Multimodal Large Language Models* (I-FAS), AAAI 2025 — [arXiv:2501.01720](https://arxiv.org/abs/2501.01720)
- Radford et al., *Learning Transferable Visual Models From Natural Language Supervision* (CLIP), ICML 2021
- Khosla et al., *Supervised Contrastive Learning*, NeurIPS 2020
- Zhang et al., *mixup: Beyond Empirical Risk Minimization*, ICLR 2018
- Huang & Sun, *Deep Networks with Stochastic Depth*, ECCV 2016
- Li et al., *BLIP-2: Bootstrapping Language-Image Pre-training with Frozen Image Encoders and Large Language Models*, ICML 2023
