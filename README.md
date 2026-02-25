# Fine-Grained Multimodal Alignment with Cross-Attention Using CLIP for Enhanced Retrieval

This repository contains the official implementation of our research work:

**“Fine-Grained Multimodal Alignment with Cross-Attention Using CLIP for Enhanced Retrieval”**

Department of Computer Science & Engineering  
KLE Technological University, Hubli, Karnataka, India  

---

## 🔍 Abstract

Large-scale vision–language models such as CLIP learn strong global image–text representations when trained on massive datasets. However, their performance deteriorates under low-resource conditions such as Flickr8k (8,092 images, 5 captions per image).

This work proposes a **lightweight two-stage alignment framework**:

1. **Stage 1:** Gentle global fine-tuning of CLIP on Flickr8k.
2. **Stage 2:** A compact cross-attention module enabling fine-grained region–token alignment between:
   - Region features extracted using Faster R-CNN
   - Token-level text embeddings from CLIP

The proposed method improves retrieval performance significantly under limited data conditions while remaining computationally efficient.

---

## 🧠 Methodology

### Stage 1 — Global Fine-Tuning

CLIP image and text encoders are fine-tuned using contrastive learning:

\[
s_{global}(I, T) = \frac{\cos(g_I, g_T)}{\tau}
\]

We optimize the standard NT-Xent contrastive loss to align matching image–caption pairs and separate mismatched pairs.

Training Details:
- Optimizer: AdamW
- Learning rate: 1e-5
- Weight decay: 1e-4
- Batch size: 64
- Temperature: 0.07
- Epochs: 12
- Mixed Precision (AMP)
- Random seed: 42

---

### Stage 2 — Fine-Grained Cross-Attention Alignment

Global representations treat the entire image as a single vector. To enable localized reasoning:

1. Region features \( r_k \) are extracted using a pretrained Faster R-CNN detector.
2. Token embeddings \( t_m \) are obtained from CLIP’s text encoder.
3. Cross-attention computes:

\[
\alpha_{mk} =
\frac{
\exp\left(t_m^\top W_q^\top W_k r_k\right)
}{
\sum_{k'} \exp\left(t_m^\top W_q^\top W_k r_{k'}\right)
}
\]

4. Region aggregation:

\[
v_m = \sum_k \alpha_{mk} W_v r_k
\]

5. Fine-grained similarity:

\[
s_{fine}(I, T) =
\frac{1}{M}
\sum_{m=1}^{M}
t_m^\top W_o v_m
\]

6. Final similarity:

\[
s(I, T) = s_{global}(I, T) + \lambda s_{fine}(I, T)
\]

In all experiments:

\[
\lambda = 1.0
\]

Stage 2 updates only:
- Cross-attention layers
- Projection layers

CLIP backbones remain frozen for stability.

---

## 📊 Dataset

**Flickr8k**
- 8,092 images
- 40,460 captions (5 per image)
- Standard split:
  - 6,000 train
  - 1,000 validation
  - 1,000 test

This represents a low-resource multimodal learning scenario.

---

## 📈 Experimental Results

### Text-to-Image Retrieval

| Model | R@1 | R@5 | R@10 |
|-------|------|------|------|
| CLIP Zero-Shot | 50.50% | 70.23% | 83.45% |
| CLIP Global Fine-Tuned | 71.41% | 89.25% | 91.87% |
| **Proposed (Global + Fine-Grained)** | **75.59%** | **92.33%** | **93.73%** |

---

### Image-to-Text Retrieval

| Model | R@1 | R@5 | R@10 |
|-------|------|------|------|
| CLIP Zero-Shot | 56.12% | 81.61% | 89.78% |
| CLIP Global Fine-Tuned | 83.06% | 84.27% | 88.48% |
| **Proposed (Global + Fine-Grained)** | **87.21%** | **89.88%** | **92.88%** |

---

## 🧪 Ablation Study

| Model Variant | Text→Image R@1 |
|---------------|----------------|
| Global Fine-Tuning Only | 71.41% |
| + Cross-Attention (λ = 1.0) | **75.59%** |

Fine-grained alignment provides consistent improvements over global-only adaptation.

---

## 🖼 Retrieval & VQA

The notebook includes:

- Image → Top-K Caption Retrieval
- Text → Top-K Image Retrieval
- Retrieval-based Visual Question Answering (VQA)

For VQA:
- The question embedding is matched with candidate caption embeddings.
- The highest similarity caption is returned as the answer.
- No separate answer classifier is trained.

This keeps the system lightweight and stable under low-resource constraints.

---

## ⚙️ Computational Efficiency

- Stage 2 updates only lightweight cross-attention modules.
- Faster R-CNN region features are precomputed and reused.
- No heavy multimodal fusion backbone is trained.
- Memory overhead remains low compared to large fusion models like ALBEF.

---

## 🚀 How to Run (Colab)

1. Open the notebook in Google Colab.
2. Mount Google Drive.
3. Set:

```python
DRIVE_ROOT = '/content/drive/MyDrive/.../flickr8k'
