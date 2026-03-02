# Fine-Grained Multimodal Alignment with Cross-Attention Using CLIP for Enhanced Retrieval


---

## 🔍 Abstract

Large-scale vision–language models such as CLIP learn strong global image–text representations when trained on massive datasets. However, their performance deteriorates under low-resource conditions such as Flickr8k (8,092 images, 5 captions per image).

This work proposes a lightweight two-stage alignment framework:

1. **Stage 1:** Gentle global fine-tuning of CLIP on Flickr8k  
2. **Stage 2:** A compact cross-attention module enabling fine-grained region–token alignment between:
   - Region features extracted using Faster R-CNN  
   - Token-level text embeddings from CLIP  

The method improves retrieval performance significantly under limited data conditions while remaining computationally efficient.

---

## 🧠 Methodology

### Stage 1 — Global Fine-Tuning

We fine-tune CLIP using contrastive learning.

Global similarity:

s_global(I, T) = cos(g_I, g_T) / τ

where:
- g_I = image embedding  
- g_T = text embedding  
- τ = temperature (0.07)

We optimize the standard NT-Xent contrastive loss.

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

Global embeddings treat the image as a single vector.  
To enable detailed grounding, we align region-level features with token-level embeddings.

Let:
- r_k ∈ R^d = region feature from Faster R-CNN  
- t_m ∈ R^d = token embedding  

Cross-attention weights:

α_mk = exp(t_mᵀ W_qᵀ W_k r_k)  
       / Σ_k' exp(t_mᵀ W_qᵀ W_k r_k')

Region aggregation:

v_m = Σ_k α_mk W_v r_k

Fine-grained similarity:

s_fine(I, T) = (1/M) Σ_m t_mᵀ W_o v_m

Final similarity:

s(I, T) = s_global(I, T) + λ s_fine(I, T)

In all experiments:

λ = 1.0

During Stage 2:
- CLIP backbones are frozen  
- Only cross-attention and projection layers are updated  

---

## 📊 Dataset

**Flickr8k**
- 8,092 images
- 40,460 captions (5 per image)
- Standard split:
  - 6,000 training
  - 1,000 validation
  - 1,000 test

This represents a low-resource multimodal setting.

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
- The question embedding is matched with caption embeddings  
- The most similar caption is returned as the answer  
- No separate answer classifier is trained  

This keeps the system lightweight and stable.

---


## 🎯 Key Contributions

- Lightweight two-stage adaptation framework  
- Explicit region–token cross-attention alignment  
- Significant Recall@1 improvement under limited data  
- Stable and computationally efficient design  

---

## ⚙️ Computational Efficiency

- Stage 2 updates only lightweight cross-attention layers  
- Faster R-CNN region features are precomputed and reused  
- No heavy multimodal fusion backbone  
- Stable under low-resource conditions  

---

## 🚀 How to Run (Colab)

1. Open the notebook in Google Colab  
2. Mount Google Drive  
3. Set: Runtime as T4 GPU
4. Run all the cells in order
