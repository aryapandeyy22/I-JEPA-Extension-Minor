Here is your **full README.md content**, clean, polished, short, and **copy-paste ready** — all in **one block** (not one line), as a proper README file.

---

#  **README.md (FINAL — Copy/Paste Entire Block)**

```markdown
# Hierarchical I-JEPA Extension – Self-Supervised Vision Learning

This project extends Meta’s **I-JEPA (Image-based Joint Embedding Predictive Architecture)** by adding a **Hierarchical Predictor Network** and a **Cross-View Consistency Objective** to improve robustness and representation quality on small-scale datasets such as Tiny-ImageNet.

---

## Proposed Framework

Below is the complete architecture used in this project:



Input Embedding (1280-dim)
│
Linear 1280→512
│
GELU
│
Linear 512→1280
│
┌──────────────┐
│  Residual +   │
│  LayerNorm    │
└──────────────┘
│
Linear 1280→1280
│
Output: Refined Embedding

```

---

##  Dataset

We use a 2,000-image subset of **Tiny-ImageNet**, containing 200 object classes (animals, vehicles, objects, scenes), resized to 224×224 and normalized using ImageNet statistics.



##  Training Objective

**Cross-View Consistency Loss:**

We generate two different augmentations of the same image and force the predictor outputs to match:

```

L = 2 – 2 ⟨normalize(p1), normalize(p2)⟩

````

This promotes stability, invariance, and semantic alignment.

---

##  How to Run the Code

### 1. Clone I-JEPA Repo
```bash
git clone https://github.com/facebookresearch/ijepa
````

### 2. Place pretrained weights (ViT-H 14×14)

Download from:
[https://github.com/facebookresearch/ijepa/blob/main/MODEL_ZOO.md](https://github.com/facebookresearch/ijepa/blob/main/MODEL_ZOO.md)
and place:

```
IN1K-vit.h.14-300e.pth.tar
```

in your project root.



## ▶️ Run Training

```python
python train_hierarchical_predictor.py
```

This script:

* loads pretrained I-JEPA backbone
* builds the hierarchical predictor
* trains on Tiny-ImageNet subset
* saves:

  ```
  hierarchical_predictor.pth
  ```



##  Run Evaluation (Linear Probe + t-SNE)

```python
python evaluate_embeddings.py
```

This script:

* loads trained predictor
* extracts embeddings
* trains logistic regression linear-probe
* reports accuracy
* generates `tsne_embeddings.png`

---

## Results Summary

* Cross-view loss rapidly converges to near-zero
* Linear probe accuracy on Tiny-ImageNet subset: **~5–6%**
* t-SNE shows partial clustering but low global semantic separation
* Indicates undertraining (small dataset) and that predictor learns invariance more than class discrimination

---

## Project Structure


├── train_hierarchical_predictor.py
├── evaluate_embeddings.py
├── hierarchical_predictor.pth
├── IN1K-vit.h.14-300e.pth.tar
├── data/tiny-imagenet-200/
└── README.md




## Future Work

To achieve stronger results:

* Train on full ImageNet-1K
* Add masked target-region prediction (true JEPA objective)
* Use stronger augmentations
* Apply predictor to downstream classification & segmentation tasks
* Use multi-scale feature aggregation



##  Author

**Arya Pandey**
IIT Guwahati | Self-Supervised Learning Research Project




