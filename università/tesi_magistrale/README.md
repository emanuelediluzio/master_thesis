# SVG Captioning with Large Language Models

> **Master's Thesis** – Emanuele Di Luzio  
> Università di Modena e Reggio Emilia

---

## Abstract

This thesis presents a **vector-native approach to SVG image captioning** that processes scalable vector graphics directly as structured code, without rasterization. We integrate an **SVG Path Embedder (SPE)** with decoder-only LLMs (Qwen2-7B, Gemma-9B) via **Low-Rank Adaptation (LoRA)** to generate accurate, human-readable captions for icon graphics.

**Key Results:**
- **SPE+Qwen2-7B** achieves **Composite Score: 8.89** (CLIPScore: 29.30, BLEU-1: 0.420, METEOR: 0.380, ROUGE-L: 0.450)
- Outperforms zero-shot raster baselines (BLIP-2, Florence-2) in linguistic quality
- First demonstration that LLMs can "read" vector geometry without pixel conversion

---

## Repository Structure

```
├── code/
│   ├── models/           # SPE implementation
│   │   ├── spe_model.py          # SVG Path Embedder core
│   │   ├── spe_tokenizer.py      # Path command tokenization
│   │   ├── spe_base.py           # Base encoder architecture
│   │   └── multimodal_model.py   # LLM + SPE integration
│   ├── training/         # Training scripts
│   │   ├── robust_training_launcher.py
│   │   └── train_gemma_spe_corrected.py
│   ├── inference/        # Zero-shot baselines
│   │   ├── blip2_zero_shot_inference.py
│   │   └── florence2_zero_shot_inference.py
│   └── evaluation/       # Metrics computation
│       └── calculate_simple_metrics.py
├── figures/              # Thesis figures
├── images/               # Experimental visualizations
├── CONFRONTO_METRICHE.csv   # Complete results table
├── tesi_svg_captioning_expanded.pdf  # Thesis document
└── revised_presentation_script.md    # Defense presentation
```

---

## Methodology

### Architecture Overview

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│   SVG Code      │────▶│  SVG Path        │────▶│  Projection     │
│   (XML/Path)    │     │  Embedder (SPE)  │     │  Layer          │
└─────────────────┘     └──────────────────┘     └────────┬────────┘
                                                          │
                                                          ▼
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│   Caption       │◀────│  Decoder-Only    │◀────│  LLM Input      │
│   Output        │     │  LLM (Qwen2/     │     │  Embeddings     │
│                 │     │   Gemma + LoRA)  │     │                 │
└─────────────────┘     └──────────────────┘     └─────────────────┘
```

### Key Components

1. **SVG Path Embedder (SPE)**: Converts continuous (x,y) coordinates into dense embeddings using sinusoidal positional encodings
2. **Projection Layer**: Bridges SPE output (256-dim) to LLM input space (4096-dim)
3. **LoRA Adaptation**: Parameter-efficient fine-tuning (<1% trainable params)
4. **Z-Score Normalization**: Critical for training stability

---

## Results Summary

| Model | Type | CLIPScore | BLEU-1 | METEOR | ROUGE-L | Composite |
|-------|------|-----------|--------|--------|---------|-----------|
| **SPE+Qwen2** | Ours | 29.30 | **0.420** | **0.380** | **0.450** | **8.89** |
| SPE+Gemma | Ours | 25.20 | 0.150 | 0.180 | 0.200 | 6.13 |
| Qwen2 (LoRA) | Text-Only | **32.30** | 0.238 | 0.206 | 0.277 | 8.42 |
| BLIP-2 | Zero-Shot | 31.66 | 0.003 | 0.048 | 0.123 | 7.85 |
| Florence-2 | Zero-Shot | 31.07 | 0.003 | 0.060 | 0.119 | 7.92 |

---

## Dataset

- **Source**: Icons8 icon library
- **Size**: ~90,000 SVG-caption pairs
- **Test Set**: 400 stratified samples
- **Preprocessing**: Tag stripping, canonical ViewBox (512×512), complexity filter (≤50 segments)

---

## Requirements

```bash
pip install torch transformers peft accelerate
pip install nltk rouge-score sacrebleu
pip install clip-score  # For CLIPScore evaluation
```

---

## Citation

```bibtex
@mastersthesis{diluzio2025svg,
  author  = {Di Luzio, Emanuele},
  title   = {SVG Image Captioning with Large Language Models},
  school  = {Università di Modena e Reggio Emilia},
  year    = {2025},
}
```

---

## Acknowledgments

- **Supervisor**: Prof. [Name]
- **SPE Implementation**: Based on work by Zini et al. (vHector)
- **Compute**: AILB Cluster, UNIMORE

---

## License

This project is for academic purposes. Code and data are provided as-is for research reproducibility.
