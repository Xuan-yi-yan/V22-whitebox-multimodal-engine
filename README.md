# V22 White-Box Multimodal Cognitive Engine

**A ∼9.9M-parameter, fully interpretable (white-box) engine that unifies Text / Vision / Audio through a single ABCDEF cognitive stack.**

- 📄 [V22 Architecture & ABCDEF Cognitive Stack](V22_ABCDEF_方案.txt)
- 🏗️ [V21 → V22 Evolution & Full Architecture Record](V21_架构全景_最终版.txt)
- 🔬 [vLLM #43090 Root-Cause Repro (hybrid KV cache)](hybrid_apc_43090_repro.py)

## Quick Start

```bash
# Train (text-only, small sample for verification):
python train_v22_stage1.py --sample 10

# Visual property extraction test:
python p3vis.py test_apple.bmp
```

## Core Components

| Component | File | Purpose |
|---|---|---|
| Full training loop (text + vision + audio) | `train_v22_stage1.py` | End-to-end multimodal training |
| P3Vis — industrial-grade visual attribute stack | `p3vis.py` | 256D white-box visual features (HSV, HuMoments, GLCM, LBP, HOG) |
| Visual token embedding | `visual_token.py` | Pixel → 256D spherical embedding (16×16 raw RGB, no CNN/ViT) |
| P3-L cross-attribute linkage | `P3_word_attr/p3l_linkage.py` | Grouped multi-head attention (312 heads) + cross-modal groups |

## Key Innovations

- **ABCDEF Hexa-Cognitive Stack**: A(Anchor) → B(Binding) → C(Composition) → D(Deduction/Relation Review) → E(Engram/Memory Read-Write Head with 3-tier cache) → F(Forecast → P6 decode)
- **Spherical Normalization**: Root-cause fix for numerical-range gradient collapse (F.normalize on unit sphere — same philosophy as ac_batch soft-norm)
- **Unified Multimodal Token**: `[Embedding 256D (spherical)] + [Attribute 384D (language 128 + vision 256)] = 640D` — one engine, three modalities, zero modality-specific branches
- **Memory Read-Write Head**: sent_vec cosine retrieval + P3 attribute Jaccard hybrid scoring, 3-tier cache (GPU hot / RAM warm / Disk cold), persisted to file
- **Hardware**: Single RTX 5070 12GB. Not scaled — verified small-scale data flow, not production training.

## Related Contributions

- **vLLM PR #47491**: Fix hybrid KV cache prefix cache truncation (Mamba group miss zeroing all hits)
- **vLLM issue #43090**: Full root-cause chain for cached/uncached output divergence (misjudged `has_initial_states`)
- **DeepSeek-V3 Community**: Graviton Anchors concept (adopted by AmoebaFPS, formalized as GDI framework)

*This is an independent research/engineering project. Architecture validated on small-scale multimodal data flow. Large-scale training and production integration are future work.*
