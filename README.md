
<p align="center">
  <a href="#quick-start"><img src="https://img.shields.io/badge/▶_Run_Demo-单卡即跑-blue?style=for-the-badge"></a>
  <a href="V22_ABCDEF_方案.txt"><img src="https://img.shields.io/badge/📐_Architecture-ABCDEF认知堆栈-green?style=for-the-badge"></a>
  <a href="https://github.com/Xuan-yi-yan/vllm"><img src="https://img.shields.io/badge/🔬_vLLM_Contributor-PR_47491-red?style=for-the-badge"></a>
</p>

# V22 White-Box Multimodal Cognitive Engine

**A ∼9.9M-parameter, fully interpretable (white-box) engine — unified Text / Vision / Audio through a single ABCDEF cognitive stack. Trained on a single RTX 5070 12GB. No ViT. No Whisper. No modality-specific silos.**

---

## ⚡ Quick Start — One Command

```bash
git clone https://github.com/Xuan-yi-yan/V22-whitebox-multimodal-engine.git
cd V22-whitebox-multimodal-engine
python run_demo.py
```

Launches `train_v22_stage1.py --sample 10` — text training → visual (apple → "苹果") → audio (FFT → text). All three modalities flow through the same engine.

---

## 📐 Core: White-Box Text Engine (Foundation, evolved from V21)

V22 is first a **text engine** — 12 modules, ∼9.9M params, evolved over 49 days (V18→V21.1) before adding vision/audio.

```
CharEmbed(256D) → P3 AttributeStack(128D, 126 slots) → P7 CrossSentence(16-head)
  → P3-L AttributeLinkage(312 heads) → ABC'(Pressurization) → ABC(3-stage logic)
  → Gate(3-channel) → P6 Tied-Weight Decoder → Output
```

| Module | Params | Role |
|---|---|---|
| **CharEmbed** | 742K | ∼2900 chars × 256D, orthogonal init — each char maps to independent direction |
| **P3 AttributeStack** | 0 | 14-module rule engine: word-class, semantic, syntactic, emotion, logic-relation, tense, person... 126 explicit attribute slots |
| **P7 CrossSentence** | 449K | 16-head × 16D, Q=A tokens, K/V=full vocabulary, RMSNorm Q/K anti-explosion |
| **P3-L AttributeLinkage** | 158K | 312 heads / 23 groups, independent per-group attention (anti-contamination) |
| **ABC 3-Stage Logic** | 65K | A(20 classes, structural decision) → B(384D content) → C(48D tone). ABC *reviews* attribute coherence — relations are already explicit in P3 slots. |
| **P6 Tied Decoder** | 8.6M | 128-head × Linear(256,256), CE loss directly outputs characters |

**Training**: 7 independent Adam optimizers with hierarchical LRs (Gate ×2.0 fast, P7 ×0.05 suppressed, ABC ×1.2 agile), 5-layer smart safeguard (CE spike clamp, aux progressive self-heal, NaN rollback, auto-stop, ac explosion detection), grad diagnostics per module.

---

## 🔀 Multimodal Extension: Vision & Audio

Vision and audio are **not separate models** — they're additional attribute + embedding paths into the same ABCDEF stack. The engine sees only 640D tokens, regardless of source.

```
Token = [Embedding 256D (spherical)] + [Attribute 384D]

Text:   char_embed(char)      + P3 linguistic attributes → 640D
Vision: VisualEmbed(pixels)   + P3Vis visual attributes  → 640D
Audio:  AudioEmbed(FFT)       + P3Aud audio attributes   → 640D  (reserved)
  ↓
P7 → ABCDEF → P6  (same engine, no modality branches)
```

| Modality | Signal | Embedding | Attributes |
|---|---|---|---|
| **Vision** | 16×16 raw RGB (768 pixels) | VisualEmbed(768→256) spherical | P3Vis: 256D — HSV, Hu 7 moments, GLCM×4, LBP, HOG, K-means |
| **Audio** | Raw FFT spectrum (512 bins) | AudioEmbed(512→256) spherical | P3Aud: pitch, formant, loudness, ZCR, spectral centroid, rhythm (reserved) |

**Why no mel-filter?** Mel = pre-encoded "human hearing model" = artificial prior. FFT = pure physical frequency decomposition, symmetric to raw RGB pixels. Let the model learn which frequencies matter.

**Verification**: Text → convergence (loss 0.0001). Vision → "image(apple) → 苹果" ✓. Audio → "FFT → 高音" ✓. Three modalities, one engine. Small-scale architectural proof-of-concept, not production training.

---

## 🧠 ABCDEF Hexa-Cognitive Stack

| Stage | Function | How |
|---|---|---|
| **A** Anchor | Modality recognition | 640D → 20 structural classes |
| **B** Binding | Cross-modal attention | Binds "red"+"round"→"apple" |
| **C** Composition | Abstract concept | 384D concept vector, cosine alignment (visual ≈ text) |
| **D** Deduction | Relation review | Verifies P3 logic-relation slots (cause, contrast, coordinate...) — relations ARE attributes, ABC reviews them |
| **E** Engram | Memory Read-Write Head | 3-tier cache + cosine + Jaccard → O(1) context, not O(N) KV-cache |
| **F** Forecast | Decode | Cognitive state → P6 128-head decoder |

### Stage E: Memory Read-Write Head (deep-dive)

```
WRITE: sent_vec[256D] + me[384D] → memory bank (FIFO 2000, persisted to V22_mem.pt)

READ:  query sent_vec ─cosine→ top-1 match
       + P3 attribute Jaccard overlap (distinguishes "天气怎么样?" vs "天气不错")
       → recalled memory, spherical fused: me = F.normalize(net(rd) + recall × 0.1)

WHY:   KV-cache = O(N) with context. Semantic memory = O(1) retrieval.
       sent_vec is already semantic (P7 output) — no external embedding DB needed.
```

---

## 🛡️ Spherical Normalization (Root-Cause Gradient Fix)

Adding D/E layers caused full gradient death (E20-160 crash: CE→5.0, all gate grads→0).

**Root cause**: D/E numerical range too large → ac_batch explodes → meta sigmoid saturates → gate path dead → chain collapse.

**Failed**: Cold-start gain (0.1→0.01) only delayed. Narrowing gate bridge (512→384) still crashed.

**Fix**: `F.normalize(dim=-1)` on all injected representations (rd/me/pixel/FFT). Spherical unit vectors control both forward magnitude and backward gradient — full-scale injection (×0.1) without collapse.

---

## 📊 Results (RTX 5070 12GB)

| Test | Result |
|---|---|
| Text `--sample 10` | Loss 6.98→0.0001 auto-stop ✓ |
| Vision (28 pairs, 30eps) | `图(苹果)→预测:苹果` ✓ |
| Audio (sim FFT, 30eps) | `声→文:高音` ✓ |
| Full pipeline | Text→Vision→Audio same engine ✓ |

*12GB VRAM = small-scale verification only. Architectural feasibility demonstrated, production training needs appropriate hardware.*

---

## 🔬 Open-Source Contributions

- **vLLM PR #47491** — Fix hybrid KV cache truncation (Mamba miss → `continue` preserves attention hits)
- **vLLM #43090** — Root-cause chain: `has_initial_states` misjudged for hybrid prefix caching
- **DeepSeek-V3** — Graviton Anchors (adopted by AmoebaFPS → GDI framework)
- **DeepSeek-V3 #1462** — Cross-framework failure mode taxonomy (HeartFlow/TAT alignment)

*Independent research. Open to collaboration and feedback.*
