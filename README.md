# V22 White-Box Multimodal Cognitive Engine

**∼9.9M params, fully interpretable. Text / Vision / Audio — one engine. Single RTX 5070.**

```bash
git clone https://github.com/Xuan-yi-yan/V22-whitebox-multimodal-engine.git
cd V22-whitebox-multimodal-engine
python run_demo.py        # --sample 10, 3-modality quick verify
```

---

## 📐 Text Engine (Core, evolved from V21)

12 modules. ABCDEF cognitive stack. 9.9M params. 49-day evolution (V18→V21.1).

```
CharEmbed(256D) → P3(128D, 126 slots) → P7(16-head) → P3-L(312 heads)
  → ABC'(Pressurization) → ABC(3-stage) → Gate(3-channel) → P6(128-head) → Output
```

| Module | Role |
|---|---|
| **P3 AttributeStack** | 0-param rule engine: word-class, syntax, emotion, logic, tense... 126 explicit slots |
| **P7 CrossSentence** | 16-head × 16D, RMSNorm Q/K anti-explosion |
| **P3-L AttributeLinkage** | 312 heads / 23 groups, per-group attention (anti-contamination) |
| **ABC** | A(20-class structural) → B(384D content) → C(48D tone). Reviews attribute coherence. |
| **P6** | 128-head tied-weight decoder, CE output |

7 optimizers, hierarchical LR (Gate ×2.0, P7 ×0.05, ABC ×1.2). 5-layer safeguard (spike clamp, progressive self-heal, NaN rollback, auto-stop).

---

## 🔀 Multimodal: Vision + Audio (not separate models)

```
Token = [Embedding 256D spherical] + [Attribute 384D]
  Text:   char_embed(char)    + P3 linguistic → 640D
  Vision: VisualEmbed(pixels) + P3Vis visual  → 640D
  Audio:  AudioEmbed(FFT)     + P3Aud audio   → 640D
  ↓
P7 → ABCDEF → P6   (same engine, zero modality branches)
```

- **Vision**: 16×16 raw RGB (768 pixels), VisualEmbed spherical. P3Vis: 256D — HSV, Hu 7 moments, GLCM×4, LBP, HOG, K-means. No CNN/ViT.
- **Audio**: Raw FFT spectrum (512 bins), AudioEmbed spherical. P3Aud: pitch, formant, loudness, ZCR, spectral centroid, rhythm (reserved). No mel-filter.
- **Engine doesn't know what a token "is"** — 640D vector. Char, pixel, or frequency. Same stack.

---

## 🧠 ABCDEF Hexa-Cognitive Stack

| Stage | Function |
|---|---|
| **A** Anchor | 640D → 20 structural classes (modality recognition) |
| **B** Binding | Cross-modal attention: "red"+"round" → "apple" |
| **C** Composition | 384D concept, cosine alignment (visual concept ≈ text concept) |
| **D** Deduction | Reviews P3 logic-relation slots (cause, contrast, coordinate...) — relations ARE attributes |
| **E** Engram | **Memory Read-Write Head**: sent_vec cosine + P3 Jaccard → O(1) context (not O(N) KV-cache). 3-tier cache (GPU/RAM/Disk). Persisted to file. |
| **F** Forecast | Cognitive state → P6 decoder |

### Stage E: Memory Read-Write Head
```
WRITE: sent_vec[256D] + me[384D] → memory bank (FIFO 2000, → V22_mem.pt)
READ:  query_sent_vec ─cosine→ top-1 + P3 Jaccard disambiguation
       → spherical fused: me = normalize(net(rd) + recall × 0.1)
       → additive injection into P6 decode (memory = detach-prior, no backprop)
```

---

## 🛡️ Spherical Normalization (root-cause gradient fix)
D/E layers: full gradient death at E20-160 (CE→5.0, gate grads→0). Root cause: numerical range → sigmoid saturation → chain collapse. Fix: `F.normalize(dim=-1)` on all injected reps (rd/me/pixel/FFT). Full-scale injection (×0.1) without collapse.

---

## 📊 Verification (RTX 5070 12GB)
| Test | Result |
|---|---|
| Text `--sample 10` | 6.98→0.0001 auto-stop |
| Vision (28 pairs) | `图(苹果)→苹果` ✓ |
| Audio (sim FFT) | `声→文:高音` ✓ |
| Full 3-modality | Same engine, all flowing ✓ |

*Small-scale only. Architectural feasibility, not production.*

---

## 🔬 Related
- **vLLM PR #47491** — Fix hybrid KV cache truncation
- **vLLM #43090** — `has_initial_states` root-cause chain
- **DeepSeek-V3** — Graviton Anchors (adopted by AmoebaFPS → GDI)

*Independent research. Open to collaboration.*
