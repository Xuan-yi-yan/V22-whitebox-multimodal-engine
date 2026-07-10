# V22.1 White-Box Multimodal Cognitive Engine

**~11M params. Text / Vision / Audio — one engine. ABCDEF cognitive stack. Spherical vMF gating. Single RTX 5070 12GB.**

> vLLM Contributor | Found & root-caused hybrid KV cache correctness bug (#43090 → #47782) | Active PR: scheduler fix #48195

```bash
git clone https://github.com/Xuan-yi-yan/V22-whitebox-multimodal-engine.git
```

---

## 📂 Repo Directory

| File | What |
|------|------|
| `V22_架构终极全貌_2026-07-06.txt` | V22.1 architecture doc (spherical gate upgrade) |
| `v22_vmf_modules.py` | 16 relay debate proposals: vMF gates, κ scheduler, topology, 7-Adam coord, probes, anti-redundancy |
| `v22_vmf_gate.py` | vMF vs sigmoid gate benchmark |
| `train_v22_stage1.py` | Training script (vMF patched) |
| `train_v22_stage1_original.py` | Original backup |
| `v22_quick_test.py` | 15-sample vMF gate validation (CE 3.87→0.002) |
| `v22_grad_diag.py` | Per-module gradient diagnostic |
| `v22_real_grad_diag.py` | Real-size 2.68M param gradient flow test |
| `v22_emotion_data.py` | Emotion dialogue dataset generator (480 samples) |
| `v22_emotion_train.py` | Emotion-conditioned dialogue training |
| `v22_relay_debate_24r.py` | 24-round relay debate script (6 models × 4 rounds) |
| `v22_audit_six_models.py` | 6-model parallel architecture audit |
| `relay_r*.txt` | 30 relay debate output files |
| `audit_r*.txt` | 18 audit output files |
| `multi_model_audit_announcement.txt` | Full audit announcement |
| `patch_v22_vmf.py` | Automated vMF patching script |
| `resume_en.html` | English resume |
| `简历_卫锦旗_详细版.html` | Chinese resume |

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

## 🔮 V22.1: Spherical vMF Gate Upgrade (2026-07-09)

24-round relay debate (6 models × 4 rounds with compressed knowledge passing) produced 16 architecture proposals. **16 modules implemented** — architecture structure unchanged, control logic upgraded.

| Layer | Modules |
|-------|---------|
| **Spherical Geometry** | TurboMeta_vMF (sigmoid→geodesic distance), P7Gate_vMF, MasterGate_vMF |
| **Head Self-Org** | KappaPhaseScheduler (explore→crystallize), CompetitiveCrystallization, Temperature Annealing |
| **Topology Monitor** | LocalBettiMonitor (β₀ collapse early-warning, O(k log n)) |
| **7-Adam Coord** | MomentumExchangePool (decoupled signals), PCACurvatureProxy, LocalCurvatureEstimator |
| **White-Box Probes** | AdaptiveProbeSet, ActiveExperimentProbe (causal matrix C) |
| **Anti-Redundancy** | SSOL (entropy-modulated repulsion), MutualInformationPenalty |

**Verified**: vMF gate gradient healthy at 0.00092/param (comparable to P6/P7). κ stable at 2.0, no saturation. 50-epoch quick test: CE 3.87→0.002. Checkpoint compatible with V22.

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
- **vLLM PR #48195** — Fix per-group prefix-hit divergence in hybrid scheduler (active)
- **vLLM #43090** — Root-cause: `has_initial_states` boundary (→ maintainer fix #47782 by njhill)
- **vLLM PR #47491** — Coordinator-level guard, closed in favor of #47782
- **vLLM PR #11314** — Progressive alignment for DeepSeek V4 spec decoding (vllm-ascend)
- **vLLM RFC #11356** — PD-disaggregated Mamba routing (vllm-ascend)
- **DeepSpec RFC #52** — Architectural reference for DSpark
- **DeepSeek-V3** — Graviton Anchors (adopted by AmoebaFPS → GDI framework)

*Independent research. Open to collaboration. Reach out: 1503163696@qq.com*
