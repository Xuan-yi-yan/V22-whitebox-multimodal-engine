# -*- coding: utf-8 -*-
"""V22.1 真实模型梯度诊断 — 2字符, 看每个模块参数变化"""
import torch, torch.nn as nn, torch.nn.functional as F, sys, os
sys.path.insert(0, 'C:/ai')
from v22_vmf_modules import TurboMeta_vMF, MasterGate_vMF

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.manual_seed(42)

# ── 用真实V22尺寸, 但只跑forward/backward一次 ──
class TurboExplore(nn.Module):
    def __init__(self, in_dim=384, hidden=512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, 256))
        for l in self.net:
            if isinstance(l, nn.Linear):
                nn.init.xavier_uniform_(l.weight, gain=0.1)
                nn.init.zeros_(l.bias)
    def forward(self, x): return self.net(x)

# ── 真实尺寸模型 ──
char_embed = nn.Embedding(3000, 256)
nn.init.orthogonal_(char_embed.weight)

# P7 (简化: 真实P7太复杂, 用等效Linear)
p7_proj = nn.Linear(256, 256)

# ABC
abcA_enc = nn.Linear(256, 128)  # hidden state
abcA_head = nn.Linear(128, 20)   # classification
abcB = nn.Sequential(nn.Linear(128+256, 256), nn.GELU(), nn.Linear(256, 384))
abcC = nn.Sequential(nn.Linear(384+5, 96), nn.GELU(), nn.Linear(96, 48))

# P6
p6_encoder = nn.Sequential(nn.Linear(256, 256), nn.GELU(), nn.Linear(256, 256))
p6_head = nn.Linear(256, 3000)  # 单头简化

# ── Gate (真实V22尺寸) ──
explore = TurboExplore(in_dim=384, hidden=512)
meta = TurboMeta_vMF(dim=256)
master_gate = MasterGate_vMF(in_dim=5, hidden=32)
gate_base_proj = nn.Linear(12, 128, bias=False)
abc_to_gate = nn.Linear(48, 128, bias=False)
sent_to_gate = nn.Linear(256, 128, bias=False)
abc_to_sent = nn.Linear(128, 256, bias=False)
nn.init.normal_(gate_base_proj.weight, std=0.03)
nn.init.normal_(abc_to_gate.weight, std=0.01)
nn.init.normal_(sent_to_gate.weight, std=0.03)

# ── 全部模块 ──
all_mods = {
    'char_embed': char_embed, 'p7_proj': p7_proj,
    'abcA_enc': abcA_enc, 'abcA_head': abcA_head, 'abcB': abcB, 'abcC': abcC,
    'p6_encoder': p6_encoder, 'p6_head': p6_head,
    'explore': explore, 'meta_vMF': meta, 'master_gate': master_gate,
    'gate_base': gate_base_proj, 'abc2gate': abc_to_gate, 'sent2gate': sent_to_gate, 'abc2sent': abc_to_sent
}

all_mods = {k: v.to(device) for k, v in all_mods.items()}

# ── 2个字符 ──
c2i = {chr(0x4e00+i): i for i in range(100)}
A_chars = ['一', '二']
B_char = '三'
A_ids = torch.tensor([c2i.get(c, 0) for c in A_chars], device=device)
B_id = torch.tensor([c2i.get(B_char, 0)], device=device)

# ── Forward ──
emb = char_embed(A_ids)                                            # [2, 256]
sv = p7_proj(emb.mean(dim=0))                                      # [256]

# ABC
ah = F.gelu(abcA_enc(sv))                                        # [128] (hidden)
ct = F.gelu(abcB(torch.cat([ah, sv], dim=-1)))                     # [384]
ac = F.gelu(abcC(torch.cat([ct, torch.zeros(5, device=device)], dim=-1)))  # [48]

# Gate (真实三路拼接)
abc_conf = torch.zeros(12, device=device).unsqueeze(0)
base = gate_base_proj(abc_conf)
abc_sig = abc_to_gate(ac).unsqueeze(0)
sent_sig = sent_to_gate(sv).unsqueeze(0)
gate_input = torch.cat([base, abc_sig, sent_sig], dim=-1)          # [1, 384]
gate_hidden = explore(gate_input)
gates = meta(gate_hidden)                                          # [1, 256] via vMF

# P6 decode
sv_rich = sv + abc_to_sent(ah) * 0.1
h = F.gelu(p6_encoder(sv_rich))
h = h * gates                                                      # vMF modulation
logits = p6_head(h)                                                 # [1, 3000]

loss = F.cross_entropy(logits, B_id)

# ── Snapshot BEFORE step ──
params = [p for m in all_mods.values() for p in m.parameters()]
opt = torch.optim.Adam(params, lr=0.001)
opt.zero_grad()

# Snapshot all parameters before backward
snaps = {name: [p.clone().detach() for p in mod.parameters()] for name, mod in all_mods.items()}

loss.backward()
opt.step()

# ── 结果 ──
total_params = 0
total_delta = 0.0
print(f"{'='*60}")
print(f"  V22.1 真实模型 | 2字符: {'+'.join(A_chars)} → {B_char} | loss={loss.item():.4f}")
print(f"{'='*60}")
print(f"  {'模块':<18} {'参数':>10} {'|Δθ|':>12} {'|Δ/param':>12}")
print(f"  {'-'*55}")

for name, mod in all_mods.items():
    mod_params = sum(p.numel() for p in mod.parameters())
    mod_delta = sum((p - s).abs().sum().item() for p, s in zip(mod.parameters(), snaps[name]))
    total_params += mod_params
    total_delta += mod_delta
    avg_delta = mod_delta / max(mod_params, 1)
    marker = " ← vMF" if 'meta' in name or 'master' in name else ""
    print(f"  {name:<18} {mod_params:>10,} {mod_delta:>12.4f} {avg_delta:>12.8f}{marker}")

print(f"  {'-'*55}")
print(f"  {'总计':<18} {total_params:>10,} {total_delta:>12.4f} {total_delta/max(total_params,1):>12.8f}")
print(f"\n  每字符: |Δθ| = {total_delta/2:.2f} | 参数总量 = {total_params:,}")
print(f"  参数活跃率: {total_delta/total_params*100:.6f}%")
