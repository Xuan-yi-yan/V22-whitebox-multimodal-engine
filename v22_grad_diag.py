# -*- coding: utf-8 -*-
"""V22.1 梯度诊断 — 1对数据, 看vMF gate每个参数的梯度流"""
import torch, torch.nn as nn, torch.nn.functional as F, sys
sys.path.insert(0, 'C:/ai')
from v22_vmf_modules import TurboMeta_vMF, MasterGate_vMF

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.manual_seed(42)
print(f"设备: {device}\n")

# ── 模型 (模拟V22核心路径) ──
DIM = 60  # 3的倍数, 三路gate拼接对齐

class MiniV22(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(100, DIM)
        self.p7_encoder = nn.Linear(DIM, DIM)
        self.abc_net = nn.Sequential(nn.Linear(DIM, 32), nn.GELU(), nn.Linear(32, 12))
        self.explore = nn.Sequential(nn.Linear(DIM, 128), nn.GELU(), nn.Linear(128, DIM))
        self.decoder = nn.Linear(DIM, 100)
        nn.init.orthogonal_(self.embed.weight)

    def forward(self, x, meta, gate_base_proj, abc_to_gate, sent_to_gate):
        h = self.embed(x).mean(dim=0)           # [seq, DIM] -> [DIM]
        sv = F.gelu(self.p7_encoder(h))          # sent_vec
        abc_out = self.abc_net(h)                # ac_batch

        # Gate: 三路拼接 → explore → vMF meta
        base = gate_base_proj(torch.zeros(1, 12, device=device))
        abc_sig = abc_to_gate(abc_out).unsqueeze(0)
        sent_sig = sent_to_gate(sv).unsqueeze(0)
        gate_input = torch.cat([base, abc_sig, sent_sig], dim=-1)
        gate_hidden = self.explore(gate_input)
        gates = meta(gate_hidden)                # ← vMF

        h = h * gates.squeeze(0)
        return self.decoder(h)

# ── 初始化 ──
model = MiniV22().to(device)
meta = TurboMeta_vMF(dim=DIM).to(device)        # ← vMF Gate
gate_base = nn.Linear(12, DIM//3, bias=False).to(device)
abc2gate = nn.Linear(12, DIM//3, bias=False).to(device)
sent2gate = nn.Linear(DIM, DIM//3, bias=False).to(device)

x = torch.tensor([1,2,3,4], device=device)       # 4个字的句子
target = torch.tensor([5], device=device)         # 预测第5个字

# ── 捕获参数变化 ──
def snap_params(mod, name):
    return {f"{name}.{n}": p.clone().detach() for n,p in mod.named_parameters() if p.requires_grad}

all_modules = {
    'embed': model.embed, 'p7': model.p7_encoder, 'abc': model.abc_net,
    'explore': model.explore, 'decoder': model.decoder,
    'meta_vMF': meta, 'gate_base': gate_base, 'abc2gate': abc2gate, 'sent2gate': sent2gate
}

snap_before = {}
for name, mod in all_modules.items():
    snap_before.update(snap_params(mod, name))

# ── Forward ──
logits = model(x, meta, gate_base, abc2gate, sent2gate).unsqueeze(0)  # [1, vocab]
loss = F.cross_entropy(logits, target)

# ── Backward ──
opt = torch.optim.Adam(sum([list(m.parameters()) for m in all_modules.values()], []), lr=0.001)
opt.zero_grad()
loss.backward()
opt.step()

# ── 计算变化 ──
snap_after = {}
for name, mod in all_modules.items():
    snap_after.update(snap_params(mod, name))

print(f"{'='*60}")
print(f"  V22.1 vMF Gate 梯度诊断 — 1对数据 | loss={loss.item():.4f}")
print(f"{'='*60}")
print(f"  {'模块':<15} {'参数数':>8} {'|grad|':>10} {'|Δparam|':>10} {'Δ/param':>12}")
print(f"  {'-'*55}")

module_stats = {}
for mname in all_modules:
    total_params = 0
    total_grad = 0.0
    total_delta = 0.0
    for full_name, before in snap_before.items():
        if full_name.startswith(mname):
            after = snap_after[full_name]
            total_params += before.numel()
            delta = (after - before).abs().sum().item()
            total_delta += delta
            if hasattr(before, 'grad') and before.grad is not None:
                total_grad += before.grad.abs().sum().item()
    module_stats[mname] = (total_params, total_grad, total_delta)

for mname, (params, grad_norm, delta) in module_stats.items():
    avg_delta = delta / max(params, 1)
    marker = " ← vMF" if 'meta' in mname else ""
    print(f"  {mname:<15} {params:>8} {grad_norm:>10.6f} {delta:>10.6f} {avg_delta:>12.8f}{marker}")

# ── vMF 专项: κ 和 μ 的变化 ──
print(f"\n{'='*60}")
print(f"  vMF Gate 专项诊断")
print(f"{'='*60}")
mu_before = snap_before['meta_vMF.mu']
mu_after = snap_after['meta_vMF.mu']
kappa_before = snap_before['meta_vMF.kappa']
kappa_after = snap_after['meta_vMF.kappa']
bias_before = snap_before['meta_vMF.bias']
bias_after = snap_after['meta_vMF.bias']

mu_delta = (mu_after - mu_before).abs()
kappa_delta = (kappa_after - kappa_before).abs()
bias_delta = (bias_after - bias_before).abs()

print(f"  μ 偏好方向:  mean(|Δ|)={mu_delta.mean().item():.8f}  max(|Δ|)={mu_delta.max().item():.8f}")
print(f"  κ 浓度参数:  mean(|Δ|)={kappa_delta.mean().item():.8f}  max(|Δ|)={kappa_delta.max().item():.8f}")
print(f"  bias偏移:    mean(|Δ|)={bias_delta.mean().item():.8f}  max(|Δ|)={bias_delta.max().item():.8f}")
with torch.no_grad():
    dummy = torch.randn(1, DIM, device=device)
    g = meta(dummy)
    print(f"  gate激活:    mean={g.mean().item():.4f} std={g.std().item():.4f}")

# ── 预算占比 ──
total = sum(s[0] for s in module_stats.values())
print(f"\n  总参数: {total} | vMF占比: {module_stats['meta_vMF'][0]/total*100:.1f}%")
print(f"  vMF Δ占比: {module_stats['meta_vMF'][2]/max(sum(s[2] for s in module_stats.values()),1e-10)*100:.1f}%")
