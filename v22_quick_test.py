# -*- coding: utf-8 -*-
"""V22.1 vMF 快速验证 — 15条数据, 50 epoch, 验证vMF gate训练稳定性"""
import torch, torch.nn as nn, torch.nn.functional as F, random, sys, os
sys.path.insert(0, 'C:/ai')
from v22_vmf_modules import (TurboMeta_vMF, MasterGate_vMF,
    SphericalSelfOrganizingLoss, MutualInformationPenalty,
    KappaPhaseScheduler, LocalBettiMonitor, SphericalVoronoiProbe)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"设备: {device} | V22.1 vMF Gate 快速验证 | 15条数据 50epoch")

# ── 简化版模型组件 (只用Gate路径验证vMF) ──
class TinyModel(nn.Module):
    def __init__(self, vocab=100, dim=256):
        super().__init__()
        self.embed = nn.Embedding(vocab, dim)
        self.encoder = nn.Linear(dim, dim)
        self.decoder = nn.Linear(dim, vocab)
        nn.init.orthogonal_(self.embed.weight)

    def forward(self, x, gate):
        h = self.embed(x)  # [bs, seq, dim]
        h = F.gelu(self.encoder(h))
        h = h * gate.unsqueeze(1)  # vMF gate modulation per seq position
        return self.decoder(h)  # [bs, seq, vocab]

# ── 合成数据 ──
vocab = 50
chars = [chr(0x4e00 + i) for i in range(vocab)]
pairs = [(random.sample(chars[:30], random.randint(2,5)),
          random.sample(chars[20:], random.randint(2,5))) for _ in range(15)]
print(f"数据: {len(pairs)} 对 | batch=2 | grad_accum=2")

# ── 模型 ──
model = TinyModel(vocab=vocab, dim=128).to(device)
explore = nn.Sequential(nn.Linear(128, 256), nn.GELU(), nn.Linear(256, 128)).to(device)
meta = TurboMeta_vMF(dim=128).to(device)  # ← vMF核心
master_gate = MasterGate_vMF(in_dim=5, hidden=16).to(device)

# ── 反冗余损失 ──
ssol = SphericalSelfOrganizingLoss(repulsion_strength=0.01)
mi_pen = MutualInformationPenalty(penalty_weight=0.005)

# ── 监控 ──
kappa_sched = KappaPhaseScheduler(total_steps=375, explore_ratio=0.4)
betti = LocalBettiMonitor(n_heads=128, k_neighbors=5)

# ── 优化器 ──
opt = torch.optim.Adam(list(model.parameters()) + list(explore.parameters()) +
    list(meta.parameters()) + list(master_gate.parameters()), lr=0.001)

# ── 训练 ──
print(f"\n{'='*50}")
print(f"  Epoch |  CE   | kappa  | gate_std | SSOL | phase")
print(f"{'='*50}")

for epoch in range(50):
    random.shuffle(pairs)
    epoch_ce = 0.0

    for i in range(0, len(pairs)-1, 2):
        A_chars, B_chars = pairs[i][0], pairs[i][1]
        A_ids = torch.tensor([c for c in range(len(A_chars))], device=device)
        B_ids = torch.tensor([c for c in range(len(B_chars))], device=device)

        # Forward
        emb = model.embed(A_ids.unsqueeze(0))  # [1, seq, 128]
        gate_input = emb.mean(dim=1)  # [1, 128]
        gate_hidden = explore(gate_input)  # [1, 128]
        gates = meta(gate_hidden)  # [1, 128] via vMF

        logits = model(A_ids.unsqueeze(0), gates)  # [1, len_A, vocab]
        # pad B_ids to match logits seq length
        target_len = min(len(B_ids), logits.shape[1])
        ce = F.cross_entropy(logits[0, :target_len, :], B_ids[:target_len])

        # SSOL: 用gate参数作为"微头"监控
        head_vecs = meta.mu.unsqueeze(0)  # [1, 128] = 128 "heads" each 1D
        ssol_loss = ssol(head_vecs.T) if head_vecs.shape[0] > 1 else torch.tensor(0.0)
        mi_loss = mi_pen(head_vecs.T) if head_vecs.shape[0] > 1 else torch.tensor(0.0)

        # MasterGate
        sub_losses = torch.tensor([0.1, 0.2, 0.3, 0.1, ce.detach()], device=device).unsqueeze(0)
        temps = master_gate(sub_losses)

        loss = ce + 0.01 * ssol_loss + 0.005 * mi_loss
        loss.backward()
        opt.step()
        opt.zero_grad()

        epoch_ce += ce.item()
        kappa_sched.step()

    avg_ce = epoch_ce / max(1, len(pairs)//2)
    k_stats = meta.get_kappa_stats()
    phase = kappa_sched.phase

    if epoch % 5 == 0:
        with torch.no_grad():
            dummy = torch.randn(1, 128, device=device)
            g = meta(explore(dummy))
        print(f"  E{epoch:3d}  | {avg_ce:.4f} | {k_stats['mean']:.2f}  "
              f"| {g.std().item():.4f}    | {ssol_loss.item():.4f} | {phase}")

print(f"\n{'='*50}")
print(f"  V22.1 vMF Gate 快速验证完成!")
print(f"  kappa_final={meta.get_kappa_stats()['mean']:.2f} "
      f"phase={kappa_sched.phase}")
print(f"{'='*50}")
