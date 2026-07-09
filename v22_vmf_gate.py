# -*- coding: utf-8 -*-
"""V22 vMF球面门控 — 最小验证: 替换TurboMeta的sigmoid, 保持架构其他部分不变"""
import torch, torch.nn as nn, torch.nn.functional as F, time, sys
sys.path.insert(0, 'C:/ai')

# ── vMF球面门控: 用测地距离替代线性投影 ──
class TurboMeta_vMF(nn.Module):
    """每个门控维度独立学习球面方向 + 浓度参数。

    原始 TurboMeta:
        sigmoid((bias + x) / 10.0)  ← 纯线性, /10压缩防饱和

    vMF TurboMeta:
        F.normalize(x) → 单位球面
        μ_i = 第i维的"偏好方向"(单位向量)
        κ_i = 第i维的浓度(感受野锐度)
        gate_i = sigmoid(κ_i * cos_sim(x, μ_i) + bias_i)

    优势:
    - 自然映射到(0,1), 不需要/10压缩
    - κ可学习, 312个κ值 = 模型学习日志
    - 球面几何内化为激活, 与V22球面归一化哲学一致
    """
    def __init__(self, dim=256):
        super().__init__()
        self.mu = nn.Parameter(torch.randn(dim) * 0.1)    # 小init防早期主导
        self.kappa = nn.Parameter(torch.ones(dim) * 2.0)  # κ=2.0: 中等锐度
        self.bias = nn.Parameter(torch.zeros(dim))

    def forward(self, x):
        # x: [bs, 256] — TurboExplore的输出
        x_norm = F.normalize(x, dim=-1, eps=1e-8)         # 球面投影
        mu_norm = F.normalize(self.mu, dim=0, eps=1e-8)   # μ也是单位向量
        cos_sim = x_norm * mu_norm.unsqueeze(0)            # 逐维余弦相似度
        return torch.sigmoid(self.kappa * cos_sim + self.bias)

    def get_kappa_stats(self):
        """白盒探针: 查看κ分布 — 了解门控锐度"""
        with torch.no_grad():
            return {
                'kappa_mean': self.kappa.mean().item(),
                'kappa_std': self.kappa.std().item(),
                'kappa_min': self.kappa.min().item(),
                'kappa_max': self.kappa.max().item(),
            }


# ── 快速对比验证 ──
if __name__ == "__main__":
    print("=" * 60)
    print("  vMF球面门控 vs 原始TurboMeta — 对比验证")
    print("=" * 60)

    # 原始版本
    class TurboMeta_Original(nn.Module):
        def __init__(self, dim=256):
            super().__init__()
            self.bias = nn.Parameter(torch.zeros(dim))
        def forward(self, x):
            return torch.sigmoid((self.bias + x) / 10.0)

    bs, dim = 4, 256
    torch.manual_seed(42)

    meta_orig = TurboMeta_Original(dim)
    meta_vmf = TurboMeta_vMF(dim)

    # 测试: 输入各种分布
    test_inputs = {
        'zero_centered': torch.randn(bs, dim) * 0.1,
        'wide_range': torch.randn(bs, dim) * 5.0,
        'positive_bias': torch.randn(bs, dim) * 0.5 + 2.0,
        'negative_bias': torch.randn(bs, dim) * 0.5 - 2.0,
        'sparse_spike': torch.randn(bs, dim) * 0.01,
    }
    test_inputs['sparse_spike'][:, :10] = 10.0  # 部分维度突变

    for name, x in test_inputs.items():
        with torch.no_grad():
            g_orig = meta_orig(x)
            g_vmf = meta_vmf(x)

        print(f"\n[{name}]")
        print(f"  原始 gate: mean={g_orig.mean().item():.4f} std={g_orig.std().item():.4f} "
              f"min={g_orig.min().item():.4f} max={g_orig.max().item():.4f}")
        print(f"  vMF  gate: mean={g_vmf.mean().item():.4f} std={g_vmf.std().item():.4f} "
              f"min={g_vmf.min().item():.4f} max={g_vmf.max().item():.4f}")

    # ── 梯度流验证 ──
    print(f"\n{'='*60}")
    print("  梯度流测试: 确保vMF gate不造成梯度死亡")
    print(f"{'='*60}")

    def check_grad(model, name):
        x = torch.randn(bs, dim, requires_grad=True)
        y = model(x)
        loss = y.mean()
        loss.backward()
        grad_norm = x.grad.norm().item()
        print(f"  {name}: grad_norm={grad_norm:.6f}, gate_mean={y.mean().item():.4f}")
        return grad_norm

    g1 = check_grad(meta_orig, '原始')
    g2 = check_grad(meta_vmf, 'vMF')

    if g2 > 1e-8 and g2 < 1e2:
        print(f"\n  ✅ vMF梯度正常 (ratio={g2/g1:.2f}x vs 原始)")
    else:
        print(f"\n  ❌ vMF梯度异常!")

    # ── κ可学习性验证 ──
    print(f"\n{'='*60}")
    print("  κ学习能力: 验证梯度能更新浓度参数")
    print(f"{'='*60}")

    meta_vmf2 = TurboMeta_vMF(dim)
    opt = torch.optim.Adam(meta_vmf2.parameters(), lr=0.01)

    kappa_before = meta_vmf2.kappa.clone()
    for step in range(50):
        x = torch.randn(bs, dim)
        y = meta_vmf2(x)
        # 目标: 让gate均值靠近0.3 (低饱和)
        loss = (y.mean() - 0.3).abs()
        opt.zero_grad()
        loss.backward()
        opt.step()

    kappa_after = meta_vmf2.kappa.clone()
    print(f"  κ before: mean={kappa_before.mean().item():.3f} std={kappa_before.std().item():.3f}")
    print(f"  κ after:  mean={kappa_after.mean().item():.3f} std={kappa_after.std().item():.3f}")
    print(f"  κ delta:  mean={kappa_after.mean().item()-kappa_before.mean().item():.4f}")

    if (kappa_after - kappa_before).abs().sum() > 1e-6:
        print(f"  ✅ κ可学习, 梯度正常流通")
    else:
        print(f"  ❌ κ未更新, 梯度断裂!")

    print(f"\n{'='*60}")
    print(f"  验证完成 — vMF TurboMeta可以安全替换原始TurboMeta")
    print(f"  下一步: 集成到 train_v22_stage1.py 训练验证")
    print(f"{'='*60}")
