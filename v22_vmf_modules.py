# -*- coding: utf-8 -*-
"""V22 架构升级模块 — 24轮环形接力辩论全部方案落地

包含:
  球面几何:  vMF TurboMeta, vMF P7 Gate, vMF MasterGate
  微头自组织: κ相变调度器, 竞争性结晶, 温度退火
  拓扑监控:   局部Betti数追踪, 持久同调重生
  7-Adam协调: 动量交换池, PCA曲率代理, LCE
  白盒探针:   自适应探针加密, 主动因果实验
  反冗余:     SSOL损失, 互信息冗余惩罚
"""
import torch, torch.nn as nn, torch.nn.functional as F, math, time
from collections import deque

# ═══════════════════════════════════════════════════════════
# 第一层: 球面几何增强 — vMF激活体系
# ═══════════════════════════════════════════════════════════

class TurboMeta_vMF(nn.Module):
    """vMF球面门控 — 替代原始TurboMeta的sigmoid。

    原始: sigmoid((bias + x) / 10.0)  — 线性投影+压缩防饱和
    vMF:  每个门控维度学习球面偏好方向μ_i + 浓度κ_i
           gate_i = sigmoid(κ_i * cos_sim(F.normalize(x), μ_i) + bias_i)

    球面拓扑内化为激活机制, 自然映射(0,1), 无需/10压缩。
    312个κ值本身就是模型学习日志。
    """
    def __init__(self, dim=256):
        super().__init__()
        self.mu = nn.Parameter(torch.randn(dim) * 0.1)       # 球面偏好方向
        self.kappa = nn.Parameter(torch.ones(dim) * 2.0)     # 浓度(感受野锐度)
        self.bias = nn.Parameter(torch.zeros(dim))

    def forward(self, x):
        x_norm = F.normalize(x, dim=-1, eps=1e-8)
        mu_norm = F.normalize(self.mu, dim=0, eps=1e-8)
        cos_sim = x_norm * mu_norm.unsqueeze(0)               # [bs, dim]
        return torch.sigmoid(self.kappa * cos_sim + self.bias)

    def get_kappa_stats(self):
        with torch.no_grad():
            return dict(mean=self.kappa.mean().item(), std=self.kappa.std().item(),
                        min=self.kappa.min().item(), max=self.kappa.max().item(),
                        saturated=(self.kappa > 8.0).sum().item())

    def trigger_phase_transition(self, mask, alpha_boost=5.0):
        """κ相变: 选中微头κ突增, 从探索→结晶"""
        with torch.no_grad():
            self.kappa[mask] *= alpha_boost
            self.kappa.clamp_(max=20.0)


class P7Gate_vMF(nn.Module):
    """P7内部vMF门控 — 替代meta_word_gate和meta_sent_gate的sigmoid。

    原始: word_gate = sigmoid(meta_word_gate(meta_input) * gate_sharpness)
          sent_gate = sigmoid(meta_sent_gate(meta_input) * gate_sharpness)

    vMF:   每个维度学习球面方向, gate基于测地距离而非线性投影强度。
    """
    def __init__(self, in_dim, out_dim, hidden=96):
        super().__init__()
        self.fc = nn.Sequential(nn.Linear(in_dim, hidden), nn.GELU(), nn.Linear(hidden, out_dim))
        self.mu = nn.Parameter(torch.randn(out_dim) * 0.1)
        self.kappa = nn.Parameter(torch.ones(out_dim) * 2.0)
        self.bias = nn.Parameter(torch.zeros(out_dim))

    def forward(self, x):
        # x: [bs, in_dim] → fc → [bs, out_dim] → normalize → vMF
        h = self.fc(x)
        h_norm = F.normalize(h, dim=-1, eps=1e-8)
        mu_norm = F.normalize(self.mu, dim=0, eps=1e-8)
        cos_sim = h_norm * mu_norm.unsqueeze(0)
        return torch.sigmoid(self.kappa * cos_sim + self.bias)

    def get_kappa(self):
        return self.kappa.detach()


class MasterGate_vMF(nn.Module):
    """vMF版MasterGate — 5个子损失温度控制。

    原始: sigmoid(net(sub_losses))  ← 纯MLP+sigmoid
    vMF:   5个损失信号投影到球面, 每个温度学自己的偏好方向
    """
    def __init__(self, in_dim=5, hidden=32):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, hidden), nn.GELU(),
                                  nn.Linear(hidden, hidden), nn.GELU(),
                                  nn.Linear(hidden, 5))
        self.mu = nn.Parameter(torch.randn(5) * 0.1)
        self.kappa = nn.Parameter(torch.ones(5) * 2.0)
        self.bias = nn.Parameter(torch.zeros(5))
        for l in self.net:
            if isinstance(l, nn.Linear):
                nn.init.xavier_uniform_(l.weight, gain=0.1)
                nn.init.zeros_(l.bias)

    def forward(self, sub_losses):
        h = self.net(sub_losses)
        h_norm = F.normalize(h, dim=-1, eps=1e-8)
        mu_norm = F.normalize(self.mu, dim=0, eps=1e-8)
        cos_sim = h_norm * mu_norm.unsqueeze(0)
        return torch.sigmoid(self.kappa * cos_sim + self.bias)


# ═══════════════════════════════════════════════════════════
# 第二层: 微头自组织 — κ相变 + 竞争性结晶 + 温度退火
# ═══════════════════════════════════════════════════════════

class KappaPhaseScheduler:
    """κ相变双阶段调度器。

    探索期: κ=0.5~2.0, 微头感受野宽, 自由竞争球面区域
    结晶期: κ突增至5.0~15.0, 微头感受野窄, 专精特定方向
    触发条件: 连续N步Voronoi密度稳定 → 相变
    """
    def __init__(self, total_steps=1000, explore_ratio=0.4):
        self.total_steps = total_steps
        self.explore_steps = int(total_steps * explore_ratio)
        self.crystallize_steps = total_steps - self.explore_steps
        self.current_step = 0
        self.phase = 'explore'  # 'explore' | 'crystallize'
        self.crystallized_heads = set()

    def step(self):
        self.current_step += 1
        if self.phase == 'explore' and self.current_step >= self.explore_steps:
            self.phase = 'crystallize'

    def get_temperature(self):
        """退火温度: 探索期高(允许迁移), 结晶期低(锁定功能)"""
        if self.phase == 'explore':
            return max(0.1, 1.0 - 0.5 * self.current_step / self.explore_steps)
        else:
            progress = (self.current_step - self.explore_steps) / self.crystallize_steps
            return 0.5 * (1.0 - progress)


class CompetitiveCrystallization:
    """竞争性结晶: 高密度区域只有Voronoi密度最高的微头获得κ突变权。

    区域内其他微头κ被压制 → 迁移到未覆盖区域。
    避免312微头冗余堆叠, 确保球面覆盖多样性。
    """
    def __init__(self, n_heads=312, region_threshold=0.7, density_window=50):
        self.n_heads = n_heads
        self.region_threshold = region_threshold
        self.density_window = density_window
        self.activation_history = deque(maxlen=density_window)

    def record_activations(self, activations):
        """activations: [n_heads] 每个微头的激活密度"""
        self.activation_history.append(activations.detach().clone())

    def detect_high_density_regions(self):
        """返回: [(region_indices, winner_idx), ...]"""
        if len(self.activation_history) < 10:
            return []
        # 平均激活密度
        avg_acts = torch.stack(list(self.activation_history)).mean(dim=0)  # [n_heads]
        # 余弦相似度矩阵 → 识别聚集区域
        regions = []
        used = set()
        sim_matrix = F.cosine_similarity(
            avg_acts.unsqueeze(1), avg_acts.unsqueeze(0), dim=-1)
        for i in range(self.n_heads):
            if i in used:
                continue
            neighbors = (sim_matrix[i] > self.region_threshold).nonzero(as_tuple=True)[0].tolist()
            neighbors = [n for n in neighbors if n != i]
            if len(neighbors) >= 2:  # 至少2个邻居 = 高密度区域
                region = [i] + neighbors
                winner = max(region, key=lambda x: avg_acts[x].item())
                regions.append((region, winner))
                used.update(region)
        return regions


# ═══════════════════════════════════════════════════════════
# 第三层: 拓扑监控 — 局部Betti数 + 持久同调
# ═══════════════════════════════════════════════════════════

class LocalBettiMonitor:
    """局部Betti数追踪 — 实时区分良性重组 vs 恶性崩溃。

    β₀(连通分量数): 每个微头追踪k近邻的β₀
      β₀保持1但近邻轮换 = 良性重组
      β₀从1→0(成为孤立点) = 恶性崩溃早期信号

    计算复杂度: O(k log n), 每步可实时追踪。
    """
    def __init__(self, n_heads=312, k_neighbors=8):
        self.n_heads = n_heads
        self.k = k_neighbors
        self.prev_neighbors = [set() for _ in range(n_heads)]
        self.collapse_warnings = 0
        self.reorg_events = 0

    def step(self, head_vectors):
        """head_vectors: [n_heads, dim] 每个微头的激活向量"""
        n = head_vectors.shape[0]
        # 计算pairwise距离
        dists = torch.cdist(head_vectors, head_vectors)  # [n, n]
        # 每头找k近邻
        for i in range(n):
            _, indices = dists[i].topk(self.k + 1, largest=False)
            curr_neighbors = set(indices[1:].tolist())  # 排除自己
            prev = self.prev_neighbors[i]

            if prev and len(curr_neighbors) == 0:
                self.collapse_warnings += 1  # β₀ 1→0: 恶性崩溃
            elif prev and len(prev & curr_neighbors) < len(prev) * 0.3:
                self.reorg_events += 1      # 近邻大轮换: 良性重组
            self.prev_neighbors[i] = curr_neighbors

        return dict(collapse=self.collapse_warnings, reorg=self.reorg_events)

    def get_status(self):
        return f"collapse_warn={self.collapse_warnings} reorg={self.reorg_events}"


# ═══════════════════════════════════════════════════════════
# 第四层: 7-Adam协调 — 动量交换池 + PCA曲率代理 + LCE
# ═══════════════════════════════════════════════════════════

class MomentumExchangePool:
    """7-Adam动量交换池 — 共享曲率预警信号, 保持独立优化器身份。

    不统一优化器, 各模块独立决定是否采纳预警。
    '松耦合'而非统一化。
    """
    def __init__(self, n_modules=7):
        self.n_modules = n_modules
        self.curvature_signals = [0.0] * n_modules
        self.accepted_signals = [0] * n_modules
        self.rejected_signals = [0] * n_modules

    def broadcast(self, module_id, curvature):
        """模块发布自己的曲率信号"""
        self.curvature_signals[module_id] = curvature

    def query(self, module_id, threshold=0.3):
        """检查是否有需要关注的预警"""
        warnings = []
        for i, sig in enumerate(self.curvature_signals):
            if i != module_id and sig > threshold:
                warnings.append((i, sig))
        return warnings

    def record_decision(self, module_id, accepted):
        if accepted:
            self.accepted_signals[module_id] += 1
        else:
            self.rejected_signals[module_id] += 1


class PCACurvatureProxy:
    """PCA方差解释比曲率代理。

    欧几里得空间中黎曼曲率恒为零 → 用统计代理。
    动量轨迹PCA第一主成分方差解释比 R_k → 接近1 = 坍缩征兆。
    复杂度 O(dT), GPU并行毫秒级。
    """
    def __init__(self, window=50, warn_threshold=0.85):
        self.window = window
        self.warn_threshold = warn_threshold
        self.history = deque(maxlen=window)
        self.r_k = 0.0
        self.warning = False

    def step(self, momentum_vec):
        """momentum_vec: [d] 当前步的动量向量"""
        self.history.append(momentum_vec.detach().clone())
        if len(self.history) < 10:
            return self.r_k, False
        # 构建轨迹矩阵 [T, d]
        traj = torch.stack(list(self.history))
        # PCA: 协方差矩阵特征值
        traj_centered = traj - traj.mean(dim=0, keepdim=True)
        cov = traj_centered.T @ traj_centered / (traj.shape[0] - 1)
        eigenvalues = torch.linalg.eigvalsh(cov)  # 升序
        total_var = eigenvalues.sum()
        if total_var < 1e-10:
            self.r_k = 1.0
        else:
            self.r_k = (eigenvalues[-1] / total_var).item()  # 第一主成分方差比
        self.warning = self.r_k > self.warn_threshold
        return self.r_k, self.warning

    def get_orthogonal_perturbation(self, momentum_vec, scale=0.01):
        """PCA正交扰动: 沿第二主成分方向偏移, 逃离一维坍缩"""
        if len(self.history) < 10:
            return torch.zeros_like(momentum_vec)
        traj = torch.stack(list(self.history))
        traj_centered = traj - traj.mean(dim=0, keepdim=True)
        cov = traj_centered.T @ traj_centered / (traj.shape[0] - 1)
        eigenvalues, eigenvectors = torch.linalg.eigh(cov)
        if len(eigenvalues) < 2:
            return torch.zeros_like(momentum_vec)
        second_pc = eigenvectors[:, -2]  # 第二主成分
        return second_pc * scale


class LocalCurvatureEstimator:
    """局部曲率估计(LCE) — 短窗口+长窗口多尺度分析。

    自适应阈值: 训练初期高阈值→允许探索, 后期低阈值→精细控制。
    """
    def __init__(self, short_window=10, long_window=50):
        self.short_win = short_window
        self.long_win = long_window
        self.short_history = deque(maxlen=short_window)
        self.long_history = deque(maxlen=long_window)
        self.threshold = 0.5  # 自适应, 随训练递减

    def step(self, grad_norm):
        self.short_history.append(grad_norm)
        self.long_history.append(grad_norm)

    def estimate(self):
        if len(self.short_history) < 5:
            return 0.0, 0.0
        s = torch.tensor(list(self.short_history), dtype=torch.float32)
        l = torch.tensor(list(self.long_history), dtype=torch.float32)
        short_cv = s.std() / (s.mean() + 1e-8)
        long_cv = l.std() / (l.mean() + 1e-8)
        curvature = float(short_cv / (long_cv + 1e-8))  # >1 = 短期曲率增大
        return curvature.item(), self.threshold

    def step_threshold(self, total_steps, current_step, initial=0.7, final=0.2):
        """自适应阈值退火"""
        self.threshold = initial - (initial - final) * current_step / total_steps


# ═══════════════════════════════════════════════════════════
# 第五层: 白盒因果探针
# ═══════════════════════════════════════════════════════════

class AdaptiveProbeSet:
    """自适应探针加密 — 50个球面探针动态重分布。

    每1000步: 计算信息增益G_j → 高G区域加密, 低G区域稀疏。
    总数保持50不变。
    """
    def __init__(self, n_probes=50, probe_dim=256):
        self.n_probes = n_probes
        self.probe_dim = probe_dim
        # 初始均匀分布球面探针
        probes = torch.randn(n_probes, probe_dim)
        self.probes = nn.Parameter(F.normalize(probes, dim=-1))
        self.info_gains = torch.zeros(n_probes)
        self.step_count = 0

    def compute_info_gain(self, head_activations):
        """每个探针的信息增益 = 附近微头激活方差"""
        gains = []
        for i in range(self.n_probes):
            # 探针与各微头的余弦相似度
            similarities = F.cosine_similarity(
                self.probes[i:i+1], head_activations, dim=-1)
            gains.append(similarities.var().item())
        return torch.tensor(gains)

    def refine(self, head_activations):
        """自适应加密: 每1000步重分布"""
        self.step_count += 1
        if self.step_count % 1000 != 0:
            return
        gains = self.compute_info_gain(head_activations)
        # 找最高/最低G的探针
        _, top_idx = gains.topk(3)
        _, low_idx = (-gains).topk(3)
        # 高G区域加密: 生成子探针
        for idx in top_idx:
            perturb = torch.randn(self.probe_dim) * 0.05
            new_probe = F.normalize(self.probes[idx] + perturb, dim=-1)
            if idx < self.n_probes:
                self.probes.data[idx] = new_probe
        self.info_gains = gains


class ActiveExperimentProbe:
    """主动实验白盒探针 — 向微头激活值注入受控扰动。

    追踪扰动→输出的因果效应, 构建微头→输出因果矩阵C。
    """
    def __init__(self, n_heads=312, output_dim=256):
        self.n_heads = n_heads
        self.output_dim = output_dim
        self.causal_matrix = torch.zeros(n_heads, output_dim)  # C[i,j] = 头i→输出j的效应
        self.perturbation_scale = 0.01

    def inject_perturbation(self, head_activations, head_idx):
        """向指定微头注入受控扰动, 返回扰动后的激活"""
        perturbed = head_activations.clone()
        perturbed[head_idx] += torch.randn_like(perturbed[head_idx]) * self.perturbation_scale
        return perturbed

    def update_causal_matrix(self, delta_output, head_idx):
        """更新因果矩阵: C[i,:] += |Δoutput|"""
        self.causal_matrix[head_idx] += delta_output.abs().mean(dim=0).detach()

    def get_top_influential_heads(self, k=10):
        """返回对输出影响最大的k个微头"""
        influence = self.causal_matrix.norm(dim=-1)
        return influence.topk(min(k, self.n_heads))


# ═══════════════════════════════════════════════════════════
# 第六层: 反冗余机制 — SSOL + 互信息惩罚
# ═══════════════════════════════════════════════════════════

class SphericalSelfOrganizingLoss(nn.Module):
    """SSOL: 熵调制斥力势场。

    微头之间根据球面距离施加斥力。
    距离过近 → 更强排斥 → 鼓励球面覆盖多样性。
    """
    def __init__(self, repulsion_strength=0.01, min_distance=0.1):
        super().__init__()
        self.strength = repulsion_strength
        self.min_dist = min_distance

    def forward(self, head_vectors):
        """head_vectors: [n_heads, dim] 球面上的微头位置"""
        n = head_vectors.shape[0]
        sims = F.cosine_similarity(
            head_vectors.unsqueeze(1), head_vectors.unsqueeze(0), dim=-1)  # [n,n]

        # 只惩罚过近的微头对
        mask = (sims > self.min_dist).float() * (1.0 - torch.eye(n, device=sims.device))
        penalty = (mask * (sims - self.min_dist)).sum() / (mask.sum() + 1e-8)
        return penalty * self.strength


class MutualInformationPenalty(nn.Module):
    """互信息冗余惩罚 — 球面余弦相似度代理。

    避免低数据量下激活互信息估计的病态问题。
    """
    def __init__(self, penalty_weight=0.005, high_sim_threshold=0.85):
        super().__init__()
        self.weight = penalty_weight
        self.threshold = high_sim_threshold

    def forward(self, head_vectors):
        """head_vectors: [n_heads, dim]"""
        n = head_vectors.shape[0]
        sims = F.cosine_similarity(
            head_vectors.unsqueeze(1), head_vectors.unsqueeze(0), dim=-1)
        # 只惩罚高冗余对
        redundant = F.relu(sims - self.threshold)
        mask = 1.0 - torch.eye(n, device=sims.device)
        return (mask * redundant).sum() / (mask.sum() + 1e-8) * self.weight


# ═══════════════════════════════════════════════════════════
# 工具: 球面Voronoi密度探针 (不修改架构, 纯监控)
# ═══════════════════════════════════════════════════════════

class SphericalVoronoiProbe:
    """球面Voronoi密度探针 — 可视化312微头功能分化。

    固定N个均匀探针, 计算每个探针附近的微头密度。
    不参与Loss, 纯白盒监控。
    """
    def __init__(self, n_probes=50, n_heads=312, head_dim=None):
        self.n_probes = n_probes
        self.n_heads = n_heads
        # 均匀探针(留到第一次step时初始化)
        self.probes = None
        self.density = torch.zeros(n_probes)

    def step(self, head_vectors):
        """head_vectors: [n_heads, dim] 微头的球面位置"""
        dim = head_vectors.shape[1]
        if self.probes is None or self.probes.shape[1] != dim:
            self.probes = F.normalize(torch.randn(self.n_probes, dim), dim=-1)

        # 每个探针的密度 = 附近微头数/总微头数
        sims = F.cosine_similarity(
            self.probes.unsqueeze(1), head_vectors.unsqueeze(0), dim=-1)  # [n_probes, n_heads]
        self.density = (sims > 0.5).float().sum(dim=-1) / self.n_heads
        return self.density

    def get_coverage_stats(self):
        """球面覆盖统计: entropy高=均匀, 低=聚集"""
        d = self.density + 1e-8
        d = d / d.sum()
        entropy = -(d * torch.log(d)).sum().item()
        max_entropy = math.log(self.n_probes)
        return dict(
            entropy=entropy,
            coverage=entropy / max_entropy,
            max_density=self.density.max().item(),
            zero_regions=(self.density < 0.01).sum().item()
        )

