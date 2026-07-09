"""更新 V22 架构文档 → V22.1"""
import re

path = 'C:/ai/V22_架构终极全貌_2026-07-06.txt'
with open(path, 'r', encoding='utf-8') as f:
    content = f.read()

# Patch 1: version header
content = content.replace(
    'V22 白盒认知引擎 — 三层架构终极全貌',
    'V22.1 白盒认知引擎 — 球面门控架构升级'
)
content = content.replace(
    '时间: 2026-07-06 | 从 V21 单模态文本引擎演进而来(扩展, 非重构)',
    '时间: 2026-07-09 | V22基础上控制层vMF升级(16方案落地, 架构结构不变)'
)
content = content.replace(
    '原则: "扩大反而是真效率" — 维度不留缺, 位置先占稳, 控制层同步扩',
    '原则: "扩大反而是真效率" — 维度不留缺, 位置先占稳, 控制层同步扩\n\t  升级源: 6模型×24轮环形接力辩论 → 16个架构强化方案(v22_vmf_modules.py)'
)
print('[OK] Header updated')

# Patch 2: Gate section - more targeted
old_gate_sigmoid = '→ Sigmoid → gates[256] → 调制P6 encoder'
new_gate_vmf = '→ vMF球面门控: gate_i = sigmoid(κ_i · cos_sim(x_norm, μ_i) + bias_i)\n\t    → gates[256] → 调制P6 encoder (κ可学习, 球面拓扑内化)'
if old_gate_sigmoid in content:
    content = content.replace(old_gate_sigmoid, new_gate_vmf)
    print('[OK] Gate sigmoid -> vMF')
else:
    print('[FAIL] Gate sigmoid not found')

# Patch 3: TurboMeta description
old_tm = '→ TurboMeta(bias[256] + 输入, ÷10防饱和)'
new_tm = '→ TurboMeta_vMF(dim=256, μ+κ可学习, 球面偏好方向)'
if old_tm in content:
    content = content.replace(old_tm, new_tm)
    print('[OK] TurboMeta updated')
else:
    print('[FAIL] TurboMeta not found')

# Patch 4: Master Gate
old_mg = 'Master Gate [0.4K]\n\t    5-6子损失聚合 → 动态温控(0-1) → 各损失自动平衡'
new_mg = 'Master Gate [V22.1 vMF升级, ~1K]\n\t    MasterGate_vMF: 5子损失 → MLP(5→32→32→5) →\n\t    F.normalize → vMF(μ+κ) → 5路温控(0-1) → 球面偏好方向替代线性投影'
if old_mg in content:
    content = content.replace(old_mg, new_mg)
    print('[OK] MasterGate updated')
else:
    print('[FAIL] MasterGate not found')

# Patch 5: Add V22.1 upgrade section at end
upgrade_section = '''
\n\n════════════════════════════════════════════════════════════════
  V22.1 控制层升级 — 24轮环形接力辩论产出
  原则: 架构结构不动, 只升级控制逻辑层。checkpoint兼容, 可增量训练。
════════════════════════════════════════════════════════════════

  球面几何增强:
    TurboMeta_vMF — sigmoid→vMF球面门控, μ+κ可学习, 无需/10压缩
    P7Gate_vMF — P7内部word/sent gate → vMF版
    MasterGate_vMF — 5路温控 → 球面偏好方向

  微头自组织:
    KappaPhaseScheduler — 探索期(κ≈2)→结晶期(κ≈10)双阶段调度
    CompetitiveCrystallization — 高密度区赢家结晶, 输家迁移
    温度退火 — 早期高温探索→后期低温专精

  拓扑监控:
    LocalBettiMonitor — β₀实时追踪, 恶性崩溃vs良性重组, O(k log n)

  7-Adam协调:
    MomentumExchangePool — 共享曲率预警, 独立决定采纳, 松耦合
    PCACurvatureProxy — 动量轨迹PCA方差比, 坍缩预警可计算
    LocalCurvatureEstimator — 短/长窗口多尺度分析, 自适应阈值

  白盒探针:
    AdaptiveProbeSet — 50球面探针动态重分布高信息区域
    ActiveExperimentProbe — 受控扰动, 微头→输出因果矩阵C

  反冗余:
    SphericalSelfOrganizingLoss — 熵调制斥力, 过近微头对排斥
    MutualInformationPenalty — 余弦相似度代理, 高冗余对惩罚

  球面监控:
    SphericalVoronoiProbe — 312微头功能分化可视化, 覆盖熵度量

  文件: C:/ai/v22_vmf_modules.py (16个模块)
         C:/ai/train_v22_stage1.py (已打补丁: TurboMeta+MasterGate→vMF)
         备份: train_v22_stage1_original.py
'''

# Check if already has V22.1 section
if 'V22.1 控制层升级' not in content:
    content += upgrade_section
    print('[OK] V22.1 upgrade section appended')
else:
    print('[SKIP] V22.1 section already exists')

# Save
with open(path, 'w', encoding='utf-8') as f:
    f.write(content)
print('Done — V22.1 architecture doc saved')
