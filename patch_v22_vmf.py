"""给 train_v22_stage1.py 打vMF升级补丁"""
filepath = 'C:/ai/train_v22_stage1.py'

with open(filepath, 'r', encoding='utf-8') as f:
    content = f.read()

# Patch 1: 替换 TurboMeta
old_turbo = 'class TurboMeta(nn.Module):\n    def __init__(self, dim=256): super().__init__(); self.bias = nn.Parameter(torch.zeros(dim))\n    def forward(self, x): return torch.sigmoid((self.bias + x) / 10.0)  # 压缩输入, 防饱和\n'
new_turbo = '# V22 vMF升级: TurboMeta球面门控\nfrom v22_vmf_modules import TurboMeta_vMF, MasterGate_vMF, P7Gate_vMF,\\\n    SphericalSelfOrganizingLoss, MutualInformationPenalty,\\\n    KappaPhaseScheduler, MomentumExchangePool, PCACurvatureProxy,\\\n    LocalCurvatureEstimator, LocalBettiMonitor, SphericalVoronoiProbe\n\nclass TurboMeta(TurboMeta_vMF):\n    """V22升级: vMF球面门控"""\n    pass\n'

if old_turbo in content:
    content = content.replace(old_turbo, new_turbo)
    print('[OK] TurboMeta → vMF')
else:
    print('[FAIL] TurboMeta not matched')
    idx = content.find('class TurboMeta')
    if idx >= 0:
        snippet = content[idx:idx+300]
        print(f'  Found at {idx}: {repr(snippet[:100])}')

# Patch 2: 替换 MasterGate
old_mg = 'class MasterGate(nn.Module):\n    """总Gate: 5个子gate → 5路温控(0-1)"""\n    def __init__(self):\n        super().__init__()\n        self.net = nn.Sequential(nn.Linear(5, 32), nn.GELU(), nn.Linear(32, 32), nn.GELU(), nn.Linear(32, 5))\n        for l in self.net:\n            if isinstance(l, nn.Linear): nn.init.xavier_uniform_(l.weight, gain=0.1); nn.init.zeros_(l.bias)\n    def forward(self, sub_losses):  # sub_losses: [5] — P7, A, B, C, cos\n        return torch.sigmoid(self.net(sub_losses))  # [5] 每路0-1温控\n'
new_mg = 'class MasterGate(MasterGate_vMF):\n    """V22升级: vMF版MasterGate — 5子gate→5路温控, 球面偏好方向"""\n    pass\n'

if old_mg in content:
    content = content.replace(old_mg, new_mg)
    print('[OK] MasterGate → vMF')
else:
    print('[FAIL] MasterGate not matched')

# Patch 3: 在 training loop 前添加 SSOL/MI/监控初始化
hook_point = "batch_id = 0  # 全局batch计数"
if hook_point in content:
    setup_code = """
# ════════════════════════ V22 vMF升级: 反冗余损失 + 监控 ════════════════════════
ssol = SphericalSelfOrganizingLoss(repulsion_strength=0.01)
mi_penalty = MutualInformationPenalty(penalty_weight=0.005)
kappa_scheduler = KappaPhaseScheduler(total_steps=args.epochs * 200)
momentum_pool = MomentumExchangePool(n_modules=7)
pca_curv = PCACurvatureProxy(window=50)
lce = LocalCurvatureEstimator(short_window=10, long_window=50)
betti_mon = LocalBettiMonitor(n_heads=312, k_neighbors=8)
voronoi = SphericalVoronoiProbe(n_probes=50, n_heads=312)
"""
    content = content.replace(hook_point, setup_code + hook_point)
    print('[OK] Monitoring hooks added')
else:
    print('[FAIL] batch_id hook not found')

with open(filepath, 'w', encoding='utf-8') as f:
    f.write(content)

print('Backup at: train_v22_stage1_original.py')
print('Patched:  train_v22_stage1.py')
