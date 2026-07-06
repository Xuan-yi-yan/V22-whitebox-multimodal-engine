"""
═══════════════════════════════════════════════════════════════
V19 Turbo — 批量训练 + P3预缓存, GPU利用率 19%→80%+
═══════════════════════════════════════════════════════════════

开关:
  --turbo     : 开启加速 (batch=8, P3预缓存GPU, 全速训练)
  不加--turbo : 普通模式 (batch=1, 低GPU占用, 可边打游戏)

用法:
  # 打游戏时
  python train_v19_turbo.py --data public --epochs 1000

  # 睡觉/出门时
  python train_v19_turbo.py --data public --epochs 1000 --turbo
"""
import torch, torch.nn as nn, torch.nn.functional as F, os, sys, random, time, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils.config import *
from P7_cross_sent.model import P7WordRouter2048
# P6: 128头→128D向量→tied weight投影到字表 (千问方案)
class P6_Tied(nn.Module):
    def __init__(self, sent_dim=256, hidden_dim=256, max_words=128, word_dim=256):
        super().__init__()
        self.max_words = max_words
        self.encoder = nn.Sequential(nn.Linear(sent_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, hidden_dim), nn.GELU())
        self.pos_embed = nn.Parameter(torch.randn(max_words, hidden_dim) * 0.1)
        self.heads = nn.ModuleList([nn.Linear(hidden_dim, word_dim) for _ in range(max_words)])
        for h in self.heads: nn.init.xavier_uniform_(h.weight, gain=0.1)
    def forward(self, sent_vec, char_weight, gate=None):
        b = sent_vec.shape[0]
        h = self.encoder(sent_vec)
        if gate is not None: h = h * gate
        words = []
        for i in range(self.max_words):
            hi = h + self.pos_embed[i].unsqueeze(0)
            words.append(self.heads[i](hi))  # [b, 256]
        pred_vecs = torch.stack(words, dim=1)
        logits = torch.matmul(pred_vecs, char_weight.T)  # tied!
        return logits
from P3_word_attr.stack import P3AttributeStack
from P3_word_attr.p3l_linkage import P3L_AttributeLinkage

parser = argparse.ArgumentParser()
parser.add_argument("--epochs", type=int, default=1000)
parser.add_argument("--display", type=int, default=10)
parser.add_argument("--lr", type=float, default=0.003)
parser.add_argument("--data", type=str, default="public")
parser.add_argument("--turbo", action="store_true", help="固定batch=8, GPU拉满")
parser.add_argument("--batch_size", type=int, default=32, help="batch大小")
parser.add_argument("--sample", type=int, default=0, help="随机采样N对训练(0=全量)")
parser.add_argument("--amp", action="store_true", help="FP16混合精度(5070加速)")
parser.add_argument("--adaptive", action="store_true", help="自适应模式: 监控GPU负载, 自动调batch 1-8")
parser.add_argument("--adaptive_interval", type=int, default=5, help="自适应检查间隔(秒)")
parser.add_argument("--quick_test", action="store_true", help="快速测试GPU占比, 几批就停")
args = parser.parse_args()

BATCH_SIZE = args.batch_size

# 自适应GPU负载检测
import pynvml
pynvml.nvmlInit()
_NVML_HANDLE = pynvml.nvmlDeviceGetHandleByIndex(0)

def get_gpu_util():
    """pynvml: C底层API, 毫秒级, 绝不阻塞"""
    try:
        return pynvml.nvmlDeviceGetUtilizationRates(_NVML_HANDLE).gpu
    except Exception:
        return 50

def get_gpu_mem_used_mb():
    """返回已用显存(MB)"""
    try:
        info = pynvml.nvmlDeviceGetMemoryInfo(_NVML_HANDLE)
        return info.used / 1024**2
    except Exception:
        return 0

MEM_LIMIT_MB = 8500
_VRAM_BASELINE = None

def set_vram_baseline():
    global _VRAM_BASELINE
    _VRAM_BASELINE = get_gpu_mem_used_mb()

MAX_BATCH_P7 = 32  # P7全词表K/V 6166字, 超32炸显存

def adaptive_batch_size(current_bs, gpu_util):
    """自适应: 目标GPU利用率85-90%, VRAM不超8.5GB"""
    mem_mb = get_gpu_mem_used_mb()
    if _VRAM_BASELINE and mem_mb - _VRAM_BASELINE < 200:
        pass  # 训练VRAM增长<200MB, 不是训练在吃, 放心加batch
    if mem_mb > MEM_LIMIT_MB:
        return max(2, current_bs - 4)
    if mem_mb > MEM_LIMIT_MB * 0.95:
        return min(current_bs, current_bs)
    if gpu_util < 0:
        return current_bs
    if gpu_util > 90:
        return max(2, current_bs - 3)
    elif gpu_util > 85:
        return max(2, current_bs - 1)
    elif gpu_util >= 75:
        return current_bs
    elif gpu_util >= 50:
        return current_bs + 1
    elif gpu_util >= 30:
        return current_bs + 2
    else:
        return current_bs + 4

MODE = f"BS{BATCH_SIZE}"
LOG_PATH = os.path.join(BASE_DIR, "logs", f"v19_{MODE}_{time.strftime('%H%M%S')}.txt")
os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
log = open(LOG_PATH, "w", encoding="utf-8")
def w(s): log.write(s+"\n"); log.flush(); print(s, flush=True)

w("="*70)
w(f"  V19 {MODE}模式: batch={BATCH_SIZE} | turbo={args.turbo}")
w(f"  epochs={args.epochs} display={args.display} lr={args.lr}")
w("="*70)

# ════════════════════════ 1. 单字符编码 ════════════════════════
w(f"\n[输入] {MODE}: 单字符直接编码 (1字→128D)...")

all_pairs = []
import re
SENT_SPLIT = re.compile(r'[。！？；\n]')
path = "C:/ai/data/public/public_combined.txt" if args.data == "public" else args.data
is_cover = ('cover' in path)  # 覆盖采样文件无需句拆分
with open(path, 'r', encoding='utf-8') as f:
    for line in f:
        line = line.strip()
        if not line or line.startswith('#'): continue
        a, b = line.split('\t', 1)
        if is_cover:
            wa = [c for c in a if c.strip() and ord(c)>32]
            wb = [c for c in b if c.strip() and ord(c)>32]
            if 3<=len(wa)<=80 and 3<=len(wb)<=80:
                all_pairs.append((wa, wb))
        else:
            sa = [p.strip() for p in SENT_SPLIT.split(a) if len(p.strip())>=3]
            sb = [p.strip() for p in SENT_SPLIT.split(b) if len(p.strip())>=3]
            for xa, xb in zip(sa, sb):
                wa = [c for c in xa if c.strip() and ord(c)>32]
                wb = [c for c in xb if c.strip() and ord(c)>32]
                if 3<=len(wa)<=80 and 3<=len(wb)<=80:
                    all_pairs.append((wa, wb))

# 采样(在c2i构建之前)
if args.sample > 0 and args.sample < len(all_pairs):
    all_pairs = random.sample(all_pairs, args.sample)

# 构建字符词典(从采样后的数据)
all_chars = set()
for A, B in all_pairs: all_chars.update(A); all_chars.update(B)
c2i = {c: i for i, c in enumerate(sorted(all_chars), start=1)}  # 0留给padding
c2i['<unk>'] = 0; c2i['<pad>'] = -1
i2c = {i:c for c,i in c2i.items()}
n_chars = len(c2i)
w(f"  字符: {n_chars}个 | 句子: {len(all_pairs)}对")

# 正交字符embeddding: 每字独立方向, 无cos混淆
char_embed = nn.Embedding(n_chars, 256, padding_idx=0).to(DEVICE)
nn.init.orthogonal_(char_embed.weight)

encoded = []
for A, B in all_pairs:
    ids_a = [c2i.get(c, 0) for c in A]
    ids_b = [c2i.get(c, 0) for c in B]
    encoded.append((ids_a, ids_b, A, B))  # 存ID而非向量, 训练时在线embed

# 把char_embed加入优化器 (替代P1, 可训练)
opt_embed = None  # char_embed冻结: 固定靶

random.shuffle(encoded)
n = len(encoded)
t80 = int(n*0.80); t10 = int(n*0.10)
train_set = encoded[:t80]; test_set = encoded[t80:t80+t10]; exam_set = encoded[t80+t10:]
torch.save(exam_set, os.path.join(SAVE_DIR, "V19_exam_set.pt"))
w(f"  数据: {len(encoded)}对 | 训练:{len(train_set)} 测试:{len(test_set)} 考试:{len(exam_set)}")
# 展示样本
# 显示训练集首对——验证模型是否至少能记住
A_ids, B_ids, A_w, B_w = train_set[0]
w(f"  展示: {len(A_w)}→{len(B_w)}字 | A={A_w} | B={B_w}")

# ════════════════════════ TURBO: P3预缓存 ════════════════════════
attr_cache = {}  # word -> attr_vec[32] on GPU
p3_stack = P3AttributeStack()

# P3预缓存常开: 6166字×64D=1.5MB GPU, 省CPU规则引擎开销
if True:
    label = "TURBO" if args.turbo else f"BS{BATCH_SIZE}"
    w(f"\n[{label}] P3属性预缓存到GPU...")
    t0 = time.time()
    for char in sorted(all_chars):
        wl = [char]  # 单字
        vec = torch.zeros(384)  # V22: 128语言 + 256视觉(纯文本字符视觉区留空)
        try:
            packets = p3_stack.process_sentence(wl)
            if packets:
                d = packets[0].to_dict()
                bt = d.get("basic_type", ("",0.0))
                if isinstance(bt, tuple) and len(bt)>=1 and bt[0]:
                    m = {"noun":0,"verb":1,"adj":2,"adv":3,"pronoun":4,"quantifier":5,
                         "preposition":6,"conjunction":7,"auxiliary":8,"interjection":9}
                    vec[m.get(bt[0],9)] = bt[1] if len(bt)>=2 else 0.8
                # Layer1: 语义 (10-23)
                sem = d.get("semantic_types", [])
                if isinstance(sem, list):
                    for s in sem[:2]:
                        if isinstance(s,(tuple,list)) and len(s)>=2:
                            slot = {"人物":0,"地点":1,"时间":2,"物体":3,"行为":4,"状态":5,
                                    "数量":6,"程度":7,"方位":8,"方式":9,"原因":10,"结果":11,
                                    "目的":12,"条件":13}.get(s[0],-1)
                            if slot>=0: vec[10+slot] = s[1]
                # Layer1: 句法 (24-33)
                syn = d.get("syntax_candidates", [])
                if isinstance(syn, list) and syn:
                    s0 = syn[0]
                    if isinstance(s0,(tuple,list)) and len(s0)>=2:
                        slot = {"主语":0,"谓语":1,"宾语":2,"定语":3,"状语":4,
                                "补语":5,"兼语":6,"连动":7,"同位":8,"独立":9}.get(s0[0],-1)
                        if slot>=0: vec[24+slot] = s0[1]
                # Layer1: 情感 (34-36)
                pol = d.get("polarity", ("neutral",0.0,"none"))
                if isinstance(pol, tuple) and len(pol)>=1:
                    vec[34] = 1.0 if pol[0]=="positive" else (-1.0 if pol[0]=="negative" else 0.0)
                    vec[35] = pol[1] if len(pol)>=2 else 0.0
                # Layer1: 时态 (37-39) — P3暂不提供, 留空
                # Layer1: 人称 (40-41) — P3暂不提供, 留空
                # Layer2: 逻辑关系(42-52) — from P3C_ConnectLogic
                conn = d.get("conn_type", "")
                cslots = {"cause":0,"adversative":1,"coordinate":2,"conditional":3,
                          "progressive":4,"concessive":5,"alternative":6,"sequential":7,
                          "summary":8,"example":9,"purpose":10}
                if conn in cslots: vec[42+cslots[conn]] = d.get("conn_confidence", 0.85)
                # Layer2: 修饰(56-63) — from P3M_Modifier
                mod = d.get("mod_type", "")
                mslots = {"adjective":0,"adverb_manner":1,"scope":2,
                          "attributive":3,"adverbial":4,"complement":5}
                if mod in mslots: vec[56+mslots[mod]] = d.get("mod_confidence", 0.8)
                if d.get("is_comparative", False): vec[63] = 0.8
                # Layer2: 语义格(64-75) — P3暂不提供, 留空
                # Layer2: 关联(76-83) — P3暂不提供, 留空
                # Layer3: 语用(84-95) — P3暂不提供, 留空
                # Layer3: 篇章(96-105) — P3暂不提供, 留空
                # Layer3: 领域(106-115) — P3暂不提供, 留空
                # Layer3: 风格(116-125) — P3暂不提供, 留空
        except Exception:
            pass
        attr_cache[char] = vec.to(DEVICE)
    w(f"  预缓存: {len(attr_cache)}字 × 128D = {len(attr_cache)*128*4/1024:.1f}KB GPU | {time.time()-t0:.0f}s")

# ════════════════════════ 2. 模型 ════════════════════════
w(f"\n[模型] {MODE}: CharEmbed→P7→P3→P3-L→ABC→Gate→P6")
p7 = P7WordRouter2048(word_dim=256, inner_dim=256, heads=16, head_dim=16, max_len=128, num_groups=4).to(DEVICE)  # 16头×16D, 总256D, 2x
p6 = P6_Tied(max_words=128, word_dim=256).to(DEVICE)  # encoder 1024D
p3l = P3L_AttributeLinkage(attr_dim=384, num_attr_values=500, enable_vision=True).to(DEVICE)  # V22: 视觉扩展 384D + 视觉组
p7_input_proj = nn.Linear(640, 256, bias=False).to(DEVICE)  # A256D+P3(384D=语言128+视觉256)→256D

# ════════════════════════ Master Gate (5子gate→5维温控) ════════════════════════
class MasterGate(nn.Module):
    """总Gate: 5个子gate是否达标 → 5路温控(0-1)"""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(5, 32), nn.GELU(), nn.Linear(32, 32), nn.GELU(), nn.Linear(32, 5))
        for l in self.net:
            if isinstance(l, nn.Linear): nn.init.xavier_uniform_(l.weight, gain=0.1); nn.init.zeros_(l.bias)
    def forward(self, sub_losses):  # sub_losses: [5] — P7_loss, A_loss, B_loss, C_loss, cos_loss
        return torch.sigmoid(self.net(sub_losses))  # [5] 每路0-1温控

master_gate = MasterGate().to(DEVICE)
opt_master = torch.optim.Adam(master_gate.parameters(), lr=args.lr * 0.5)

# torch.compile: Windows无triton, 跳过

# ABC模块 (同全模块版)
class ABC_StageA(nn.Module):
    """A: 结构决策 20类 — 对齐 14语言模块 + 5视觉大类 P3颗粒度 (V22)"""
    def __init__(self, in_dim=640, hidden=128, n_classes=20):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(in_dim, hidden*2), nn.GELU(), nn.Linear(hidden*2, hidden), nn.GELU())
        self.pool = nn.Linear(hidden, hidden)
        self.classifier = nn.Linear(hidden, n_classes)
        self.n_classes = n_classes
        for m in self.modules():
            if isinstance(m, nn.Linear): nn.init.xavier_uniform_(m.weight, gain=0.5); nn.init.zeros_(m.bias)
    def forward(self, attr_vec, word_out):
        x = torch.cat([attr_vec, word_out], dim=-1)
        h = self.encoder(x).mean(dim=1)
        h_pooled = self.pool(h)
        logits = self.classifier(h_pooled)
        return logits, h_pooled

class ABC_StageB(nn.Module):
    """B: 内容填充 384D — 对齐P3 (V22: 128语言+256视觉, mask 自动过滤未激活槽)"""
    def __init__(self, hidden=128, content_dim=384):
        super().__init__()
        self.fuse = nn.Sequential(nn.Linear(128+384+128, 256), nn.GELU(), nn.Linear(256, 128), nn.GELU(), nn.Linear(128, content_dim))  # V22: attr_sum 128→384
        for m in self.modules():
            if isinstance(m, nn.Linear): nn.init.xavier_uniform_(m.weight, gain=0.5); nn.init.zeros_(m.bias)
    def forward(self, a_hidden, attr_vec, p3l_feat):
        attr_sum = attr_vec.mean(dim=1)
        return self.fuse(torch.cat([a_hidden, attr_sum, p3l_feat], dim=-1))

class ABC_StageC(nn.Module):
    """C: 语气精炼 48D"""
    def __init__(self, content_dim=384, hidden=96, out_dim=48):  # V22: ct 128→384
        super().__init__()
        self.refine = nn.Sequential(nn.Linear(content_dim+5, hidden), nn.GELU(), nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, out_dim))
        for m in self.modules():
            if isinstance(m, nn.Linear): nn.init.xavier_uniform_(m.weight, gain=0.5); nn.init.zeros_(m.bias)
    def forward(self, content, attr_vec):
        ef = attr_vec[:, :, 34:39].mean(dim=1)  # 情感槽位34-39
        return self.refine(torch.cat([content, ef], dim=-1))

abcA = ABC_StageA().to(DEVICE)
abcB = ABC_StageB().to(DEVICE)
abcC = ABC_StageC().to(DEVICE)

# ════════════════════════ ABC' 内容加压 (V21.1) ════════════════════════
class ABC_Prime(nn.Module):
    """ABC': P3-L后内容加压 128→192→128, 冷启动gain=0.01不冲现有信号"""
    def __init__(self, in_dim=128, hidden=192, out_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, out_dim))
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.01)
                nn.init.zeros_(m.bias)
    def forward(self, p3l_feat):
        return self.net(p3l_feat)

abc_prime = ABC_Prime().to(DEVICE)

GATE_DIM = 256  # 主Gate: 仅管P6 encoder (cos loss → P6调制)

class TurboExplore(nn.Module):
    def __init__(self, in_dim=384, hidden=512):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, hidden), nn.GELU(), nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, 256))
        for l in self.net:
            if isinstance(l, nn.Linear): nn.init.xavier_uniform_(l.weight, gain=0.1); nn.init.zeros_(l.bias)
    def forward(self, x): return self.net(x)

class TurboMeta(nn.Module):
    def __init__(self, dim=256): super().__init__(); self.bias = nn.Parameter(torch.zeros(dim))
    def forward(self, x): return torch.sigmoid((self.bias + x) / 10.0)  # 压缩输入, 防饱和

explore = TurboExplore().to(DEVICE)
meta = TurboMeta().to(DEVICE)
abc_to_gate = nn.Linear(48, 128, bias=False).to(DEVICE)   # ABC→128D
sent_to_gate = nn.Linear(256, 128, bias=False).to(DEVICE)  # sent→128D
gate_base_proj = nn.Linear(12, 128, bias=False).to(DEVICE) # prev_loss→128D
nn.init.normal_(abc_to_gate.weight, std=0.01)  # 0.1→0.01, 防主导
nn.init.normal_(sent_to_gate.weight, std=0.03)  # 0.01→0.03, 平衡
nn.init.normal_(gate_base_proj.weight, std=0.03)

# V21.1: ABC'内容→sent加法注入 (保sv_batch不变, abc_prime渐进参与)
abc_to_sent = nn.Linear(128, 256, bias=False).to(DEVICE)
nn.init.xavier_uniform_(abc_to_sent.weight, gain=0.01)

n_embed = sum(p.numel() for p in char_embed.parameters())
n_all = sum(p.numel() for m in [p7,p6,p3l,abcA,abcB,abcC,abc_prime,explore,meta,abc_to_sent,char_embed] for p in m.parameters())
w(f"  CharEmbed:{n_embed:,} | P7:{sum(p.numel() for p in p7.parameters()):,} | P6:{sum(p.numel() for p in p6.parameters()):,} | 总:{n_all:,}")
w(f"  总参数: {n_all:,} | batch_size={BATCH_SIZE}")

opt_embed = None  # char_embed冻结: 固定靶

# ════════════════════════ 维度校验 ════════════════════════
w(f"\n[校验] 架构维度一致性...")
assert p7.word_dim == 256, f"P7 word_dim={p7.word_dim}"
assert p7.heads == 16 and p7.head_dim == 16, f"P7头"
assert p7_input_proj.in_features == 640, f"p7_in in={p7_input_proj.in_features}"
assert abc_to_gate.in_features == 48, f"abc_to_gate in={abc_to_gate.in_features}"
assert abc_to_gate.out_features == 128, f"abc_to_gate out={abc_to_gate.out_features}"
assert sent_to_gate.in_features == 256, f"sent_to_gate in={sent_to_gate.in_features}"
assert sent_to_gate.out_features == 128, f"sent_to_gate out={sent_to_gate.out_features}"
assert meta.bias.shape[0] == 256, f"Meta dim={meta.bias.shape[0]}"
assert abc_to_sent.in_features == 128, f"abc_to_sent in={abc_to_sent.in_features}"  # V21.1
assert abc_to_sent.out_features == 256, f"abc_to_sent out={abc_to_sent.out_features}"
w(f"  [OK] 全部通过")

# ── 自学习率: 每个优化器根据loss趋势自适应调整lr ──
class SelfLR:
    def __init__(self, name, base_lr, lr_min=1e-6, lr_max=0.1):
        self.name = name; self.base_lr = base_lr; self.lr = base_lr
        self.lr_min = lr_min; self.lr_max = lr_max
        self.ema = None; self.mom = 0.95
    def step_lr(self, batch_loss):
        if self.ema is None: self.ema = batch_loss
        else: self.ema = self.mom * self.ema + (1 - self.mom) * batch_loss
        if batch_loss < self.ema: self.lr *= 1.003   # 涨0.3%
        else: self.lr *= 0.997                        # 降0.3% 对称
        self.lr = max(self.lr_min, min(self.base_lr * 3.0, self.lr))  # 上限3x base
    def apply(self, opt):
        for pg in opt.param_groups: pg['lr'] = self.lr
    def __repr__(self): return f"{self.name}: lr={self.lr:.2e}"

# 千问计算: base_lr=0.003, 分层乘数
slr_p7_in = SelfLR("p7_in", args.lr*1.0)     # 投影层, 标准
slr_p7    = SelfLR("p7",    args.lr*0.05)    # [OK] 1.5e-4, 狠压! P7是爆炸源头
slr_p6    = SelfLR("p6",    args.lr*1.0)     # 36M巨无霸, 标准lr
slr_p3l   = SelfLR("p3l",   args.lr*0.8)
slr_abc   = SelfLR("abc",   args.lr*3.0)     # 参数少, 需敏捷
slr_gate  = SelfLR("gate",  args.lr*5.0)     # 阀门, 需极敏捷
slr_master= SelfLR("master",args.lr*0.5)
slr_abc_prime = SelfLR("abc_prime", args.lr*0.5)  # V21.1: 慢速, 防冲现有信号
slrs = [slr_p7_in, slr_p7, slr_p6, slr_p3l, slr_abc, slr_gate, slr_master, slr_abc_prime]

opt_p7_in = torch.optim.Adam(p7_input_proj.parameters(), lr=slr_p7_in.lr)
opt_p7  = torch.optim.Adam(p7.parameters(), lr=slr_p7.lr)
opt_p6  = torch.optim.Adam(p6.parameters(), lr=slr_p6.lr)
opt_p3l = torch.optim.Adam(p3l.parameters(), lr=slr_p3l.lr)
opt_abc = torch.optim.Adam(list(abcA.parameters())+list(abcB.parameters())+list(abcC.parameters()), lr=slr_abc.lr)
opt_gate= torch.optim.Adam(list(explore.parameters())+list(meta.parameters())+list(abc_to_gate.parameters())+list(sent_to_gate.parameters())+list(gate_base_proj.parameters())+list(abc_to_sent.parameters()), lr=slr_gate.lr)
opt_abc_prime = torch.optim.Adam(abc_prime.parameters(), lr=slr_abc_prime.lr)  # V21.1


# ── V20热启动 ──
v20_path = ""  # 冷启动
if os.path.exists(v20_path):
    w(f"\n[热启动] 加载V20权重...")
    ckpt = torch.load(v20_path, map_location=DEVICE)
    v20_c2i = ckpt.get('c2i', {})
    if v20_c2i and len(v20_c2i) > 0:
        v20_i2c = {i:c for c,i in v20_c2i.items()}
        transferred = 0
        with torch.no_grad():
            for c, idx in c2i.items():
                if idx > 0 and c in v20_c2i:
                    v20_idx = v20_c2i[c]
                    if v20_idx < ckpt['char_embed']['weight'].shape[0]:
                        char_embed.weight[idx] = ckpt['char_embed']['weight'][v20_idx].to(DEVICE)
                        transferred += 1
        w(f"  char_embed: 迁移{transferred}字")
    for name, mod in [('p6', p6), ('p3l', p3l),
                       ('abcA', abcA), ('abcB', abcB), ('abcC', abcC),
                       ('explore', explore), ('meta', meta), ('master_gate', master_gate)]:
        if name in ckpt:
            try: mod.load_state_dict(ckpt[name], strict=False); w(f"  {name}: OK")
            except Exception as e: w(f"  {name}: 跳过({e})")
    # p7in: 128→256, V20不兼容
    if 'p7in' in ckpt:
        w(f"  p7in: V20(128D)→V21(256D), 形状不兼容, 重新初始化")
    # P7特殊处理: 32头×4D→8头×16D, 形状不兼容的跳过
    if 'p7' in ckpt:
        p7_state = p7.state_dict()
        loaded, skipped = 0, 0
        for k, v in p7_state.items():
            if k in ckpt['p7'] and ckpt['p7'][k].shape == v.shape:
                p7_state[k] = ckpt['p7'][k].to(DEVICE)
                loaded += 1
            else:
                skipped += 1
        p7.load_state_dict(p7_state)
        w(f"  p7: 迁移{loaded}参数, 重初始化{skipped}(v_heads/out_proj/group_bias)")
    if 'abc_to_gate' in ckpt:
        abc_to_gate.load_state_dict(ckpt['abc_to_gate']); w(f"  abc_to_gate: OK")
    if 'sent_to_gate' in ckpt:
        sent_to_gate.load_state_dict(ckpt['sent_to_gate']); w(f"  sent_to_gate: OK")
    w(f"  起始epoch: {ckpt.get('epoch', '?')} loss={ckpt.get('loss', '?'):.4f}")
else:
    w(f"\n[冷启动] 未找到V20权重")

# ── V21热启动 (断点续训 + V21.1新模块冷启动) ──
v21_path = os.path.join(SAVE_DIR, "V21_best.pt")
if os.path.exists(v21_path):
    w(f"\n[热启动] 加载V21权重 → V21.1...")
    ckpt = torch.load(v21_path, map_location=DEVICE)
    for name, mod in [('p6', p6), ('p3l', p3l), ('p7in', p7_input_proj),
                       ('abcA', abcA), ('abcB', abcB), ('abcC', abcC),
                       ('explore', explore), ('meta', meta), ('master', master_gate),
                       ('abc_to_gate', abc_to_gate), ('sent_to_gate', sent_to_gate),
                       ('gate_base_proj', gate_base_proj)]:
        if name in ckpt:
            try: mod.load_state_dict(ckpt[name], strict=False); w(f"  {name}: OK")
            except Exception as e: w(f"  {name}: 跳过({e})")
    # char_embed: 按字迁移(新旧词表大小可能不同)
    if 'char_embed' in ckpt and 'c2i' in ckpt:
        old_c2i = ckpt['c2i']
        old_i2c = {i: c for c, i in old_c2i.items()}
        transferred = 0
        with torch.no_grad():
            for c, new_idx in c2i.items():
                if new_idx > 0 and c in old_c2i:
                    old_idx = old_c2i[c]
                    if old_idx < ckpt['char_embed']['weight'].shape[0]:
                        char_embed.weight[new_idx] = ckpt['char_embed']['weight'][old_idx].to(DEVICE)
                        transferred += 1
        w(f"  char_embed: 按字迁移{transferred}/{len(c2i)-1}字")
    elif 'char_embed' in ckpt:
        # 无c2i, 按形状尽力复制
        with torch.no_grad():
            n = min(ckpt['char_embed']['weight'].shape[0], char_embed.weight.shape[0])
            char_embed.weight[:n] = ckpt['char_embed']['weight'][:n].to(DEVICE)
        w(f"  char_embed: 按索引迁移{n}字")
    # P7: 形状兼容迁移
    if 'p7' in ckpt:
        loaded = 0; state = p7.state_dict()
        for k, v in state.items():
            if k in ckpt['p7'] and ckpt['p7'][k].shape == v.shape:
                state[k] = ckpt['p7'][k].to(DEVICE); loaded += 1
        p7.load_state_dict(state)
        w(f"  p7: 迁移{loaded}/{len(state)}参数")
    best_epoch = ckpt.get('epoch', 0)
    best_loss = ckpt.get('loss', float('inf'))
    w(f"  起始epoch: {best_epoch} loss={best_loss:.4f}")
    w(f"  abc_prime + abc_to_sent: 冷启动 (V21.1新模块, gain=0.01)")
else:
    w(f"\n[冷启动] 未找到V21权重, 全新训练")


# ════════════════════════ 辅助函数 ════════════════════════
ATTR_ID_MAP = {}; _cid = [0]
def _aid(k):
    if k not in ATTR_ID_MAP: ATTR_ID_MAP[k] = _cid[0]; _cid[0] += 1
    return ATTR_ID_MAP[k]

def words_to_attr_vec(words):
    """从预缓存组装attr_vec — 有缓存用缓存, 无缓存实时P3"""
    if args.turbo or len(attr_cache) > 100:  # 有预缓存就用
        vecs = []
        for w in words[:80]:
            if w in attr_cache: vecs.append(attr_cache[w])
            else: vecs.append(torch.zeros(384, device=DEVICE))  # V22: attr 384
        if len(vecs) < 80:
            pad = torch.zeros(80-len(vecs), 384, device=DEVICE)  # V22: attr 384
            return torch.cat([torch.stack(vecs), pad])
        return torch.stack(vecs)
    else:
        # ECO模式: 实时P3处理
        packets = p3_stack.process_sentence(words[:80])
        return _packets_to_vec([p.to_dict() for p in packets])

def words_to_attr_ids(words):
    packets = p3_stack.process_sentence(words)
    ids = []
    for p in packets:
        d = p.to_dict()
        bt = d.get("basic_type", ("",))
        if isinstance(bt, tuple): bt = bt[0]
        ids.append(_aid(f"bt:{bt}"))
        sem = d.get("semantic_types", [])
        for s in sem[:2]:
            if isinstance(s,(tuple,list)) and len(s)>=1: ids.append(_aid(f"sem:{s[0]}"))
        syn = d.get("syntax_candidates", [])
        if syn and isinstance(syn[0],(tuple,list)): ids.append(_aid(f"syn:{syn[0][0]}"))
    return list(set(ids))

def p3l_features(attr_ids, wo_feat=None):
    """P3-L特征: 离散ID + 可选连续word_out特征"""
    if len(attr_ids) < 2:
        base = torch.zeros(1, 128, device=DEVICE)
        if wo_feat is not None:
            return torch.cat([base, wo_feat[:64].unsqueeze(0)], dim=-1)
        return base
    all_scores, _ = p3l(torch.tensor(attr_ids, device=DEVICE))
    feats = []
    for gname, scores in all_scores.items():
        if scores.numel()>0: feats.extend([scores.mean(), scores.std(), (scores.max()-scores.min())])
    flat = torch.cat(feats) if feats else torch.zeros(60, device=DEVICE)
    if flat.shape[0] < 64: flat = F.pad(flat, (0, 64-flat.shape[0]))
    else: flat = flat[:64]
    result = flat.unsqueeze(0)
    if wo_feat is not None:
        result = torch.cat([result, wo_feat[:64].unsqueeze(0)], dim=-1)  # [1, 128]
    return result

def _packets_to_vec(pkt_dicts, max_n=80, dim=128):
    vec = torch.zeros(max_n, dim, device=DEVICE)
    for i, d in enumerate(pkt_dicts[:max_n]):
        # 0-7: 基础词类
        bt = d.get("basic_type", ("",0.0))
        if isinstance(bt,tuple) and len(bt)>=1 and bt[0]:
            m={"noun":0,"verb":1,"pronoun":2,"question":3,"function":4,"content_word":5}
            vec[i, m.get(bt[0],6)] = bt[1] if len(bt)>=2 else 0.8
        # 8-15: 语义族
        sem = d.get("semantic_types", [])
        if isinstance(sem,list):
            for s in sem[:3]:
                if isinstance(s,(tuple,list)) and len(s)>=2:
                    slot={"人物":0,"地点":1,"时间":2,"物体":3,"行为":4,"状态":5}.get(s[0],-1)
                    if slot>=0: vec[i,8+slot]=s[1]
        # 16-23: 句法
        syn = d.get("syntax_candidates",[])
        if isinstance(syn,list) and syn:
            s0=syn[0]
            if isinstance(s0,(tuple,list)) and len(s0)>=2:
                slot={"主语":0,"谓语":1,"宾语":2,"定语":3,"状语":4}.get(s0[0],-1)
                if slot>=0: vec[i,16+slot]=s0[1]
        # 24-31: 情感+人称+方向+疑问+位置
        pol = d.get("polarity",("neutral",0.0,"none"))
        if isinstance(pol,tuple) and len(pol)>=1:
            vec[i,24]=1.0 if pol[0]=="positive" else (-1.0 if pol[0]=="negative" else 0.0)
            vec[i,25]=pol[1] if len(pol)>=2 else 0.0
        qs = d.get("question_slot", ("none",0.0))
        if isinstance(qs,tuple) and len(qs)>=1:
            vec[i,26]={"subject":0.2,"object":0.4,"place":0.6,"manner":0.8,"reason":1.0}.get(qs[0],0.0)
            vec[i,27]=qs[1] if len(qs)>=2 else 0.0
        person = d.get("person", ("none",0.0))
        if isinstance(person,tuple) and len(person)>=1:
            vec[i,28]={"first":0.33,"second":0.66,"third":0.99}.get(person[0],0.0)
            vec[i,29]=person[1] if len(person)>=2 else 0.0
        direction = d.get("direction", ("unknown","unknown"))
        if isinstance(direction,tuple) and len(direction)>=1:
            vec[i,30]={"outward":0.25,"inward":0.5,"mutual":0.75,"cognitive":1.0}.get(direction[0],0.0)
        # 32-37: 时态体态
        tense = d.get("tense_type","")
        aspect = d.get("aspect_type","")
        vec[i,32] = {"past":0.3,"present":0.6,"future":0.9,"timeless":0.0}.get(tense,0.0)
        vec[i,33] = d.get("tense_confidence",0.0)
        vec[i,34] = 1.0 if d.get("is_time_noun",False) else 0.0
        # 38-41: 数量程度
        vec[i,38] = 1.0 if d.get("is_number",False) else 0.0
        vec[i,39] = 1.0 if d.get("is_measure",False) else 0.0
        vec[i,40] = {"high":0.8,"medium":0.5,"low":0.2,"excess":1.0}.get(d.get("degree_level",""),0.0)
        vec[i,41] = 1.0 if d.get("is_quantifier",False) else 0.0
        # 42-45: 连接逻辑
        vec[i,42] = 1.0 if d.get("is_connective",False) else 0.0
        conn_map = {"cause":0.2,"adversative":0.4,"coordinate":0.6,"conditional":0.8,"sequential":1.0}
        vec[i,43] = conn_map.get(d.get("conn_type",""),0.0)
        # 46-49: 语态语气
        vec[i,46] = 1.0 if d.get("voice_type","") == "passive" else (0.5 if d.get("voice_type","")=="disposal" else 0.0)
        mood_map = {"suggestion":0.2,"inquiry":0.4,"yes_no_question":0.6,"exclamation":0.8,"obvious":1.0}
        vec[i,47] = mood_map.get(d.get("mood_type",""),0.0)
        modal_map = {"ability":0.2,"permission":0.4,"obligation":0.6,"necessity":0.8,"desire":1.0}
        vec[i,48] = modal_map.get(d.get("modal_type",""),0.0)
        # 50-53: 修饰限定
        vec[i,50] = 1.0 if d.get("is_modifier",False) else 0.0
        de_map = {"attributive":0.3,"adverbial":0.6,"complement":0.9}
        vec[i,51] = de_map.get(d.get("de_type",""),0.0)
        vec[i,52] = 1.0 if d.get("is_comparative",False) else 0.0
        # 54-57: 代码技术
        vec[i,54] = 1.0 if d.get("is_code",False) else 0.0
        code_map = {"python":0.2,"javascript":0.4,"cpp":0.6,"sql":0.8,"chinese_tech":1.0}
        vec[i,55] = code_map.get(d.get("code_language",""),0.0)
        role_map = {"declaration":0.2,"control_flow":0.4,"function_call":0.6,"import_export":0.8,"data_access":1.0}
        vec[i,56] = role_map.get(d.get("code_role",""),0.0)
        vec[i,57] = 1.0 if d.get("cn_tech",False) else 0.0
        # 58-61: 标点符号
        vec[i,58] = 1.0 if d.get("is_punct",False) else 0.0
        punct_map = {"period":0.1,"comma":0.2,"semicolon":0.3,"colon":0.4,"question":0.6,"exclamation":0.7}
        vec[i,59] = punct_map.get(d.get("punct_type",""),0.0)
        vec[i,60] = 1.0 if d.get("is_sent_boundary",False) else 0.0
        # 62-63: 保留
        vec[i,63] = i/max(max_n,1)
    return vec


# ════════════════════════ 3. 批量训练 ════════════════════════
w(f"\n{'='*70}\n  {MODE}训练 {args.epochs}轮 (batch={BATCH_SIZE})\n{'='*70}\n")
total_t0 = time.time()

scaler = torch.amp.GradScaler('cuda') if args.amp else None
best_loss = float('inf')
best_epoch = 0
import signal
def save_best(sig=None, frame=None):
    path = os.path.join(SAVE_DIR, "V21_best.pt")
    torch.save({"char_embed":char_embed.state_dict(),"p7in":p7_input_proj.state_dict(),"p7":p7.state_dict(),"p6":p6.state_dict(),"p3l":p3l.state_dict(),
                "abcA":abcA.state_dict(),"abcB":abcB.state_dict(),"abcC":abcC.state_dict(),
                "abc_prime":abc_prime.state_dict(),  # V21.1
                "abc_to_sent":abc_to_sent.state_dict(),  # V21.1
                "explore":explore.state_dict(),"meta":meta.state_dict(),"master":master_gate.state_dict(),
                "abc_to_gate":abc_to_gate.state_dict(),
                "sent_to_gate":sent_to_gate.state_dict(),
                "c2i": c2i, "epoch": best_epoch, "loss": best_loss}, path)
    w(f"\n[保存] 最佳E{best_epoch} loss={best_loss:.4f} → {path}")
    if sig: sys.exit(0)
signal.signal(signal.SIGINT, save_best)
signal.signal(signal.SIGTERM, save_best)

def make_batches(data, bs):
    """按长度分组做batch (减少padding浪费)"""
    if bs == 1:
        return [[x] for x in data]
    data_sorted = sorted(data, key=lambda x: len(x[2]))  # sort by A_words length
    batches = []
    for i in range(0, len(data_sorted), bs):
        batches.append(data_sorted[i:i+bs])
    return batches

current_bs = BATCH_SIZE

for ep in range(1, args.epochs+1):
    t0 = time.time(); total_loss, n = 0.0, 0
    total_ce, total_A, total_B, total_C, total_gate = 0.0, 0.0, 0.0, 0.0, 0.0
    last_temps, last_scale = None, 0.0
    p7.g_base_sum = p7.g_aux_sum = p7.g_sent_sum = p7.g_ac_sum = 0.0
    pairs_done = 0
    remaining = train_set[:]  # 本epoch剩余数据
    total_batches = 0

    while remaining:
        batch = remaining[:current_bs]
        remaining = remaining[current_bs:]
        bs = len(batch)
        prev_loss = getattr(p7, '_loss_vec', None)

        # ── 在线embedding + 批量Padding ──
        max_len_a = max(len(x[0]) for x in batch)
        max_len_b = max(len(x[1]) for x in batch)
        max_attr = max_len_a

        Av_batch_ids = torch.zeros(bs, max_len_a, dtype=torch.long, device=DEVICE)
        Bv_batch_ids = torch.zeros(bs, max_len_b, dtype=torch.long, device=DEVICE)
        for i, (ids_a, ids_b, A_words, _) in enumerate(batch):
            Av_batch_ids[i, :len(ids_a)] = torch.tensor(ids_a, device=DEVICE)
            Bv_batch_ids[i, :len(ids_b)] = torch.tensor(ids_b, device=DEVICE)

        # ── P7: 批量 (输入升级为192D) ──
        a_lens = [len(batch[i][0]) for i in range(bs)]
        # P7: 词向量+P3属性拼接→p7_input_proj→128D→全词表路由
        Av_emb = char_embed(Av_batch_ids[:, :max_len_a])
        attr_for_p7 = torch.zeros(bs, max_len_a, 384, device=DEVICE)  # V22: 128语言+256视觉
        for i, (ids_a, ids_b, A_words, _) in enumerate(batch):
            av = words_to_attr_vec(A_words)[:max_len_a, :]
            attr_for_p7[i, :av.shape[0], :] = av
        Av_rich = torch.cat([Av_emb, attr_for_p7], dim=-1)
        vocab = char_embed.weight.detach()
        with torch.amp.autocast('cuda', enabled=args.amp):
            wo_batch, sv_batch, _, wg_batch, _ = p7.forward_batch(
                p7_input_proj(Av_rich), vocab.unsqueeze(0).expand(bs, -1, -1),
                a_lens, [vocab.shape[0]]*bs, last_loss=0.0)

        # ABC: P3属性 + P7词级路由
        attr_b = attr_for_p7.clone()  # 独立副本防别名
        for i, (ids_a, ids_b, A_words, _) in enumerate(batch):
            av = words_to_attr_vec(A_words)[:max_len_a, :]
            attr_b[i, :av.shape[0], :] = av
        # P3-L: 属性联动特征
        p3lf_list = []
        for i in range(bs):
            aids = words_to_attr_ids(batch[i][2])
            if len(aids) >= 2:
                _, attns = p3l(torch.tensor(aids, device=DEVICE))
                feats = []
                for g, scores in attns.items():
                    if scores.numel() > 0: feats.append(torch.stack([scores.mean(), scores.std(), scores.max() - scores.min()]))
                f = torch.cat(feats) if feats else torch.zeros(60, device=DEVICE)
            else:
                f = torch.zeros(60, device=DEVICE)
            p3lf_list.append(F.pad(f, (0, max(0, 128 - f.shape[0])))[:128])
        p3lf_b = torch.stack(p3lf_list)
        # V21.1: ABC'内容加压 → 双路径(提升ABC控制输入 + 灌入P6解码)
        abc_prime_out = abc_prime(p3lf_b)              # [bs, 128]
        p3lf_rich = p3lf_b + abc_prime_out * 0.1       # 渐进注入, 初始0.1防冲
        a_logits, ah = abcA(attr_b, wo_batch[:, :max_len_a, :])
        ct = abcB(ah, attr_b, p3lf_rich)               # V21.1: 加压后特征
        ac_batch = abcC(ct, attr_b)
        ac_batch = ac_batch / (ac_batch.norm(dim=-1, keepdim=True) + 1.0)  # 软归一化, 保动态范围

        # 辅助信号智能缩放: 震荡→退, 收敛→退, 健康→全开 (abc+sent同步)
        if not hasattr(p7, 'ce_ema'): p7.ce_ema = 8.0
        if not hasattr(p7, 'ce_std'): p7.ce_std = 1.0
        # 实时检测: ac_batch突然暴增→立刻压(不等CE, ac算在gate之前)
        if not hasattr(p7, 'ac_ema'): p7.ac_ema = 1.0
        p7.ac_ema = 0.95 * p7.ac_ema + 0.05 * ac_batch.norm().item()
        if not hasattr(p7, 'cool_down'): p7.cool_down = 0
        ac_exploding = (ac_batch.norm().item() > p7.ac_ema * 3.0)
        if ac_exploding: p7.cool_down = max(p7.cool_down, 30)
        if not hasattr(p7, 'aux_val'): p7.aux_val = 1.0
        if p7.cool_down > 0:
            p7.aux_val = max(0.2, p7.aux_val * 0.95)   # 最低0.2, 保梯度
            p7.cool_down -= 1
        else:
            p7.aux_val = min(1.0, p7.aux_val * 1.01)   # 涨: 1%/步, ~70步恢复
        aux_scale = p7.aux_val

        # ── Gate + P6 ── gate_base用ABC置信度, 训练推理一致
        with torch.amp.autocast('cuda', enabled=args.amp):
            # gate_base用ABC的12个真实统计量(不用零), NaN保护
            a_prob = a_logits.softmax(-1)
            def _safe(x): return torch.nan_to_num(x, nan=0.0, posinf=1.0, neginf=-1.0).detach()
            abc_conf = torch.stack([
                _safe(a_prob.max(-1).values.mean()),
                _safe(-(a_prob * torch.log(a_prob+1e-8)).sum(-1).mean()),
                _safe(a_logits.argmax(-1).float().mean()),
                _safe(ac_batch.norm(dim=-1).mean()),
                _safe(ac_batch.std(dim=-1).mean()),
                _safe(ac_batch.max(dim=-1).values.mean()),
                _safe(ct.norm(dim=-1).mean()),
                _safe(ct.std(dim=-1).mean()),
                _safe(ac_batch[:,:5].norm(dim=-1).mean()),
                _safe(ac_batch[:,5:15].norm(dim=-1).mean()),
                _safe(ac_batch[:,15:].norm(dim=-1).mean()),
                _safe(ac_batch.norm(dim=-1).mean()),
            ]).unsqueeze(0).expand(bs, -1)
            gb, ga, gs = gate_base_proj(abc_conf).norm().item(), (aux_scale*abc_to_gate(ac_batch)).norm().item(), (aux_scale*sent_to_gate(sv_batch)).norm().item()
            if not hasattr(p7, 'g_base_sum'): p7.g_base_sum = p7.g_aux_sum = p7.g_sent_sum = p7.g_ac_sum = 0.0
            p7.g_base_sum += gb; p7.g_aux_sum += ga; p7.g_sent_sum += gs; p7.g_ac_sum += ac_batch.norm().item()
            # 三路独立通道拼接: 128+128+128=384D, 不竞争不截幅
            gate_input = torch.cat([gate_base_proj(abc_conf), aux_scale*abc_to_gate(ac_batch), aux_scale*sent_to_gate(sv_batch)], dim=-1)
            gates = meta(explore(gate_input))
            # V21.1: abc_prime加法注入sent (保sv_batch主导, abc_prime渐进参与)
            sv_rich = sv_batch + abc_to_sent(abc_prime_out) * 0.1
            pred_all = p6(sv_rich, char_embed.weight, gate=gates)

        # 主loss: cross_entropy直出字
        ce_loss = 0.0; ok_count = 0; total_pos = 0
        for i in range(bs):
            nB = min(len(batch[i][1]), pred_all.shape[1])
            b_ids = torch.tensor([c2i.get(c, 0) for c in batch[i][3][:nB]], device=DEVICE)
            ce_loss += F.cross_entropy(pred_all[i, :nB, :], b_ids, ignore_index=0)
            pred_ids = pred_all[i, :nB, :].argmax(dim=-1)
            ok_count += (pred_ids == b_ids).sum().item()
            total_pos += nB
        ce_loss = ce_loss / bs
        cos_val = ok_count / max(total_pos, 1)
        p7._loss_vec = F.pad(torch.tensor([cos_val, ce_loss.item()], device=DEVICE), (0, 10))

        # ABC (不归一化)
        a_targets = torch.zeros(bs, dtype=torch.long, device=DEVICE)
        for i in range(bs):
            words_i = batch[i][2]
            has_q = any(w in '？谁哪怎为什么何时怎么' for w in words_i)
            has_excl = any(w in '！啊呀呵哇' for w in words_i)
            has_neg = any(w in '不没未否非' for w in words_i)
            if has_q: a_targets[i] = 0
            elif has_excl: a_targets[i] = 12
            elif has_neg: a_targets[i] = 13
            else: a_targets[i] = 6
        # ABC辅助loss (P3规则监督)
        a_tgt = torch.zeros(bs, dtype=torch.long, device=DEVICE)
        for i in range(bs):
            aw = batch[i][2]; w_str = ''.join(aw)
            if any(c in w_str for c in '？谁哪怎为'): a_tgt[i] = 0
            elif any(c in w_str for c in '！啊哇呵'): a_tgt[i] = 12
            elif any(c in w_str for c in '不没未否'): a_tgt[i] = 13
            else: a_tgt[i] = 6
        loss_A_c = F.cross_entropy(a_logits, a_tgt)
        # loss_B: 只用非零槽位(活跃P3属性), 不用全量mean
        attr_active_mask = (attr_b.abs().sum(dim=(0,1)) > 1e-6).float()  # [128]
        if attr_active_mask.sum() > 0:
            loss_B_c = torch.clamp(F.mse_loss(ct * attr_active_mask, attr_b.mean(dim=1) * attr_active_mask), max=10.0)
        else:
            loss_B_c = torch.tensor(0.0, device=DEVICE)
        loss_C_c = torch.clamp(F.mse_loss(ac_batch[:, :5], attr_b[:, :, 34:39].mean(dim=1)), max=10.0)
        gate_loss = (wg_batch.std() * 10).clamp(0.01, 1.0) if wg_batch.numel() > 1 else torch.tensor(0.1, device=DEVICE)
        sub_losses = torch.stack([gate_loss.detach(), loss_A_c.detach(), loss_B_c.detach(), loss_C_c.detach(), ce_loss.detach()])
        temps = master_gate(sub_losses)
        # 单次CE尖峰检测: 超EMA的3倍或绝对值>5 → 触发冷却
        ce_now = ce_loss.item()
        ce_spike = (ce_now > max(p7.ce_ema * 3.0, 5.0))
        if ce_spike:
            p7.cool_down = max(p7.cool_down, 50)
            ce_loss = torch.clamp(ce_loss, max=5.0)
        # EMA用clamp后的CE(防爆炸值进EMA)
        p7.ce_ema = 0.90 * p7.ce_ema + 0.10 * ce_loss.item()
        p7.ce_std = 0.95 * p7.ce_std + 0.05 * abs(ce_loss.item() - p7.ce_ema)
        p7.prev_ce = ce_loss.item()
        batch_loss = ce_loss + aux_scale*(temps[1]*loss_A_c + temps[2]*loss_B_c + temps[3]*loss_C_c) + temps[0]*gate_loss

        # 自学习率: 每批更新 (暂时关闭, 用固定lr)
        # for slr in slrs: slr.step_lr(batch_loss.item())
        opts = [opt_p7_in, opt_p7, opt_p6, opt_p3l, opt_abc, opt_gate, opt_master]
        # for s, o in zip(slrs, opts): s.apply(o)  # SelfLR关闭

        for o in opts: o.zero_grad()
        if args.amp:
            scaler.scale(batch_loss).backward()
        else:
            batch_loss.backward()
        # NaN检测+自动回滚 (检测所有参数, 报告来源)
        has_nan = False
        nan_source = None
        for m_name, module in [('char_embed', char_embed), ('p7', p7), ('p6', p6),
                                ('p7in', p7_input_proj), ('p3l', p3l),
                                ('abc_prime', abc_prime), ('abc_to_sent', abc_to_sent)]:
            for p in module.parameters():
                if p.grad is not None and torch.isnan(p.grad).any():
                    has_nan = True; nan_source = m_name; break
            if has_nan: break
        if has_nan:
            w(f" [NaN]梯度异常! 来源={nan_source}, 跳过step, 从best恢复")
            ckpt_r = torch.load(os.path.join(SAVE_DIR, "V21_best.pt"), map_location=DEVICE)
            # char_embed: 按索引迁移(新旧词表可能不同)
            if 'char_embed' in ckpt_r:
                with torch.no_grad():
                    ce_ckpt = ckpt_r['char_embed']['weight']
                    ce_curr = char_embed.weight
                    n_match = min(ce_ckpt.shape[0], ce_curr.shape[0])
                    ce_curr[:n_match] = ce_ckpt[:n_match].to(DEVICE)
            for key in ['p7in','p7','p6','p3l','abcA','abcB','abcC',
                        'abc_to_gate','sent_to_gate','explore','meta','master']:
                if key in ckpt_r:
                    try:
                        mod = {'p7in':p7_input_proj,'p7':p7,'p6':p6,'p3l':p3l,
                               'abcA':abcA,'abcB':abcB,'abcC':abcC,
                               'abc_to_gate':abc_to_gate,'sent_to_gate':sent_to_gate,
                               'explore':explore,'meta':meta,'master':master_gate}[key]
                        mod.load_state_dict(ckpt_r[key], strict=False)
                    except: pass
            # V21.1新模块: 冷启动(不加回滚, 保留当前初始化)
            del ckpt_r
            for o in opts: o.zero_grad()
        else:
            for o in opts: o.step()

        del wo_batch, sv_batch, wg_batch, Av_emb, pred_all, a_logits, ah, ct, ac_batch, attr_b
        import gc; gc.collect()
        torch.cuda.synchronize(); torch.cuda.empty_cache()

        total_loss += batch_loss.item(); total_ce += ce_loss.item()
        total_A += loss_A_c.item(); total_B += loss_B_c.item()
        total_C += loss_C_c.item(); total_gate += gate_loss.item()
        last_temps = temps; last_scale = aux_scale; n += 1
        pct = (1 - len(remaining) / len(train_set)) * 100

        if n % 100 == 0:  # 进度条
            mem = get_gpu_mem_used_mb()
            delta = f'+{mem-_VRAM_BASELINE:.0f}' if _VRAM_BASELINE and mem > _VRAM_BASELINE else ''
            gpu_str = ''
            elapsed_ep = time.time()-t0
            eta_ep = elapsed_ep / max(pct, 1) * (100 - pct)
            w(f"  E{ep} [{pct:.0f}%] batch={current_bs} {gpu_str} loss={batch_loss.item():.4f} | {elapsed_ep:.0f}s ETA{eta_ep:.0f}s")

    elapsed = time.time()-t0; avg_loss = total_loss/max(n,1)

    if avg_loss < best_loss:
        best_loss = avg_loss
        best_epoch = ep
        try: save_best()
        except: pass
    if avg_loss < 0.0001:  # 收敛到底, 继续训只会炸
        w(f"  [OK] loss={avg_loss:.6f}<0.0001, 自动停训")
        break

    # 梯度诊断 (每5轮)
    if ep % 5 == 0:
        with torch.no_grad():
            mods = {'p7_in':p7_input_proj,'p7':p7,'p6':p6,'p3l':p3l,
                    'abcA':abcA,'abcB':abcB,'abcC':abcC,
                    'explore':explore,'meta':meta,'abc_to_gate':abc_to_gate,'sent_to_gate':sent_to_gate}
            for name, mod in mods.items():
                gn = sum(p.grad.norm().item() for p in mod.parameters() if p.grad is not None)
                pn = sum(p.norm().item() for p in mod.parameters())
                w(f"  grad:{name}={gn:.4f} param={pn:.1f} ratio={gn/(pn+1e-8):.6f}")

    if (ep<=3 or ep%args.display==0 or ep==args.epochs) and len(A_w) > 5:  # 跳过太小样本的展示
        with torch.no_grad():
            # 推理全链: P3属性+P7+P3-L+ABC+Gate+P6
            p7._loss_vec = None  # 推理前清除训练_loss_vec残留
            A_emb = char_embed(torch.tensor(A_ids, device=DEVICE))
            attr_a = words_to_attr_vec(A_w).to(DEVICE)[:len(A_w),:]
            Av_rich = torch.cat([A_emb, attr_a], dim=-1).unsqueeze(0)
            vocab = char_embed.weight.detach()
            wo, sv, _, wg, _ = p7.forward_batch(p7_input_proj(Av_rich), vocab.unsqueeze(0), [len(A_ids)], [n_chars])
            # P3-L + ABC (推理前向, 不传梯度)
            test_ids_attr = words_to_attr_ids(A_w)
            if len(test_ids_attr) >= 2:
                _, attns = p3l(torch.tensor(test_ids_attr, device=DEVICE))
                p3l_f = torch.cat([torch.stack([s.mean(),s.std(),s.max()-s.min()]) for s in attns.values() if s.numel()>0])
                p3l_f = F.pad(p3l_f, (0,max(0,128-p3l_f.shape[0])))[:128].unsqueeze(0)
            else:
                p3l_f = torch.zeros(1, 128, device=DEVICE)
            a_log, ah = abcA(attr_a.unsqueeze(0), wo[:,:attr_a.shape[0],:])
            ct = abcB(ah, attr_a.unsqueeze(0), p3l_f)
            ac = abcC(ct, attr_a.unsqueeze(0))
            # Gate用完整信号(base+ABC+sent, 训练推理一致)
            a_prob_d = a_log.softmax(-1)
            abc_conf_d = torch.stack([a_prob_d.max(-1).values.mean(),-(a_prob_d*torch.log(a_prob_d+1e-8)).sum(-1).mean(),a_log.argmax(-1).float().mean(),ac.norm(dim=-1).mean(),ac.std(dim=-1).mean(),ac.max(dim=-1).values.mean(),ct.norm(dim=-1).mean(),ct.std(dim=-1).mean(),ac[:,:5].norm(dim=-1).mean(),ac[:,5:15].norm(dim=-1).mean(),ac[:,15:].norm(dim=-1).mean(),ac.norm(dim=-1).mean()]).unsqueeze(0)
            gate = meta(explore(torch.cat([gate_base_proj(abc_conf_d), abc_to_gate(ac), sent_to_gate(sv)], dim=-1)))
            logits = p6(sv, char_embed.weight, gate=gate)[:, :len(B_w), :]
            pred_ids = logits[0].argmax(dim=-1).tolist()
            pred = [i2c.get(i, '?') for i in pred_ids]
            ok = sum(1 for p,t in zip(pred, B_w) if p==t)

        ttl = time.time()-total_t0; eta = ttl/ep*(args.epochs-ep) if ep>0 else 0
        gpu_u = get_gpu_util()
        gpu_str = f"GPU={gpu_u}%" if gpu_u >= 0 else "GPU=?"
        # P3-L探针: 取attention最大值看梯度流通
        p3l_max = 0.0
        with torch.no_grad():
            aids = words_to_attr_ids(A_w[:5])
            if len(aids) >= 2:
                _, attns = p3l(torch.tensor(aids, device=DEVICE))
                for gname, scores in attns.items():
                    p3l_max = max(p3l_max, scores.detach().max().item())
        w(f"E{ep:5d}/{args.epochs} | loss={avg_loss:.4f} | ok={ok}/{len(B_w)} | P3L_max={p3l_max:.4f} | {elapsed:.0f}s")
        # Loss构成 + 学习率
        avg_ce = total_ce/max(n,1); avg_A = total_A/max(n,1); avg_B = total_B/max(n,1)
        avg_C = total_C/max(n,1); avg_g = total_gate/max(n,1)
        tA,tB,tC,tG = last_temps[1].item(),last_temps[2].item(),last_temps[3].item(),last_temps[0].item() if last_temps is not None else (0,0,0,0)
        lr_str = ' '.join([f"{s.name}={s.lr:.1e}" for s in slrs])
        trend_up = (p7.prev_ce > p7.prev2_ce) if hasattr(p7,'prev_ce') and hasattr(p7,'prev2_ce') else False
        st = "[OK]" if (p7.ce_std>p7.ce_ema*0.5 and p7.ce_ema>1 and trend_up) else ("[OK]" if p7.ce_ema<0.5 else "↓")
        gb = p7.g_base_sum/max(n,1); ga = p7.g_aux_sum/max(n,1); gs = p7.g_sent_sum/max(n,1)
        gsum = gb+ga+gs+1e-8
        w(f"  CE={avg_ce:.3f} A={avg_A:.3f} B={avg_B:.2e} C={avg_C:.2e} gate={avg_g:.3f} | ema={p7.ce_ema:.2f} aux_s={last_scale:.2f} {st}")
        w(f"  gate占比: base={gb/gsum*100:.0f}% abc={ga/gsum*100:.0f}% sent={gs/gsum*100:.0f}% | ac_raw={p7.g_ac_sum/max(n,1):.2f}")
        p7.g_base_sum = p7.g_aux_sum = p7.g_sent_sum = p7.g_ac_sum = 0.0
        w(f"  lr: {lr_str}")
        w(f"  预测: {''.join(pred[:20])}")
        w(f"  正确: {''.join(B_w[:20])}")
        w(f"{'='*70}")
    elif ep%200==0:
        ttl=time.time()-total_t0; eta=ttl/ep*(args.epochs-ep)
        w(f"  E{ep:5d} | loss={avg_loss:.4f} | ETA {eta/3600:.1f}h")

# ════════════════════════ 保存 ════════════════════════
save_path = os.path.join(SAVE_DIR, f"V19_{MODE}.pt")
torch.save({"char_embed":char_embed.state_dict(),"p7in":p7_input_proj.state_dict(),"p7":p7.state_dict(),"p6":p6.state_dict(),"p3l":p3l.state_dict(),
            "abcA":abcA.state_dict(),"abcB":abcB.state_dict(),"abcC":abcC.state_dict(),
            "explore":explore.state_dict(),"meta":meta.state_dict(),"master":master_gate.state_dict(),
            "abc_to_gate":abc_to_gate.state_dict(),
            "sent_to_gate":sent_to_gate.state_dict()}, save_path)
w(f"\n[保存] {save_path}")

# ════════════════════════ 属性关联分析 ════════════════════════
w(f"\n[属性关联度] 训练集首对详情:")
A0, B0 = train_set[0][2], train_set[0][3]
with torch.no_grad():
    attr_vec = words_to_attr_vec(A0).unsqueeze(0).to(DEVICE)[:, :min(len(A0), 30), :]
    wo_pool = torch.zeros(1, 256, device=DEVICE)
    wo_exp = wo_pool.unsqueeze(1).expand(1, attr_vec.shape[1], 256)
    a_log, ah = abcA(attr_vec, wo_exp)
    ct = abcB(ah, attr_vec, torch.zeros(1, 128, device=DEVICE))
    ac = abcC(ct, attr_vec)
w(f"  A句: {''.join(A0[:30])}")
w(f"  B句: {''.join(B0[:30])}")
w(f"  ┌─ ABC输出 ───────────────────────")
w(f"  │ StageA 结构logits(前5): {a_log[0,:5].tolist()}")
w(f"  │ StageA 预测类: {a_log[0].argmax().item()} / 15")
w(f"  │ StageB 内容: mean={ct[0].mean().item():.3f} std={ct[0].std().item():.3f} (前5)={ct[0,:5].tolist()}")
w(f"  │ StageC 语气: mean={ac[0].mean().item():.3f} std={ac[0].std().item():.3f} (前5)={ac[0,:5].tolist()}")
w(f"  └──────────────────────────────────")
w(f"  ┌─ P3属性(前8字) ─────────────────")
pkts = p3_stack.process_sentence(A0[:8])
for p in pkts:
    w(f"  │ [{p.word}] type={p.basic_type} sem={p.best_semantic()} syn={p.best_syntax()} person={p.person} emot={p.polarity}")
w(f"  └──────────────────────────────────")
w(f"  ┌─ P3-L 属性关联 (312头前5组) ────")
aid_list = list(range(min(50, n_chars)))
if len(aid_list) >= 2:
    _, attns = p3l(torch.tensor(aid_list, device=DEVICE))
    for i, (gname, scores) in enumerate(attns.items()):
        if i >= 5: break
        s = scores.detach()
        w(f"  │ {gname}({s.numel()}值): mean={s.mean().item():.3f} max={s.max().item():.3f}")
w(f"  └──────────────────────────────────")
w(f"  ┌─ Gate状态 ───────────────────────")
w(f"  │ explore: signal_mean={explore(torch.randn(12,device=DEVICE)).mean().item():.3f}")
w(f"  │ meta: bias_mean={meta.bias.mean().item():.3f} bias_std={meta.bias.std().item():.3f}")
w(f"  │ Master: temps={master_gate(torch.randn(5,device=DEVICE)).tolist()}")
w(f"  └──────────────────────────────────")

if exam_set:
    # 用训练集首对做"考试" — 验证模型至少学会训练数据
    w("\n[训练记忆验证]...")
    A_ids, B_ids, A, B = train_set[0]
    with torch.no_grad():
        p7._loss_vec = None  # 清除训练残留
        Av_emb = char_embed(torch.tensor(A_ids,device=DEVICE)).unsqueeze(0)
        attr_a = words_to_attr_vec(A).to(DEVICE).unsqueeze(0)[:,:len(A),:]
        Av_rich = torch.cat([Av_emb, attr_a], dim=-1)
        wo, sv, _, _, _ = p7.forward_batch(p7_input_proj(Av_rich), vocab.unsqueeze(0), [len(A_ids)], [n_chars])
        # 真实gate
        aids_v = words_to_attr_ids(A)
        if len(aids_v) >= 2:
            _, attns_v = p3l(torch.tensor(aids_v,device=DEVICE))
            feats_v = [torch.stack([s.mean(),s.std(),s.max()-s.min()]) for s in attns_v.values() if s.numel()>0]
            p3lf_v = torch.cat(feats_v) if feats_v else torch.zeros(60,device=DEVICE)
        else: p3lf_v = torch.zeros(60,device=DEVICE)
        p3lf_v = F.pad(p3lf_v,(0,max(0,128-p3lf_v.shape[0])))[:128].unsqueeze(0)
        a_log_v, ah_v = abcA(attr_a.squeeze(0), wo[:,:attr_a.shape[1],:])
        ct_v = abcB(ah_v, attr_a, p3lf_v)
        ac_v = abcC(ct_v, attr_a)
        a_prob_v = a_log_v.softmax(-1)
        abc_conf_v = torch.stack([a_prob_v.max(-1).values.mean(),-(a_prob_v*torch.log(a_prob_v+1e-8)).sum(-1).mean(),a_log_v.argmax(-1).float().mean(),ac_v.norm(dim=-1).mean(),ac_v.std(dim=-1).mean(),ac_v.max(dim=-1).values.mean(),ct_v.norm(dim=-1).mean(),ct_v.std(dim=-1).mean(),ac_v[:,:5].norm(dim=-1).mean(),ac_v[:,5:15].norm(dim=-1).mean(),ac_v[:,15:].norm(dim=-1).mean(),ac_v.norm(dim=-1).mean()]).unsqueeze(0)
        gate_v = meta(explore(torch.cat([gate_base_proj(abc_conf_v), abc_to_gate(ac_v), sent_to_gate(sv)], dim=-1)))
        logits = p6(sv, char_embed.weight, gate=gate_v)[:,:len(B),:]
        preds = logits[0].argmax(-1).tolist()
        ok_train = sum(1 for i,p in enumerate(preds) if i2c.get(p,'?')==B[i])
        w(f"  训练记忆: {ok_train}/{len(B)} = {ok_train/len(B)*100:.0f}%")
        w(f"  预测: {''.join([i2c.get(p,'?') for p in preds[:30]])}")
        w(f"  正确: {''.join(B[:30])}")

    w("\n[考试集评测]...")
    t0=time.time();tok,tn=0,0
    vocab = char_embed.weight.detach()
    for A_ids,B_ids,A,B in exam_set[:100]:
        with torch.no_grad():
            p7._loss_vec = None
            Av_emb = char_embed(torch.tensor(A_ids,device=DEVICE)).unsqueeze(0)
            attr_a = words_to_attr_vec(A).to(DEVICE).unsqueeze(0)[:,:len(A),:]
            Av_rich = torch.cat([Av_emb, attr_a], dim=-1)
            wo, sv, _, _, _ = p7.forward_batch(p7_input_proj(Av_rich), vocab.unsqueeze(0),
                                              [len(A_ids)], [n_chars])
            # 考试gate: 完整推理链
            aids = words_to_attr_ids(A)
            if len(aids) >= 2:
                _, attns = p3l(torch.tensor(aids,device=DEVICE))
                feats = [torch.stack([s.mean(),s.std(),s.max()-s.min()]) for s in attns.values() if s.numel()>0]
                p3lf_e = torch.cat(feats) if feats else torch.zeros(60,device=DEVICE)
            else: p3lf_e = torch.zeros(60,device=DEVICE)
            p3lf_e = F.pad(p3lf_e,(0,max(0,128-p3lf_e.shape[0])))[:128].unsqueeze(0)
            a_log_e, ah_e = abcA(attr_a.squeeze(0), wo[:,:attr_a.shape[1],:])
            ct_e = abcB(ah_e, attr_a, p3lf_e)
            ac_e = abcC(ct_e, attr_a)
            a_prob_e = a_log_e.softmax(-1)
            abc_conf_e = torch.stack([a_prob_e.max(-1).values.mean(),-(a_prob_e*torch.log(a_prob_e+1e-8)).sum(-1).mean(),a_log_e.argmax(-1).float().mean(),ac_e.norm(dim=-1).mean(),ac_e.std(dim=-1).mean(),ac_e.max(dim=-1).values.mean(),ct_e.norm(dim=-1).mean(),ct_e.std(dim=-1).mean(),ac_e[:,:5].norm(dim=-1).mean(),ac_e[:,5:15].norm(dim=-1).mean(),ac_e[:,15:].norm(dim=-1).mean(),ac_e.norm(dim=-1).mean()]).unsqueeze(0)
            gate_e = meta(explore(torch.cat([gate_base_proj(abc_conf_e), abc_to_gate(ac_e), sent_to_gate(sv)], dim=-1)))
            logits = p6(sv, char_embed.weight, gate=gate_e)[:,:len(B),:]
            preds = logits[0].argmax(-1).tolist()
            tok += sum(1 for i,p in enumerate(preds) if i2c.get(p,'?')==B[i]); tn += len(B)
    w(f"  考试集: {float(tok)/float(tn)*100:.1f}% ({int(tok)}/{int(tn)}) | {time.time()-t0:.0f}s")

w(f"[日志] {LOG_PATH}")
log.close()
