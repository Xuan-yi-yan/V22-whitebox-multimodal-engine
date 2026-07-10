# -*- coding: utf-8 -*-
"""V22.1 情感对话训练 — 6情绪分类+回复生成, 480条数据 50epoch"""
import torch, torch.nn as nn, torch.nn.functional as F, random, sys, os
sys.path.insert(0, 'C:/ai')
from v22_vmf_modules import TurboMeta_vMF, KappaPhaseScheduler
from collections import defaultdict

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.manual_seed(42)
random.seed(42)

# ── 数据加载 ──
data_path = "C:/ai/data/emotion_dialogue.txt"
if not os.path.exists(data_path):
    # 自动生成
    import subprocess
    subprocess.run(["python", "C:/ai/v22_emotion_data.py"])

emotions = ["真诚感谢", "愤怒指责", "阴阳怪气", "无奈吐槽", "撒娇卖萌", "中性陈述"]
e2i = {e: i for i, e in enumerate(emotions)}

pairs = []
with open(data_path, "r", encoding="utf-8") as f:
    for line in f:
        parts = line.strip().split("\t")
        if len(parts) == 3:
            pairs.append((parts[0], parts[1], parts[2]))
print(f"数据: {len(pairs)} 条 | 情绪: {len(emotions)} 类")

# ── 构建字表 ──
chars = set()
for e, a, b in pairs:
    for c in a + b:
        chars.add(c)
chars = sorted(chars)
c2i = {c: i+2 for i, c in enumerate(chars)}  # 0=PAD 1=EOS
c2i['<PAD>'] = 0
c2i['<EOS>'] = 1
i2c = {i: c for c, i in c2i.items()}
V = len(c2i)
print(f"字表: {V} | 最大长度: {max(len(a)+len(b) for _,a,b in pairs)}")

# ── 模型 (V22 真实结构简化) ──
DIM = 256
N_EMOTION = 6

# Embed
char_embed = nn.Embedding(V, DIM)
emotion_embed = nn.Embedding(N_EMOTION, 32)
nn.init.orthogonal_(char_embed.weight)

# Encoder (P7 + ABC 等效)
encoder = nn.Sequential(
    nn.Linear(DIM, DIM), nn.GELU(),
    nn.Linear(DIM, DIM), nn.GELU(),
    nn.Linear(DIM, DIM))

# Gate (vMF)
class MiniExplore(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(DIM, 512), nn.GELU(), nn.Linear(512, DIM))
    def forward(self, x): return self.net(x)

explore = MiniExplore()
meta = TurboMeta_vMF(dim=DIM)

# Decoder (P6)
decoder = nn.Linear(DIM, V)

# ── 优化器 ──
params = list(char_embed.parameters()) + list(emotion_embed.parameters()) + \
         list(encoder.parameters()) + list(explore.parameters()) + \
         list(meta.parameters()) + list(decoder.parameters())

# 情绪投影层 (放到device)
e_proj = nn.Linear(DIM+32, DIM, bias=False).to(device)

# 移动所有模块到device
for m in [char_embed, emotion_embed, encoder, explore, meta, decoder]:
    m.to(device)

params += list(e_proj.parameters())
opt = torch.optim.Adam(params, lr=0.002)
total_params = sum(p.numel() for p in params)

# ── 工具 ──
def encode(text, max_len=20):
    ids = [c2i.get(c, 0) for c in text[:max_len]]
    ids.append(1)  # EOS
    return torch.tensor(ids, device=device)

def decode(ids):
    return ''.join(i2c.get(i, '?') for i in ids if i > 1)

# ── 训练 ──
kappa_sched = KappaPhaseScheduler(total_steps=50*len(pairs), explore_ratio=0.3)
bs = 8

print(f"\n{'='*55}")
print(f"  V22.1 情感对话训练 | {total_params:,}参数 | {len(pairs)}条")
print(f"  vMF激活 | κ相变 | 6情绪分类+回复生成")
print(f"{'='*55}")
print(f"  {'Epoch':<6} {'CE':>8} {'Acc%':>7} {'κ':>6} {'phase':>11}")
print(f"  {'-'*50}")

for epoch in range(300):
    random.shuffle(pairs)
    total_loss = 0.0
    total_ok = 0
    total_tokens = 0

    for i in range(0, len(pairs), bs):
        batch = pairs[i:i+bs]
        if len(batch) < 2: continue

        batch_loss = 0.0
        batch_ok = 0
        batch_tokens = 0

        for emotion, a_text, b_text in batch:
            e_id = e2i.get(emotion, 0)
            a_ids = encode(a_text)
            b_ids = encode(b_text)

            if len(a_ids) == 0 or len(b_ids) == 0: continue

            # Encoder: 输入字 + 情绪标记
            emb_a = char_embed(a_ids)                          # [La, DIM]
            emb_e = emotion_embed(torch.tensor([e_id], device=device))  # [1, 32]
            # 情绪注入: 拼接到每个位置
            emb_a = torch.cat([emb_a, emb_e.expand(len(a_ids), -1)], dim=-1)  # [La, DIM+32]
            # 投影回DIM
            h = F.gelu(encoder(e_proj(emb_a)))                # [La, DIM]

            # Gate: 用编码输出的均值作为gate信号
            gate_input = h.mean(dim=0).unsqueeze(0)            # [1, DIM]
            gate_hidden = explore(gate_input)
            gates = meta(gate_hidden)                          # [1, DIM] vMF

            # Decoder: 从h的最后状态预测B句
            ctx = h[-1] * gates.squeeze(0)                     # [DIM]
            logits = decoder(ctx)                              # [V]

            # 对B句每个字计算CE
            b_emb = char_embed(b_ids)                         # [Lb, DIM]
            ctx_expanded = ctx.unsqueeze(0).expand(len(b_ids), -1)  # [Lb, DIM]
            b_logits = decoder(ctx_expanded * gates)           # [Lb, V]

            ce = F.cross_entropy(b_logits, b_ids)
            batch_loss += ce
            batch_tokens += len(b_ids)

            # 准确率：argmax是否命中
            preds = b_logits.argmax(dim=-1)
            batch_ok += (preds == b_ids).sum().item()

        if batch_tokens == 0: continue

        loss = batch_loss / len(batch)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()

        total_loss += loss.item()
        total_ok += batch_ok
        total_tokens += batch_tokens
        kappa_sched.step()

    avg_loss = total_loss / max(1, len(pairs)//bs)
    acc = total_ok / max(1, total_tokens) * 100
    k_stats = meta.get_kappa_stats()
    phase = kappa_sched.phase

    if epoch % 20 == 0 or epoch < 5:
        print(f"  E{epoch:<5} {avg_loss:>8.4f} {acc:>7.1f} {k_stats['mean']:>5.2f} {phase:>11}")

# ── 最终测试 ──
print(f"\n{'='*55}")
print(f"  最终测试 — 各情绪对话生成")
print(f"{'='*55}")

meta.eval()
encoder.eval()
decoder.eval()

test_cases = [
    ("真诚感谢", "谢谢你帮了我"),
    ("愤怒指责", "你太过分了"),
    ("阴阳怪气", "你可真厉害啊"),
    ("无奈吐槽", "又加班到十点"),
    ("撒娇卖萌", "人家想要那个嘛"),
    ("中性陈述", "今天天气不错"),
]

for emotion, text in test_cases:
    with torch.no_grad():
        e_id = e2i[emotion]
        a_ids = encode(text)
        emb_a = char_embed(a_ids)
        emb_e = emotion_embed(torch.tensor([e_id], device=device))
        emb_in = torch.cat([emb_a, emb_e.expand(len(a_ids), -1)], dim=-1)
        h_all = F.gelu(encoder(e_proj(emb_in)))
        ctx = h_all[-1]
        gate_hidden = explore(ctx.unsqueeze(0))
        gates = meta(gate_hidden)

        # 自回归生成
        gen = []
        current = ctx * gates.squeeze(0)
        for _ in range(12):
            logits = decoder(current.unsqueeze(0))
            probs = F.softmax(logits[0] / 0.5, dim=-1)  # 低温度 = 更确定
            # Top-3采样提高质量
            top_probs, top_ids = probs.topk(3)
            top_probs = top_probs / top_probs.sum()
            next_id = top_ids[torch.multinomial(top_probs, 1)].item()
            if next_id == 1: break
            if next_id <= 1: continue
            gen.append(i2c.get(next_id, '?'))
            # 用生成的字更新上下文
            current = F.gelu(encoder(e_proj(torch.cat([
                char_embed(torch.tensor([next_id], device=device)),
                emotion_embed(torch.tensor([e_id], device=device))
            ], dim=-1))))[-1] * gates.squeeze(0)
    print(f"  [{emotion}] {text}")
    print(f"         → {''.join(gen)}")

print(f"\n  训练完成 — kappa={meta.get_kappa_stats()['mean']:.2f}")
