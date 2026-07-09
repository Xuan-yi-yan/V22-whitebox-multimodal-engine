# -*- coding: utf-8 -*-
"""V22 24轮环形接力辩论 — 6模型×4轮, 信息压缩传递, 只追技术极限"""
import time, requests, sys, os

COMET_BASE="https://api.cometapi.com/v1"
K2="sk-JQ275W3heOmJvsrEf64uB12qfUCXDjkNcawSx9HorrPskf5V"
DS_KEY="sk-f4b7a4abc6234e78987d51e7ce4d19b0"
QW_KEY="sk-3940b50f25a945789e6438ad11e434f8"

def ask_comet(key, model, msg, sys="你是资深AI架构师,用中文回答。", temp=0.5):
    t0=time.time()
    try:
        r=requests.post(f"{COMET_BASE}/chat/completions",
            headers={"Authorization":f"Bearer {key}","Content-Type":"application/json"},
            json={"model":model,"messages":[{"role":"system","content":sys},
                  {"role":"user","content":msg}],"temperature":temp},timeout=300,
            proxies={"http":None,"https":None})
        d=r.json()
        if "choices" in d: return d["choices"][0]["message"]["content"], time.time()-t0
        return f"[ERR]{d}",0
    except Exception as e:
        return f"[EXC]{e}",0

def ask_ds(msg, temp=0.5):
    t0=time.time()
    try:
        r=requests.post("https://api.deepseek.com/v1/chat/completions",
            headers={"Authorization":f"Bearer {DS_KEY}","Content-Type":"application/json"},
            json={"model":"deepseek-chat","messages":[{"role":"user","content":msg}],"temperature":temp},timeout=300,
            proxies={"http":None,"https":None})
        d=r.json()
        if "choices" in d: return d["choices"][0]["message"]["content"], time.time()-t0
        return f"[ERR]{d}",0
    except Exception as e:
        return f"[EXC]{e}",0

def ask_qw(msg, temp=0.5):
    t0=time.time()
    try:
        r=requests.post("https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions",
            headers={"Authorization":f"Bearer {QW_KEY}","Content-Type":"application/json"},
            json={"model":"qwen-max","messages":[{"role":"user","content":msg}],"temperature":temp},timeout=300,
            proxies={"http":None,"https":None})
        d=r.json()
        if "choices" in d: return d["choices"][0]["message"]["content"], time.time()-t0
        return f"[ERR]{d}",0
    except Exception as e:
        return f"[EXC]{e}",0

# ── 加载背景材料 ──
with open("C:/ai/V22_架构终极全貌_2026-07-06.txt","r",encoding="utf-8") as f:
    V22_DOC=f.read()[:6000]
with open("C:/ai/V21_Bug猎人_执着修复全记录.txt","r",encoding="utf-8") as f:
    BUG_LOG=f.read()
with open("C:/ai/eval_debug.py","r",encoding="utf-8") as f:
    DEBUG_CODE=f.read()[:2000]

# ── 架构师核心立场 ──
ARCHITECT_STANCE="""【架构师核心立场 — 本场辩论的不可动摇前提】

1. 不考虑现实落地。不考虑生产部署。不考虑算力成本。只追求技术极限。
2. 高颗粒度(312头) + 低量高质量数据 = 真正的突破点。
   所有模型在第1-3轮说"312头太多要砍"——这个判断基于"数据量大→头需要宽"的Scaling惯性。
   但V22走的是反方向: 每个头被压缩到1-2D, 在高质量数据下反而学到更干净的特征方向,
   避免了宽头内部的噪声混叠。这才是技术壁垒所在。
3. 球面归一化是V22的灵魂,不是"维稳工具"。它使得312个微头在几何约束下各司其职。
4. 7独立Adam不是显存浪费——是"每个模块有自己的收敛节奏"的精细控制。统一优化器=失去控制粒度。
5. 白盒不是为了可解释性报告——是为了"跑通一个实验,立刻吸收能力"。
   每一层的激活值可直接阅读、直接调试、直接修改。这是黑盒永远做不到的。
6. Bug猎人记录证明: 不跑验证不接受结论。25个假说推翻了18个。
   模型们的建议如果不经过实验验证,一律视为待验证假说。

请站在这个立场上发言。不要建议"砍头""减少模块""降低复杂度"。
要建议"怎么让312头在12GB上跑起来""怎么让7Adam更高效""怎么让球面归一化发挥更大威力"。
如果觉得某个设计有问题,给出在保持设计意图前提下的改进方案,而不是建议删掉它。"""

# ── 六模型环形顺序 ──
TEAM=[
    ("Claude-Opus-4.5",  lambda m,t=0.5: ask_comet(K2,"claude-opus-4-5-20251101",m,temp=t)),
    ("Gemini-2.5-Pro",   lambda m,t=0.5: ask_comet(K2,"gemini-2.5-pro",m,temp=t)),
    ("Claude-Sonnet-4",  lambda m,t=0.5: ask_comet(K2,"claude-sonnet-4-20250514",m,temp=t)),
    ("DeepSeek-V4(C)",   lambda m,t=0.5: ask_comet(K2,"deepseek-v4-pro",m,temp=t)),
    ("DeepSeek-V4(国内)", lambda m,t=0.5: ask_ds(m,t)),
    ("Qwen3.7(国内)",    lambda m,t=0.5: ask_qw(m,t)),
]
N=len(TEAM)  # 6
TOTAL_ROUNDS=24  # 每人4轮

# ── 状态 ──
# 每个模型的"认同摘要"——这是传递给下一个人的压缩信息
agreement_chain=[""]*N  # 每人持有的"我认同前人的什么"
last_full_response=[""]*N  # 每人上一轮完整回答
round_counter=[0]*N  # 每人已发言次数

print("="*60)
print("  V22 24轮环形接力辩论")
print("  6模型 × 4轮 = 24次顶级推理")
print("  架构师立场: 只追技术极限, 不考虑落地")
print("  信息传递: 每人压缩前人观点 → 传递给下一人")
print("="*60)

# ═══════════════════════════════════════════════════════════
# 第1轮: 并行 — 所有模型阅读全部材料, 发表首次立场
# ═══════════════════════════════════════════════════════════
print(f"\n{'#'*60}")
print(f"  第1轮: 全体阅读材料, 发表初始立场")
print(f"{'#'*60}")

R1_PROMPT=f"""你是AI架构师。请仔细阅读以下全部材料:

═══════════════════════════════════════════════
材料1: V22白盒多模态引擎架构
{V22_DOC}
═══════════════════════════════════════════════
材料2: Bug猎人全记录 (V21全部Bug定位与修复, 展示了"不跑验证不接受结论"的方法论)
{BUG_LOG[:4000]}
═══════════════════════════════════════════════
材料3: 架构师核心立场 (不可动摇的辩论前提)
{ARCHITECT_STANCE}
═══════════════════════════════════════════════

这是第1轮。请发表你的初始立场(不少于500字):

1. 你认同架构师立场中的哪些点? 为什么?
2. 在"不考虑落地、只追技术极限"的前提下, 你认为V22最值得深挖的3个技术方向是什么?
3. 对312头P3-L、7独立Adam、球面归一化这三个被其他模型质疑的设计,
   在保持设计意图的前提下, 你能给出什么强化方案?

最后, 请写一段100字以内的"认同摘要", 用「我认同: ...」开头,
这段摘要将被传递给下一个模型。"""

for i,(name,fn) in enumerate(TEAM):
    print(f"\n[{i+1}/6] {name}",end=" ",flush=True)
    ans,lat=fn(R1_PROMPT, 0.5)
    fname=f"C:/ai/relay_r01_{name.replace(' ','_')}.txt"
    with open(fname,"w",encoding="utf-8") as f: f.write(ans)
    # 提取"我认同:"摘要
    ag_start=ans.find("我认同:")
    if ag_start>=0:
        ag_end=ans.find("\n", ag_start+5)
        if ag_end<0: ag_end=ag_start+150
        agreement_chain[i]=ans[ag_start:ag_end].strip()[:200]
    else:
        agreement_chain[i]=f"[{name}]认同V22高颗粒度+低量高质量数据的突破方向"
    last_full_response[i]=ans
    round_counter[i]=1
    print(f"✓ {lat:.1f}s ({len(ans)}字) 摘要:{len(agreement_chain[i])}字")
    print(f"─── {fname} ───")

# ═══════════════════════════════════════════════════════════
# 第2-24轮: 环形接力
# ═══════════════════════════════════════════════════════════
for rnd in range(2, TOTAL_ROUNDS+1):
    idx = (rnd-1) % N  # 当前发言的模型
    prev_idx = (idx-1) % N  # 前一个模型
    prev2_idx = (idx-2) % N  # 前前一个模型
    name, fn = TEAM[idx]
    prev_name = TEAM[prev_idx][0]
    prev2_name = TEAM[prev2_idx][0]

    print(f"\n{'='*60}")
    print(f"  第{rnd}轮 — {name} (第{round_counter[idx]+1}次发言)")
    print(f"  接收: {prev_name}的完整观点 + {prev2_name}→{prev_name}的摘要链")
    print(f"{'='*60}")

    prev_full = last_full_response[prev_idx]
    prev_ag = agreement_chain[prev_idx]
    prev2_ag = agreement_chain[prev2_idx]

    relay_prompt=f"""第{rnd}轮环形辩论 — 轮到你了: {name}

═══════════════════════════════════════════════
架构师核心立场 (始终有效):
{ARCHITECT_STANCE[:2000]}
═══════════════════════════════════════════════

你的前两个人:
[{prev2_name}] 认同摘要: {prev2_ag}
    ↓ 传递给 ↓
[{prev_name}] 认同摘要: {prev_ag}
    ↓ 他们的完整观点如下 ↓

═══════════════════════════════════════════════
[{prev_name}] 上一轮的完整观点 (请仔细阅读):
{prev_full[:4000]}
═══════════════════════════════════════════════

你的任务:
1. 阅读{prev_name}的完整观点后, 提炼你认同的2-3个核心点,
   用「我认同{prev_name}: ...」格式写出(150字以内)。
2. 对你不认同的点, 给出技术层面的反驳(300字以上)。
3. 在"只追技术极限"的前提下, 提出你的新洞察或深化方案(300字以上)。
4. 最后, 写一段100字以内的"认同摘要"(从你本轮的完整观点中提炼),
   用「我认同: ...」开头, 这段将被传递给下一个模型({TEAM[(idx+1)%N][0]})。

注意:
- 你是{name}。你的前一轮发言是第{round_counter[idx]}轮。
- 只讨论技术深度, 不考虑落地/部署/成本。
- 如果同意某个设计方向, 给出让它更强的方案, 而不是建议简化。"""

    print(f"  [{name}] 思考中...", end=" ", flush=True)
    ans,lat=fn(relay_prompt, 0.5)
    fname=f"C:/ai/relay_r{rnd:02d}_{name.replace(' ','_')}.txt"
    with open(fname,"w",encoding="utf-8") as f: f.write(ans)

    # 提取新一轮的"我认同:"摘要
    ag_start=ans.find("我认同:")
    if ag_start>=0:
        ag_end=ans.find("\n", ag_start+5)
        if ag_end<0: ag_end=ag_start+200
        agreement_chain[idx]=ans[ag_start:ag_end].strip()[:200]
    else:
        # 尝试找"认同摘要"段
        ag_start2=ans.rfind("认同摘要")
        if ag_start2>=0:
            agreement_chain[idx]=ans[ag_start2:ag_start2+200].strip()[:200]
        else:
            agreement_chain[idx]=f"[{name}]认同前人的核心技术方向,新洞察见全文"

    last_full_response[idx]=ans
    round_counter[idx]+=1

    print(f"✓ {lat:.1f}s ({len(ans)}字)")
    print(f"  摘要: {agreement_chain[idx][:100]}...")
    print(f"─── {fname} ───")

# ═══════════════════════════════════════════════════════════
# 终局统计
# ═══════════════════════════════════════════════════════════
print(f"\n{'='*60}")
print(f"  V22 24轮环形接力辩论 完成!")
print(f"  6模型 × 4轮 = 24次推理")
print(f"  发言统计:")
for i,(name,_) in enumerate(TEAM):
    print(f"    {name}: {round_counter[i]}次发言")
print(f"{'='*60}")
print(f"  全部文件: C:/ai/relay_r*.txt")
