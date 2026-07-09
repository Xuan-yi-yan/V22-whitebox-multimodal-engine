# -*- coding: utf-8 -*-
"""V22架构大检查 — 6模型×3轮 多核审查"""
import time, requests

COMET_BASE="https://api.cometapi.com/v1"

# Key1 ($2.60) → 便宜国外模型
K1="sk-JQ275W3heOmJvsrEf64uB12qfUCXDjkNcawSx9HorrPskf5V"
# Key2 ($5.00) → 最贵顶级模型
K2="sk-JQ275W3heOmJvsrEf64uB12qfUCXDjkNcawSx9HorrPskf5V"
# 国内
DS_KEY="sk-f4b7a4abc6234e78987d51e7ce4d19b0"
QW_KEY="sk-3940b50f25a945789e6438ad11e434f8"

def ask_comet(key, model, msg, sys="你是资深AI架构师,用中文回答。"):
    t0=time.time()
    r=requests.post(f"{COMET_BASE}/chat/completions",
        headers={"Authorization":f"Bearer {key}","Content-Type":"application/json"},
        json={"model":model,"messages":[{"role":"system","content":sys},
              {"role":"user","content":msg}],"temperature":0.7,},timeout=180,
        proxies={"http":None,"https":None})
    d=r.json()
    if "choices" in d: return d["choices"][0]["message"]["content"], time.time()-t0
    return f"[ERR]{d}",0

def ask_ds(msg, temp=0.7):
    t0=time.time()
    r=requests.post("https://api.deepseek.com/v1/chat/completions",
        headers={"Authorization":f"Bearer {DS_KEY}","Content-Type":"application/json"},
        json={"model":"deepseek-chat","messages":[{"role":"user","content":msg}],"temperature":temp},timeout=180,
        proxies={"http":None,"https":None})
    d=r.json()
    if "choices" in d: return d["choices"][0]["message"]["content"], time.time()-t0
    return f"[ERR]{d}",0

def ask_qw(msg, temp=0.7):
    t0=time.time()
    r=requests.post("https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions",
        headers={"Authorization":f"Bearer {QW_KEY}","Content-Type":"application/json"},
        json={"model":"qwen-max","messages":[{"role":"user","content":msg}],"temperature":temp},timeout=180,
        proxies={"http":None,"https":None})
    d=r.json()
    if "choices" in d: return d["choices"][0]["message"]["content"], time.time()-t0
    return f"[ERR]{d}",0

with open("C:/ai/V22_架构终极全貌_2026-07-06.txt","r",encoding="utf-8") as f:
    V22=f.read()[:5000]

# 6个模型, 分3层
TEAM=[
    ("Claude-Opus-4.5", lambda m,t=0.7:ask_comet(K2,"claude-opus-4-5-20251101",m), "Key2"),
    ("Gemini-2.5-Pro",  lambda m,t=0.7:ask_comet(K2,"gemini-2.5-pro",m), "Key2"),
    ("Claude-Sonnet-4", lambda m,t=0.7:ask_comet(K2,"claude-sonnet-4-20250514",m), "Key2"),
    ("DeepSeek-V4(C)",  lambda m,t=0.7:ask_comet(K2,"deepseek-v4-pro",m), "Key2"),
    ("DeepSeek-V4(国内)",lambda m,t=0.7:ask_ds(m,t), "国内"),
    ("Qwen3.7(国内)",   lambda m,t=0.7:ask_qw(m,t), "国内"),
]

print("="*60)
print("  V22白盒引擎 六核大检查 (3轮)")
print("  限制: 单卡RTX 5070 12GB, 仅小规模训练验证")
print("="*60)
for rnd in range(1,4):
    print(f"\n{'#'*60}")
    print(f"  第{rnd}轮 — 每个模型独立发表观点")
    print(f"{'#'*60}")
    for name, fn, tier in TEAM:
        prompt=f"""你是{name}。审查V22白盒多模态引擎(完整架构: github.com/Xuan-yi-yan/V22-whitebox-multimodal-engine/blob/master/V22_架构终极全貌_2026-07-06.txt)

核心摘要: CharEmbed→P3(0参数属性栈384D)→P7(16头RMA)→P3-L(312头,23组属性联动)→ABC'→ABCDEF六层(A结构,B内容,C语气,D关系审查,E记忆读写头三层缓存,F→P6)。球面归一化根治梯度崩溃。文字/视觉/声学→640D同构token。Gate三路门控,7独立Adam,5层防护。训练:11.2M参数,单卡5070,仅小规模验证。

现实约束: 单卡RTX 5070 12GB, 无法大规模训练。第{rnd}轮。请发表你的观点(不限字数):
- 第1轮: 分析V22的发展前景, 给出3个具体的优化方案
- 第2轮: 针对之前其他模型的观点, 补充或反驳
- 第3轮: 给出最终的改进路线图, 考虑硬件限制"""
        print(f"\n[{tier}] {name}",end=" ",flush=True)
        ans,lat=fn(prompt, 0.7 if rnd==1 else 0.5)
        fname=f"C:/ai/audit_r{rnd}_{name.replace(' ','_')}.txt"
        with open(fname,"w",encoding="utf-8") as f: f.write(ans)
        print(f"✓ {lat:.1f}s ({len(ans)}字)")
        print(ans)
        print(f"─── {fname} ───")

print(f"\n{'='*60}")
print(f"  六核大检查完成 — 6模型×3轮=18次顶级推理")
print(f"  覆盖: Claude-Opus-4.5, Gemini-2.5-Pro, Claude-Sonnet-4,")
print(f"        DeepSeek-V4(Comet+国内), Qwen3.7(国内)")
print(f"{'='*60}")
