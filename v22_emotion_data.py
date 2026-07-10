# -*- coding: utf-8 -*-
"""V22 情感对话数据生成 — 6种情绪 × 80对 = 480条, 种子模板+规则变体"""
import random

EMOTIONS = {
    "真诚感谢": {
        "seeds": [
            "谢谢您|不客气",
            "太感谢了|别这么说应该的",
            "帮大忙了|小事一桩",
            "感激不尽|能帮上忙就好",
            "多谢照顾|应该的应该的",
            "有你真好|嘿嘿过奖了",
            "感恩遇见|我也是",
            "谢啦|客气啥",
        ],
        "tone_words": ["谢谢", "感谢", "多谢", "感恩", "好心", "帮", "照顾"],
        "reply_words": ["不客气", "应该的", "小事", "能帮", "嘿嘿", "客气"],
    },
    "愤怒指责": {
        "seeds": [
            "你太过分了|冷静一下听我说",
            "简直不可理喻|你听我解释",
            "气死我了|消消气消消气",
            "你什么意思|我没什么意思",
            "太让人失望了|我会改的",
            "你骗我|我没有骗你",
            "滚开|别这样",
            "烦死了|我走开行了吧",
        ],
        "tone_words": ["过分", "气死", "骗", "失望", "烦", "滚", "什么", "不可理喻"],
        "reply_words": ["冷静", "消气", "解释", "改", "没有", "别这样"],
    },
    "阴阳怪气": {
        "seeds": [
            "你可真是个天才|你这是在夸我吗",
            "好厉害哦|哼你阴阳谁呢",
            "对对对你都对|本来就是",
            "真会说话啊|彼此彼此",
            "太阳打西边出来了|说谁呢你",
            "你开心就好|我当然开心",
            "没毛病老铁|这都行？",
            "哦是吗|爱信不信",
        ],
        "tone_words": ["天才", "厉害", "对对对", "会说话", "太阳", "开心", "哦"],
        "reply_words": ["夸", "哼", "谁", "彼此", "爱信", "当然"],
    },
    "无奈吐槽": {
        "seeds": [
            "又加班了|唉打工人的命",
            "老板又改需求|习惯了麻木了",
            "追的剧烂尾了|白熬那么多夜",
            "今天又堵车|哪天不堵",
            "手机又没电了|充电宝呢",
            "考试又挂了|下次一定过",
            "工资还没发|等着吧兄弟",
            "外卖又迟了|饿死了都",
        ],
        "tone_words": ["又", "唉", "无语", "废", "习惯了", "都", "还没", "算了"],
        "reply_words": ["打工人", "习惯", "麻木", "等着", "唉", "下次"],
    },
    "撒娇卖萌": {
        "seeds": [
            "人家想要嘛~|好好好给你买",
            "你都不理我|没有不理你呀",
            "求求你啦|好吧就这一次",
            "你最好了~|知道就好",
            "陪我玩嘛|好呀玩什么",
            "我今天好看吗|好看好看",
            "你凶我|我哪有凶你",
            "饿了想吃好吃的|走带你去",
        ],
        "tone_words": ["嘛", "啦", "呀", "~", "人家", "求求", "陪我", "最"],
        "reply_words": ["好", "给你", "哪有", "陪你", "好看", "带你去"],
    },
    "中性陈述": {
        "seeds": [
            "今天天气不错|是啊适合出去走走",
            "我刚到家|好的早点休息",
            "这份报告做好了|收到辛苦了",
            "明天开会|知道了准时到",
            "饭做好了|来了来了",
            "帮我拿一下快递|放门口了",
            "这个多少钱|二十块",
            "地铁还有几站|快了还有两站",
        ],
        "tone_words": ["天气", "到家", "报告", "开会", "快递", "多少", "地铁"],
        "reply_words": ["是啊", "好的", "收到", "知道", "来了", "快了"],
    },
}

def expand_seeds(seeds, tone_words, reply_words, count=80):
    """种子对 + 同义词变体 → 扩充数据"""
    data = []
    while len(data) < count:
        pair = random.choice(seeds)
        a_raw, b_raw = pair.split("|")
        # 随机变体: 加入语气词
        if random.random() < 0.3 and tone_words:
            a_raw = a_raw + random.choice(["呢", "啊", "吧", "哦", "呀", "呢"])
        if random.random() < 0.3 and reply_words:
            b_raw = b_raw + random.choice(["呢", "哈", "哦", "嗯", "呀"])
        data.append((a_raw, b_raw))
    return data

def generate_dataset(total_per_emotion=80):
    """生成完整数据集"""
    all_data = []
    for emotion, config in EMOTIONS.items():
        pairs = expand_seeds(config["seeds"], config["tone_words"],
                            config["reply_words"], total_per_emotion)
        for a, b in pairs:
            all_data.append((a, b, emotion))
    random.shuffle(all_data)
    return all_data

if __name__ == "__main__":
    random.seed(42)
    data = generate_dataset(80)
    print(f"生成数据: {len(data)} 条 (6情绪 × 80对)")
    # 每类抽样展示
    from collections import Counter
    cnt = Counter(d[2] for d in data)
    for emotion, n in cnt.items():
        print(f"  {emotion}: {n}条")
    # 展示样例
    print("\n样例:")
    for i in range(0, 60, 10):
        a, b, e = data[i]
        print(f"  [{e}] {a} → {b}")
    # 保存
    with open("C:/ai/data/emotion_dialogue.txt", "w", encoding="utf-8") as f:
        for a, b, e in data:
            f.write(f"{e}\t{a}\t{b}\n")
    print(f"\n已保存: C:/ai/data/emotion_dialogue.txt")
