"""
P3-L v2: 分组多头属性联动 — 128头, 每组专管一类属性
=====================================================
person/syntax/semantic/emotion/question/direction/basic 各独立
消除属性污染: "我"不再关联positive/question_thing等异类属性
"""
import torch, torch.nn as nn, torch.nn.functional as F
from typing import Dict, List, Tuple, Optional


class P3L_AttributeLinkage(nn.Module):
    """分组多头属性联动: 每类属性独立注意力, 128头总计"""

    def __init__(self, attr_dim=128, num_attr_values=500, enable_vision=False):
        super().__init__()
        self.attr_dim = attr_dim
        self.num_attr_values = num_attr_values
        self.enable_vision = enable_vision  # V22: True 时激活视觉组+跨模态组

        # 属性嵌入
        self.attr_embed = nn.Embedding(num_attr_values, attr_dim)

        # 分组定义: 词级(9组) + 句间短语级(5组) + 句间(5组) + 新增(3组) = 22组
        self.groups = {
            # 词级属性组 (扩展9组: 原有7组 + 细分语义2组)
            "person":      24,   # first/second/third/interrogative
            "syntax":      24,   # 主语/谓语/宾语/定语/状语/补语
            "semantic":    20,   # 人物/食物/地点/物体/知识/行为/状态 (粗粒度)
            "sem_entity":  16,   # 细分: 具体实体 (人物/食物/地点/物体/工具/自然)
            "sem_abstract":12,   # 细分: 抽象概念 (知识/行为/状态/抽象/未知/方式)
            "emotion":     16,   # polarity + emotion_cat
            "question":    16,   # question_slot
            "direction":   8,    # outward/inward/mutual/cognitive
            "basic":       16,   # basic_type + number
            # 句间短语上下文组 (5组)
            "func":        16,   # 功能承接: 主语↔主语, 谓语↔谓语, 宾语↔疑问槽
            "context":     16,   # 语境转接: 陈述→疑问, 肯定→否定, 直陈→反问
            "seq":         16,   # 顺序关系: 前句→后句承接, 因果, 条件
            "emotion_x":   8,    # 情感跨句: 积极→同向回应, 消极→安抚
            "persp":       8,    # 视角切换: USER↔SELF, 我↔你
            # 词组级属性组 (新增3组)
            "phrase_syn":  12,   # 词组句法: NP/VP/PP/subject_phrase/predicate_phrase/object_phrase
            "phrase_sem":  12,   # 词组语义: action_phrase/entity_phrase/query_phrase/answer_phrase
            "phrase_func": 8,    # 词组功能: concrete/abstract/relational/modal
            # 句子级关联组 (5组)
            "sent_func":    12,  # 句功能: 陈述↔疑问, 祈使↔回应, 疑问↔回答
            "sent_context": 12,  # 句语境: 对话轮次关系, 话题延续/转换
            "sent_seq":     12,  # 句顺序: A→B承接类型(直接回答/反问/补充)
            "sent_emotion": 8,   # 句情感: 整体情感倾向匹配
            "sent_persp":   8,   # 句视角: 句级人称一致性
            "sent_attr":    12,  # 句属性: sent_type/sent_domain/sent_mood/sent_tense
        }
        # ===== V22 视觉扩展 (enable_vision=True 时激活, V21 默认 False 零影响) =====
        if enable_vision:
            self.groups.update({
                # 视觉属性组 (与语言组平级, 各自独立注意力, 消除跨模态污染)
                "vis_color":    16,  # 色彩物理: 主色/曝光/色温/心理
                "vis_shape":    16,  # 几何形态: 轮廓/Hu矩/拓扑
                "vis_texture":  16,  # 纹理材质: GLCM/LBP/反光
                "vis_geometry": 12,  # 空间几何: bbox/尺度/填充
                "vis_space":    12,  # 空间深度: 九宫格方位/显著性
                # 跨模态组 (看图说话逻辑联动: 语言属性 <-> 视觉属性)
                "cross_vl":     32,  # 语言 <-> 视觉 双向关联
            })
        total_heads = sum(self.groups.values())  # 语言308 (+视觉104 = 412 when vision on)

        # 每组的Q/K/V投影 + 输出投影
        self.group_projs = nn.ModuleDict()
        for gname, nheads in self.groups.items():
            gdim = nheads  # head_dim=1, attn_dim=nheads
            self.group_projs[gname] = nn.ModuleDict({
                "q": nn.Linear(attr_dim, gdim, bias=False),
                "k": nn.Linear(attr_dim, gdim, bias=False),
                "v": nn.Linear(attr_dim, gdim, bias=False),
                "out": nn.Linear(gdim, attr_dim, bias=False),
            })
            # init: eye用于方阵, xavier用于非方阵
            for pname in ["q", "k", "v", "out"]:
                w = self.group_projs[gname][pname].weight
                if w.shape[0] == w.shape[1]:
                    nn.init.eye_(w)
                else:
                    nn.init.xavier_uniform_(w, gain=0.1)

        self.scale = 1.0
        self.attn_temperature = 0.67  # 手术2: T<1锐化注意力, 恢复特征区分度
        self.group_ids = {}  # 外部注入: registry.group_ids

    def sentence_linkage(self, A_attr_emb, B_attr_emb):
        """句子级关联: A句属性向量 ↔ B句属性向量, 每组独立评分
        A_attr_emb: [nA, attr_dim] A句词属性嵌入
        B_attr_emb: [nB, attr_dim] B句词属性嵌入
        返回: sent_scores[group_name] = scalar (句子对关联分)
        """
        # 句向量: 词属性均值
        A_sent = A_attr_emb.mean(dim=0, keepdim=True)  # [1, attr_dim]
        B_sent = B_attr_emb.mean(dim=0, keepdim=True)  # [1, attr_dim]

        sent_groups = ["sent_func", "sent_context", "sent_seq", "sent_emotion", "sent_persp"]
        results = {}
        for gname in sent_groups:
            if gname not in self.groups:
                continue
            nheads = self.groups[gname]
            proj = self.group_projs[gname]

            q = proj["q"](A_sent).view(1, nheads, 1).transpose(0, 1)  # [h, 1, 1]
            k = proj["k"](B_sent).view(1, nheads, 1).transpose(0, 1)
            v = proj["v"](B_sent).view(1, nheads, 1).transpose(0, 1)

            scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # [h, 1, 1]
            attn = F.softmax(scores / self.attn_temperature, dim=-1)  # T<1锐化
            attn_out = torch.matmul(attn, v)
            attn_out = attn_out.transpose(0, 1).contiguous().view(1, nheads)
            out = proj["out"](attn_out)
            out_n = F.normalize(out, dim=-1)
            B_n = F.normalize(B_sent, dim=-1)
            raw_score = (out_n * B_n).sum().item()
            # 手术3: 负数关联截断 — 上下文可以是"弱(0)"但不能是"反向排斥(负)"
            # score*0.5+0.5 将[-1,1]映射到[0,1], 保留梯度信息
            score = raw_score * 0.5 + 0.5
            results[gname] = round(score, 4)
        return results

    def cross_modal_linkage(self, lang_ids, vis_ids):
        """V22 跨模态关联: 语言属性 <-> 视觉属性 (cross_vl 组, 双向)

        这是"看图说话"逻辑联动的物理基础: 语言的"苹果"去 Query 视觉的
        "圆形+红色+光滑", 视觉也反向 Query 语言。单独一组, 既联动又不
        污染各模态内部属性。

        lang_ids: 语言属性id列表
        vis_ids:  视觉离散属性id列表 (来自 register_from_vision)
        返回: {"L2V": [nL,nV] 语言查视觉关联分, "V2L": [nV,nL] 视觉查语言}
        """
        if not self.enable_vision or "cross_vl" not in self.group_projs:
            return {"L2V": None, "V2L": None}
        if len(lang_ids) == 0 or len(vis_ids) == 0:
            return {"L2V": None, "V2L": None}

        gname = "cross_vl"
        proj = self.group_projs[gname]
        nheads = self.groups[gname]
        dev = self.attr_embed.weight.device
        L = self.attr_embed(torch.tensor(lang_ids, device=dev))  # [nL, attr_dim]
        V = self.attr_embed(torch.tensor(vis_ids, device=dev))   # [nV, attr_dim]

        def _attend(Q_emb, K_emb):
            nQ, nK = Q_emb.shape[0], K_emb.shape[0]
            q = proj["q"](Q_emb).view(nQ, nheads, 1).transpose(0, 1)  # [h,nQ,1]
            k = proj["k"](K_emb).view(nK, nheads, 1).transpose(0, 1)  # [h,nK,1]
            v = proj["v"](K_emb).view(nK, nheads, 1).transpose(0, 1)
            scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # [h,nQ,nK]
            attn = F.softmax(scores / self.attn_temperature, dim=-1)     # T<1锐化
            _ = torch.matmul(attn, v)  # attn_out (备用: 可接out投影做特征)
            return scores.mean(dim=0)  # [nQ,nK] 头平均关联分

        return {"L2V": _attend(L, V), "V2L": _attend(V, L)}

    def phrase_linkage(self, A_word_attrs, B_word_attrs, A_syntax, B_syntax):
        """跨句短语级关联: A句词→B句词, 按句法角色分组

        A_word_attrs: [nA, attr_dim] A句每个词的属性嵌入
        B_word_attrs: [nB, attr_dim] B句每个词的属性嵌入
        A_syntax: [nA] A句每个词的句法角色(0=主语,1=谓语,2=宾语...)
        B_syntax: [nB] B句每个词的句法角色
        返回: phrase_scores[group][nA, nB] 每对词组的各组分值
        """
        # 短语级KV: 同角色的词归为一个"短语" → 用均值作段向量
        def phrase_vecs(word_attrs, syntax):
            n = len(syntax)
            roles = {}
            for i in range(min(n, len(word_attrs))):
                r = syntax[i]
                roles.setdefault(r, []).append(word_attrs[i])
            # 均值
            return {r: torch.stack(vs).mean(dim=0) for r, vs in roles.items()}, roles

        A_phrases, A_role_map = phrase_vecs(A_word_attrs, A_syntax)
        B_phrases, B_role_map = phrase_vecs(B_word_attrs, B_syntax)

        # 对每个句间组计算交叉注意力分数
        cross_groups = ["func", "context", "seq", "emotion_x", "persp"]
        results = {}
        for gname in cross_groups:
            if gname not in self.groups:
                continue
            nheads = self.groups[gname]
            proj = self.group_projs[gname]

            # 取所有短语向量 → 交叉注意力
            a_keys = sorted(A_phrases.keys())
            b_keys = sorted(B_phrases.keys())
            if not a_keys or not b_keys:
                continue

            A_stack = torch.stack([A_phrases[k] for k in a_keys])  # [nP_A, attr_dim]
            B_stack = torch.stack([B_phrases[k] for k in b_keys])  # [nP_B, attr_dim]

            nPA, nPB = len(a_keys), len(b_keys)
            q = proj["q"](A_stack).view(nPA, nheads, 1).transpose(0, 1)  # [h, nPA, 1]
            k = proj["k"](B_stack).view(nPB, nheads, 1).transpose(0, 1)  # [h, nPB, 1]
            v = proj["v"](B_stack).view(nPB, nheads, 1).transpose(0, 1)

            scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
            attn = F.softmax(scores / self.attn_temperature, dim=-1)  # T<1锐化
            attn_out = torch.matmul(attn, v)
            attn_out = attn_out.transpose(0, 1).contiguous().view(nPA, nheads)
            out = proj["out"](attn_out)
            out_n = F.normalize(out, dim=-1)
            B_n = F.normalize(B_stack, dim=-1)
            phrase_scores = torch.mm(out_n, B_n.T)

            # 展开到词级: 每个词的短语关联度 = 其所属短语的关联度
            word_scores = torch.zeros(len(A_word_attrs), len(B_word_attrs),
                                      device=A_word_attrs.device)
            for ai in range(min(len(A_word_attrs), len(A_syntax))):
                ra = A_syntax[ai]
                if ra in a_keys:
                    pa = a_keys.index(ra)
                    for bi in range(min(len(B_word_attrs), len(B_syntax))):
                        rb = B_syntax[bi]
                        if rb in b_keys:
                            pb = b_keys.index(rb)
                            word_scores[ai, bi] = phrase_scores[pa, pb]
            results[gname] = word_scores

        return results

    def phrase_topk(self, A_words, B_words, A_syntax, B_syntax, topk=3):
        """返回每组top-k短语关联对"""
        A_attrs = self.attr_embed(torch.arange(min(len(A_words), self.num_attr_values),
                                                device=self.attr_embed.weight.device))
        # 简化: 直接用嵌入权重
        a_emb = self.attr_embed.weight[:min(len(A_words), self.num_attr_values)]
        b_emb = self.attr_embed.weight[:min(len(B_words), self.num_attr_values)]
        # Pad or trim
        nA, nB = len(A_words), len(B_words)
        if nA > a_emb.shape[0]: nA = a_emb.shape[0]
        if nB > b_emb.shape[0]: nB = b_emb.shape[0]

        results = {}
        cross_groups = ["func", "context", "seq", "emotion_x", "persp"]
        for gname in cross_groups:
            if gname not in self.groups:
                continue
            nheads = self.groups[gname]
            proj = self.group_projs[gname]
            a_use = a_emb[:nA]; b_use = b_emb[:nB]

            q = proj["q"](a_use).view(nA, nheads, 1).transpose(0, 1)
            k = proj["k"](b_use).view(nB, nheads, 1).transpose(0, 1)
            scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
            # 取词对分数: mean over heads
            pair_scores = scores.mean(dim=0).squeeze(-1)  # [nA, nB]
            # top-k词对
            flat = pair_scores.view(-1)
            kk = min(topk, flat.shape[0])
            topv, topi = torch.topk(flat, kk)
            pairs = []
            for idx, val in zip(topi.tolist(), topv.tolist()):
                ai, bi = idx // nB, idx % nB
                pairs.append((A_words[ai], B_words[bi], round(val, 3)))
            results[gname] = pairs
        return results

    def forward_group(self, gname: str, q_ids, kv_ids):
        """单组前向: head_dim=1, nheads=group_dim"""
        proj = self.group_projs[gname]
        nheads = self.groups[gname]
        nQ = len(q_ids)
        nKV = len(kv_ids)

        q_emb = self.attr_embed(q_ids)
        k_emb = self.attr_embed(kv_ids)
        v_emb = self.attr_embed(kv_ids)

        q = proj["q"](q_emb).view(nQ, nheads, 1).transpose(0, 1)  # [h, nQ, 1]
        k = proj["k"](k_emb).view(nKV, nheads, 1).transpose(0, 1)  # [h, nKV, 1]
        v = proj["v"](v_emb).view(nKV, nheads, 1).transpose(0, 1)  # [h, nKV, 1]

        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = F.softmax(scores / self.attn_temperature, dim=-1)  # T<1锐化
        attn_out = torch.matmul(attn, v)

        attn_out = attn_out.transpose(0, 1).contiguous().view(nQ, nheads)
        out = proj["out"](attn_out)
        out_n = F.normalize(out, dim=-1)
        kv_n = F.normalize(k_emb, dim=-1)
        return torch.mm(out_n, kv_n.T), attn

    def forward(self, query_attr_ids, key_value_ids=None):
        if key_value_ids is None:
            key_value_ids = torch.arange(self.num_attr_values, device=query_attr_ids.device)
        all_scores, all_attns = {}, {}
        for gname in self.groups:
            scores, attn = self.forward_group(gname, query_attr_ids, key_value_ids)
            all_scores[gname] = scores
            all_attns[gname] = attn
        return all_scores, all_attns

    def topk_per_group(self, query_attr_ids, key_value_ids=None, topk=5):
        if key_value_ids is None:
            key_value_ids = torch.arange(self.num_attr_values, device=query_attr_ids.device)
        results = {}
        for gname in self.groups:
            scores, _ = self.forward_group(gname, query_attr_ids, key_value_ids)
            s = scores[0]
            k = min(topk, s.shape[0])
            topv, topi = torch.topk(s, k, dim=-1)
            results[gname] = (topi.cpu(), topv.cpu())
        return results


class AttributeValueRegistry:
    """属性值注册表: 按组管理属性值, 支持分组查询"""

    def __init__(self):
        self.attr_map = {}       # (attr_type, value) → global_id
        self.id_to_attr = {}     # global_id → (attr_type, value)
        self.next_id = 0
        # 类型→组映射 (扩展: 词组级+句级+细分语义)
        self.type_to_group = {
            # 词级
            "person": "person", "syntax": "syntax", "semantic": "semantic",
            "sem_entity": "sem_entity", "sem_abstract": "sem_abstract",
            "polarity": "emotion", "emotion_cat": "emotion",
            "direction": "direction",
            "question_slot": "question",
            "basic_type": "basic", "number": "basic",
            # 词组级 (新增)
            "phrase_role": "phrase_syn", "phrase_type": "phrase_sem", "phrase_sem": "phrase_sem",
            # 句级 (新增)
            "sent_type": "sent_attr", "sent_domain": "sent_attr",
            "sent_mood": "sent_attr", "sent_tense": "sent_attr",
            # ===== V22 视觉属性映射 (离散视觉值 → 视觉组) =====
            "vis_hue": "vis_color", "vis_brightness": "vis_color",
            "vis_saturation": "vis_color", "vis_warmth": "vis_color",
            "vis_shape": "vis_shape", "vis_symmetry": "vis_shape",
            "vis_texture": "vis_texture", "vis_material": "vis_texture",
            "vis_bbox": "vis_geometry", "vis_scale": "vis_geometry",
            "vis_position": "vis_space", "vis_saliency": "vis_space",
        }
        # basic_type中以question_开头的 → 路由到question组
        self.question_basic_types = {
            "question_person", "question_thing", "question_place",
            "question_manner", "question_reason", "question_time",
            "question_subject_object", "question_time_duration",
            "question_quantity", "question_distance", "question_size_age",
            "question_manner_state",
        }
        # 每组属性ID列表
        self.group_ids = {}

    def register(self, attr_type: str, value: str) -> int:
        key = (attr_type, value)
        if key not in self.attr_map:
            gid = self.next_id
            self.attr_map[key] = gid
            self.id_to_attr[gid] = key
            self.next_id += 1
            # 路由到对应组: question_xxx basic_types → question组
            if attr_type == "basic_type" and value in self.question_basic_types:
                group = "question"
            else:
                group = self.type_to_group.get(attr_type, "basic")
            self.group_ids.setdefault(group, []).append(gid)
        return self.attr_map[key]

    def get_group_ids(self, group_name: str) -> List[int]:
        return self.group_ids.get(group_name, [])

    def get_id(self, attr_type: str, value: str) -> Optional[int]:
        return self.attr_map.get((attr_type, value))

    def register_from_packet(self, packet, level="word") -> List[int]:
        """从属性包注册属性ID。level: word(词级) phrase(词组级) sent(句级) all(全部)"""
        ids = []
        if level in ("word", "all"):
            if packet.basic_type:
                ids.append(self.register("basic_type", packet.basic_type))
            for role, _ in packet.syntax_candidates:
                ids.append(self.register("syntax", role))
            for sem, _ in packet.semantic_types:
                ids.append(self.register("semantic", sem))
                # 细分语义: concrete vs abstract
                sem_name = sem
                if sem_name in ("人物", "食物", "地点", "物体", "工具", "自然"):
                    ids.append(self.register("sem_entity", sem_name))
                elif sem_name in ("知识", "行为", "状态", "抽象", "未知", "方式"):
                    ids.append(self.register("sem_abstract", sem_name))
            if packet.polarity and packet.polarity != "neutral":
                ids.append(self.register("polarity", packet.polarity))
                ids.append(self.register("emotion_cat", packet.emotion_category))
            if packet.direction and packet.direction != "unknown":
                ids.append(self.register("direction", packet.direction))
            if packet.person and packet.person != "none":
                ids.append(self.register("person", packet.person))
            if packet.question_slot and packet.question_slot != "none":
                ids.append(self.register("question_slot", packet.question_slot))
            if packet.number:
                ids.append(self.register("number", packet.number))
        if level in ("phrase", "all"):
            if packet.phrase_role:
                ids.append(self.register("phrase_role", packet.phrase_role))
            if packet.phrase_type:
                ids.append(self.register("phrase_type", packet.phrase_type))
            if packet.phrase_sem:
                ids.append(self.register("phrase_sem", packet.phrase_sem))
        if level in ("sent", "all"):
            if packet.sent_type:
                ids.append(self.register("sent_type", packet.sent_type))
            if packet.sent_domain:
                ids.append(self.register("sent_domain", packet.sent_domain))
            if packet.sent_mood:
                ids.append(self.register("sent_mood", packet.sent_mood))
            if packet.sent_tense:
                ids.append(self.register("sent_tense", packet.sent_tense))
        return ids

    def register_from_vision(self, vis_attrs: dict) -> List[int]:
        """V22: 从 P3Vis 的离散映射注册视觉属性值 → id (供 cross_vl 跨模态关联)

        vis_attrs: {"vis_hue":"red", "vis_shape":"round", "vis_texture":"rough",
                    "vis_brightness":"bright", "vis_position":"center", ...}
        连续精度另走 attr_vec 进 ABC; 这里只把离散视觉值注册进关联层。
        """
        ids = []
        for vtype, vval in vis_attrs.items():
            if vval is None or vval == "":
                continue
            ids.append(self.register(vtype, str(vval)))
        return ids

    def __len__(self):
        return self.next_id
