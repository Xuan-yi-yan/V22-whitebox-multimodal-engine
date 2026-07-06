# -*- coding: utf-8 -*-
"""
视觉 token 构造 — V22 多模态
============================
视觉 token 与文本 token 同构: [256d embedding] + [384d 属性]

  embedding: 16x16 图 → 768 RGB 数字 → Linear(768,256) → 256d
             (对应文本的 char_embed 256d, 正交初始化同哲学)
  属性:      P3Vis(图) → 256d 连续视觉特征 → 填 attr[128:384]
             (语言区 attr[0:128] 留空)

两者都是 256+384=640, 与文本 token 一起进 p7_input_proj。
"""
import numpy as np
import cv2
import torch
import torch.nn as nn

from p3vis import P3Vis


class VisualEmbed(nn.Module):
    """视觉 token 的 256d embedding: 16x16 → 768 RGB → 256d (对应 char_embed)"""

    def __init__(self, grid: int = 16, out_dim: int = 256):
        super().__init__()
        self.grid = grid
        self.in_dim = grid * grid * 3          # 768
        self.proj = nn.Linear(self.in_dim, out_dim, bias=False)
        nn.init.orthogonal_(self.proj.weight)  # 正交初始化, 同 char_embed 哲学

    def forward(self, pixels_768: torch.Tensor) -> torch.Tensor:
        """pixels_768: [..., 768] 归一化 RGB → [..., 256]"""
        return self.proj(pixels_768)


def image_to_pixels(img_bgr: np.ndarray, grid: int = 16) -> np.ndarray:
    """图 → 768 维 RGB 向量 (16x16x3, /255 归一化)"""
    small = cv2.resize(img_bgr, (grid, grid), interpolation=cv2.INTER_AREA)
    rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return rgb.flatten()  # 768


def image_to_token(img_bgr, vembed: VisualEmbed, p3vis: P3Vis, device):
    """图 → 视觉 token (与文本 token 同构)
    返回: (v_emb[256], v_attr[384])  —— 语言区 attr[0:128] 留空, 视觉区[128:384]=P3Vis
    """
    pixels = torch.tensor(image_to_pixels(img_bgr, vembed.grid), device=device)
    v_emb = vembed(pixels)                                   # [256]
    v_attr = torch.zeros(384, device=device)
    v_attr[128:384] = torch.tensor(p3vis.process_image(img_bgr),
                                   dtype=torch.float32, device=device)  # 视觉区
    return v_emb, v_attr


# ================================================================
# 独立验证: python visual_token.py
# ================================================================
if __name__ == "__main__":
    def bg(s=128): return np.full((s, s, 3), 235, np.uint8)
    def red_circle(s=128):
        img = bg(s); cv2.circle(img, (s//2, s//2), s//3, (40, 40, 220), -1); return img

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vembed = VisualEmbed().to(device)
    p3vis = P3Vis()

    img = red_circle()
    v_emb, v_attr = image_to_token(img, vembed, p3vis, device)

    print("=" * 64)
    print("视觉 token 构造验证 (红色圆)")
    print("=" * 64)
    print(f"  16x16 像素展平       : 768 维")
    print(f"  VisualEmbed 输出     : {tuple(v_emb.shape)}  (期望 [256])")
    print(f"  属性向量             : {tuple(v_attr.shape)}  (期望 [384])")
    print(f"    语言区[0:128] 激活 : {int((v_attr[:128].abs()>1e-6).sum())}  (期望 0)")
    print(f"    视觉区[128:384]激活: {int((v_attr[128:].abs()>1e-6).sum())}  (P3Vis)")
    full = torch.cat([v_emb, v_attr])
    print(f"  拼接视觉 token       : {tuple(full.shape)}  (期望 [640] = 256+384)")
    print(f"  VisualEmbed 参数量   : {sum(p.numel() for p in vembed.parameters()):,}  (768x256)")
    print("=" * 64)
    print("  ✓ 与文本 token 同构 (256 emb + 384 attr = 640), 可一起进 p7_input_proj")
    print("=" * 64)
