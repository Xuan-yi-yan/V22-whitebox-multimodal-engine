# -*- coding: utf-8 -*-
"""
P3Vis — V22 工业级视觉属性栈 (白盒规则引擎, 0 可学习参数)
===========================================================
填 P3 属性向量的视觉区间 [128:384] (256D), 与语言区间 [0:128] 同层并列。

设计对齐 P3 语言侧的分层颗粒度:
  语言:  词类 → 语义 → 句法 → 逻辑 → 语用
  视觉:  V1色彩物理 → V2几何形态 → V3纹理材质 → V4空间深度 → V5时序语义

槽位布局 (全局绝对索引):
  V1 色彩物理层  128-175 (48)  [本模块相对 0-47]    — 规则可算, 已实现
  V2 几何形态层  176-239 (64)  [相对 48-111]        — 规则可算, 已实现
  V3 纹理材质层  240-287 (48)  [相对 112-159]       — 规则可算, 已实现
  V4 空间深度层  288-335 (48)  [相对 160-207]       — 部分需模型, 能算的已填
  V5 时序语义层  336-383 (48)  [相对 208-255]       — 需模型/多帧, 预留占位

原则(架构师定): 不做归一化, 原始特征值直填, 由下游多路 loss + 控制层自适应。
"""
import numpy as np
import cv2

try:
    from skimage.feature import local_binary_pattern, graycomatrix, graycoprops
    _HAS_SKIMAGE = True
except Exception:
    _HAS_SKIMAGE = False


# 视觉区间总维度 & 各层相对偏移
VIS_DIM = 256
V1_OFF, V1_LEN = 0, 48     # 色彩物理
V2_OFF, V2_LEN = 48, 64    # 几何形态
V3_OFF, V3_LEN = 112, 48   # 纹理材质
V4_OFF, V4_LEN = 160, 48   # 空间深度
V5_OFF, V5_LEN = 208, 48   # 时序语义


class P3Vis:
    """视觉属性规则引擎: image -> 256D 视觉属性向量 (填 P3 的 [128:384])"""

    def __init__(self, kmeans_k: int = 4):
        self.kmeans_k = kmeans_k

    # ============================================================
    # 主入口
    # ============================================================
    def process_image(self, img_bgr: np.ndarray) -> np.ndarray:
        """输入 BGR 图 (cv2 格式), 返回 256D 视觉属性向量。"""
        vec = np.zeros(VIS_DIM, dtype=np.float32)
        if img_bgr is None or img_bgr.size == 0:
            return vec

        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        fg_mask, main_cnt = self._foreground(gray)

        vec[V1_OFF:V1_OFF + V1_LEN] = self._v1_color(img_bgr)
        vec[V2_OFF:V2_OFF + V2_LEN] = self._v2_geometry(gray, fg_mask, main_cnt)
        vec[V3_OFF:V3_OFF + V3_LEN] = self._v3_texture(gray)
        vec[V4_OFF:V4_OFF + V4_LEN] = self._v4_space_depth(gray, fg_mask, main_cnt)
        vec[V5_OFF:V5_OFF + V5_LEN] = self._v5_temporal_semantic(img_bgr)
        return vec

    # ============================================================
    # 离散映射: 连续特征 → 离散视觉值 (供 P3-L cross_vl 跨模态关联)
    # 双流互补: process_image 连续256维进 ABC(精度);
    #           to_discrete 离散值进 P3-L(关联/看图说话)。
    # ============================================================
    def to_discrete(self, img_bgr) -> dict:
        d = {}
        if img_bgr is None or img_bgr.size == 0:
            return d
        hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        sat = hsv[:, :, 1]
        mask = sat > 30
        fg = hsv[mask] if mask.any() else hsv.reshape(-1, 3)

        # --- 色相 + 冷暖 ---
        h_deg = float(np.median(fg[:, 0])) * 2 if len(fg) else 0.0  # H 0-180 → 0-360
        if h_deg < 15 or h_deg > 330:   hue = "red"
        elif h_deg < 45:                hue = "orange"
        elif h_deg < 75:                hue = "yellow"
        elif h_deg < 165:               hue = "green"
        elif h_deg < 195:               hue = "cyan"
        elif h_deg < 270:               hue = "blue"
        else:                           hue = "magenta"
        d["vis_hue"] = hue
        d["vis_warmth"] = "warm" if (h_deg < 60 or h_deg > 300) else "cool"

        # --- 亮度 / 饱和 ---
        v_mean = float(hsv[:, :, 2].mean()) / 255
        d["vis_brightness"] = "bright" if v_mean > 0.6 else ("dark" if v_mean < 0.3 else "medium")
        s_mean = float(sat.mean()) / 255
        d["vis_saturation"] = "vivid" if s_mean > 0.4 else ("dull" if s_mean < 0.15 else "moderate")

        # --- 形状 / 对称 / 位置 (前景轮廓) ---
        _, fgm = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        cnts, _ = cv2.findContours(fgm, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if cnts:
            cnt = max(cnts, key=cv2.contourArea)
            area = cv2.contourArea(cnt) + 1e-6
            peri = cv2.arcLength(cnt, True) + 1e-6
            circ = 4 * np.pi * area / peri ** 2
            x, y, bw, bh = cv2.boundingRect(cnt)
            ar = bw / max(bh, 1)
            if circ > 0.7:              d["vis_shape"] = "round"
            elif ar > 1.8 or ar < 0.55: d["vis_shape"] = "elongated"
            else:                       d["vis_shape"] = "irregular"
            d["vis_symmetry"] = "symmetric" if 0.8 < ar < 1.25 else "asymmetric"
            M = cv2.moments(cnt)
            if M['m00'] > 0:
                cx, cy = M['m10'] / M['m00'], M['m01'] / M['m00']
                h_, w_ = gray.shape
                col = ["left", "center", "right"][min(2, int(cx / w_ * 3))]
                row = ["top", "center", "bottom"][min(2, int(cy / h_ * 3))]
                d["vis_position"] = "center" if (col == "center" and row == "center") else f"{row}-{col}"

        # --- 纹理 (LBP 主峰) ---
        if _HAS_SKIMAGE:
            lbp = local_binary_pattern(cv2.resize(gray, (128, 128)), P=8, R=1, method="uniform")
            hist, _ = np.histogram(lbp, bins=10, range=(0, 10), density=True)
            peak = int(np.argmax(hist))
            d["vis_texture"] = "smooth" if peak < 3 else ("gradient" if peak < 6 else "rough")

        return d

    # ============================================================
    # 前景分离 (供 V2/V4 共用)
    # ============================================================
    def _foreground(self, gray):
        _, fg = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        cnts, _ = cv2.findContours(fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        main = max(cnts, key=cv2.contourArea) if cnts else None
        return fg, main

    # ============================================================
    # V1 色彩物理层 (48) — 主色/曝光/色温/心理/主色块
    # ============================================================
    def _v1_color(self, bgr):
        v = np.zeros(V1_LEN, dtype=np.float32)
        h, w = bgr.shape[:2]
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32)

        # 0-2 主色调 HSV 均值
        v[0:3] = hsv.reshape(-1, 3).mean(0) / np.array([180, 255, 255])
        # 3-5 主色 Lab 均值
        v[3:6] = lab.reshape(-1, 3).mean(0) / 255.0
        # 6-7 色彩方差 / 丰富度(colorfulness, Hasler-Süsstrunk)
        R, G, B = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
        rg, yb = R - G, 0.5 * (R + G) - B
        v[6] = float(np.sqrt(rg.std() ** 2 + yb.std() ** 2))
        v[7] = float(np.sqrt(rg.mean() ** 2 + yb.mean() ** 2)) + v[6]
        # 8-10 H通道直方图 偏度/峰度/熵
        hist_h = cv2.calcHist([hsv], [0], None, [18], [0, 180]).flatten()
        hist_h = hist_h / (hist_h.sum() + 1e-8)
        mu = (hist_h * np.arange(18)).sum()
        sig = np.sqrt(((np.arange(18) - mu) ** 2 * hist_h).sum()) + 1e-8
        v[8] = float(((np.arange(18) - mu) ** 3 * hist_h).sum() / sig ** 3)  # 偏度
        v[9] = float(((np.arange(18) - mu) ** 4 * hist_h).sum() / sig ** 4)  # 峰度
        v[10] = float(-(hist_h * np.log2(hist_h + 1e-8)).sum())              # 色相熵

        # === 光学与曝光 11-18 ===
        Y = 0.299 * R + 0.587 * G + 0.114 * B
        v[11] = float(Y.mean() / 255)                       # 全局亮度
        v[12] = float(Y.std() / 128)                        # 对比度
        v[13] = float((Y.max() - Y.min()) / 255)            # 动态范围
        # 曝光均匀度: 4x4 分块亮度方差的逆
        blk = cv2.resize(Y, (4, 4)).flatten()
        v[14] = float(1.0 / (1.0 + blk.std()))
        v[15] = float((Y > 250).mean())                     # 高光溢出率
        v[16] = float((Y < 5).mean())                       # 暗部死黑率
        v[17] = float(((Y > 5) & (Y < 250)).mean())         # 有效动态占比
        v[18] = float(np.median(Y) / 255)                   # 亮度中位

        # === 色温与白平衡 19-23 ===
        r_avg, g_avg, b_avg = R.mean(), G.mean(), B.mean()
        v[19] = float((r_avg - b_avg) / 255)                # 暖冷偏移(色温代理)
        v[20] = float((g_avg - 0.5 * (r_avg + b_avg)) / 255)  # 品绿偏移(tint)
        v[21] = float(r_avg / (b_avg + 1e-6))               # R/B 比(白平衡)
        v[22] = float(hsv[:, :, 1].mean() / 255)            # 全局饱和
        v[23] = float((hsv[:, :, 1] > 30).mean())           # 有效彩色占比

        # === 色彩心理 24-27 ===
        H = hsv[:, :, 0]
        warm = ((H < 30) | (H > 150)).mean()
        cool = ((H >= 60) & (H <= 150)).mean()
        v[24] = float(warm)
        v[25] = float(cool)
        v[26] = float(warm - cool)                          # 冷暖倾向
        v[27] = float(hsv[:, :, 1].std() / 128)             # 饱和活跃度

        # === K-Means Top-4 主色块 28-43 (4色 × [H,S,V,占比]) ===
        try:
            Z = hsv.reshape(-1, 3).astype(np.float32)
            if len(Z) > self.kmeans_k:
                crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
                _, labels, centers = cv2.kmeans(Z, self.kmeans_k, None, crit, 2,
                                                cv2.KMEANS_PP_CENTERS)
                counts = np.bincount(labels.flatten(), minlength=self.kmeans_k)
                order = np.argsort(-counts)
                for i, idx in enumerate(order[:4]):
                    c = centers[idx] / np.array([180, 255, 255])
                    v[28 + i * 4:28 + i * 4 + 3] = c
                    v[28 + i * 4 + 3] = counts[idx] / len(labels)
        except Exception:
            pass
        # 44-47 预留(色彩和谐度等)
        return v

    # ============================================================
    # V2 几何形态层 (64) — 包围盒/边缘/Hu矩/拓扑
    # ============================================================
    def _v2_geometry(self, gray, fg, cnt):
        v = np.zeros(V2_LEN, dtype=np.float32)
        h, w = gray.shape

        # --- 基础图元与包围盒 0-15 ---
        v[0] = float((fg > 0).mean())                       # 前景占比
        if cnt is not None and cv2.contourArea(cnt) > 5:
            area = cv2.contourArea(cnt)
            x, y, bw, bh = cv2.boundingRect(cnt)
            v[1:5] = [x / w, y / h, bw / w, bh / h]         # bbox 归一坐标
            hull = cv2.convexHull(cnt)
            harea = cv2.contourArea(hull) + 1e-6
            v[5] = float(area / harea)                       # solidity 凸包比
            v[6] = float(bw / max(bh, 1))                    # 长宽比
            rect = cv2.minAreaRect(cnt)
            (rw, rh) = rect[1]
            v[7] = float(max(rw, rh) / (min(rw, rh) + 1e-6)) # 最小外接矩形长宽比
            v[8] = float(area / (bw * bh + 1e-6))            # extent 填充度
            peri = cv2.arcLength(cnt, True) + 1e-6
            v[9] = float(4 * np.pi * area / peri ** 2)       # 圆度
            v[10] = float(peri ** 2 / (4 * np.pi * area + 1e-6))  # 紧凑度

            # --- 轮廓与边缘 16-31 ---
            edges = cv2.Canny(gray, 50, 150)
            v[16] = float((edges > 0).mean())               # 边缘密度
            gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0)
            gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1)
            ang = (np.arctan2(gy, gx) + np.pi)               # 0..2pi
            mag = np.sqrt(gx ** 2 + gy ** 2)
            hog, _ = np.histogram(ang, bins=8, range=(0, 2 * np.pi), weights=mag)
            hog = hog / (hog.sum() + 1e-8)
            v[17:25] = hog                                   # HOG 8方向
            v[25] = float(mag.mean() / 255)                  # 平均梯度强度
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
            v[26] = float(len(approx))                       # 多边形边数
            v[27] = float(cv2.isContourConvex(approx))       # 是否凸

            # --- Hu矩 + 形状描述子 32-47 ---
            hu = cv2.HuMoments(cv2.moments(cnt)).flatten()
            hu = -np.sign(hu) * np.log10(np.abs(hu) + 1e-30)  # 对数标度
            v[32:39] = hu / 30.0                              # 7个不变矩
            v[39] = float(bw / max(bh, 1))                    # 矩形度代理

            # --- 拓扑与连通性 48-63 ---
            n_labels, _, stats, _ = cv2.connectedComponentsWithStats(fg)
            v[48] = float(n_labels - 1)                       # 连通域数(去背景)
            if n_labels > 1:
                areas = stats[1:, cv2.CC_STAT_AREA]
                v[49] = float(areas.max() / (h * w))          # 最大连通域占比
            inv = cv2.bitwise_not(fg)
            n_holes, _, _, _ = cv2.connectedComponentsWithStats(inv)
            v[50] = float(max(0, n_holes - 2))                # 孔洞数(近似)
            v[51] = float((n_labels - 1) - max(0, n_holes - 2))  # 欧拉数近似
        return v

    # ============================================================
    # V3 纹理材质层 (48) — GLCM/LBP/材质/缺陷
    # ============================================================
    def _v3_texture(self, gray):
        v = np.zeros(V3_LEN, dtype=np.float32)
        g_small = cv2.resize(gray, (128, 128))

        # --- GLCM 0-15 (4指标 × 4方向) ---
        if _HAS_SKIMAGE:
            q = (g_small // 32).astype(np.uint8)  # 量化到8级
            glcm = graycomatrix(q, distances=[1],
                                angles=[0, np.pi/4, np.pi/2, 3*np.pi/4],
                                levels=8, symmetric=True, normed=True)
            for i, prop in enumerate(['contrast', 'energy', 'homogeneity', 'correlation']):
                vals = graycoprops(glcm, prop).flatten()  # 4方向
                v[i * 4:i * 4 + 4] = vals / (vals.max() + 1e-8) if prop == 'contrast' else vals

        # --- LBP 16-23 ---
        if _HAS_SKIMAGE:
            lbp = local_binary_pattern(g_small, P=8, R=1, method='uniform')
            hist, _ = np.histogram(lbp, bins=10, range=(0, 10), density=True)
            v[16] = float(np.argmax(hist)) / 10              # 主峰位置
            v[17] = float(hist.max())                        # 主峰强度
            v[18] = float(-(hist * np.log2(hist + 1e-8)).sum())  # 纹理熵(粗糙度)
            v[19:24] = hist[:5]                              # 直方图前5

        # --- 材质物理估计 24-31 (简化) ---
        v[24] = float((g_small > 240).mean())               # 高光/镜面反射代理
        v[25] = float(g_small.std() / 128)                  # 表面粗糙度代理
        v[26] = float(g_small.mean() / 255)                 # 漫反射率代理(反照率)
        lap = cv2.Laplacian(g_small, cv2.CV_32F)
        v[27] = float(lap.var() / 1000)                     # 清晰度/细节丰富度

        # --- 规律性与缺陷 32-47 ---
        f = np.fft.fftshift(np.abs(np.fft.fft2(g_small)))
        v[32] = float(f.std() / (f.mean() + 1e-6))          # 频域能量分散(周期性)
        gx = cv2.Sobel(g_small, cv2.CV_32F, 1, 0)
        gy = cv2.Sobel(g_small, cv2.CV_32F, 0, 1)
        v[33] = float(np.abs(gx).mean() / (np.abs(gy).mean() + 1e-6))  # 纹理方向性
        return v

    # ============================================================
    # V4 空间深度层 (48) — 能算的填(方位/显著性), 深度/透视留 TODO
    # ============================================================
    def _v4_space_depth(self, gray, fg, cnt):
        v = np.zeros(V4_LEN, dtype=np.float32)
        h, w = gray.shape

        # 0-2 深度粗估(TODO: 接单目深度模型; 暂用聚焦/模糊梯度做近远景代理)
        lap = cv2.Laplacian(gray, cv2.CV_32F)
        focus = np.abs(lap)
        v[0] = float(focus.mean() / 100)                    # 全局清晰度(近景代理)
        v[1] = float(focus.std() / 100)                     # 景深方差代理
        # 3-7 透视(TODO: 消失点检测, 留空)

        # 16-31 空间方位: 前景质心落在九宫格哪格 (one-hot)
        if cnt is not None and cv2.contourArea(cnt) > 5:
            M = cv2.moments(cnt)
            cx, cy = M['m10'] / (M['m00'] + 1e-6), M['m01'] / (M['m00'] + 1e-6)
            gx_, gy_ = min(2, int(cx / w * 3)), min(2, int(cy / h * 3))
            v[16 + gy_ * 3 + gx_] = 1.0                      # 九宫格 one-hot
            v[25] = float(cx / w)                            # 质心 x
            v[26] = float(cy / h)                            # 质心 y

        # 32-39 显著性(用 spectral residual, 需 opencv-contrib; 无则跳过)
        try:
            sal = cv2.saliency.StaticSaliencySpectralResidual_create()
            ok, smap = sal.computeSaliency(gray)
            if ok:
                smap = (smap * 255).astype(np.uint8)
                v[32] = float(smap.mean() / 255)             # 显著性均值
                v[33] = float(smap.max() / 255)              # 显著性峰值
                my, mx = np.unravel_index(np.argmax(smap), smap.shape)
                v[34], v[35] = float(mx / w), float(my / h)  # 显著点坐标
                v[36] = float((smap > 128).mean())           # 显著区占比
        except Exception:
            pass  # opencv-contrib 未装则 V4显著性留空
        return v

    # ============================================================
    # V5 时序/运动/高层语义 (48) — 需模型/多帧, 预留占位
    # ============================================================
    def _v5_temporal_semantic(self, bgr):
        v = np.zeros(V5_LEN, dtype=np.float32)
        # 0-15  运动光流   TODO: 需多帧输入
        # 16-31 场景分类   TODO: 需场景分类模型(室内/外, 天气)
        # 32-47 跨模态锚点 TODO: OCR/标志检测 → 视觉-文本强对齐
        # 能免费算的: 场景复杂度(边缘熵) 放 v[16]
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        v[16] = float((edges > 0).mean())                    # 场景复杂度(边缘密度)
        return v


# ================================================================
# 独立验证: python p3vis.py <图片路径>
# ================================================================
if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else "test_apple.bmp"
    img = cv2.imread(path)
    if img is None:
        print(f"[ERROR] 无法读取图片: {path}")
        sys.exit(1)

    p3v = P3Vis()
    vis = p3v.process_image(img)  # 256D 视觉属性

    # 拼成完整 384D P3 向量(语言区 0-127 此处留空, 因为输入是纯图)
    full = np.zeros(384, dtype=np.float32)
    full[128:384] = vis

    print("=" * 64)
    print(f"P3Vis 工业级视觉属性 Forward Pass  |  图: {path}  |  shape: {img.shape}")
    print("=" * 64)
    layers = [("V1 色彩物理", 0, 48), ("V2 几何形态", 48, 112),
              ("V3 纹理材质", 112, 160), ("V4 空间深度", 160, 208),
              ("V5 时序语义", 208, 256)]
    for name, s, e in layers:
        seg = vis[s:e]
        nz = int((np.abs(seg) > 1e-6).sum())
        print(f"  [{name}] 槽位 {128+s:>3}-{128+e-1:<3} | 激活 {nz:>2}/{e-s} | "
              f"范围[{seg.min():+.3f},{seg.max():+.3f}] 均值{seg.mean():+.3f}")
    total_nz = int((np.abs(vis) > 1e-6).sum())
    print("-" * 64)
    print(f"  视觉区间 [128:384] 共 256 槽, 激活 {total_nz} 个")
    print(f"  完整 P3 向量 shape = {full.shape} (语言0-127留空 + 视觉128-383)")
    print(f"  skimage(GLCM/LBP) = {'可用' if _HAS_SKIMAGE else '缺失, V3部分留空'}")
    print("=" * 64)
