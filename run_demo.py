# -*- coding: utf-8 -*-
"""
V22 一键 DEMO — 文本→视觉→声学 三模态验证
用法: python run_demo.py

首次运行会检查依赖, 缺失自动提示安装。
--sample 10 控制训练规模(小样本快速验证架构, 跑通即停)。
"""
import sys, subprocess, os

def check_deps():
    deps = {"torch": "pip install torch", "cv2": "pip install opencv-python",
            "numpy": "pip install numpy", "skimage": "pip install scikit-image"}
    missing = []
    for mod, cmd in deps.items():
        try:
            if mod == "cv2": import cv2
            elif mod == "skimage": from skimage import feature
            else: __import__(mod)
        except Exception:
            missing.append(cmd)
    if missing:
        print("[依赖缺失] 请先安装:")
        for c in missing: print(f"  {c}")
        print("安装后重跑: python run_demo.py"); sys.exit(1)

if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    check_deps()
    print("═" * 60)
    print("  V22 白盒多模态引擎 — 一键 DEMO")
    print("  文本训练 → 视觉图→文 → 声学频谱→文")
    print("═" * 60)
    subprocess.run([sys.executable, "train_v22_stage1.py", "--sample", "10"])
