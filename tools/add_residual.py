#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
add_residual.py — 给已训练好的"无残差" embed GLSL 直接追加残差修正层（无需重训练）

原理：
    高斯参数已经固化在 GLSL 的 kData 数组中，残差层只需
    残差 = 原图 - 高斯渲染
    无需重新训练。本脚本用 fitsplat_gl.exe 离线渲染一遍（与显示完全一致的
    alpha 合成），再与原图求差、量化成 4bit/通道 kRes 数组，插入到 GLSL 中。
    fitsplat_gl 渲染器本身已支持 kRes 解析与叠加，追加后立即可用。

用法：
    python add_residual.py input.glsl original.png [--out out.glsl] [--exe path]

示例：
    python add_residual.py fitsplat.glsl input.png --out fitsplat_res.glsl
"""
import argparse
import os
import re
import shutil
import subprocess
import sys
import tempfile

import numpy as np
from PIL import Image


# ---------- GLSL 解析 ----------

def parse_img_size(glsl):
    m = re.search(r"IMG_W = ([\d.]+)", glsl)
    n = re.search(r"IMG_H = ([\d.]+)", glsl)
    if not m or not n:
        raise SystemExit("无法从 GLSL 解析 IMG_W/IMG_H，请确认是 embed 格式")
    return int(float(m.group(1))), int(float(n.group(1)))


# ---------- 原图 ----------

def load_target(path, w, h):
    """读原图 -> (rgb 0-1 浮点 (H,W,3), 有效像素 mask (H,W))。
    mask 规则与训练器一致：alpha >= 16 视为有效，透明像素不参与残差。"""
    im = Image.open(path).convert("RGBA")
    if im.size != (w, h):
        im = im.resize((w, h), Image.BILINEAR)   # 与训练器双线性下采样保持一致
    alpha = np.asarray(im)[..., 3]
    rgb = np.asarray(im.convert("RGB"), dtype=np.float32) / 255.0
    return rgb, alpha >= 16


# ---------- 调用渲染器离线渲染 ----------

def render_via_exe(exe, glsl, tag):
    """把 glsl 临时放到 exe 同目录 input.glsl，跑 --export，读回 8bit BMP 渲染图 (0-1)。"""
    d = os.path.dirname(os.path.abspath(exe))
    dst = os.path.join(d, "input.glsl")
    bak = None
    if os.path.exists(dst):
        bak = dst + ".addres.bak"
        shutil.copy2(dst, bak)
    tmpdir = tempfile.mkdtemp(prefix="addres_")
    try:
        shutil.copy2(glsl, dst)
        outbase = os.path.join(tmpdir, tag)
        subprocess.run([exe, "--export", outbase], check=True,
                       capture_output=True, text=True)
        bmp = outbase + ".bmp"
        if not os.path.exists(bmp):
            raise SystemExit("渲染器未生成 " + bmp)
        im = Image.open(bmp).convert("RGB")
        return np.asarray(im, dtype=np.float32) / 255.0
    finally:
        if bak:
            shutil.move(bak, dst)   # 还原 exe 目录的 input.glsl
        shutil.rmtree(tmpdir, ignore_errors=True)


# ---------- 残差计算与打包 ----------

def build_kres_text(target, render, mask):
    """残差 = 原图 - 渲染；量化 v = round((t-r)*255*15/128) + 8，范围 [0,15]。
    与训练器 --residual 1 的公式逐位一致；每像素 RGB 3 个 4bit 打包进 1 个 uint。"""
    h, w = target.shape[:2]
    vv = np.full((h, w), (8 | (8 << 4) | (8 << 8)), dtype=np.uint32)   # 透明像素 v=8 -> res=0
    d = (target - render) * 255.0 * (15.0 / 128.0)
    q = np.clip(np.round(d) + 8.0, 0.0, 15.0).astype(np.uint32)
    m = mask
    vv[m] = q[m][:, 0] | (q[m][:, 1] << 4) | (q[m][:, 2] << 8)

    vals = vv.ravel()
    line = []
    chunks = []
    for i, v in enumerate(vals):
        line.append("%du" % v + ("" if i + 1 == vals.size else ","))
        if len(line) == 12 or i + 1 == vals.size:
            chunks.append("".join(line))
            line = []
    body = ",\n".join(chunks)
    return "const uint kRes[%d] = uint[%d](\n%s\n);\n\n" % (vals.size, vals.size, body)


def insert_kres(glsl, ktext):
    if "const uint kRes[" in glsl:
        raise SystemExit("GLSL 已包含 kRes，无需追加")
    m = re.search(r"(const uint kData\[\d+\] = uint\[\d+\]\(.*?\);\n)", glsl, re.S)
    if not m:
        raise SystemExit("未找到 kData 数组段，请确认是无残差的 embed GLSL")
    pos = m.end()
    return glsl[:pos] + ktext + glsl[pos:]


# ---------- 验证 ----------

def psnr(a, b):
    """a、b 均为 0-1 域浮点图像。"""
    mse = float(np.mean((a - b) ** 2))
    if mse <= 1e-9:
        return 99.99
    return -10.0 * np.log10(mse)


# ---------- main ----------

def main():
    ap = argparse.ArgumentParser(
        description="给无残差的 embed GLSL 追加残差修正层（无需重训练）")
    ap.add_argument("glsl", help="无残差的 embed GLSL（含 kData）")
    ap.add_argument("target", help="原图（PNG/JPEG/WebP，与训练输入一致）")
    ap.add_argument("--out", default=None, help="输出 GLSL（默认 <输入名>_res.glsl）")
    ap.add_argument("--exe", default=None,
                    help="fitsplat_gl.exe 路径（默认自动探测 ../fitsplat_gl/build/）")
    args = ap.parse_args()

    if not os.path.isfile(args.glsl) or not os.path.isfile(args.target):
        raise SystemExit("输入文件不存在")

    exe = args.exe
    if not exe:
        cand = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "..", "fitsplat_gl", "build", "fitsplat_gl.exe")
        if os.path.isfile(cand):
            exe = cand
    if not exe or not os.path.isfile(exe):
        raise SystemExit("找不到 fitsplat_gl.exe，请用 --exe 指定")

    out = args.out or (os.path.splitext(args.glsl)[0] + "_res.glsl")

    with open(args.glsl, "r", encoding="utf-8", errors="replace") as f:
        glsl = f.read()

    w, h = parse_img_size(glsl)
    print("画布: %dx%d" % (w, h))

    print("渲染纯高斯预览 ...")
    render = render_via_exe(exe, args.glsl, "render")
    print("渲染完成")

    target, mask = load_target(args.target, w, h)
    base_psnr = psnr(render, target)
    print("纯高斯 PSNR: %.2f dB" % base_psnr)

    ktext = build_kres_text(target, render, mask)
    out_glsl = insert_kres(glsl, ktext)
    with open(out, "w", encoding="utf-8") as f:
        f.write(out_glsl)
    print("已输出: %s（追加 kRes[%d]，%.2f MB）" % (out, w * h, len(ktext) / 1048576.0))

    print("验证残差叠加 ...")
    final = render_via_exe(exe, out, "verify")
    print("最终 PSNR: %.2f dB（提升 +%.2f dB）" % (psnr(final, target),
                                                psnr(final, target) - base_psnr))
    print("完成。将 %s 放到 fitsplat_gl.exe 同目录即可渲染。" % out)


if __name__ == "__main__":
    main()
