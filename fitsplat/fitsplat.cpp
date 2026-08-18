// fitsplat.cpp — 把一张图片拟合成"高斯泼溅参数 + 渲染算法"（离线优化，允许高耗时）
//
// 遵循业界主流 3D Gaussian Splatting 做法：
//   1) 可微渲染：每个 splat 是不透明度 α=a*gauss（位置/尺寸/旋转参数化的 2D 高斯），
//      按固定顺序做 alpha 合成  acc += c*α*T, T *= (1-α)，输出天然 ≤1，数值稳定。
//   2) 梯度优化：L1 损失 + 解析反传（标准 alpha 合成 backward 递推）+ Adam，
//      学习率随迭代衰减，单步幅值 clamp 防发散。
//   3) 密度控制：累积位置梯度分裂"欠拟合" splat，删除低不透明度 splat。
//
// 产物是一小段 GLSL（const 数组参数 + 十几行 alpha 合成渲染），运行时实时合成图像。
//
// 用法：
//   fitsplat input.png [--splats 600] [--iters 1000] [--width 320] [--height 0] [--batch 2048]
//            [--out out.glsl] [--threads 0] [--bg 0.0] [--seed 42] [--preview 10] [--embed 1]
//
// 依赖：stb_image.h（仅读 PNG）

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstdint>
#include <vector>
#include <string>
#include <thread>
#include <atomic>
#include <chrono>
#include <random>
#include <algorithm>
#include <array>
#include <functional>   // std::function：线程池任务分派
#include <immintrin.h>   // _mm_pause：持久线程池 spin-wait

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX   // 阻止 windows.h 定义 min/max 宏，避免与 std::min/std::max 冲突
#endif
#include <windows.h>
#endif

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

// ---------- 基础工具 ----------
static inline float ClampF(float x, float lo, float hi) { return x < lo ? lo : (x > hi ? hi : x); }
static inline float Sigmoid(float x) { return 1.0f / (1.0f + expf(-x)); }
static inline float Logit(float a)   { return logf(a / (1.0f - a)); }

// ---------- 数据结构 ----------
struct Splat {
    float mx, my;   // 中心（像素坐标）
    float sx, sy;   // 主轴半径（正）
    float rot;      // 旋转角
    float r, g, b;  // 颜色
    float a;        // 基础不透明度 (0..1)
    float r2;       // 剔除半径平方 = (3*max(sx,sy))^2，加速用
    float cs, sn;   // cos(rot)/sin(rot) 缓存：渲染热路径每像素每 splat 都要用，避免重复调三角函数
};

// 每个 splat 的 9 个可优化参数：
//   0=mx 1=my 2=log(sx) 3=log(sy) 4=rot 5=r 6=g 7=b 8=logit(a)
static const float kLr[9] = { 0.010f, 0.010f, 0.004f, 0.004f, 0.010f, 0.020f, 0.020f, 0.020f, 0.030f };

// 训练 tile 尺寸：8x8 比 16x16 每像素遍历的 splat 列表更精准（16x16 平均 572 splat/像素，
// 8x8 降到 ~390），正传/反传内存流量降 ~30%（DDR4 带宽瓶颈下的提速）。输出 GLSL 仍用 16x16。
static constexpr int TS = 8;

static int   W = 0, H = 0;                 // 降采样后画布
static std::vector<float> target;          // W*H*3 RGB 目标
static std::vector<unsigned char> valid;   // W*H，1=有效像素（alpha 不透明）：mask 训练，透明区域不参与损失
static std::vector<Splat> splats;
static std::vector<std::array<float, 9>> mAdam, vAdam;
static std::vector<std::array<float, 2>> gradAcc;   // 密度控制：累积位置梯度

static inline void SplatFromParams(Splat& s, const std::array<float, 9>& p) {
    s.mx  = p[0]; s.my = p[1];
    s.sx  = expf(p[2]); s.sy = expf(p[3]);
    s.rot = p[4];
    s.r = ClampF(p[5], 0, 1); s.g = ClampF(p[6], 0, 1); s.b = ClampF(p[7], 0, 1);
    s.a = Sigmoid(p[8]);
    float m = std::max(s.sx, s.sy);
    s.r2 = (3.0f * m) * (3.0f * m);
    s.cs = cosf(s.rot); s.sn = sinf(s.rot);
}

// 单像素 alpha 合成渲染（tile 局部版；与最终 GLSL 一致；输出 clamp 到 [0,1]）
static inline void RenderPixel(const Splat* S, const uint32_t* idx, int n, float x, float y, float& r, float& g, float& b) {
    float T = 1.0f; r = g = b = 0.0f;
    for (int k = 0; k < n; ++k) {
        const Splat& s = S[idx[k]];
        float dx = x - s.mx, dy = y - s.my;
        if (dx * dx + dy * dy > s.r2) continue;          // 快速剔除
        float ex = (s.cs * dx + s.sn * dy) / s.sx;
        float ey = (-s.sn * dx + s.cs * dy) / s.sy;
        float gv = expf(-0.5f * (ex * ex + ey * ey));
        float alpha = s.a * gv;
        r += s.r * alpha * T; g += s.g * alpha * T; b += s.b * alpha * T;
        T *= 1.0f - alpha;
    }
    r = ClampF(r, 0, 1); g = ClampF(g, 0, 1); b = ClampF(b, 0, 1);
}

// ---------- 可微渲染：正传 + 反传（3DGS 标准 alpha 合成 backward 递推）----------
struct PixelFwd2 {
    std::vector<float> alpha, T, g, ex, ey;
    void resize(size_t K) { alpha.resize(K); T.resize(K); g.resize(K); ex.resize(K); ey.resize(K); }
};

// 正传（tile 局部版）：只遍历本像素所在 tile 的 splat（idx 为全局索引列表，n 为个数）
static inline void ForwardPixel2(const Splat* S, const uint32_t* idx, int n, float x, float y, PixelFwd2& fwd, float& accR, float& accG, float& accB) {
    float T = 1.0f; accR = accG = accB = 0.0f;
    fwd.resize((size_t)n);
    for (int k = 0; k < n; ++k) {
        const Splat& s = S[idx[k]];
        float dx = x - s.mx, dy = y - s.my;
        float gv = 0.0f, ex = 0.0f, ey = 0.0f;
        if (dx * dx + dy * dy <= s.r2) {
            ex = (s.cs * dx + s.sn * dy) / s.sx;
            ey = (-s.sn * dx + s.cs * dy) / s.sy;
            gv = expf(-0.5f * (ex * ex + ey * ey));
        }
        float alpha = s.a * gv;
        fwd.g[k] = gv; fwd.ex[k] = ex; fwd.ey[k] = ey;
        fwd.alpha[k] = alpha; fwd.T[k] = T;
        accR += s.r * alpha * T; accG += s.g * alpha * T; accB += s.b * alpha * T;
        T *= 1.0f - alpha;
    }
}

// 反传（tile 局部版）：对局部列表里的每个 splat 累加 9 参数梯度到全局 gacc[idx[ii]]
// touched/mark：记录本迭代被触碰的 splat（稀疏梯度：Adam 更新只处理触碰集，
// 消除每迭代 tGrads 全量清零 + 全量交叉读取两大内存带宽开销）
static inline void BackwardPixel2(const Splat* S, const uint32_t* idx, int n, const PixelFwd2& fwd,
                                  float eR, float eG, float eB,
                                  std::vector<std::array<float, 9>>& gacc,
                                  std::vector<uint32_t>& touched, std::vector<unsigned char>& mark) {
    float gradT = 0.0f;   // ∂L/∂T_{i+1}
    for (int ii = n - 1; ii >= 0; --ii) {
        size_t gi = idx[ii];                          // 全局 splat 索引
        if (!mark[gi]) { mark[gi] = 1; touched.push_back((uint32_t)gi); }   // 稀疏梯度：首次触碰才入列表
        const Splat& s = S[gi];
        float T_i = fwd.T[ii];
        float alp = fwd.alpha[ii];
        float gv  = fwd.g[ii];
        // E·c_i = Σ_c e_c * c_{i,c}（e_c 由 L1/L2 损失的 ∂L/∂acc_c 决定）
        float eC = eR * s.r + eG * s.g + eB * s.b;
        float dAlp = eC * T_i - gradT * T_i;          // ∂L/∂α_i
        float dTi  = eC * alp + gradT * (1.0f - alp); // ∂L/∂T_i
        gradT = dTi;

        std::array<float, 9>& g = gacc[gi];
        // α = a * gauss
        float dA = dAlp * gv;                         // ∂L/∂a
        float dG = dAlp * s.a;                        // ∂L/∂gauss
        g[8] += dA * s.a * (1.0f - s.a);              // ∂L/∂logit(a)
        // 颜色：∂L/∂c_c = e_c * α_i * T_i
        g[5] += eR * alp * T_i; g[6] += eG * alp * T_i; g[7] += eB * alp * T_i;

        float w = dG * gv;
        if (w == 0.0f) continue;
        float ex = fwd.ex[ii], ey = fwd.ey[ii];
        float cs = s.cs, sn = s.sn;
        // 形状参数梯度：dg/dmx = g*(ex*cs/sx - ey*sn/sy)，∂L/∂p = w * (dg/dp 去除 g)
        g[0] += w * (ex * cs / s.sx - ey * sn / s.sy);   // ∂L/∂mx
        g[1] += w * (ex * sn / s.sx + ey * cs / s.sy);   // ∂L/∂my
        g[2] += w * (ex * ex);                           // ∂L/∂log(sx)
        g[3] += w * (ey * ey);                           // ∂L/∂log(sy)
        // d(dex)/drot 的分子需要像素-中心差值 dx,dy：由 ex/ey 反解
        float dxp = cs * (ex * s.sx) - sn * (ey * s.sy);
        float dyp = sn * (ex * s.sx) + cs * (ey * s.sy);
        float u1r = -sn * dxp + cs * dyp;
        float v1r = -cs * dxp - sn * dyp;
        g[4] += w * (-ex * u1r / s.sx - ey * v1r / s.sy); // ∂L/∂rot
    }
}

// ---------- tile 分块（业界 3DGS 核心：每像素只遍历本 tile 的 splat）----------
struct TileMap {
    int TCOLS = 0, TROWS = 0;
    std::vector<uint32_t> off, cnt, idx;   // 每 tile (起始,个数) + 全局 splat 索引列表
};
// cap>0 时每 tile 索引数截断（输出 GLSL 保护用）；训练用 cap=0 不截断
static void BuildTileMap(const std::vector<Splat>& S, int ts, TileMap& tm, uint32_t cap = 0) {
    tm.TCOLS = (W + ts - 1) / ts; tm.TROWS = (H + ts - 1) / ts;
    int TCNT = tm.TCOLS * tm.TROWS;
    std::vector<std::vector<uint32_t>> lists(TCNT);
    for (size_t i = 0; i < S.size(); ++i) {
        const Splat& s = S[i];
        float r = 3.0f * std::max(s.sx, s.sy);           // 3σ 覆盖半径
        int x0 = std::max((int)std::floor((s.mx - r) / ts), 0);
        int x1 = std::min((int)std::floor((s.mx + r) / ts), tm.TCOLS - 1);
        int y0 = std::max((int)std::floor((s.my - r) / ts), 0);
        int y1 = std::min((int)std::floor((s.my + r) / ts), tm.TROWS - 1);
        for (int ty = y0; ty <= y1; ++ty)
            for (int tx = x0; tx <= x1; ++tx)
                lists[ty * tm.TCOLS + tx].push_back((uint32_t)i);
    }
    tm.off.resize(TCNT); tm.cnt.resize(TCNT); tm.idx.clear();
    for (int t = 0; t < TCNT; ++t) {
        uint32_t n = cap ? std::min((size_t)cap, lists[t].size()) : (uint32_t)lists[t].size();
        tm.off[t] = (uint32_t)tm.idx.size(); tm.cnt[t] = n;
        for (uint32_t k = 0; k < n; ++k) tm.idx.push_back(lists[t][k]);
    }
}

// ---------- 全图评估（写 BMP / 算 MSE）----------
static double Evaluate(std::vector<float>* outRGB = nullptr, const char* bmpPath = nullptr) {
    double mse = 0; long long nv = 0;
    TileMap evMap;
    BuildTileMap(splats, 16, evMap);   // tile 化：全图评估只遍历本 tile 的 splat
    std::vector<unsigned char> bmp;
    if (outRGB) outRGB->assign((size_t)W * H * 3, 0.0f);
    if (bmpPath) bmp.resize((size_t)W * H * 3);
    for (int j = 0; j < H; ++j) {
        for (int i = 0; i < W; ++i) {
            float r, g, b;
            int tile = (j / 16) * evMap.TCOLS + (i / 16);
            RenderPixel(splats.data(), evMap.idx.data() + evMap.off[tile], (int)evMap.cnt[tile], i + 0.5f, j + 0.5f, r, g, b);
            size_t k = ((size_t)j * W + i) * 3;
            if (valid[(size_t)j * W + i]) {   // PSNR 只统计有效（非透明）像素
                mse += (double)((r - target[k]) * (r - target[k]) + (g - target[k + 1]) * (g - target[k + 1]) + (b - target[k + 2]) * (b - target[k + 2]));
                ++nv;
            }
            if (outRGB) { (*outRGB)[k] = r; (*outRGB)[k + 1] = g; (*outRGB)[k + 2] = b; }
            if (bmpPath) {
                bmp[k]     = (unsigned char)(b * 255.0f);
                bmp[k + 1] = (unsigned char)(g * 255.0f);
                bmp[k + 2] = (unsigned char)(r * 255.0f);
            }
        }
    }
    if (bmpPath) {
        int rowSize = (W * 3 + 3) & ~3;
        int dataSize = rowSize * H;
        int fileSize = 54 + dataSize;
        FILE* f = nullptr;
        if (fopen_s(&f, bmpPath, "wb") == 0 && f) {
            unsigned char hdr[54] = { 0 };
            hdr[0] = 'B'; hdr[1] = 'M';
            hdr[2] = (unsigned char)(fileSize & 0xFF); hdr[3] = (unsigned char)((fileSize >> 8) & 0xFF); hdr[4] = (unsigned char)((fileSize >> 16) & 0xFF); hdr[5] = (unsigned char)((fileSize >> 24) & 0xFF);
            hdr[10] = 54;
            hdr[14] = 40;
            hdr[18] = (unsigned char)(W & 0xFF); hdr[19] = (unsigned char)((W >> 8) & 0xFF); hdr[20] = (unsigned char)((W >> 16) & 0xFF); hdr[21] = (unsigned char)((W >> 24) & 0xFF);
            hdr[22] = (unsigned char)(H & 0xFF); hdr[23] = (unsigned char)((H >> 8) & 0xFF); hdr[24] = (unsigned char)((H >> 16) & 0xFF); hdr[25] = (unsigned char)((H >> 24) & 0xFF);
            hdr[26] = 1; hdr[28] = 24;
            fwrite(hdr, 1, 54, f);
            std::vector<unsigned char> row(rowSize, 0);
            for (int j = H - 1; j >= 0; --j) {
                memcpy(row.data(), &bmp[(size_t)j * W * 3], (size_t)W * 3);
                fwrite(row.data(), 1, rowSize, f);
            }
            fclose(f);
        }
    }
    return mse / ((double)std::max(nv, 1LL) * 3.0);
}

// ---------- SSIM 损失（业界 3DGS 标配 0.8·L1 + 0.2·SSIM，5x5 窗口）----------
// 输入 25 像素的 (目标, 渲染) RGB，输出损失 = 平均(1-SSIM_c) 及每像素 ∂L/∂render
static float SsimLossGrad(const float* tgt, const float* ren, float* g) {
    const float C1 = 1e-4f, C2 = 9e-4f;   // (0.01)^2, (0.03)^2（像素范围 0..1）
    float loss = 0.0f;
    for (int c = 0; c < 3; ++c) {
        float mx = 0, my = 0;
        for (int i = 0; i < 25; ++i) { mx += tgt[i * 3 + c]; my += ren[i * 3 + c]; }
        mx /= 25.0f; my /= 25.0f;
        float sx2 = 0, sy2 = 0, sxy = 0;
        for (int i = 0; i < 25; ++i) {
            float dx = tgt[i * 3 + c] - mx, dy = ren[i * 3 + c] - my;
            sx2 += dx * dx; sy2 += dy * dy; sxy += dx * dy;
        }
        sx2 /= 25.0f; sy2 /= 25.0f; sxy /= 25.0f;   // 总体方差/协方差
        float A1 = 2.0f * mx * my + C1;
        float A2 = 2.0f * sxy + C2;
        float B1 = mx * mx + my * my + C1;
        float B2 = sx2 + sy2 + C2;
        float ssim = (A1 * A2) / (B1 * B2);
        loss += 1.0f - ssim;
        // ∂SSIM/∂A1 = A2/(B1B2)，∂SSIM/∂A2 = A1/(B1B2)，∂SSIM/∂B1 = -A1A2/(B1^2 B2)，∂SSIM/∂B2 = -A1A2/(B1 B2^2)
        float dA1 = A2 / (B1 * B2);
        float dA2 = A1 / (B1 * B2);
        float dB1 = -(A1 * A2) / (B1 * B1 * B2);
        float dB2 = -(A1 * A2) / (B1 * B2 * B2);
        for (int i = 0; i < 25; ++i) {
            float dy = ren[i * 3 + c] - my;
            // ∂SSIM/∂ren_i = dA1·(2μx/N) + dA2·(2(x_i-μx)/N) + dB1·(2μy/N) + dB2·(2(y_i-μy)/N)
            float dssim = dA1 * (2.0f * mx / 25.0f) + dA2 * (2.0f * (tgt[i * 3 + c] - mx) / 25.0f)
                        + dB1 * (2.0f * my / 25.0f) + dB2 * (2.0f * dy / 25.0f);
            g[i * 3 + c] = -dssim;   // ∂L/∂ren_c，L = 1 - SSIM
        }
    }
    for (int i = 0; i < 25 * 3; ++i) g[i] /= 3.0f;   // RGB 三通道平均
    return loss / 3.0f;
}

// 采样 nSamples 个随机 5x5 窗口的平均 SSIM（全图质量指标，用于训练前后对比）
static float EvaluateSSIM(int nSamples = 512) {
    TileMap evMap;
    BuildTileMap(splats, 16, evMap);
    std::mt19937 rng(1234);
    std::uniform_int_distribution<int> wxr(0, std::max(0, W - 5)), wyr(0, std::max(0, H - 5));
    float total = 0;
    float t25[75], r25[75], g25[75];
    for (int s = 0; s < nSamples; ++s) {
        int x0 = wxr(rng), y0 = wyr(rng);
        if (!valid[(size_t)(y0 + 2) * W + (x0 + 2)]) { --s; continue; }   // 窗口中心透明：重采样（SSIM 只统计有效区域）
        for (int q = 0; q < 25; ++q) {
            int px = x0 + q % 5, py = y0 + q / 5;
            size_t k = ((size_t)py * W + px) * 3;
            t25[q * 3] = target[k]; t25[q * 3 + 1] = target[k + 1]; t25[q * 3 + 2] = target[k + 2];
            int tile = (py / 16) * evMap.TCOLS + (px / 16);
            float r, g, b;
            RenderPixel(splats.data(), evMap.idx.data() + evMap.off[tile], (int)evMap.cnt[tile], px + 0.5f, py + 0.5f, r, g, b);
            r25[q * 3] = r; r25[q * 3 + 1] = g; r25[q * 3 + 2] = b;
        }
        total += 1.0f - SsimLossGrad(t25, r25, g25);   // 平均 SSIM
    }
    return total / nSamples;
}

// ---------- 密度控制：分裂高梯度 splat，删除低不透明度 splat ----------
static void Densify(size_t maxK, std::mt19937& rng) {
    if (splats.empty()) return;
    size_t K = splats.size();
    std::vector<size_t> byA(K), byG(K);
    for (size_t i = 0; i < K; ++i) { byA[i] = i; byG[i] = i; }
    std::sort(byA.begin(), byA.end(), [&](size_t x, size_t y) { return splats[x].a < splats[y].a; });
    std::sort(byG.begin(), byG.end(), [&](size_t x, size_t y) {
        float gx = gradAcc[x][0] + gradAcc[x][1];
        float gy = gradAcc[y][0] + gradAcc[y][1];
        return gx > gy;
    });
    std::vector<char> drop(K, 0);
    size_t nDrop = std::max<size_t>(1, K / 30);
    for (size_t i = 0; i < nDrop; ++i)
        if (splats[byA[i]].a < 0.02f) drop[byA[i]] = 1;
    size_t nSplit = std::max<size_t>(1, K / 30);
    std::uniform_real_distribution<float> ud(-1.0f, 1.0f);
    std::vector<Splat> add;
    std::vector<std::array<float, 9>> addM, addV;
    std::vector<std::array<float, 2>> addGA;
    for (size_t i = 0; i < nSplit && splats.size() + add.size() + nDrop <= maxK; ++i) {
        size_t si = byG[i];
        if (drop[si]) continue;
        for (int c = 0; c < 2; ++c) {
            Splat n = splats[si];
            float off = 0.3f * std::max(n.sx, n.sy);
            n.mx += ud(rng) * off; n.my += ud(rng) * off;
            n.sx *= 0.7f; n.sy *= 0.7f;
            n.r2 = (3.0f * std::max(n.sx, n.sy)) * (3.0f * std::max(n.sx, n.sy));
            add.push_back(n);
            addM.push_back(std::array<float, 9>{});
            addV.push_back(std::array<float, 9>{});
            addGA.push_back(std::array<float, 2>{});
        }
    }
    std::vector<Splat> nS; nS.reserve(splats.size() + add.size());
    std::vector<std::array<float, 9>> nM, nV;
    std::vector<std::array<float, 2>> nGA;
    for (size_t i = 0; i < K; ++i) if (!drop[i]) {
        nS.push_back(splats[i]); nM.push_back(mAdam[i]); nV.push_back(vAdam[i]); nGA.push_back(gradAcc[i]);
    }
    for (size_t i = 0; i < add.size(); ++i) {
        nS.push_back(add[i]); nM.push_back(addM[i]); nV.push_back(addV[i]); nGA.push_back(addGA[i]);
    }
    splats.swap(nS); mAdam.swap(nM); vAdam.swap(nV); gradAcc.swap(nGA);
}

// ---------- 主流程 ----------
static void PrintUsage() {
    printf("用法: fitsplat input.png [选项]\n");
    printf("  --splats K    初始 splat 数（默认 600）\n");
    printf("  --iters N     迭代次数（默认 1000）\n");
    printf("  --width W     拟合画布宽度，保持纵横比（默认 320）\n");
    printf("  --height H    强制拟合画布高度（默认 0=按源图比例）；与 --width 组合可精确控制画布如 1920x1080\n");
    printf("  --batch B     每迭代随机采样像素数（默认 2048）\n");
    printf("  --out FILE    输出 GLSL（默认 fitsplat.glsl）\n");
    printf("  --threads T   线程数，0=自动（默认 0，上限 8）\n");
    printf("  --ckpt FILE   每 --ckpt-every 迭代保存 checkpoint（分段训练/断点续训）\n");
    printf("  --ckpt-every N checkpoint 保存间隔（默认 2000）\n");
    printf("  --resume FILE 从 checkpoint 恢复继续训练（跳过已完成迭代）\n");
    printf("  --bg F        透明像素合成背景灰度（默认 0.0 黑）\n");
    printf("  --seed N      随机种子（默认 42）\n");
    printf("  --preview N   每 N%% 存一次预览 BMP（默认 10）\n");
    printf("  --embed 1     输出无纹理纯 GLSL（参数内嵌 const 数组）\n");
    printf("  --l1only 1    纯 L1 损失（对照实验，默认 0 = 0.8L1+0.2SSIM）\n");
    printf("  --l2 1        像素梯度用 L2(∝误差) 替代 L1 次梯度(±1)，高精度收敛更干净（默认 0）\n");
}

static double NowSec() {
    using namespace std::chrono;
    return duration<double>(steady_clock::now().time_since_epoch()).count();
}

int main(int argc, char** argv) {
#ifdef _WIN32
    // 正常优先级：用户要求 CPU 压榨（此前 BELOW_NORMAL 导致调度份额低、CPU 利用率只有 ~20%）
    SetPriorityClass(GetCurrentProcess(), NORMAL_PRIORITY_CLASS);
#endif
    if (argc < 2) { PrintUsage(); return 1; }
    std::string inPath = argv[1];
    int    K0     = 600, iters = 1000, width = 320, height = 0, batch = 2048, threads = 0, seed = 42, previewPct = 10, embed = 0, l1only = 0, l2loss = 0;
    int    tilecap = 1024;   // 输出产物每 tile 索引上限；0 = 不截断（渲染全部 splat，画质最佳）
    int    dumpTarget = 0;   // 调试：dump target 像素（float）到 target.raw
    int    residual = 0;     // embed 产物附加"残差修正层"（4bit/通道 ±64，叠加补高频细节，可再提 ~7dB）
    float  bg     = 0.0f;
    std::string outPath = "fitsplat.glsl";
    std::string ckptPath, resumePath;
    int ckptEvery = 2000;
    for (int i = 2; i + 1 < argc; i += 2) {
        std::string k = argv[i];
        auto next = [&]() { return argv[i + 1]; };
        if      (k == "--splats")   K0     = atoi(next());
        else if (k == "--iters")    iters  = atoi(next());
        else if (k == "--width")    width  = atoi(next());
        else if (k == "--height")   height = atoi(next());
        else if (k == "--batch")    batch  = atoi(next());
        else if (k == "--out")      outPath = next();
        else if (k == "--threads")  threads = atoi(next());
        else if (k == "--bg")       bg     = (float)atof(next());
        else if (k == "--seed")     seed   = atoi(next());
        else if (k == "--preview")  previewPct = atoi(next());
        else if (k == "--embed")    embed  = atoi(next());
        else if (k == "--l1only")   l1only = atoi(next());
        else if (k == "--l2")       l2loss = atoi(next());
        else if (k == "--ckpt")     ckptPath = next();      // 每 ckptEvery 迭代自动保存 checkpoint（断点续训/分段训练）
        else if (k == "--ckpt-every") ckptEvery = atoi(next());
        else if (k == "--resume")   resumePath = next();    // 从 checkpoint 恢复，跳过已完成的迭代
        else if (k == "--tilecap")  tilecap = atoi(next()); // 输出 tile 索引上限（0=不截断）
        else if (k == "--residual") residual = atoi(next()); // 1=embed 产物附带残差修正层
        else if (k == "--dump-target") dumpTarget = atoi(next());
        else { printf("未知参数: %s\n", k.c_str()); return 1; }
    }
    if (threads <= 0) threads = std::min(8, (int)std::thread::hardware_concurrency());  // 默认上限 8 线程，避免吃满所有核导致系统卡死
    if (threads <= 0) threads = 4;
    if (batch < 64)   batch = 64;
    if (K0 < 16)      K0 = 16;

    // ---------- 读图 ----------
    int iw, ih, comp;
    unsigned char* img = stbi_load(inPath.c_str(), &iw, &ih, &comp, 4);
    if (!img) { printf("无法读取图片: %s\n", inPath.c_str()); return 1; }
    printf("已读入 %s (%dx%d, %dch)\n", inPath.c_str(), iw, ih, comp);

    int Wsrc = iw, Hsrc = ih;
    W = width;
    H = height > 0 ? height : std::max(1, (int)llround((double)W * Hsrc / Wsrc));
    target.assign((size_t)W * H * 3, 0.0f);
    valid.assign((size_t)W * H, 1);   // 默认全有效（不透明图）
    if (W == Wsrc && H == Hsrc) {
        // 无缩放：直接拷贝（之前总做 0.5/0.5 双线性混合，导致 target 是模糊版，
        // 训练拟合到模糊目标 -> 渲染 PSNR 卡在 ~32dB 的根因）
        for (int j = 0; j < H; ++j)
            for (int i = 0; i < W; ++i) {
                const unsigned char* p = img + ((size_t)j * Wsrc + i) * 4;
                size_t k = ((size_t)j * W + i) * 3;
                target[k] = p[0] / 255.0f; target[k + 1] = p[1] / 255.0f; target[k + 2] = p[2] / 255.0f;
                valid[(size_t)j * W + i] = (p[3] >= 16);   // alpha 阈值：低于视为透明，mask 掉
            }
    } else {
    for (int j = 0; j < H; ++j) {
        float sy = (j + 0.5f) * Hsrc / H;
        int   y0 = (int)sy; if (y0 > Hsrc - 1) y0 = Hsrc - 1;
        int   y1 = y0 + 1 < Hsrc ? y0 + 1 : y0;
        float fy = sy - y0;
        for (int i = 0; i < W; ++i) {
            float sx = (i + 0.5f) * Wsrc / W;
            int   x0 = (int)sx; if (x0 > Wsrc - 1) x0 = Wsrc - 1;
            int   x1 = x0 + 1 < Wsrc ? x0 + 1 : x0;
            float fx = sx - x0;
            for (int c = 0; c < 3; ++c) {
                float v = 0;
                v += (1 - fx) * (1 - fy) * (img[((size_t)y0 * Wsrc + x0) * 4 + c] / 255.0f);
                v += fx * (1 - fy) * (img[((size_t)y0 * Wsrc + x1) * 4 + c] / 255.0f);
                v += (1 - fx) * fy * (img[((size_t)y1 * Wsrc + x0) * 4 + c] / 255.0f);
                v += fx * fy * (img[((size_t)y1 * Wsrc + x1) * 4 + c] / 255.0f);
                float a = (img[((size_t)y0 * Wsrc + x0) * 4 + 3] + img[((size_t)y1 * Wsrc + x0) * 4 + 3] +
                           img[((size_t)y0 * Wsrc + x1) * 4 + 3] + img[((size_t)y1 * Wsrc + x1) * 4 + 3]) / (4.0f * 255.0f);
                target[((size_t)j * W + i) * 3 + c] = v * a + bg * (1.0f - a);
            }
            valid[(size_t)j * W + i] = ((img[((size_t)y0 * Wsrc + x0) * 4 + 3] +
                                         img[((size_t)y1 * Wsrc + x0) * 4 + 3] +
                                         img[((size_t)y0 * Wsrc + x1) * 4 + 3] +
                                         img[((size_t)y1 * Wsrc + x1) * 4 + 3]) / 4 >= 16);
        }
    }
    }
    stbi_image_free(img);
    printf("拟合画布: %dx%d，初始 splat: %d，迭代: %d，线程: %d\n", W, H, K0, iters, threads);

    if (dumpTarget) {
        FILE* f = nullptr;
        if (fopen_s(&f, "target.raw", "wb") == 0 && f) {
            fwrite(target.data(), 4, target.size(), f); fclose(f);
            printf("已 dump target.raw (%dx%dx3 float)\n", W, H);
        }
    }

    // ---------- 初始化：网格铺开 ----------
    std::mt19937 rng((unsigned)seed);
    int n = (int)ceil(sqrt((double)K0));
    float cellX = (float)W / n, cellY = (float)H / n;
    splats.clear(); mAdam.clear(); vAdam.clear(); gradAcc.clear();
    for (int j = 0; j < n; ++j) {
        for (int i = 0; i < n; ++i) {
            Splat s;
            s.mx = (i + 0.5f) * cellX; s.my = (j + 0.5f) * cellY;
            s.sx = 1.1f * cellX;       s.sy = 1.1f * cellY;
            s.rot = 0.0f;
            size_t k = ((size_t)(int)s.my * W + (int)s.mx) * 3;
            k = std::min(k, target.size() - 3);
            s.r = ClampF(target[k], 0, 1); s.g = ClampF(target[k + 1], 0, 1); s.b = ClampF(target[k + 2], 0, 1);
            s.a = valid[(size_t)(int)s.my * W + (int)s.mx] ? 0.5f : 0.01f;   // 透明区域 splat 初始近透明，密度控制会快速删除
            s.r2 = (3.0f * std::max(s.sx, s.sy)) * (3.0f * std::max(s.sx, s.sy));
            splats.push_back(s);
            mAdam.push_back(std::array<float, 9>{});
            vAdam.push_back(std::array<float, 9>{});
            gradAcc.push_back(std::array<float, 2>{});
        }
    }

    auto splatToParams = [](const Splat& s) {
        std::array<float, 9> p;
        p[0] = s.mx; p[1] = s.my; p[2] = logf(s.sx); p[3] = logf(s.sy); p[4] = s.rot;
        p[5] = s.r;  p[6] = s.g;  p[7] = s.b;        p[8] = Logit(s.a);
        return p;
    };

    // checkpoint 保存/恢复：写/读 9 参数 + Adam m/v + 密度累积 + 迭代数（二进制）
    int startIt = 0;
    auto saveCkpt = [&](const std::string& path, int it) {
        FILE* f = nullptr; fopen_s(&f, path.c_str(), "wb");
        if (!f) { printf("\n[checkpoint] 无法写入: %s\n", path.c_str()); return; }
        uint32_t magic = 0x46535350u;   // "FSSP"
        int ns = (int)splats.size();
        fwrite(&magic, 4, 1, f); fwrite(&it, 4, 1, f); fwrite(&ns, 4, 1, f);
        for (int i = 0; i < ns; ++i) {
            std::array<float, 9> p = splatToParams(splats[i]);
            fwrite(p.data(), 4, 9, f);
            fwrite(mAdam[i].data(), 4, 9, f);
            fwrite(vAdam[i].data(), 4, 9, f);
            fwrite(gradAcc[i].data(), 4, 2, f);
        }
        fclose(f);
        printf("\n[checkpoint] it=%d splats=%d -> %s\n", it, ns, path.c_str());
    };
    if (!resumePath.empty()) {
        FILE* f = nullptr; fopen_s(&f, resumePath.c_str(), "rb");
        if (!f) { printf("无法打开 checkpoint: %s\n", resumePath.c_str()); return 1; }
        uint32_t magic = 0; int ckIt = 0, ns = 0;
        size_t got = fread(&magic, 4, 1, f);
        if (got == 1 && magic == 0x46535350u && fread(&ckIt, 4, 1, f) == 1 && fread(&ns, 4, 1, f) == 1 && ns > 0) {
            splats.resize(ns); mAdam.resize(ns); vAdam.resize(ns); gradAcc.resize(ns);
            for (int i = 0; i < ns; ++i) {
                std::array<float, 9> p{};
                fread(p.data(), 4, 9, f);
                SplatFromParams(splats[i], p);
                fread(mAdam[i].data(), 4, 9, f);
                fread(vAdam[i].data(), 4, 9, f);
                fread(gradAcc[i].data(), 4, 2, f);
            }
            startIt = ckIt;
            printf("恢复 checkpoint: %s (it=%d, splats=%d)\n", resumePath.c_str(), ckIt, ns);
        } else {
            printf("checkpoint 格式错误: %s\n", resumePath.c_str()); fclose(f); return 1;
        }
        fclose(f);
    }

    // ---------- 训练 ----------
    double t0 = NowSec();
    // 单元 = 训练 tile（8x8，见文件级常量 TS）：tile 越小每像素遍历的 splat 列表越精准，
    // 每迭代选 nTile 个 tile，tile 内 64 像素共享同一 splat 列表全部渲染，
    // 损失 = 0.8·L1 + 0.2·SSIM（tile 内 5x5 滑窗）。
    // --l1only 1 时退化为纯 L1（对照实验用）
    int nTile = std::max(1, batch / (TS * TS));
    float wL1 = l1only ? 1.0f : 0.8f, wSsim = l1only ? 0.0f : 0.2f;
    std::vector<int> tileX(nTile), tileY(nTile);
    std::uniform_int_distribution<int> txr(0, (W + TS - 1) / TS - 1), tyr(0, (H + TS - 1) / TS - 1);
    std::vector<std::vector<std::array<float, 9>>> tGrads((size_t)threads + 1);   // 最后 1 槽给主线程
    for (auto& tg : tGrads) tg.assign(splats.size(), std::array<float, 9>{});   // 稀疏梯度：初始清零一次，之后只清 touched
    std::vector<std::vector<PixelFwd2>> fwdTiles((size_t)threads + 1);   // 每线程 tile 内像素的 fwd
    std::vector<std::vector<float>> eSsim((size_t)threads + 1);          // 每线程 tile 内 SSIM 像素梯度累积
    // 稀疏梯度：每线程本迭代触碰的 splat 列表（去重标记）+ 触碰标记位图
    std::vector<std::vector<uint32_t>> touched((size_t)threads + 1);
    std::vector<std::vector<unsigned char>> tmark((size_t)threads + 1);
    for (auto& tk : touched) tk.reserve(1 << 16);
    for (auto& mk : tmark) mk.assign(splats.size(), 0);
    for (auto& ft : fwdTiles) {
        ft.resize(TS * TS);
        for (auto& f : ft) f.resize(2048);   // 预留按 tile 索引 cap(1024) 双份即可；原 K0*2 会预留几十 GB 虚拟地址
    }
    for (auto& es : eSsim) es.assign(TS * TS * 3, 0.0f);

    // ---------- 持久线程池（消除每迭代 2x16 次线程创建/join 开销，占迭代时间 ~30%）----------
    // 两阶段复用：phase 1 = tile 正传反传，phase 2 = Adam 稀疏更新。
    // 关键设计：worker 静态 interleaved 分块（b = w, w+NW, ...），不用原子抢任务计数器——
    // 17 线程 fetch_add 同一原子 + phase/done 缓存行乒乓实测让 16 线程比单线程还慢 2 倍。
    // phase/done 用 alignas(64) 分开缓存行避免 false sharing；主线程只发布阶段并等待。
    const int NW = threads;
    struct alignas(64) PoolCtrl { std::atomic<int> phase{0}; };   // 0=空闲 1=tile 2=adam -1=退出
    PoolCtrl pool;
    std::atomic<int> perDone[64] = {};   // per-worker 完成标志：16 worker 各自写独立原子，避免单一 done 计数器的缓存行竞争
    int poolNTask = 0;
    std::function<void(int, int)> poolTask;   // (任务 id, worker id)：每迭代按阶段重新赋值
    std::vector<std::thread> workers;
    for (int w = 0; w < NW; ++w) {
        workers.emplace_back([&, w]() {
            for (;;) {
                int ph = pool.phase.load(std::memory_order_acquire);
                while (ph == 0) {
                    if (pool.phase.load(std::memory_order_acquire) == -1) return;
                    _mm_pause();
                    ph = pool.phase.load(std::memory_order_acquire);
                }
                if (ph == -1) return;
                // 静态 interleaved 分块：负载均衡（tile 大小不均时交错分布最稳），零原子竞争
                for (int b = w; b < poolNTask; b += NW) poolTask(b, w);
                perDone[w].store(1, std::memory_order_release);   // 本 worker 完成当前阶段
                while (pool.phase.load(std::memory_order_acquire) == ph) _mm_pause();   // 等阶段切换
            }
        });
    }

    // 每线程损失/有效像素数汇总（worker + 主线程，大小 threads+1）
    std::vector<float> threadL1((size_t)threads + 1, 0.0f), threadL2((size_t)threads + 1, 0.0f);
    std::vector<int> threadNV((size_t)threads + 1, 0);
    // 等待所有 worker 完成当前阶段并重置标志
    auto poolWaitAll = [&]() {
        for (int w = 0; w < NW; ++w) while (perDone[w].load(std::memory_order_acquire) == 0) _mm_pause();
        for (int w = 0; w < NW; ++w) perDone[w].store(0, std::memory_order_release);
    };
    float lossEma = 0, mseEma = 0;
    int   prevPct = -1, prevBmpPct = -1;
    size_t maxK = (size_t)(K0 * 3.0);   // 密度控制上限放宽到 3x（接近无损复刻需要更大 splat 规模）

    // tile 分块（训练核心加速：每像素只遍历本 tile 的 splat，剔除 99% 无关 splat）
    TileMap tMap;
    BuildTileMap(splats, TS, tMap);

    // 分段 profile：定位迭代时间分布（tile 正传反传 / Adam / 其他）
    double prof_tile = 0, prof_adam = 0, prof_other = 0;

    for (int it = startIt + 1; it <= iters; ++it) {
        double tIt0 = NowSec();
        for (int b = 0; b < nTile; ++b) { tileX[b] = txr(rng); tileY[b] = tyr(rng); }

        // 每迭代参与线程 = NW worker + 主线程（主线程用 tid=threads 槽）。
        // tGrads 不再每迭代全量清零（230MB/迭代 写带宽）；梯度只写 touched 的 splat，
        // 由 Adam 更新读后清零（稀疏梯度）
        const int nw = threads + 1;   // 全部潜在参与线程数
        std::fill(threadL1.begin(), threadL1.end(), 0.0f);
        std::fill(threadL2.begin(), threadL2.end(), 0.0f);
        std::fill(threadNV.begin(), threadNV.end(), 0);
        float totalL1 = 0, totalL2 = 0; int totalNV = 0;

        // ---- phase 1：tile 正传/反传（持久 worker 静态分块，主线程只发布并等待）----
        poolNTask = nTile;
        poolTask = [&](int b, int tid) {
            auto& gacc = tGrads[(size_t)tid];
            auto& ft = fwdTiles[(size_t)tid];
            auto& es = eSsim[(size_t)tid];
            auto& tk = touched[(size_t)tid];
            auto& mk = tmark[(size_t)tid];
            float l1 = 0, l2 = 0; int nv = 0;
            int tx = tileX[b], ty = tileY[b];
            int x0 = tx * TS, y0 = ty * TS;
            int tile = ty * tMap.TCOLS + tx;
            const uint32_t* idx = tMap.idx.data() + tMap.off[tile];
            int n = (int)tMap.cnt[tile];
            float r[TS*TS], g[TS*TS], bv[TS*TS];
            std::fill(es.begin(), es.end(), 0.0f);   // 每 tile 清零 SSIM 梯度累积
            // ---- 正传：tile 内 TS*TS 像素全部渲染（共享 splat 列表，只越界像素跳过）----
            for (int q = 0; q < TS*TS; ++q) {
                int pxi = x0 + q % TS, pyi = y0 + q / TS;
                if (pxi >= W || pyi >= H) { r[q] = g[q] = bv[q] = 0.0f; continue; }
                if (!valid[(size_t)pyi * W + pxi]) {
                    // 透明像素（mask 掉）：渲染值=目标值，误差恒 0，不污染 SSIM 窗口也不反传
                    size_t k3 = ((size_t)pyi * W + pxi) * 3;
                    r[q] = target[k3]; g[q] = target[k3 + 1]; bv[q] = target[k3 + 2];
                    continue;
                }
                ForwardPixel2(splats.data(), idx, n, pxi + 0.5f, pyi + 0.5f, ft[q], r[q], g[q], bv[q]);
                r[q] = ClampF(r[q], 0, 1); g[q] = ClampF(g[q], 0, 1); bv[q] = ClampF(bv[q], 0, 1);
                ++nv;
            }
            // ---- SSIM：tile 内滑窗（隔 2px 采样，5x5 窗口完全在 tile 内），梯度累积到 es ----
            if (!l1only) {
                float t25[75], r25[75], g25[75];
                for (int wy = 2; wy <= TS - 3; wy += 2) {
                    for (int wx = 2; wx <= TS - 3; wx += 2) {
                        // 窗口越界（tile 部分超出画布）则跳过
                        if (x0 + wx - 2 >= W || y0 + wy - 2 >= H || x0 + wx + 2 >= W || y0 + wy + 2 >= H) continue;
                        int cq = wy * TS + wx;   // 窗口中心像素
                        if (!valid[(size_t)(y0 + cq / TS) * W + (x0 + cq % TS)]) continue;   // 中心透明：整个窗口跳过
                        for (int q25 = 0; q25 < 25; ++q25) {
                            int qq = (wy + q25 / 5 - 2) * TS + (wx + q25 % 5 - 2);
                            int kt = ((size_t)(y0 + qq / TS) * W + (x0 + qq % TS)) * 3;
                            r25[q25 * 3] = r[qq]; r25[q25 * 3 + 1] = g[qq]; r25[q25 * 3 + 2] = bv[qq];
                            t25[q25 * 3] = target[kt]; t25[q25 * 3 + 1] = target[kt + 1]; t25[q25 * 3 + 2] = target[kt + 2];
                        }
                        l1 += SsimLossGrad(t25, r25, g25);   // SSIM 损失计入进度显示
                        for (int q25 = 0; q25 < 25; ++q25) {
                            int qq = (wy + q25 / 5 - 2) * TS + (wx + q25 % 5 - 2);
                            es[qq * 3]     += wSsim * g25[q25 * 3];
                            es[qq * 3 + 1] += wSsim * g25[q25 * 3 + 1];
                            es[qq * 3 + 2] += wSsim * g25[q25 * 3 + 2];
                        }
                    }
                }
            }
            // ---- 反传：每像素 e = 像素误差梯度(L1 次梯度 或 L2 线性) + es(SSIM 累积) ----
            for (int q = 0; q < TS*TS; ++q) {
                int pxi = x0 + q % TS, pyi = y0 + q / TS;
                if (pxi >= W || pyi >= H) continue;
                if (!valid[(size_t)pyi * W + pxi]) continue;   // 透明像素不参与损失/反传
                size_t k = ((size_t)pyi * W + pxi) * 3;
                float dR = r[q] - target[k], dG = g[q] - target[k + 1], dB = bv[q] - target[k + 2];
                float eR, eG, eB;
                if (l2loss) {
                    // L2：梯度 ∝ 误差，接近收敛时比 L1(sign 恒定) 收敛更干净（Adam 尺度自适应）
                    eR = wL1 * 2.0f * dR + es[q * 3];
                    eG = wL1 * 2.0f * dG + es[q * 3 + 1];
                    eB = wL1 * 2.0f * dB + es[q * 3 + 2];
                } else {
                    eR = wL1 * copysignf(1.0f, dR) + es[q * 3];
                    eG = wL1 * copysignf(1.0f, dG) + es[q * 3 + 1];
                    eB = wL1 * copysignf(1.0f, dB) + es[q * 3 + 2];
                }
                l1 += fabsf(dR) + fabsf(dG) + fabsf(dB);
                l2 += dR * dR + dG * dG + dB * dB;
                BackwardPixel2(splats.data(), idx, n, ft[q], eR, eG, eB, gacc, tk, mk);
            }
            threadL1[(size_t)tid] = l1; threadL2[(size_t)tid] = l2; threadNV[(size_t)tid] = nv;
        };
        pool.phase.store(1, std::memory_order_release);   // 唤醒 worker 开始 tile 阶段
        poolWaitAll();   // 主线程纯等待，不抢任务（避免竞争）
        double tTileEnd = NowSec(); prof_tile += tTileEnd - tIt0;
        for (int t = 0; t < nw; ++t) { totalL1 += threadL1[t]; totalL2 += threadL2[t]; totalNV += threadNV[t]; }
        if (totalNV <= 0) totalNV = 1;   // 整批 tile 全透明时防除零
        totalL1 /= (float)totalNV; totalL2 /= (float)totalNV;

        // 学习率随迭代衰减（位置/尺寸/旋转线性衰减到 30%）
        float lrScale = 0.3f + 0.7f * (1.0f - (float)it / iters);
        // Adam bias correction 的 powf 每迭代只算一次（原在内层循环 9 参数 × 每 splat，O(N) 次 powf）
        float bc1 = 1.0f - powf(0.9f, (float)it);
        float bc2 = 1.0f - powf(0.999f, (float)it);
        // ---- phase 2：Adam 稀疏更新（只更新本迭代被触碰的 splat）----
        // 随机采 tile 下每迭代触碰的 splat 通常 <5%，稀疏后读写流量比全量版降 ~25 倍
        {
            const size_t NA = splats.size();
            // 跨线程去重收集触碰并集（touched[tid] 线程内已去重，跨线程可能重复）
            std::vector<uint32_t> allT;
            allT.reserve(1 << 17);
            std::vector<unsigned char> seen(NA, 0);
            for (int t = 0; t < nw; ++t)
                for (uint32_t i : touched[(size_t)t])
                    if (!seen[i]) { seen[i] = 1; allT.push_back(i); }
            // 按 splat 索引升序排序：tGrads[tt] 各槽（每槽 9MB，按 splat 索引连续存储）改随机访问为
            // 顺序访问，消除 TLB/缓存未命中——实测无序时 16 线程 adam 比单线程还慢 5 倍
            std::sort(allT.begin(), allT.end());
            // 注意：tmark 不在 adam 前清零——adam 用 tmark[tt][i] 判断"槽 tt 是否触碰过 splat i"，
            // 只读有梯度的槽（平均 2~4 个而非全部 17 个），清零延后到本阶段结束（见 touched.clear 处）
            const size_t NT = allT.size();
            if (NT > 0) {
                const int nChunk = NW;   // 每个 worker 一个静态块
                size_t chunkA = (NT + nChunk - 1) / nChunk;
                poolNTask = nChunk;
                poolTask = [&](int b, int tid) {
                    size_t i0 = (size_t)b * chunkA, i1 = std::min(NT, i0 + chunkA);
                    for (size_t ii = i0; ii < i1; ++ii) {
                        size_t i = allT[ii];
                        std::array<float, 9> g = {};
                        for (int tt = 0; tt < nw; ++tt) {   // 只累加实际触碰过该 splat 的槽（tmark 标记）
                            if (!tmark[(size_t)tt][i]) continue;
                            for (int c = 0; c < 9; ++c) g[c] += tGrads[(size_t)tt][i][c];
                            tGrads[(size_t)tt][i] = std::array<float, 9>{};   // 读后清零（替代每迭代全量 assign）
                        }
                        for (int c = 0; c < 9; ++c) g[c] /= (float)totalNV;   // 只按有效（非透明）像素平均
                        // 累积位置梯度（密度控制用）
                        gradAcc[i][0] += fabsf(g[0]); gradAcc[i][1] += fabsf(g[1]);
                        auto p = splatToParams(splats[i]);
                        for (int c = 0; c < 9; ++c) {
                            mAdam[i][c] = 0.9f * mAdam[i][c] + 0.1f * g[c];
                            vAdam[i][c] = 0.999f * vAdam[i][c] + 0.001f * g[c] * g[c];
                            float mh = mAdam[i][c] / bc1;
                            float vh = vAdam[i][c] / bc2;
                            float step = mh / (sqrtf(vh) + 1e-3f);
                            step = ClampF(step, -1.0f, 1.0f);
                            p[c] -= kLr[c] * lrScale * step;
                        }
                        Splat ns; SplatFromParams(ns, p);
                        ns.r = ClampF(ns.r, 0, 1); ns.g = ClampF(ns.g, 0, 1); ns.b = ClampF(ns.b, 0, 1);
                        ns.mx = ClampF(ns.mx, 0, (float)W); ns.my = ClampF(ns.my, 0, (float)H);
                        ns.sx = ClampF(ns.sx, 0.05f, (float)W); ns.sy = ClampF(ns.sy, 0.05f, (float)H);
                        splats[i] = ns;
                    }
                };
                pool.phase.store(2, std::memory_order_release);
                poolWaitAll();   // 等所有 worker 完成 Adam 稀疏更新
                double tP2 = NowSec(); prof_adam += tP2 - tTileEnd;
            }
            pool.phase.store(0, std::memory_order_release);   // worker 回空闲，等下阶段
            // 清触碰标记（供下迭代 tile 阶段重新收集）；tGrads 已在上方读后清零
            for (uint32_t i : allT)
                for (int tt = 0; tt < nw; ++tt) tmark[(size_t)tt][i] = 0;
            for (auto& tk : touched) tk.clear();   // 本迭代触碰集已消费，清空供下迭代重新收集
        }
        double tAdamEnd = NowSec();   // prof_adam 已在 if (NT>0) 内累计
        prof_other += NowSec() - tAdamEnd;

        lossEma = (it == 1) ? totalL1 : (0.95f * lossEma + 0.05f * totalL1);
        mseEma  = (it == 1) ? totalL2 : (0.95f * mseEma  + 0.05f * totalL2);

        // 进度反馈
        int pct = (int)(100.0 * it / iters);
        if (pct != prevPct && pct % 2 == 0) {
            prevPct = pct;
            double dt = NowSec() - t0;
            double eta = dt * (iters - it) / std::max(1, it);
            double psnr = 10.0 * log10(1.0 / std::max(mseEma, 1e-6f));
            int barN = 20, fill = (int)(barN * pct / 100.0);
            printf("\r[%s] %3d%% iter=%d/%d  l1=%.4f psnr=%.1fdb  splats=%d  dt=%.0fs  eta=%.0fs",
                   std::string(fill, '=').c_str(), pct, it, iters, lossEma, psnr, (int)splats.size(), dt, eta);
            if (it >= 50) printf("  [tile %.2fms adam %.2fms other %.2fms]", prof_tile / it * 1000, prof_adam / it * 1000, prof_other / it * 1000);
            fflush(stdout);
        }
        if (previewPct > 0 && pct / previewPct != prevBmpPct && pct >= previewPct) {
            prevBmpPct = pct / previewPct;
            Evaluate(nullptr, "fitsplat_progress.bmp");
        }

        // 密度控制
        if (it % 100 == 0) {
            Densify(maxK, rng);
            for (auto& ga : gradAcc) { ga[0] = 0; ga[1] = 0; }
            for (auto& tg : tGrads) tg.assign(splats.size(), std::array<float, 9>{});
            for (auto& tk : touched) tk.clear();             // splat 数量/顺序变化，重建稀疏结构
            for (auto& mk : tmark) mk.assign(splats.size(), 0);
            BuildTileMap(splats, TS, tMap);   // splat 位置/数量变化，重建 tile 分块
        }

        // 分段训练：每 ckptEvery 迭代存 checkpoint（可 --resume 继续，用时间换空间）
        if (!ckptPath.empty() && it % ckptEvery == 0) saveCkpt(ckptPath, it);
    }
    if (!ckptPath.empty()) saveCkpt(ckptPath, iters);   // 训练结束也保存一份
    // 通知持久线程池 worker 退出并回收（否则程序退出时 vector 析构 std::terminate 崩溃）
    pool.phase.store(-1, std::memory_order_release);
    for (auto& th : workers) th.join();
    printf("\n");

    // ---------- 最终评估 ----------
    double mse = Evaluate(nullptr, "fitsplat_preview.bmp");
    double psnr = 10.0 * log10(1.0 / std::max(mse, 1e-6));
    float ssim = EvaluateSSIM();
    printf("训练完成: 最终 splat=%d, 全图 MSE=%.5f, PSNR=%.1f dB, SSIM=%.4f\n",
           (int)splats.size(), mse, psnr, ssim);

    // ---------- 输出：3DGS 标准前向渲染（参数纹理 + tile 分块剔除）----------
    //
    // 业界 3DGS 渲染不会把数千 splat 写成 const 数组逐像素全量循环（核显动态索引
    // 灾难），而是：
    //   1) 参数存纹理：每行一个 splat，texelFetch 硬件随机访问（流畅无压力）
    //   2) tile 分块：画布切成 16x16 tile，CPU 预计算每个 tile 覆盖哪些 splat，
    //      每像素只遍历本 tile 的索引列表（通常十几个），剔除 99% 无关 splat
    {
        const int TS_OUT = 16;                               // 输出 GLSL tile 尺寸（与训练 TS=8 独立）
        // ---- 构建 tile -> splat 索引列表（复用训练用 BuildTileMap，截断保护性能）----
        const uint32_t MAX_PER_TILE = (uint32_t)tilecap;    // 大 splat 规模下必须放宽（256 会让 65% tile 截断失真）；0=不截断
        TileMap om;
        BuildTileMap(splats, TS_OUT, om, MAX_PER_TILE);
        int TCOLS = om.TCOLS, TROWS = om.TROWS, TCNT = TCOLS * TROWS;
        std::vector<uint32_t> offsets = om.off, counts = om.cnt, indices = om.idx;
        size_t totalIdx = om.idx.size(), worst = 0;
        for (uint32_t c : counts) worst = std::max<size_t>(worst, c);
        printf("tile 分块: %dx%d tile=%d, 每 tile 平均 %.1f 个 splat, 最多 %d\n",
               TCOLS, TROWS, TCNT, (double)totalIdx / TCNT, (int)worst);

        // ---- 写参数纹理 + tile 数据（与 outPath 同目录，基名相同）----
        std::string base = outPath;
        size_t dot = base.find_last_of('.');
        if (dot != std::string::npos) base = base.substr(0, dot);
        auto writeBin = [&](const std::string& path, const void* p, size_t bytes) {
            FILE* f = nullptr;
            if (fopen_s(&f, path.c_str(), "wb") == 0 && f) { fwrite(p, 1, bytes, f); fclose(f); }
            else printf("无法写入 %s\n", path.c_str());
        };
        std::vector<float> A(splats.size() * 4), B(splats.size() * 4), C(splats.size() * 4);
        for (size_t i = 0; i < splats.size(); ++i) {
            A[i*4+0]=splats[i].mx; A[i*4+1]=splats[i].my; A[i*4+2]=splats[i].sx; A[i*4+3]=splats[i].sy;
            B[i*4+0]=splats[i].r;  B[i*4+1]=splats[i].g;  B[i*4+2]=splats[i].b;  B[i*4+3]=splats[i].a;
            C[i*4+0]=cosf(splats[i].rot); C[i*4+1]=sinf(splats[i].rot); C[i*4+2]=0; C[i*4+3]=0;
        }
        writeBin(base + "_splatA.raw", A.data(), A.size()*sizeof(float));
        writeBin(base + "_splatB.raw", B.data(), B.size()*sizeof(float));
        writeBin(base + "_splatC.raw", C.data(), C.size()*sizeof(float));

        std::vector<uint32_t> off(TCNT*2);               // (offset,count) 对
        for (int t = 0; t < TCNT; ++t) { off[t*2]=offsets[t]; off[t*2+1]=counts[t]; }
        writeBin(base + "_tileOff.raw", off.data(), off.size()*sizeof(uint32_t));
        writeBin(base + "_tileIdx.raw", indices.data(), indices.size()*sizeof(uint32_t));

        // ---- 写 GLSL（--embed 时参数内嵌 const 数组，无纹理纯 GLSL）----
        FILE* f = nullptr;
        if (fopen_s(&f, outPath.c_str(), "w") == 0 && f) {
            fprintf(f, "#version 330 core\nout vec4 FragColor;\n\n");
            fprintf(f, "uniform vec2  iResolution;\nuniform float iTime;\n\n");
            fprintf(f, "// 由 fitsplat 生成：%d 个高斯泼溅（alpha 合成 + tile 分块剔除）拟合图片\n", (int)splats.size());
            if (embed) {
                // 位置量化系数按画布自适应：16bit 需覆盖 [0,W]/[0,H]。
                // 旧版固定 128/64（mx 上限 512px、my 上限 1024px），大画布（如 800x1165）会
                // 16bit 溢出回绕 -> 右侧/底部内容叠到左侧/顶部。自适应后任意画布不溢出。
                const uint32_t smx = std::max(1u, 65535u / (uint32_t)std::max(1, W));
                const uint32_t smy = std::max(1u, 65535u / (uint32_t)std::max(1, H));
                fprintf(f, "// [无纹理压缩模式] 参数量化为 uint 打包：每 splat 4 个 uint（vs 12 个 float，体积降 ~10 倍）\n");
                fprintf(f, "//   u0: mx(16bit,1/%upx) | my(16bit,1/%upx)（系数按画布自适应，防止大画布溢出回绕）\n", smx, smy);
                fprintf(f, "//   u1: r,g,b,a 各 8bit\n");
                fprintf(f, "//   u2: sx,sy log2 量化各 12bit（[2^-5,2^9]，相对误差 <0.25%%）\n");
                fprintf(f, "//   u3: cos,sin 各 16bit（[0,65535] 映射 [-1,1]）\n");
                fprintf(f, "const float kMXS = %u.0; const float kMYS = %u.0;\n\n", smx, smy);
                std::string ND = std::to_string(splats.size() * 4);
                fprintf(f, "const uint kData[%s] = uint[%s](\n", ND.c_str(), ND.c_str());
                auto q12 = [](float v) {
                    float l = log2f(ClampF(v, 1e-3f, 1e3f));
                    return (uint32_t)ClampF((l + 5.0f) * (4095.0f / 14.0f), 0.0f, 4095.0f);
                };
                for (size_t i = 0; i < splats.size(); ++i) {
                    const Splat& s = splats[i];
                    // 注意：先乘后截断（(uint32_t)(v*scale)），否则小数部分先被截掉
                    uint32_t u0 = ((uint32_t)(ClampF(s.mx, 0, (float)W) * (float)smx)) |
                                  (((uint32_t)(ClampF(s.my, 0, (float)H) * (float)smy)) << 16);
                    uint32_t u1 = ((uint32_t)(ClampF(s.r, 0, 1) * 255.0f)) |
                                  ((uint32_t)(ClampF(s.g, 0, 1) * 255.0f) << 8) |
                                  ((uint32_t)(ClampF(s.b, 0, 1) * 255.0f) << 16) |
                                  ((uint32_t)(ClampF(s.a, 0, 1) * 255.0f) << 24);
                    uint32_t u2 = q12(s.sx) | (q12(s.sy) << 12);
                    uint32_t u3 = (uint32_t)ClampF((s.cs + 1.0f) * 32767.0f, 0, 65535) |
                                  ((uint32_t)ClampF((s.sn + 1.0f) * 32767.0f, 0, 65535) << 16);
                    fprintf(f, "%u,%u,%u,%u%s\n", u0, u1, u2, u3, (i + 1 < splats.size() ? "," : ""));
                }
                fprintf(f, ");\n\n");
                // ---- 残差修正层（--residual 1）：全分辨率 4bit/通道 ±64，叠加补 splat 高频细节 ----
                // 残差 = 原图 - 高斯渲染；量化 v = round((res+64)*15/128)，解码 res = v*(128/15)-64
                if (residual) {
                    std::vector<float> ren;
                    Evaluate(&ren, nullptr);   // 全图重渲染（0-1 域）
                    fprintf(f, "const uint kRes[%d] = uint[%d](\n", W * H, W * H);
                    int nCol = 0;
                    for (int j = 0; j < H; ++j) {
                        for (int i = 0; i < W; ++i) {
                            size_t p = (size_t)j * W + i;
                            uint32_t vR = 8, vG = 8, vB = 8;   // 透明像素残差 0（v=8 -> res=0）
                            if (valid[p]) {
                                auto q4 = [&](float t, float r) {
                                    return (uint32_t)ClampF(roundf((t - r) * 255.0f * (15.0f / 128.0f)) + 8.0f, 0.0f, 15.0f);
                                };
                                vR = q4(target[p * 3], ren[p * 3]);
                                vG = q4(target[p * 3 + 1], ren[p * 3 + 1]);
                                vB = q4(target[p * 3 + 2], ren[p * 3 + 2]);
                            }
                            fprintf(f, "%uu%s", vR | (vG << 4) | (vB << 8), ((size_t)j * W + i + 1 < (size_t)W * H) ? "," : "");
                            if (++nCol % 12 == 0) fprintf(f, "\n");
                        }
                    }
                    fprintf(f, ");\n\n");
                }
                std::string NT = std::to_string(TCNT);
                std::string NI = std::to_string(indices.size());
                fprintf(f, "const uvec2 kTileOff[%s] = uvec2[%s](\n", NT.c_str(), NT.c_str());
                for (int t = 0; t < TCNT; ++t)
                    fprintf(f, "uvec2(%du,%du)%s\n", off[t*2], off[t*2+1], (t+1 < TCNT ? "," : ""));
                fprintf(f, ");\n\n");
                fprintf(f, "const uint kTileIdx[%s] = uint[%s](\n", NI.c_str(), NI.c_str());
                for (size_t i = 0; i < indices.size(); ++i)
                    fprintf(f, "%du%s\n", indices[i], (i+1 < indices.size() ? "," : ""));
                fprintf(f, ");\n\n");
            } else {
                fprintf(f, "uniform sampler2D  uSplatA;   // (mx,my,sx,sy)     每行一个 splat\n");
                fprintf(f, "uniform sampler2D  uSplatB;   // (r,g,b,opacity)\n");
                fprintf(f, "uniform sampler2D  uSplatC;   // (cos,sin,0,0)\n");
                fprintf(f, "uniform usampler2D uTileOff;  // 每 tile (offset,count)\n");
                fprintf(f, "uniform usampler2D uTileIdx;  // splat 索引列表\n\n");
            }
            fprintf(f, "const float IMG_W = %d.0;\nconst float IMG_H = %d.0;\n", W, H);
            fprintf(f, "const int IMG_WI = %d;\n", W);
            fprintf(f, "const float TILE = %d.0;\nconst int TCOLS = %d;\nconst int TROWS = %d;\n\n", TS_OUT, TCOLS, TROWS);
            fprintf(f, "vec4 sampleImage(vec2 p){\n");
            fprintf(f, "    ivec2 tl = clamp(ivec2(floor(p / TILE)), ivec2(0), ivec2(TCOLS-1, TROWS-1));\n");
            fprintf(f, "    int ti = tl.y * TCOLS + tl.x;\n");
            if (embed) {
                fprintf(f, "    uvec2 oc = kTileOff[ti];\n");
                fprintf(f, "    vec3 acc = vec3(0.0);\n");
                fprintf(f, "    float T = 1.0;\n");
                fprintf(f, "    for (uint k = 0u; k < oc.y; ++k){\n");
                fprintf(f, "        int si = int(kTileIdx[int(oc.x) + int(k)]);\n");
                fprintf(f, "        uint u0 = kData[si*4], u1 = kData[si*4+1], u2 = kData[si*4+2], u3 = kData[si*4+3];\n");
                fprintf(f, "        vec4 sa = vec4(float(u0 & 0xFFFFu) / kMXS, float(u0 >> 16) / kMYS,\n");
                fprintf(f, "                       exp2(float(u2 & 0xFFFu) * (14.0 / 4095.0) - 5.0),\n");
                fprintf(f, "                       exp2(float((u2 >> 12) & 0xFFFu) * (14.0 / 4095.0) - 5.0));\n");
                fprintf(f, "        vec4 sb = vec4(float(u1 & 0xFFu), float((u1 >> 8) & 0xFFu), float((u1 >> 16) & 0xFFu), float(u1 >> 24)) / 255.0;\n");
                fprintf(f, "        vec4 sc = vec4(float(u3 & 0xFFFFu) / 32767.0 - 1.0,\n");
                fprintf(f, "                       float((u3 >> 16) & 0xFFFFu) / 32767.0 - 1.0, 0.0, 0.0);\n");
            } else {
                fprintf(f, "    uvec2 oc = texelFetch(uTileOff, ivec2(ti,0), 0).xy;\n");
                fprintf(f, "    vec3 acc = vec3(0.0);\n");
                fprintf(f, "    float T = 1.0;\n");
                fprintf(f, "    for (uint k = 0u; k < oc.y; ++k){\n");
                fprintf(f, "        int si = int(texelFetch(uTileIdx, ivec2(int(oc.x)+int(k),0), 0).x);\n");
                fprintf(f, "        vec4 sa = texelFetch(uSplatA, ivec2(si,0), 0);\n");
                fprintf(f, "        vec4 sb = texelFetch(uSplatB, ivec2(si,0), 0);\n");
                fprintf(f, "        vec4 sc = texelFetch(uSplatC, ivec2(si,0), 0);\n");
            }
            fprintf(f, "        vec2 d = p - sa.xy;\n");
            fprintf(f, "        float ex = (sc.x*d.x + sc.y*d.y) / sa.z;\n");
            fprintf(f, "        float ey = (-sc.y*d.x + sc.x*d.y) / sa.w;\n");
            fprintf(f, "        float g = exp(-0.5*(ex*ex + ey*ey));\n");
            fprintf(f, "        float al = sb.w * g;\n");
            fprintf(f, "        acc += sb.rgb * (al * T);\n");
            fprintf(f, "        T *= (1.0 - al);\n");
            fprintf(f, "    }\n");
            if (residual) {
                fprintf(f, "    uint rv = kRes[int(clamp(p.y, 0.0, IMG_H - 1.0)) * IMG_WI + int(clamp(p.x, 0.0, IMG_W - 1.0))];\n");
                fprintf(f, "    acc += vec3(float(rv & 15u), float((rv >> 4u) & 15u), float((rv >> 8u) & 15u)) * (8.0 / 3825.0) - (64.0 / 255.0);\n");
            }
            fprintf(f, "    return vec4(clamp(acc, 0.0, 1.0), 1.0);\n");
            fprintf(f, "}\n\n");
            fprintf(f, "void mainImage(out vec4 fragColor, in vec2 fragCoord){\n");
            fprintf(f, "    float s = min(iResolution.x / IMG_W, iResolution.y / IMG_H);\n");
            fprintf(f, "    vec2 sz = vec2(IMG_W, IMG_H) * s;\n");
            fprintf(f, "    vec2 o = (iResolution.xy - sz) * 0.5;\n");
            fprintf(f, "    vec2 q = (fragCoord - o) / sz;\n");
            fprintf(f, "    if (q.x < 0.0 || q.y < 0.0 || q.x > 1.0 || q.y > 1.0) { fragColor = vec4(0.0); return; }\n");
            fprintf(f, "    fragColor = sampleImage(q * vec2(IMG_W, IMG_H));\n");
            fprintf(f, "}\n\n");
            fprintf(f, "void main() { mainImage(FragColor, gl_FragCoord.xy); }\n");
            fclose(f);
            printf("已输出 GLSL: %s %s\n", outPath.c_str(),
                   embed ? "（无纹理纯 GLSL，参数内嵌 const 数组）" : "（含参数纹理 *_splatA/B/C.raw 与 tile 数据 *_tileOff/Idx.raw）");
        } else {
            printf("无法写入 %s\n", outPath.c_str());
        }
    }
    printf("预览: fitsplat_preview.bmp\n");
    return 0;
}
