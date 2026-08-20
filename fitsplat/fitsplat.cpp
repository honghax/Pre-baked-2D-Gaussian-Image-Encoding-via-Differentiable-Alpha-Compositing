// fitsplat.cpp — 把一张图片拟合成"高斯泼溅参数 + 渲染算法"（离线优化，允许高耗时）
//
// 支持两种渲染模型（--model 0|1，默认 1）：
//   model 1 = 加和模型（image-gs / playpiano 风格）：
//       每个 splat 是 2D 高斯（位置/尺寸/旋转参数化），权重 w = a*gauss，
//       逐 splat 加和  acc += c*w，顺序无关、无 transmittance 链，输出最后 clamp ≤1。
//   model 0 = alpha 合成（3DGS 标准，旧版 fitsplat_git 行为）：
//       逐 splat 按序合成  acc += c*w*T; T *= (1-w)，依赖渲染顺序。
//   梯度优化：L1 损失 + 解析反传（加和 backward 逐 splat 独立；alpha 需要 T 递推）+ Adam，
//      学习率随迭代衰减，单步幅值 clamp 防发散。
//   密度控制：累积位置梯度分裂"欠拟合" splat，删除低权重 splat；
//      渐进模式（--prog）额外按误差采样补残差 splat（image-gs 风格）。
//
// 产物是一小段 GLSL（const 数组参数 + 十几行渲染），文件头带 MODEL 标识，
// 运行时由 fitsplat_gl 按标识自动选择混合方式渲染。
//
// 用法：
//   fitsplat input.png [--splats 600] [--iters 1000] [--width 320] [--height 0] [--batch 2048]
//            [--out out.glsl] [--threads 0] [--bg 0.0] [--seed 42] [--preview 10] [--embed 1]
//            [--model 0|1]
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

// GPU 前向渲染基准（--gpu）：OpenGL 4.3 compute shader（glad43 为训练器专用 loader，
// 与播放器 fitsplat_gl 的 3.3 glad 各自链接，互不冲突）
#include <glad/glad.h>
#define GLFW_INCLUDE_NONE   // 阻止 glfw3.h 引入系统 gl.h 与 glad 冲突
#include <GLFW/glfw3.h>

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
static float gInitScale = 5.0f;            // 内容自适应初始化/误差引导添加的初始高斯尺度（像素）
static int   g_model = 1;                  // 渲染模型：0=alpha 合成（3DGS 旧版），1=加和模型（默认）

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

// 单像素渲染（tile 局部版；与最终 GLSL 一致；输出 clamp 到 [0,1]）
// model 1=加和（顺序无关）；model 0=alpha 合成（依赖 idx 顺序，T 递推）
static inline void RenderPixel(const Splat* S, const uint32_t* idx, int n, float x, float y, float& r, float& g, float& b) {
    r = g = b = 0.0f;
    if (g_model == 1) {
        for (int k = 0; k < n; ++k) {
            const Splat& s = S[idx[k]];
            float dx = x - s.mx, dy = y - s.my;
            if (dx * dx + dy * dy > s.r2) continue;          // 快速剔除
            float ex = (s.cs * dx + s.sn * dy) / s.sx;
            float ey = (-s.sn * dx + s.cs * dy) / s.sy;
            float gv = expf(-0.5f * (ex * ex + ey * ey));
            float w = s.a * gv;
            r += s.r * w; g += s.g * w; b += s.b * w;        // 加和：顺序无关，无 transmittance
        }
    } else {
        float T = 1.0f;
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
    }
    r = ClampF(r, 0, 1); g = ClampF(g, 0, 1); b = ClampF(b, 0, 1);
}

// ---------- 可微渲染：正传 + 反传（加和 backward 逐 splat 独立 / alpha 需要 T 递推）----------
struct PixelFwd2 {
    std::vector<float> alpha, T, g, ex, ey;   // T 仅 alpha 模式使用（加和模式恒为 1.0）
    void resize(size_t K) { alpha.resize(K); T.resize(K); g.resize(K); ex.resize(K); ey.resize(K); }
};

// 正传（tile 局部版）：只遍历本像素所在 tile 的 splat（idx 为全局索引列表，n 为个数）
static inline void ForwardPixel2(const Splat* S, const uint32_t* idx, int n, float x, float y, PixelFwd2& fwd, float& accR, float& accG, float& accB) {
    accR = accG = accB = 0.0f;
    fwd.resize((size_t)n);
    if (g_model == 1) {
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
            fwd.alpha[k] = alpha; fwd.T[k] = 1.0f;
            accR += s.r * alpha; accG += s.g * alpha; accB += s.b * alpha;   // 加和：不乘 T
        }
    } else {
        float T = 1.0f;
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
}

// 反传（tile 局部版）：对局部列表里的每个 splat 累加 9 参数梯度到全局 gacc[idx[ii]]
// touched/mark：记录本迭代被触碰的 splat（稀疏梯度：Adam 更新只处理触碰集，
// 消除每迭代 tGrads 全量清零 + 全量交叉读取两大内存带宽开销）
// 加和模型：∂L/∂α_i = eC（无 transmittance 递推），各 splat 梯度完全独立
// alpha 模型：∂L/∂α_i = eC*T_i - gradT*T_i，逆序递推 gradT
static inline void BackwardPixel2(const Splat* S, const uint32_t* idx, int n, const PixelFwd2& fwd,
                                  float eR, float eG, float eB,
                                  std::vector<std::array<float, 9>>& gacc,
                                  std::vector<uint32_t>& touched, std::vector<unsigned char>& mark) {
    if (g_model == 1) {
        for (int ii = 0; ii < n; ++ii) {
            size_t gi = idx[ii];                          // 全局 splat 索引
            if (!mark[gi]) { mark[gi] = 1; touched.push_back((uint32_t)gi); }   // 稀疏梯度：首次触碰才入列表
            const Splat& s = S[gi];
            float alp = fwd.alpha[ii];
            float gv  = fwd.g[ii];
            // E·c_i = Σ_c e_c * c_{i,c}（e_c 由 L1/L2 损失的 ∂L/∂acc_c 决定）
            float eC = eR * s.r + eG * s.g + eB * s.b;
            float dAlp = eC;                              // ∂L/∂α_i = Σ_c e_c c_c（加和无 T 耦合）

            std::array<float, 9>& g = gacc[gi];
            // α = a * gauss
            float dA = dAlp * gv;                         // ∂L/∂a
            float dG = dAlp * s.a;                        // ∂L/∂gauss
            g[8] += dA * s.a * (1.0f - s.a);              // ∂L/∂logit(a)
            // 颜色：∂L/∂c_c = e_c * α_i
            g[5] += eR * alp; g[6] += eG * alp; g[7] += eB * alp;

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
    } else {
        float gradT = 0.0f;   // ∂L/∂T_{i+1}（alpha 合成逆序递推）
        for (int ii = n - 1; ii >= 0; --ii) {
            size_t gi = idx[ii];                          // 全局 splat 索引
            if (!mark[gi]) { mark[gi] = 1; touched.push_back((uint32_t)gi); }   // 稀疏梯度：首次触碰才入列表
            const Splat& s = S[gi];
            float T_i = fwd.T[ii];
            float alp = fwd.alpha[ii];
            float gv  = fwd.g[ii];
            float eC = eR * s.r + eG * s.g + eB * s.b;
            float dAlp = eC * T_i - gradT * T_i;          // ∂L/∂α_i
            float dTi  = eC * alp + gradT * (1.0f - alp); // ∂L/∂T_i
            gradT = dTi;

            std::array<float, 9>& g = gacc[gi];
            float dA = dAlp * gv;                         // ∂L/∂a
            float dG = dAlp * s.a;                        // ∂L/∂gauss
            g[8] += dA * s.a * (1.0f - s.a);              // ∂L/∂logit(a)
            // 颜色：∂L/∂c_c = e_c * α_i * T_i
            g[5] += eR * alp * T_i; g[6] += eG * alp * T_i; g[7] += eB * alp * T_i;

            float w = dG * gv;
            if (w == 0.0f) continue;
            float ex = fwd.ex[ii], ey = fwd.ey[ii];
            float cs = s.cs, sn = s.sn;
            g[0] += w * (ex * cs / s.sx - ey * sn / s.sy);   // ∂L/∂mx
            g[1] += w * (ex * sn / s.sx + ey * cs / s.sy);   // ∂L/∂my
            g[2] += w * (ex * ex);                           // ∂L/∂log(sx)
            g[3] += w * (ey * ey);                           // ∂L/∂log(sy)
            float dxp = cs * (ex * s.sx) - sn * (ey * s.sy);
            float dyp = sn * (ex * s.sx) + cs * (ey * s.sy);
            float u1r = -sn * dxp + cs * dyp;
            float v1r = -cs * dxp - sn * dyp;
            g[4] += w * (-ex * u1r / s.sx - ey * v1r / s.sy); // ∂L/∂rot
        }
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

// ---------- 内容自适应初始化：梯度引导采样（image-gs 风格） ----------
// 按图像梯度幅值² 的概率采样像素作为 splat 中心，细节区集中分配、平滑区少分配；
// 30% 均匀随机兜底平滑区。颜色直接用目标色（color init，与网格初始化一致）。
static void InitSplatGradient(int K, std::mt19937& rng) {
    std::vector<float> grad((size_t)W * H, 0.0f);
    double gsum = 0;
    for (int j = 0; j < H; ++j) {
        for (int i = 0; i < W; ++i) {
            if (!valid[(size_t)j * W + i]) continue;
            int i0 = std::max(i - 1, 0), i1 = std::min(i + 1, W - 1);
            int j0 = std::max(j - 1, 0), j1 = std::min(j + 1, H - 1);
            float gx = 0, gy = 0;
            for (int c = 0; c < 3; ++c) {
                gx += fabsf(target[((size_t)j * W + i1) * 3 + c] - target[((size_t)j * W + i0) * 3 + c]);
                gy += fabsf(target[((size_t)j1 * W + i) * 3 + c] - target[((size_t)j0 * W + i) * 3 + c]);
            }
            float g = gx + gy;
            grad[(size_t)j * W + i] = g * g;
            gsum += (double)g * g;
        }
    }
    if (gsum <= 0) gsum = 1.0;
    std::vector<double> cdf((size_t)W * H + 1, 0.0);
    for (size_t i = 0; i < grad.size(); ++i) cdf[i + 1] = cdf[i] + (double)grad[i] / gsum;
    std::uniform_real_distribution<double> u01(0.0, 1.0);
    std::uniform_int_distribution<int> upix(0, W * H - 1);
    for (int k = 0; k < K; ++k) {
        size_t pi;
        float ps;
        if (k % 10 < 3) { pi = (size_t)upix(rng); ps = gInitScale * 2.5f; }   // 30% 随机：大尺度兜底平滑区
        else {                                                               // 70% 按梯度概率采样：细节区小尺度
            double u = u01(rng);
            auto it = std::upper_bound(cdf.begin(), cdf.end(), u);
            pi = (size_t)(it - cdf.begin()) - 1;
            ps = gInitScale;
        }
        if (!valid[pi]) continue;                                // 透明像素跳过
        Splat s;
        s.mx = (float)(pi % W) + 0.5f; s.my = (float)(pi / W) + 0.5f;
        s.sx = ps; s.sy = ps; s.rot = 0.0f;
        s.r = ClampF(target[pi * 3], 0, 1); s.g = ClampF(target[pi * 3 + 1], 0, 1); s.b = ClampF(target[pi * 3 + 2], 0, 1);
        s.a = 0.5f;
        s.r2 = (3.0f * s.sx) * (3.0f * s.sx);
        s.cs = 1.0f; s.sn = 0.0f;
        splats.push_back(s);
        mAdam.push_back(std::array<float, 9>{});
        vAdam.push_back(std::array<float, 9>{});
        gradAcc.push_back(std::array<float, 2>{});
    }
}

// ---------- 密度控制：分裂高梯度 splat，删除低不透明度 splat ----------
// errAdd=1 时（渐进模式）额外做误差引导添加：全图渲染后在"欠拟合区"（渲染<目标）
// 采样新 splat，颜色=正残差（加和模型下 splat 颜色仍 clamp 到 [0,1]，只能补亮缺口；
// 若未来放开颜色负值约束可直接用 image-gs 的 diff_map 双向残差），位置=误差像素。
static void Densify(size_t maxK, std::mt19937& rng, bool errAdd) {
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
    // 误差引导添加（渐进模式）：补足到 maxK 的缺口，残差色初始化
    if (errAdd) {
        size_t finalCnt = K - nDrop + add.size();
        if (finalCnt < maxK) {
            std::vector<float> render;
            Evaluate(&render);   // 全图渲染（复用评估渲染）
            std::vector<double> prob((size_t)W * H, 0.0);
            double psum = 0;
            for (size_t i = 0; i < (size_t)W * H; ++i) {
                if (!valid[i]) continue;
                float d = (target[i * 3] - render[i * 3]) + (target[i * 3 + 1] - render[i * 3 + 1]) + (target[i * 3 + 2] - render[i * 3 + 2]);
                if (d > 0) { prob[i] = (double)(d * d); psum += prob[i]; }
            }
            size_t want = maxK - finalCnt;
            if (psum > 1e-12) {
                std::vector<double> cdf(prob.size() + 1, 0.0);
                for (size_t i = 0; i < prob.size(); ++i) cdf[i + 1] = cdf[i] + prob[i] / psum;
                std::uniform_real_distribution<double> u01(0.0, 1.0);
                for (size_t w = 0; w < want; ++w) {
                    double u = u01(rng);
                    auto it = std::upper_bound(cdf.begin(), cdf.end(), u);
                    size_t pi = (size_t)(it - cdf.begin()) - 1;
                    if (!valid[pi]) continue;
                    Splat n;
                    n.mx = (float)(pi % W) + 0.5f; n.my = (float)(pi / W) + 0.5f;
                    n.sx = gInitScale * 0.6f; n.sy = gInitScale * 0.6f; n.rot = 0.0f;   // 细节补丁：小尺度高浓度
                    n.r = ClampF(target[pi * 3] - render[pi * 3], 0, 1);
                    n.g = ClampF(target[pi * 3 + 1] - render[pi * 3 + 1], 0, 1);
                    n.b = ClampF(target[pi * 3 + 2] - render[pi * 3 + 2], 0, 1);
                    n.a = 0.6f;
                    n.r2 = (3.0f * n.sx) * (3.0f * n.sx);
                    n.cs = 1.0f; n.sn = 0.0f;
                    add.push_back(n);
                    addM.push_back(std::array<float, 9>{});
                    addV.push_back(std::array<float, 9>{});
                    addGA.push_back(std::array<float, 2>{});
                }
            }
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
    printf("  --target F    目标 PSNR(dB)，每 10%% 迭代全图评估，达到即提前停止（默认 0 = 不启用）\n");
    printf("  --gputrain 1  训练切到 GPU compute shader（正传/反传/Adam 全 GPU，需 OpenGL 4.3；默认 0 = CPU）\n");
    printf("  --init 1      梯度引导初始化（内容自适应：按图像梯度概率采样，细节区集中分配；默认 0 = 网格）\n");
    printf("  --prog 1      误差引导渐进优化（初始 50%% splat，逐步在欠拟合区补残差 splat，预算=--splats；默认 0）\n");
    printf("  --init-scale F 初始/新增高斯尺度（像素，默认 5.0；--init 1 / --prog 1 时生效）\n");
    printf("  --model 0|1    渲染模型：1=加和（image-gs 风格，默认），0=alpha 合成（3DGS 旧版）\n");
}

static double NowSec() {
    using namespace std::chrono;
    return duration<double>(steady_clock::now().time_since_epoch()).count();
}

// ---------- GPU 前向 tile 渲染基准（--gpu）----------
// 验证思路（只移植最容易验证的前向）：
//   CPU：BuildTileMap 构建 tile → splat 索引表
//   GPU：每个 workgroup 渲染一个 16x16 tile，逐像素按列表顺序加和（无 transmittance）
//   CPU：读回结果，与 CPU Evaluate 全量渲染对比（数值一致性 + 耗时加速比）
static const char* kGpuFwdShader = R"GLSL(
#version 430 core
layout(local_size_x = 16, local_size_y = 16) in;

// 每 splat 3xvec4：(mx,my,sx,sy) (cs,sn,r,g) (b,a,0,0)
layout(std430, binding = 0) readonly buffer SSplats { vec4 splat[]; };
layout(std430, binding = 1) readonly buffer SOff   { uint off[]; };
layout(std430, binding = 2) readonly buffer SCnt   { uint cnt[]; };
layout(std430, binding = 3) readonly buffer SIdx   { uint idx[]; };
layout(std430, binding = 4) writeonly buffer SImg  { vec4 img[]; };

uniform uint uTCOLS;
uniform uint uImgW;
uniform uint uImgH;
uniform int  uModel;    // 1=加和（顺序无关），0=alpha 合成（T 递推）

void main() {
    uvec2 px = gl_GlobalInvocationID.xy;
    uint tile = uTCOLS * gl_WorkGroupID.y + gl_WorkGroupID.x;
    uint n    = cnt[tile];
    uint base = off[tile];
    vec2 p = vec2(float(px.x) + 0.5f, float(px.y) + 0.5f);
    vec3 acc = vec3(0.0f);
    float T = 1.0f;
    for (uint k = 0u; k < n; ++k) {
        uint gi = idx[base + k];
        vec4 sa = splat[gi * 3u + 0u];
        vec4 sb = splat[gi * 3u + 1u];
        vec4 sc = splat[gi * 3u + 2u];
        float dx = p.x - sa.x;
        float dy = p.y - sa.y;
        float m = max(sa.z, sa.w);
        if (dx * dx + dy * dy > 9.0f * m * m) continue;   // 3σ 快速剔除（与 CPU RenderPixel 一致）
        float ex = (sb.x * dx + sb.y * dy) / sa.z;
        float ey = (-sb.y * dx + sb.x * dy) / sa.w;
        float gv = exp(-0.5f * (ex * ex + ey * ey));
        float alpha = sc.y * gv;
        if (uModel == 1) {
            acc += vec3(sb.z, sb.w, sc.x) * alpha;       // 加和：顺序无关，无 transmittance
        } else {
            acc += vec3(sb.z, sb.w, sc.x) * (alpha * T);
            T *= 1.0f - alpha;
        }
    }
    if (px.x < uImgW && px.y < uImgH)
        img[px.y * uImgW + px.x] = vec4(clamp(acc, 0.0f, 1.0f), 1.0f);
}
)GLSL";

static double GpuForwardTest()
{
    // 1) CPU 构建 16x16 tile（与 Evaluate 完全一致），单独计时
    double t0b = NowSec();
    TileMap tm;
    BuildTileMap(splats, 16, tm);
    double buildTime = NowSec() - t0b;
    int TCOLS = tm.TCOLS, TROWS = tm.TROWS;
    int NS = (int)splats.size();
    printf("[gpu] CPU BuildTileMap(16x16) %g ms, %d 个 tile, 每 tile 平均 %.1f splat\n",
           buildTime * 1e3, TCOLS * TROWS, (double)tm.idx.size() / (TCOLS * TROWS));

    // 2) 隐藏窗口 4.3 core 上下文
    if (!glfwInit()) { printf("[gpu] glfwInit 失败\n"); return -1; }
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    glfwWindowHint(GLFW_VISIBLE, GLFW_FALSE);
    GLFWwindow* win = glfwCreateWindow(16, 16, "fitsplat-gpu", nullptr, nullptr);
    if (!win) {
        printf("[gpu] 创建 4.3 core 上下文失败（驱动/核显不支持 compute shader）\n");
        glfwTerminate(); return -1;
    }
    glfwMakeContextCurrent(win);
    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) {
        printf("[gpu] gladLoadGL 失败\n"); glfwTerminate(); return -1;
    }
    printf("[gpu] 上下文: OpenGL %s, GLSL %s, 渲染器 %s\n",
           glGetString(GL_VERSION), glGetString(GL_SHADING_LANGUAGE_VERSION), glGetString(GL_RENDERER));

    // 3) 编译链接 compute shader
    GLuint cs = glCreateShader(GL_COMPUTE_SHADER);
    glShaderSource(cs, 1, &kGpuFwdShader, nullptr);
    glCompileShader(cs);
    GLint ok = 0;
    glGetShaderiv(cs, GL_COMPILE_STATUS, &ok);
    if (!ok) {
        char log[8192]; glGetShaderInfoLog(cs, 8192, nullptr, log);
        printf("[gpu] compute 编译失败:\n%s\n", log);
        glDeleteShader(cs); glfwTerminate(); return -1;
    }
    GLuint prog = glCreateProgram();
    glAttachShader(prog, cs); glLinkProgram(prog);
    glGetProgramiv(prog, GL_LINK_STATUS, &ok);
    if (!ok) {
        char log[8192]; glGetProgramInfoLog(prog, 8192, nullptr, log);
        printf("[gpu] compute 链接失败:\n%s\n", log);
        glDeleteShader(cs); glDeleteProgram(prog); glfwTerminate(); return -1;
    }
    glDeleteShader(cs);

    // 4) splat 参数打包：每 splat 12 float = 3x vec4
    std::vector<float> sp((size_t)NS * 12);
    for (int i = 0; i < NS; ++i) {
        const Splat& s = splats[i];
        float* d = &sp[(size_t)i * 12];
        d[0]=s.mx; d[1]=s.my; d[2]=s.sx; d[3]=s.sy;
        d[4]=s.cs; d[5]=s.sn; d[6]=s.r;  d[7]=s.g;
        d[8]=s.b;  d[9]=s.a;  d[10]=0;   d[11]=0;
    }

    // 5) SSBO 上传
    GLuint sbo[5]; glGenBuffers(5, sbo);
    auto upload = [&](int idx, size_t bytes, const void* data) {
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, sbo[idx]);
        glBufferData(GL_SHADER_STORAGE_BUFFER, (GLsizeiptr)bytes, data, GL_STATIC_DRAW);
    };
    double t0u = NowSec();
    upload(0, sp.size() * 4, sp.data());
    upload(1, (size_t)TCOLS * TROWS * 4, tm.off.data());
    upload(2, (size_t)TCOLS * TROWS * 4, tm.cnt.data());
    upload(3, tm.idx.size() * 4, tm.idx.data());
    upload(4, (size_t)W * H * 16, nullptr);
    double upTime = NowSec() - t0u;
    printf("[gpu] SSBO 上传 %.1f MB %g ms\n",
           (double)(sp.size() * 4 + tm.idx.size() * 4 + (size_t)TCOLS * TROWS * 8 + (size_t)W * H * 16) / 1e6, upTime * 1e3);

    glUseProgram(prog);
    for (int i = 0; i < 5; ++i) glBindBufferBase(GL_SHADER_STORAGE_BUFFER, i, sbo[i]);
    glUniform1ui(glGetUniformLocation(prog, "uTCOLS"), (GLuint)TCOLS);
    glUniform1ui(glGetUniformLocation(prog, "uImgW"),  (GLuint)W);
    glUniform1ui(glGetUniformLocation(prog, "uImgH"),  (GLuint)H);
    glUniform1i (glGetUniformLocation(prog, "uModel"), g_model);

    // 6) dispatch 计时（5 次取最小，排除首帧驱动编译）
    double gpuBest = 1e18;
    for (int rep = 0; rep < 5; ++rep) {
        double t0 = NowSec();
        glDispatchCompute((GLuint)TCOLS, (GLuint)TROWS, 1);
        glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT | GL_BUFFER_UPDATE_BARRIER_BIT);
        glFinish();
        double t1 = NowSec();
        gpuBest = std::min(gpuBest, t1 - t0);
    }
    printf("[gpu] dispatch %dx%d workgroup（%d 个 16x16 tile）, 最佳 %g ms\n",
           TCOLS, TROWS, TCOLS * TROWS, gpuBest * 1e3);

    // 7) 读回
    std::vector<float> img((size_t)W * H * 4);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, sbo[4]);
    void* mp = glMapBufferRange(GL_SHADER_STORAGE_BUFFER, 0, (GLsizeiptr)(img.size() * 4), GL_MAP_READ_BIT);
    if (mp) memcpy(img.data(), mp, img.size() * 4);
    glUnmapBuffer(GL_SHADER_STORAGE_BUFFER);

    // 8) CPU Evaluate 参考（计时，含内部 BuildTileMap）
    double t0c = NowSec();
    std::vector<float> refRGB;
    double mseCpu = Evaluate(&refRGB);
    double cpuTime = NowSec() - t0c;

    // 9) GPU vs CPU 参考一致性 + GPU vs 原图 PSNR
    double mseDiff = 0, mseTgt = 0; long long nv = 0;
    for (int j = 0; j < H; ++j)
        for (int i = 0; i < W; ++i) {
            size_t p = (size_t)j * W + i;
            if (!valid[p]) continue;
            const float* g = &img[p * 4];
            const float* c = &refRGB[p * 3];
            float dr = g[0]-c[0], dg = g[1]-c[1], db = g[2]-c[2];
            mseDiff += (double)(dr*dr + dg*dg + db*db);
            mseTgt  += (double)((g[0]-target[p*3])*(g[0]-target[p*3]) +
                                (g[1]-target[p*3+1])*(g[1]-target[p*3+1]) +
                                (g[2]-target[p*3+2])*(g[2]-target[p*3+2]));
            ++nv;
        }
    mseDiff /= (nv * 3.0); mseTgt /= (nv * 3.0);   // 每通道 MSE（与 Evaluate 口径一致，PSNR 可直比）
    double dbConsist = 10.0 * log10(1.0 / std::max(mseDiff, 1e-12));
    printf("[gpu] GPU vs CPU 参考差 MSE=%.6g → 一致性 %.2f dB（越高越一致）\n", mseDiff, dbConsist);
    printf("[gpu] GPU vs 原图 PSNR=%.2f dB（CPU Evaluate=%.2f dB）\n",
           10.0 * log10(1.0 / std::max(mseTgt, 1e-12)), 10.0 * log10(1.0 / std::max(mseCpu, 1e-12)));
    printf("[gpu] CPU 全量渲染 %g ms vs GPU dispatch %g ms → 纯渲染加速 %.2fx\n",
           cpuTime * 1e3, gpuBest * 1e3, cpuTime / std::max(gpuBest, 1e-12));
    printf("[gpu] GPU 全流程（上传+dispatch+读回）%g ms\n", (upTime + gpuBest) * 1e3);

    glDeleteProgram(prog); glDeleteBuffers(5, sbo);
    glfwTerminate();
    return gpuBest;
}

// ================= GPU 训练（--gputrain）=================
// 架构：每 workgroup = 1 个 8x8 训练 tile（local 8x8，每线程 1 像素），四个 compute pass：
//   zero  清零 stat/touchMark/touchCnt（每迭代开头）
//   fwd   正传：加和，fwd 数据(alpha,ex,ey,unused)写 SSBO，像素颜色写 pcol，原子累加 loss/像素数
//   bwd   反传：读 fwd/pcol，加和 backward（逐 splat 独立，无递推），原子加 9 参数梯度，稀疏 touch 收集
//   adam  稀疏 Adam：只更新本迭代 touch 的 splat（读回 touchCnt 后 dispatch），梯度除 nv，清 grad，累积 gradAcc
// 每迭代仅上传 nTile*2 个 tile 坐标（KB 级）+ 读回 3 个 uint（loss/nv/touchCnt），无大传输；
// 密度控制每 100 迭代读回全量参数跑 CPU Densify 后重传（~70MB/100 迭代，摊薄后可忽略）。
static const char* kGpuTrainZero = R"GLSL(
#version 430 core
layout(local_size_x=256) in;
layout(std430,binding=11) buffer BStat { uint stat[]; };   // [loss*1e5, l2*1e5, nv, touchCnt]
layout(std430,binding=12) buffer BMark { uint mark[]; };   // touchMark[NS]
uniform uint uN;
void main(){
  uint i = gl_GlobalInvocationID.x;
  if (i < uN) mark[i] = 0u;
  if (i == 0u) { stat[0] = 0u; stat[1] = 0u; stat[2] = 0u; stat[3] = 0u; }
}
)GLSL";

// 正传：每个 workgroup 处理一个 8x8 tile，64 线程 = 64 像素（共享同一 splat 索引列表）
static const char* kGpuTrainFwd = R"GLSL(
#version 430 core
layout(local_size_x=8, local_size_y=8) in;
layout(std430,binding=0) buffer BP { float p[]; };         // NS*9 参数
layout(std430,binding=1) buffer BT { float tgt[]; };       // W*H*3 目标
layout(std430,binding=2) buffer BV { uint valid[]; };      // W*H mask
layout(std430,binding=3) buffer BOff { uint off[]; };
layout(std430,binding=4) buffer BCnt { uint cnt[]; };
layout(std430,binding=5) buffer BIdx { uint idx[]; };
layout(std430,binding=6) buffer BTxy { uint txy[]; };      // nTile*2 tile 坐标
layout(std430,binding=7) buffer BFwd { float fwd[]; };     // seq*(uMaxPl*4): (alpha,T,ex,ey)  T 加和模式恒 1.0
layout(std430,binding=8) buffer BPc  { float pcol[]; };    // seq*3: 像素颜色
layout(std430,binding=11) buffer BStat { uint stat[]; };
uniform uint uImgW, uImgH, uTCOLS, uMaxPl;
uniform int uModel;   // 1=加和，0=alpha 合成
float SigmoidF(float x){ return 1.0f/(1.0f+exp(-x)); }
void main(){
  uint wg = gl_WorkGroupID.x;
  uint t  = gl_LocalInvocationIndex;
  uint tx = txy[wg*2u], ty = txy[wg*2u+1u];
  uint tile = ty*uTCOLS + tx;
  uint n = cnt[tile], base = off[tile];
  uint px = tx*8u + t%8u, py = ty*8u + t/8u;
  uint gid = py*uImgW + px;
  uint seq = wg*64u + t;
  if (px >= uImgW || py >= uImgH) return;
  float cx = float(px)+0.5f, cy = float(py)+0.5f;
  vec3 acc = vec3(0.0f);
  float T = 1.0f;
  uint fbase = seq*uMaxPl*4u;
  for (uint k=0u;k<n;++k){
    uint gi = idx[base+k];
    uint g9 = gi*9u;
    float mx=p[g9], my=p[g9+1u];
    float sx=exp(p[g9+2u]), sy=exp(p[g9+3u]);
    float cs=cos(p[g9+4u]), sn=sin(p[g9+4u]);
    float r=clamp(p[g9+5u],0.0f,1.0f), g=clamp(p[g9+6u],0.0f,1.0f), b=clamp(p[g9+7u],0.0f,1.0f);
    float a=SigmoidF(p[g9+8u]);
    float dx=cx-mx, dy=cy-my;
    float m=max(sx,sy); float r2=(3.0f*m)*(3.0f*m);
    float ex=0.0f, ey=0.0f, gv=0.0f;
    if (dx*dx+dy*dy <= r2){
      ex=(cs*dx+sn*dy)/sx; ey=(-sn*dx+cs*dy)/sy;
      gv=exp(-0.5f*(ex*ex+ey*ey));
    }
    float alpha=a*gv;
    uint fp=fbase+k*4u;
    fwd[fp]=alpha; fwd[fp+1u]=T; fwd[fp+2u]=ex; fwd[fp+3u]=ey;
    if (uModel == 1) {
      acc += vec3(r,g,b)*alpha;          // 加和：顺序无关，无 transmittance
    } else {
      acc += vec3(r,g,b)*(alpha*T);
      T *= 1.0f-alpha;
    }
  }
  acc = clamp(acc, 0.0f, 1.0f);
  pcol[seq*3u]=acc.x; pcol[seq*3u+1u]=acc.y; pcol[seq*3u+2u]=acc.z;
  if (valid[gid] != 0u){
    uint k3 = gid*3u;
    vec3 d = acc - vec3(tgt[k3], tgt[k3+1u], tgt[k3+2u]);
    // loss/l2 用 1e5 定点（1e6 在 batch 大时 uint32 累加可能溢出）
    atomicAdd(stat[0], uint((abs(d.x)+abs(d.y)+abs(d.z))*1e5f+0.5f));
    atomicAdd(stat[1], uint((d.x*d.x+d.y*d.y+d.z*d.z)*1e5f+0.5f));
    atomicAdd(stat[2], 1u);
  }
}
)GLSL";

// 反传：与 CPU BackwardPixel2 完全一致（加和逐 splat 独立 / alpha 逆序 T 递推）+ 9 参数解析梯度。
// 注意：Intel 核显 GLSL 4.30 不支持 float atomicAdd（GL_ARB_shader_atomic_float 未启用），
// 梯度用 int32 定点（×65536）原子累加，Adam pass 转回 float——精度 1.5e-5，足够训练。
static const char* kGpuTrainBwd = R"GLSL(
#version 430 core
layout(local_size_x=8, local_size_y=8) in;
layout(std430,binding=0) buffer BP { float p[]; };
layout(std430,binding=1) buffer BT { float tgt[]; };
layout(std430,binding=2) buffer BV { uint valid[]; };
layout(std430,binding=3) buffer BOff { uint off[]; };
layout(std430,binding=4) buffer BCnt { uint cnt[]; };
layout(std430,binding=5) buffer BIdx { uint idx[]; };
layout(std430,binding=6) buffer BTxy { uint txy[]; };
layout(std430,binding=7) buffer BFwd { float fwd[]; };
layout(std430,binding=8) buffer BPc  { float pcol[]; };
layout(std430,binding=9) buffer BG   { int grad[]; };     // int32 定点 ×65536
layout(std430,binding=11) buffer BStat { uint stat[]; };
layout(std430,binding=12) buffer BMark { uint mark[]; };
layout(std430,binding=13) buffer BTch  { uint tch[]; };
uniform uint uImgW, uImgH, uTCOLS, uMaxPl;
uniform int uL2, uModel;
float SigmoidF(float x){ return 1.0f/(1.0f+exp(-x)); }
void main(){
  uint wg=gl_WorkGroupID.x, t=gl_LocalInvocationIndex;
  uint tx=txy[wg*2u], ty=txy[wg*2u+1u];
  uint tile=ty*uTCOLS+tx;
  uint n=cnt[tile], base=off[tile];
  uint px=tx*8u+t%8u, py=ty*8u+t/8u;
  uint gid=py*uImgW+px;
  uint seq=wg*64u+t;
  if (px>=uImgW||py>=uImgH) return;
  if (valid[gid]==0u) return;
  uint k3=gid*3u;
  float accR=pcol[seq*3u], accG=pcol[seq*3u+1u], accB=pcol[seq*3u+2u];
  float dR=accR-tgt[k3], dG=accG-tgt[k3+1u], dB=accB-tgt[k3+2u];
  float eR,eG,eB;
  if (uL2==1){ eR=2.0f*dR; eG=2.0f*dG; eB=2.0f*dB; }
  // copysign 在部分 Intel 核显驱动未实现，用三元等价（copysign(1.0,0)=+1.0，与 CPU copysignf 一致）
  else { eR=(dR>=0.0f)?1.0f:-1.0f; eG=(dG>=0.0f)?1.0f:-1.0f; eB=(dB>=0.0f)?1.0f:-1.0f; }
  uint fbase=seq*uMaxPl*4u;
  float gradT=0.0f;
  for (uint i2=0u;i2<n;++i2){
    // 加和模型顺序无关：正序即可；alpha 合成必须逆序递推 gradT
    uint ii = (uModel == 1) ? i2 : n-1u-i2;
    uint gi=idx[base+ii];
    uint g9=gi*9u;
    uint fp=fbase+ii*4u;
    float alp=fwd[fp], Ti=fwd[fp+1u], ex=fwd[fp+2u], ey=fwd[fp+3u];
    float a=SigmoidF(p[g9+8u]);
    float r=clamp(p[g9+5u],0.0f,1.0f), g=clamp(p[g9+6u],0.0f,1.0f), b=clamp(p[g9+7u],0.0f,1.0f);
    float eC=eR*r+eG*g+eB*b;
    float dAlp;
    if (uModel == 1) {
      dAlp=eC;   // 加和：∂L/∂α_i = eC，无 transmittance 耦合
    } else {
      dAlp=eC*Ti-gradT*Ti;   // alpha：∂L/∂α_i = (eC-gradT)*T_i
      float dTi=eC*alp+gradT*(1.0f-alp);
      gradT=dTi;
    }
    float gv = (a>1e-12f) ? alp/a : 0.0f;
    float dA=dAlp*gv;
    float dG=dAlp*a;
    // 定点 ×256：8 位小数，除以 nv 后截断误差 ~1e-5；比 ×65536 抗早期大 splat 的多像素累加溢出
    atomicAdd(grad[g9+8u], int(dA*a*(1.0f-a)*256.0f));
    atomicAdd(grad[g9+5u], int(eR*alp*((uModel==1)?1.0f:Ti)*256.0f));
    atomicAdd(grad[g9+6u], int(eG*alp*((uModel==1)?1.0f:Ti)*256.0f));
    atomicAdd(grad[g9+7u], int(eB*alp*((uModel==1)?1.0f:Ti)*256.0f));
    float w=dG*gv;
    if (w!=0.0f){
      float cs=cos(p[g9+4u]), sn=sin(p[g9+4u]);
      float sx=exp(p[g9+2u]), sy=exp(p[g9+3u]);
      atomicAdd(grad[g9], int(w*(ex*cs/sx-ey*sn/sy)*256.0f));
      atomicAdd(grad[g9+1u], int(w*(ex*sn/sx+ey*cs/sy)*256.0f));
      atomicAdd(grad[g9+2u], int(w*ex*ex*256.0f));
      atomicAdd(grad[g9+3u], int(w*ey*ey*256.0f));
      float dxp=cs*(ex*sx)-sn*(ey*sy);
      float dyp=sn*(ex*sx)+cs*(ey*sy);
      float u1r=-sn*dxp+cs*dyp;
      float v1r=-cs*dxp-sn*dyp;
      atomicAdd(grad[g9+4u], int(w*(-ex*u1r/sx-ey*v1r/sy)*256.0f));
    }
    // 稀疏 touch 收集：w==0 时所有梯度为 0，无需进 Adam
    if (w!=0.0f && mark[gi]==0u){
      uint old=atomicExchange(mark[gi],1u);
      if (old==0u){ uint slot=atomicAdd(stat[3],1u); tch[slot]=gi; }
    }
  }
}
)GLSL";

// 稀疏 Adam：只更新本迭代 touch 的 splat（零梯度 splat 无需更新，与 CPU 稀疏梯度一致）
static const char* kGpuTrainAdam = R"GLSL(
#version 430 core
layout(local_size_x=256) in;
layout(std430,binding=0) buffer BP { float p[]; };
layout(std430,binding=9) buffer BG { int grad[]; };       // int32 定点 ×65536
layout(std430,binding=10) buffer BM { float mom[]; };
layout(std430,binding=13) buffer BTch { uint tch[]; };
layout(std430,binding=14) buffer BGA { float ga[]; };
layout(std430,binding=15) buffer BVel { float vel[]; };
uniform uint uTouch;
uniform float uInvNv, uLrScale, uBc1, uBc2;
uniform float kLr[9];
uniform uint uImgW, uImgH;
void main(){
  uint i = gl_GlobalInvocationID.x;
  if (i >= uTouch) return;
  uint gi = tch[i];
  uint g9 = gi*9u;
  float g0=0.0f, g1=0.0f;
  for (uint c=0u;c<9u;++c){
    uint idx=g9+c;
    float g = float(grad[idx]) / 256.0f * uInvNv;   // 定点转回 float 并按有效像素平均
    if (c==0u) g0=abs(g); if (c==1u) g1=abs(g);
    float m=0.9f*mom[idx]+0.1f*g;
    float v=0.999f*vel[idx]+0.001f*g*g;
    mom[idx]=m; vel[idx]=v;
    float mh=m/uBc1, vh=v/uBc2;
    float step=clamp(mh/(sqrt(vh)+1e-3f), -1.0f, 1.0f);
    p[idx] -= kLr[c]*uLrScale*step;
    grad[idx]=0;
  }
  ga[gi*2u]+=g0; ga[gi*2u+1u]+=g1;   // gradAcc 非原子：每个 splat 只被一个 workgroup 处理一次
  p[g9]=clamp(p[g9],0.0f,float(uImgW));
  p[g9+1u]=clamp(p[g9+1u],0.0f,float(uImgH));
  float sx=exp(p[g9+2u]); if (sx<0.05f)sx=0.05f; if (sx>float(uImgW))sx=float(uImgW); p[g9+2u]=log(sx);
  float sy=exp(p[g9+3u]); if (sy<0.05f)sy=0.05f; if (sy>float(uImgH))sy=float(uImgH); p[g9+3u]=log(sy);
  p[g9+5u]=clamp(p[g9+5u],0.0f,1.0f);
  p[g9+6u]=clamp(p[g9+6u],0.0f,1.0f);
  p[g9+7u]=clamp(p[g9+7u],0.0f,1.0f);
}
)GLSL";

// 创建 GLFW 隐藏窗口 4.3 core 上下文并加载 glad（与 GpuForwardTest 共用模式）
static GLFWwindow* GpuCreateCtx() {
    if (!glfwInit()) { printf("[gputrain] glfwInit 失败\n"); return nullptr; }
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    glfwWindowHint(GLFW_VISIBLE, GLFW_FALSE);
    GLFWwindow* win = glfwCreateWindow(16, 16, "fitsplat-gputrain", nullptr, nullptr);
    if (!win) { printf("[gputrain] 创建 4.3 core 上下文失败\n"); glfwTerminate(); return nullptr; }
    glfwMakeContextCurrent(win);
    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) {
        printf("[gputrain] gladLoadGL 失败\n"); glfwTerminate(); return nullptr;
    }
    printf("[gputrain] 上下文: OpenGL %s, GLSL %s, 渲染器 %s\n",
           glGetString(GL_VERSION), glGetString(GL_SHADING_LANGUAGE_VERSION), glGetString(GL_RENDERER));
    return win;
}

// 编译单个 compute program，失败打印日志并返回 0
static GLuint GpuCompile(const char* src, const char* name) {
    GLuint cs = glCreateShader(GL_COMPUTE_SHADER);
    glShaderSource(cs, 1, &src, nullptr);
    glCompileShader(cs);
    GLint ok = 0; glGetShaderiv(cs, GL_COMPILE_STATUS, &ok);
    if (!ok) {
        char log[8192]; glGetShaderInfoLog(cs, 8192, nullptr, log);
        printf("[gputrain] %s 编译失败:\n%s\n", name, log);
        glDeleteShader(cs); return 0;
    }
    GLuint prog = glCreateProgram();
    glAttachShader(prog, cs); glLinkProgram(prog);
    glGetProgramiv(prog, GL_LINK_STATUS, &ok);
    if (!ok) {
        char log[8192]; glGetProgramInfoLog(prog, 8192, nullptr, log);
        printf("[gputrain] %s 链接失败:\n%s\n", name, log);
        glDeleteShader(cs); glDeleteProgram(prog); return 0;
    }
    glDeleteShader(cs);
    return prog;
}

// GPU 训练主循环：正传/反传/Adam 全部在 compute shader 执行，CPU 只做 tile 采样、
// 密度控制（每 100 迭代读回跑 Densify）与进度显示。结束后把参数写回全局 splats。
static void GpuTrainLoop(int startIt, int iters, int nTile, size_t maxK, int previewPct,
                         const std::string& ckptPath, int ckptEvery, int l2loss, std::mt19937& rng,
                         int prog, int initK) {
    const int TSg = TS;   // 8x8 训练 tile
    GLFWwindow* win = GpuCreateCtx();
    if (!win) return;
    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) { glfwTerminate(); return; }

    // 渐进模式：目标 splat 数从 initK 逐步增长到 maxK（每 100 迭代一批，共 ~20 批）
    int progK = initK;
    int progStep = prog ? std::max(1, (int)ceil((double)((int)maxK - initK) / 20)) : 0;

    // 编译 4 个 pass
    GLuint progZero = GpuCompile(kGpuTrainZero, "zero");
    GLuint progFwd  = GpuCompile(kGpuTrainFwd, "fwd");
    GLuint progBwd  = GpuCompile(kGpuTrainBwd, "bwd");
    GLuint progAdam = GpuCompile(kGpuTrainAdam, "adam");
    if (!progZero || !progFwd || !progBwd || !progAdam) {
        glfwTerminate(); return;
    }

    // 构建 tile 分块（8x8，训练口径，cap=0 不截断）
    TileMap tMap;
    BuildTileMap(splats, TSg, tMap);
    int TCOLS = tMap.TCOLS;
    uint32_t maxPl = 0;
    for (uint32_t c : tMap.cnt) maxPl = std::max(maxPl, c);
    uint32_t maxPlInit = maxPl;   // fwd buffer 的分配上限，密度控制后只增不减
    int NS = (int)splats.size();
    printf("[gputrain] NS=%d tile=%dx%d maxPl=%d nTile=%d\n", NS, TCOLS, tMap.TROWS, (int)maxPl, nTile);

    // ---------- 打包上传数据 ----------
    std::vector<float> p((size_t)NS * 9), mom((size_t)NS * 9), vel((size_t)NS * 9), ga((size_t)NS * 2);
    for (int i = 0; i < NS; ++i) {
        std::array<float, 9> q = { splats[i].mx, splats[i].my, logf(splats[i].sx), logf(splats[i].sy),
                                   splats[i].rot, splats[i].r, splats[i].g, splats[i].b, Logit(splats[i].a) };
        for (int c = 0; c < 9; ++c) { p[(size_t)i * 9 + c] = q[c]; mom[(size_t)i * 9 + c] = mAdam[i][c]; vel[(size_t)i * 9 + c] = vAdam[i][c]; }
        ga[(size_t)i * 2] = gradAcc[i][0]; ga[(size_t)i * 2 + 1] = gradAcc[i][1];
    }
    std::vector<uint32_t> validU(W * H);
    for (int i = 0; i < W * H; ++i) validU[i] = valid[i] ? 1u : 0u;

    // SSBO 布局（binding 号与 shader 一致）
    enum { BP=0, BT=1, BV=2, BO=3, BC=4, BI=5, BTX=6, BFW=7, BPC=8, BGR=9, BMM=10, BST=11, BMK=12, BTC=13, BGA=14, BVL=15 };
    GLuint sbo[16] = {};
    auto mk = [&](int b, size_t bytes, const void* data) {
        glGenBuffers(1, &sbo[b]);
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, sbo[b]);
        glBufferData(GL_SHADER_STORAGE_BUFFER, (GLsizeiptr)bytes, data, GL_DYNAMIC_DRAW);
    };
    size_t npix = (size_t)nTile * 64;
    mk(BP,  p.size() * 4, p.data());
    mk(BT,  target.size() * 4, target.data());
    mk(BV,  validU.size() * 4, validU.data());
    mk(BO,  tMap.off.size() * 4, tMap.off.data());
    mk(BC,  tMap.cnt.size() * 4, tMap.cnt.data());
    mk(BI,  tMap.idx.size() * 4, tMap.idx.data());
    mk(BTX, (size_t)nTile * 2 * 4, nullptr);
    mk(BFW, npix * maxPl * 4 * 4, nullptr);
    mk(BPC, npix * 3 * 4, nullptr);
    mk(BGR, p.size() * 4, nullptr);          // grad 初始 0
    mk(BMM, p.size() * 4, mom.data());
    mk(BST, 4 * 4, nullptr);          // stat[4] = {loss*1e5, l2*1e5, nv, touchCnt}（4 个 uint，勿缩）
    mk(BMK, (size_t)NS * 4, nullptr);
    mk(BTC, (size_t)NS * 4, nullptr);
    mk(BGA, ga.size() * 4, ga.data());
    mk(BVL, p.size() * 4, vel.data());
    // grad 显式清零
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, sbo[BGR]);
    std::vector<float> zero(p.size(), 0.0f);
    glBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, (GLsizeiptr)(zero.size() * 4), zero.data());

    // uniform 定位
    GLuint uF_W=glGetUniformLocation(progFwd,"uImgW"), uF_H=glGetUniformLocation(progFwd,"uImgH"),
           uF_T=glGetUniformLocation(progFwd,"uTCOLS"), uF_M=glGetUniformLocation(progFwd,"uMaxPl"),
           uF_Md=glGetUniformLocation(progFwd,"uModel");
    GLuint uB_W=glGetUniformLocation(progBwd,"uImgW"), uB_H=glGetUniformLocation(progBwd,"uImgH"),
           uB_T=glGetUniformLocation(progBwd,"uTCOLS"), uB_M=glGetUniformLocation(progBwd,"uMaxPl"),
           uB_L=glGetUniformLocation(progBwd,"uL2"), uB_Md=glGetUniformLocation(progBwd,"uModel");
    GLuint uZ_N=glGetUniformLocation(progZero,"uN");
    GLuint uA_T=glGetUniformLocation(progAdam,"uTouch"), uA_I=glGetUniformLocation(progAdam,"uInvNv"),
           uA_L=glGetUniformLocation(progAdam,"uLrScale"), uA_C1=glGetUniformLocation(progAdam,"uBc1"),
           uA_C2=glGetUniformLocation(progAdam,"uBc2"), uA_K=glGetUniformLocation(progAdam,"kLr"),
           uA_W=glGetUniformLocation(progAdam,"uImgW"), uA_H=glGetUniformLocation(progAdam,"uImgH");
    const float kLrG[9] = { 0.010f,0.010f,0.004f,0.004f,0.010f,0.020f,0.020f,0.020f,0.030f };
    glUseProgram(progAdam); glUniform1fv(uA_K, 9, kLrG);

    auto bindAll = [&]() {
        glUseProgram(progZero);
        for (int i = 0; i < 16; ++i) glBindBufferBase(GL_SHADER_STORAGE_BUFFER, i, sbo[i]);
        glUniform1ui(uZ_N, (GLuint)NS);
        glUseProgram(progFwd);
        glUniform1ui(uF_W, (GLuint)W); glUniform1ui(uF_H, (GLuint)H);
        glUniform1ui(uF_T, (GLuint)TCOLS); glUniform1ui(uF_M, maxPl);
        glUniform1i(uF_Md, g_model);
        glUseProgram(progBwd);
        glUniform1ui(uB_W, (GLuint)W); glUniform1ui(uB_H, (GLuint)H);
        glUniform1ui(uB_T, (GLuint)TCOLS); glUniform1ui(uB_M, maxPl);
        glUniform1i(uB_L, l2loss);
        glUniform1i(uB_Md, g_model);
        glUseProgram(progAdam);
        glUniform1ui(uA_W, (GLuint)W); glUniform1ui(uA_H, (GLuint)H);
    };
    bindAll();

    std::vector<uint32_t> txy((size_t)nTile * 2);
    std::uniform_int_distribution<int> txr(0, (W + TSg - 1) / TSg - 1), tyr(0, (H + TSg - 1) / TSg - 1);

    // 计时与进度
    double t0 = NowSec(); float lossEma = 0; int prevPct = -1, prevBmpPct = -1;
    double prof_pass = 0;   // GPU pass 耗时
    double dispPsnr = 0;    // 进度显示 PSNR：每 previewPct 用全图 Evaluate 刷新（采样 l1 无法直接转 PSNR）
    for (int it = startIt + 1; it <= iters; ++it) {
        double tIt0 = NowSec();
        // 1) 随机 tile 采样 + 上传
        for (int b = 0; b < nTile; ++b) { txy[(size_t)b * 2] = (uint32_t)txr(rng); txy[(size_t)b * 2 + 1] = (uint32_t)tyr(rng); }
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, sbo[BTX]);
        glBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, (GLsizeiptr)(txy.size() * 4), txy.data());
        // 2) zero -> fwd -> bwd -> adam（每个 dispatch 间加 barrier，保证前一个 pass 的
        //    SSBO 写入对下一个 pass 可见；缺 barrier 在部分驱动上 stat/fwd 数据错乱）
        glUseProgram(progZero); glDispatchCompute((GLuint)((NS + 255) / 256), 1, 1);
        glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
        glUseProgram(progFwd);  glDispatchCompute((GLuint)nTile, 1, 1);
        glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
        glUseProgram(progBwd);  glDispatchCompute((GLuint)nTile, 1, 1);
        glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
        // 3) 读回统计（loss/l2/nv/touchCnt）：glFinish 等 fwd/bwd 写完 stat 再 map 读
        glFinish();
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, sbo[BST]);
        uint32_t st3[4] = {};
        void* sp = glMapBufferRange(GL_SHADER_STORAGE_BUFFER, 0, 16, GL_MAP_READ_BIT);
        if (sp) memcpy(st3, sp, 16); glUnmapBuffer(GL_SHADER_STORAGE_BUFFER);
        uint32_t touch = st3[3];
        // 4) adam（只处理 touch 的 splat）
        glUseProgram(progAdam);
        glUniform1ui(uA_T, touch);
        glUniform1f(uA_I, touch > 0 ? 1.0f / std::max(1, (int)st3[2]) : 0.0f);
        glUniform1f(uA_L, 0.3f + 0.7f * (1.0f - (float)it / iters));
        float bc1 = 1.0f - powf(0.9f, (float)it), bc2 = 1.0f - powf(0.999f, (float)it);
        glUniform1f(uA_C1, bc1); glUniform1f(uA_C2, bc2);
        if (touch > 0) glDispatchCompute((GLuint)((touch + 255) / 256), 1, 1);
        glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT | GL_BUFFER_UPDATE_BARRIER_BIT);
        prof_pass += NowSec() - tIt0;

        // 5) 损失/进度显示（loss = L1 采样值，psnr = L2 每通道口径 EMA，与 CPU 一致）
        float l1 = (float)st3[0] / 1e5f / std::max(1u, st3[2]);
        float l2e = (float)st3[1] / 1e5f / std::max(1u, st3[2]);
        lossEma = (it == 1) ? l1 : (0.95f * lossEma + 0.05f * l1);
        double psnrEma = 10.0 * log10(3.0 / std::max((double)l2e, 1e-6));
        int pct = (int)(100.0 * it / iters);
        if (pct != prevPct && pct % 2 == 0) {
            prevPct = pct;
            double dt = NowSec() - t0, eta = dt * (iters - it) / std::max(1, it);
            double ps = dispPsnr > 0 ? dispPsnr : psnrEma;   // 有全图评估用全图值，否则用采样 EMA
            printf("\r[%s] %3d%% iter=%d/%d  l1=%.4f psnr=%.1fdb  splats=%d  dt=%.0fs  eta=%.0fs [gpu %.2fms]",
                   std::string(pct / 5, '=').c_str(), pct, it, iters, lossEma, ps, NS, dt, eta, prof_pass / it * 1000);
            fflush(stdout);
        }
        if (previewPct > 0 && pct / previewPct != prevBmpPct && pct >= previewPct) {
            prevBmpPct = pct / previewPct;
            // 预览需要 CPU splats：读回 p 后 Evaluate（只读回，不改 GPU 状态）
            glBindBuffer(GL_SHADER_STORAGE_BUFFER, sbo[BP]);
            void* mp = glMapBufferRange(GL_SHADER_STORAGE_BUFFER, 0, (GLsizeiptr)(p.size() * 4), GL_MAP_READ_BIT);
            if (mp) {
                memcpy(p.data(), mp, p.size() * 4); glUnmapBuffer(GL_SHADER_STORAGE_BUFFER);
                for (int i = 0; i < NS; ++i) {
                    std::array<float, 9> q = { p[(size_t)i*9], p[(size_t)i*9+1], p[(size_t)i*9+2], p[(size_t)i*9+3],
                                               p[(size_t)i*9+4], p[(size_t)i*9+5], p[(size_t)i*9+6], p[(size_t)i*9+7], p[(size_t)i*9+8] };
                    SplatFromParams(splats[i], q);
                }
                double mE = Evaluate(nullptr, "fitsplat_progress.bmp");
                dispPsnr = 10.0 * log10(1.0 / std::max(mE, 1e-6));
            }
        }

        // 6) 密度控制：每 100 迭代读回参数跑 CPU Densify，重传（数据结构调整只能在 CPU 做）
        if (it % 100 == 0) {
            glBindBuffer(GL_SHADER_STORAGE_BUFFER, sbo[BP]);
            void* mp = glMapBufferRange(GL_SHADER_STORAGE_BUFFER, 0, (GLsizeiptr)(p.size() * 4), GL_MAP_READ_BIT);
            if (mp) memcpy(p.data(), mp, p.size() * 4);
            glUnmapBuffer(GL_SHADER_STORAGE_BUFFER);
            glBindBuffer(GL_SHADER_STORAGE_BUFFER, sbo[BGA]);
            void* mg = glMapBufferRange(GL_SHADER_STORAGE_BUFFER, 0, (GLsizeiptr)(ga.size() * 4), GL_MAP_READ_BIT);
            if (mg) memcpy(ga.data(), mg, ga.size() * 4);
            glUnmapBuffer(GL_SHADER_STORAGE_BUFFER);
            // 动量/二阶矩也在 GPU 上累积，读回才能避免重传时用陈旧 mAdam 重置 Adam 状态
            glBindBuffer(GL_SHADER_STORAGE_BUFFER, sbo[BMM]);
            void* mm = glMapBufferRange(GL_SHADER_STORAGE_BUFFER, 0, (GLsizeiptr)(mom.size() * 4), GL_MAP_READ_BIT);
            if (mm) memcpy(mom.data(), mm, mom.size() * 4);
            glUnmapBuffer(GL_SHADER_STORAGE_BUFFER);
            glBindBuffer(GL_SHADER_STORAGE_BUFFER, sbo[BVL]);
            void* mv = glMapBufferRange(GL_SHADER_STORAGE_BUFFER, 0, (GLsizeiptr)(vel.size() * 4), GL_MAP_READ_BIT);
            if (mv) memcpy(vel.data(), mv, vel.size() * 4);
            glUnmapBuffer(GL_SHADER_STORAGE_BUFFER);
            for (int i = 0; i < NS; ++i) {
                std::array<float, 9> q = { p[(size_t)i*9], p[(size_t)i*9+1], p[(size_t)i*9+2], p[(size_t)i*9+3],
                                           p[(size_t)i*9+4], p[(size_t)i*9+5], p[(size_t)i*9+6], p[(size_t)i*9+7], p[(size_t)i*9+8] };
                SplatFromParams(splats[i], q);
                gradAcc[i][0] = ga[(size_t)i * 2]; gradAcc[i][1] = ga[(size_t)i * 2 + 1];
                for (int c = 0; c < 9; ++c) { mAdam[i][c] = mom[(size_t)i * 9 + c]; vAdam[i][c] = vel[(size_t)i * 9 + c]; }
            }
            Densify(prog ? (size_t)progK : maxK, rng, prog != 0);
            if (prog) progK = std::min((int)maxK, progK + progStep);
            for (auto& g : gradAcc) { g[0] = 0; g[1] = 0; }
            BuildTileMap(splats, TSg, tMap);
            TCOLS = tMap.TCOLS;
            maxPl = 0; for (uint32_t c : tMap.cnt) maxPl = std::max(maxPl, c);
            NS = (int)splats.size();
            // 重传 p/mom/vel/ga（NS 已变）+ 重建 tile buffer + grad/mark/tch 重分配
            p.resize((size_t)NS * 9); mom.resize((size_t)NS * 9); vel.resize((size_t)NS * 9); ga.resize((size_t)NS * 2);
            for (int i = 0; i < NS; ++i) {
                std::array<float, 9> q = { splats[i].mx, splats[i].my, logf(splats[i].sx), logf(splats[i].sy),
                                           splats[i].rot, splats[i].r, splats[i].g, splats[i].b, Logit(splats[i].a) };
                for (int c = 0; c < 9; ++c) { p[(size_t)i * 9 + c] = q[c]; mom[(size_t)i * 9 + c] = mAdam[i][c]; vel[(size_t)i * 9 + c] = vAdam[i][c]; }
                ga[(size_t)i * 2] = 0; ga[(size_t)i * 2 + 1] = 0;
            }
            auto up = [&](int b, size_t bytes, const void* d) {
                glBindBuffer(GL_SHADER_STORAGE_BUFFER, sbo[b]);
                glBufferData(GL_SHADER_STORAGE_BUFFER, (GLsizeiptr)bytes, d, GL_DYNAMIC_DRAW);
            };
            std::vector<float> zeroG(p.size(), 0.0f);   // NS 变化后按新大小重建清零缓冲
            up(BP, p.size() * 4, p.data());
            up(BMM, p.size() * 4, mom.data());
            up(BVL, p.size() * 4, vel.data());
            up(BGA, ga.size() * 4, ga.data());
            up(BGR, p.size() * 4, zeroG.data());
            up(BO, tMap.off.size() * 4, tMap.off.data());
            up(BC, tMap.cnt.size() * 4, tMap.cnt.data());
            up(BI, tMap.idx.size() * 4, tMap.idx.data());
            std::vector<uint32_t> mz((size_t)NS, 0u);
            up(BMK, mz.size() * 4, mz.data());
            up(BTC, mz.size() * 4, mz.data());
            // splat 分裂可能让某些 tile 覆盖更多 splat（maxPl 增大），fwd buffer 按新 maxPl 重分配防越界
            if (maxPl > maxPlInit) {
                up(BFW, npix * maxPl * 4 * 4, nullptr);
                maxPlInit = maxPl;
            }
            glUniform1ui(uZ_N, (GLuint)NS);
            glUseProgram(progFwd); glUniform1ui(uF_T, (GLuint)TCOLS); glUniform1ui(uF_M, maxPl); glUniform1i(uF_Md, g_model);
            glUseProgram(progBwd); glUniform1ui(uB_T, (GLuint)TCOLS); glUniform1ui(uB_M, maxPl); glUniform1i(uB_Md, g_model);
            glUseProgram(progAdam); glUniform1ui(uA_W, (GLuint)W); glUniform1ui(uA_H, (GLuint)H);
            printf("\n[gputrain] 密度控制 iter=%d splats=%d\n", it, NS);
        }

        // checkpoint（GPU 版每 ckptEvery 迭代读回参数保存，供 --resume 继续）
        if (!ckptPath.empty() && it % ckptEvery == 0) {
            glBindBuffer(GL_SHADER_STORAGE_BUFFER, sbo[BP]);
            void* mp = glMapBufferRange(GL_SHADER_STORAGE_BUFFER, 0, (GLsizeiptr)(p.size() * 4), GL_MAP_READ_BIT);
            if (mp) memcpy(p.data(), mp, p.size() * 4);
            glUnmapBuffer(GL_SHADER_STORAGE_BUFFER);
            for (int i = 0; i < NS; ++i) {
                std::array<float, 9> q = { p[(size_t)i*9], p[(size_t)i*9+1], p[(size_t)i*9+2], p[(size_t)i*9+3],
                                           p[(size_t)i*9+4], p[(size_t)i*9+5], p[(size_t)i*9+6], p[(size_t)i*9+7], p[(size_t)i*9+8] };
                SplatFromParams(splats[i], q);
            }
            // 复用 main 里的 checkpoint 保存逻辑（独立实现，避免依赖 main 作用域 lambda）
            FILE* f = nullptr; fopen_s(&f, ckptPath.c_str(), "wb");
            if (f) {
                uint32_t magic = 0x46535350u; int ns = (int)splats.size();
                fwrite(&magic, 4, 1, f); fwrite(&it, 4, 1, f); fwrite(&ns, 4, 1, f);
                for (int i = 0; i < ns; ++i) {
                    std::array<float, 9> q = { splats[i].mx, splats[i].my, logf(splats[i].sx), logf(splats[i].sy),
                                               splats[i].rot, splats[i].r, splats[i].g, splats[i].b, Logit(splats[i].a) };
                    fwrite(q.data(), 4, 9, f);
                    fwrite(mAdam[i].data(), 4, 9, f);
                    fwrite(vAdam[i].data(), 4, 9, f);
                    fwrite(gradAcc[i].data(), 4, 2, f);
                }
                fclose(f);
                printf("\n[gputrain] checkpoint it=%d splats=%d -> %s\n", it, ns, ckptPath.c_str());
            }
        }
    }
    printf("\n");

    // 训练结束：读回最终参数到全局 splats（供 main 后续 Evaluate/输出 GLSL）
    p.resize((size_t)NS * 9);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, sbo[BP]);
    void* mp = glMapBufferRange(GL_SHADER_STORAGE_BUFFER, 0, (GLsizeiptr)(p.size() * 4), GL_MAP_READ_BIT);
    if (mp) memcpy(p.data(), mp, p.size() * 4);
    glUnmapBuffer(GL_SHADER_STORAGE_BUFFER);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, sbo[BGA]);
    void* mg = glMapBufferRange(GL_SHADER_STORAGE_BUFFER, 0, (GLsizeiptr)(ga.size() * 4), GL_MAP_READ_BIT);
    if (mg) memcpy(ga.data(), mg, ga.size() * 4);
    glUnmapBuffer(GL_SHADER_STORAGE_BUFFER);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, sbo[BMM]);
    void* mm2 = glMapBufferRange(GL_SHADER_STORAGE_BUFFER, 0, (GLsizeiptr)(mom.size() * 4), GL_MAP_READ_BIT);
    if (mm2) memcpy(mom.data(), mm2, mom.size() * 4);
    glUnmapBuffer(GL_SHADER_STORAGE_BUFFER);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, sbo[BVL]);
    void* mv2 = glMapBufferRange(GL_SHADER_STORAGE_BUFFER, 0, (GLsizeiptr)(vel.size() * 4), GL_MAP_READ_BIT);
    if (mv2) memcpy(vel.data(), mv2, vel.size() * 4);
    glUnmapBuffer(GL_SHADER_STORAGE_BUFFER);
    for (int i = 0; i < NS; ++i) {
        std::array<float, 9> q = { p[(size_t)i*9], p[(size_t)i*9+1], p[(size_t)i*9+2], p[(size_t)i*9+3],
                                   p[(size_t)i*9+4], p[(size_t)i*9+5], p[(size_t)i*9+6], p[(size_t)i*9+7], p[(size_t)i*9+8] };
        SplatFromParams(splats[i], q);
        gradAcc[i][0] = ga[(size_t)i * 2]; gradAcc[i][1] = ga[(size_t)i * 2 + 1];
        for (int c = 0; c < 9; ++c) { mAdam[i][c] = mom[(size_t)i * 9 + c]; vAdam[i][c] = vel[(size_t)i * 9 + c]; }
    }
    if ((int)splats.size() > NS) splats.resize(NS);
    else if ((int)splats.size() < NS) { splats.resize(NS); mAdam.resize(NS); vAdam.resize(NS); gradAcc.resize(NS); }
    printf("[gputrain] 训练结束: splats=%d 平均 %.2f ms/迭代\n", NS, prof_pass / std::max(1, iters - startIt) * 1000);

    for (int i = 0; i < 16; ++i) if (sbo[i]) glDeleteBuffers(1, &sbo[i]);
    glDeleteProgram(progZero); glDeleteProgram(progFwd); glDeleteProgram(progBwd); glDeleteProgram(progAdam);
    glfwTerminate();
}

int main(int argc, char** argv) {
#ifdef _WIN32
    SetConsoleOutputCP(CP_UTF8);   // 中文日志按 UTF-8 输出，避免 GBK(936) 终端乱码
    // 正常优先级：用户要求 CPU 压榨（此前 BELOW_NORMAL 导致调度份额低、CPU 利用率只有 ~20%）
    SetPriorityClass(GetCurrentProcess(), NORMAL_PRIORITY_CLASS);
#endif
    if (argc < 2) { PrintUsage(); return 1; }
    std::string inPath = argv[1];
    int    K0     = 600, iters = 1000, width = 320, height = 0, batch = 2048, threads = 0, seed = 42, previewPct = 10, embed = 0, l1only = 0, l2loss = 0;
    int    tilecap = 1024;   // 输出产物每 tile 索引上限；0 = 不截断（渲染全部 splat，画质最佳）
    int    dumpTarget = 0;   // 调试：dump target 像素（float）到 target.raw
    int    residual = 0;     // embed 产物附加"残差修正层"（4bit/通道 ±64，叠加补高频细节，可再提 ~7dB）
    int    gpu = 0;          // 1=训练完成后跑 GPU 前向 tile 渲染基准（OpenGL 4.3 compute，需核显支持）
    int    gputrain = 0;     // 1=训练模式切到 GPU compute（正传/反传/Adam 全 GPU，密度控制 CPU 化）
    int    initMode = 0;     // 0=网格（默认） 1=梯度引导（内容自适应，细节区集中分配）
    int    prog = 0;         // 1=误差引导渐进优化（初始 50% splat，逐步在欠拟合区补残差 splat，预算=--splats）
    float  bg     = 0.0f;
    float  targetPsnr = 0.0f;   // --target：EMA PSNR 达到该值即提前停止训练（0 = 不启用）
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
        else if (k == "--target")   targetPsnr = (float)atof(next());   // 达到目标 PSNR 提前停止
        else if (k == "--gputrain") gputrain = atoi(next());            // 训练切到 GPU compute
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
        else if (k == "--gpu")      gpu = atoi(next());       // 1=训练完成后跑 GPU 前向 tile 渲染基准
        else if (k == "--init")     initMode = atoi(next());  // 0=网格 1=梯度引导（内容自适应）
        else if (k == "--prog")     prog = atoi(next());      // 1=误差引导渐进优化
        else if (k == "--init-scale") gInitScale = (float)atof(next());  // 初始高斯尺度（像素）
        else if (k == "--model")    g_model = atoi(next());   // 渲染模型：1=加和（默认），0=alpha 合成
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

    // ---------- 初始化：网格铺开 / 梯度引导（内容自适应） ----------
    std::mt19937 rng((unsigned)seed);
    int initK = prog ? std::max(16, (int)ceil(K0 * 0.5)) : K0;   // 渐进模式先只铺 50%
    splats.clear(); mAdam.clear(); vAdam.clear(); gradAcc.clear();
    if (initMode == 1) {
        InitSplatGradient(initK, rng);
    } else {
        int n = (int)ceil(sqrt((double)initK));
        float cellX = (float)W / n, cellY = (float)H / n;
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
    int   lastPct10 = 0;   // --target 检查点：每跨过总迭代 10% 检查一次（固定 10 次全图评估，与迭代总数无关）
    // 密度控制上限：非渐进 = 3x 初始（现有语义）；渐进模式 --splats 即总预算，目标从 50% 逐步长到 K0
    size_t maxK = (size_t)(prog ? K0 : K0 * 3);
    int progK = initK;   // 渐进模式当前目标 splat 数（每 100 迭代增长一批）
    int progStep = prog ? std::max(1, (int)ceil((double)((int)maxK - initK) / 20)) : 0;

    // tile 分块（训练核心加速：每像素只遍历本 tile 的 splat，剔除 99% 无关 splat）
    TileMap tMap;
    BuildTileMap(splats, TS, tMap);

    // 分段 profile：定位迭代时间分布（tile 正传反传 / Adam / 其他）
    double prof_tile = 0, prof_adam = 0, prof_other = 0;

    for (int it = startIt + 1; it <= iters; ++it) {
        // GPU 训练模式：正传/反传/Adam 全部在 compute shader 执行，跑完后直接收尾
        if (gputrain) {
            GpuTrainLoop(startIt, iters, nTile, maxK, previewPct, ckptPath, ckptEvery, l2loss, rng, prog, initK);
            it = iters;   // 循环结束（GpuTrainLoop 内部跑完所有迭代）
            break;
        }
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

        // 每通道口径 EMA PSNR（mseEma 是三通道和，除以 3 与全图 Evaluate 的 PSNR 口径一致）
        double psnr = 10.0 * log10(3.0 / std::max(mseEma, 1e-6f));
        double dispPsnr = psnr;   // 进度条显示值：每 200 迭代刷新为全图真实 PSNR，避免 EMA 采样口径虚高

        // 提前停止：每跨过总迭代 10% 检查一次全图真实 PSNR（Evaluate 口径，随机采样 tile 的
        // EMA 会系统性偏高 10+ dB，不能用于目标判断），达到 --target 即结束。
        // 动态 10% 粒度：15K 迭代 ≈ 每 1500 次检查，50K 迭代 ≈ 每 5000 次检查，
        // 全图评估固定只跑 ~10 次，不会随迭代数放大开销
        int pctNow = (int)(100.0 * it / iters);
        if (targetPsnr > 0 && pctNow >= lastPct10 + 10) {
            lastPct10 = pctNow - pctNow % 10;
            double mEval = Evaluate();   // 全图 MSE（每通道）
            double pEval = 10.0 * log10(1.0 / std::max(mEval, 1e-6));
            dispPsnr = pEval;
            if (pEval >= targetPsnr) {
                printf("\n[目标] iter=%d 全图 PSNR=%.2f dB 达到目标 %.1f dB，提前停止训练\n", it, pEval, targetPsnr);
                iters = it;   // 后续 saveCkpt / 最终评估以实际迭代数为准
                break;
            }
        }

        // 进度反馈
        int pct = (int)(100.0 * it / iters);
        if (pct != prevPct && pct % 2 == 0) {
            prevPct = pct;
            double dt = NowSec() - t0;
            double eta = dt * (iters - it) / std::max(1, it);
            int barN = 20, fill = (int)(barN * pct / 100.0);
            printf("\r[%s] %3d%% iter=%d/%d  l1=%.4f psnr=%.1fdb  splats=%d  dt=%.0fs  eta=%.0fs",
                   std::string(fill, '=').c_str(), pct, it, iters, lossEma, dispPsnr, (int)splats.size(), dt, eta);
            if (it >= 50) printf("  [tile %.2fms adam %.2fms other %.2fms]", prof_tile / it * 1000, prof_adam / it * 1000, prof_other / it * 1000);
            fflush(stdout);
        }
        if (previewPct > 0 && pct / previewPct != prevBmpPct && pct >= previewPct) {
            prevBmpPct = pct / previewPct;
            Evaluate(nullptr, "fitsplat_progress.bmp");
        }

        // 密度控制
        if (it % 100 == 0) {
            Densify(prog ? (size_t)progK : maxK, rng, prog != 0);
            if (prog) progK = std::min((int)maxK, progK + progStep);
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

    // ---------- GPU 前向 tile 渲染基准（--gpu）----------
    if (gpu) {
        printf("\n---- GPU 前向 tile 渲染基准（--gpu）----\n");
        GpuForwardTest();
    }

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
            fprintf(f, "// MODEL=%s\n", g_model == 1 ? "additive" : "alpha");   // fitsplat_gl 按此标识自动选择混合方式
            fprintf(f, "// 由 fitsplat 生成：%d 个高斯泼溅（%s + tile 分块剔除）拟合图片\n",
                    (int)splats.size(), g_model == 1 ? "加和模型" : "alpha 合成模型");
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
                if (g_model == 0) fprintf(f, "    float T = 1.0;\n");
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
                if (g_model == 0) fprintf(f, "    float T = 1.0;\n");
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
            if (g_model == 1) {
                fprintf(f, "        acc += sb.rgb * al;   // 加和：顺序无关，无 transmittance\n");
            } else {
                fprintf(f, "        acc += sb.rgb * (al * T);\n");
                fprintf(f, "        T *= (1.0 - al);\n");
            }
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
