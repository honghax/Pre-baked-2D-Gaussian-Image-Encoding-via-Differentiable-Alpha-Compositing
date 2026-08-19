// fitsplat_gl — 3DGS 实例化四边形渲染器（业界标准做法）
//
// 与 playpiano_gl 底层同架构：每个 splat 渲染为一个实例化四边形
//   [顶点着色器] 按 gl_InstanceID 从参数纹理取 SA/SB/SC，计算 3σ 包围盒
//                四边形角点（旋转椭圆外接），展开为 2 个三角形
//   [片元着色器] 只对四边形内像素算高斯 g，输出 (颜色, 不透明度*a*g)
//   [混合]       GL_SRC_ALPHA / GL_ONE_MINUS_SRC_ALPHA 标准 alpha 合成，
//                渲染顺序 = 实例顺序 = 训练时的合成顺序，逐项复现
//   [硬件]       光栅化自动剔除无关像素 —— O(覆盖面积) 而非 O(像素*splat数)
//
// 数据：加载与 input.glsl 同目录的 input_splatA/B/C.raw（RGBA32F，宽=NS 高=1）
// 画布：IMG_W/IMG_H 从 input.glsl 的常量解析
// 窗口：1920x1080，ESC 退出

#ifndef NOMINMAX
#define NOMINMAX   // 阻止 windows.h 定义 min/max 宏，避免与 std::min/std::max 冲突
#endif
#include <windows.h>

#include <glad/glad.h>
#include <GLFW/glfw3.h>

#include <cstdio>
#include <cmath>
#include <string>
#include <vector>
#include <cstring>
#include <chrono>

static long long NowMs()
{
    using namespace std::chrono;
    return duration_cast<milliseconds>(steady_clock::now().time_since_epoch()).count();
}

static inline float ClampF(float x, float lo, float hi) { return x < lo ? lo : (x > hi ? hi : x); }

static int g_winW = 1920, g_winH = 1080;
static float g_imgW = 320.f, g_imgH = 180.f;
static std::vector<unsigned> g_resVals;   // 残差修正层 kRes（供 ExportAll 叠加，与 pass2 逻辑一致）

// ---- 工具 ----
static std::string ExeDir()
{
    char buf[MAX_PATH];
    DWORD n = GetModuleFileNameA(nullptr, buf, MAX_PATH);
    std::string p(buf, n);
    size_t s = p.find_last_of("\\/");
    return s == std::string::npos ? "." : p.substr(0, s);
}

static bool ReadFileBin(const std::string& path, std::vector<unsigned char>& out)
{
    FILE* f = nullptr;
    if (fopen_s(&f, path.c_str(), "rb") != 0 || !f) return false;
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    out.resize((size_t)sz);
    size_t got = fread(out.data(), 1, (size_t)sz, f);
    fclose(f);
    out.resize(got);
    return got > 0;
}

// 解析 input.glsl 中的 "IMG_W = 320.0" 常量
static bool ParseImgSize(const std::string& glsl, float& w, float& h)
{
    auto findFloat = [&](const char* key, float& v) {
        std::string k = key;
        size_t p = glsl.find(k);
        if (p == std::string::npos) return false;
        p = glsl.find('=', p);
        if (p == std::string::npos) return false;
        v = (float)atof(glsl.c_str() + p + 1);
        return v > 0.f;
    };
    return findFloat("IMG_W", w) && findFloat("IMG_H", h);
}

// 从无纹理纯 GLSL（--embed）解析 const vec4 kName[N] = vec4[N](vec4(f,f,f,f), ...);
static bool ParseVec4Array(const std::string& src, const char* name, std::vector<float>& out)
{
    std::string pat = name;
    pat += "[";
    size_t p = src.find(pat);
    if (p == std::string::npos) return false;
    p = src.find('=', p);
    if (p == std::string::npos) return false;
    size_t end = src.find(';', p);           // 数组初始化以 ");" 结尾，限定解析范围
    if (end == std::string::npos) return false;
    p = src.find("vec4(", p);                // 定位第一个 vec4 元素（声明是 vec4[N](，不匹配）
    while (p != std::string::npos && p < end) {
        size_t r = src.find(')', p);
        if (r == std::string::npos || r >= end) break;
        float v[4];
        if (sscanf_s(src.c_str() + p + 5, "%f,%f,%f,%f", &v[0], &v[1], &v[2], &v[3]) != 4) break;
        out.insert(out.end(), v, v + 4);
        p = src.find("vec4(", r);
    }
    return out.size() >= 4;
}

// 从无纹理纯 GLSL（新版 --embed，uint 打包压缩）解析 const uint kData[N] = uint[N](...);
// CPU 端解码：把每 splat 4 个 uint 反量化为参数（与 fitsplat.cpp 的导出格式一一对应）——
// 渲染程序当"解码器"用，GPU 端只编译小着色器，避免驱动编译 34MB 常量数组崩溃。
static bool ParseUintArray(const std::string& src, const char* name, std::vector<unsigned>& out)
{
    std::string pat = name;
    pat += "[";
    size_t p = src.find(pat);
    if (p == std::string::npos) return false;
    p = src.find('=', p);
    if (p == std::string::npos) return false;
    size_t end = src.find(");", p);          // 数组初始化以 ");" 结尾（全文件后续数组无分号，不能用 ';' 定位）
    if (end == std::string::npos) return false;
    size_t lp = src.find('(', p);            // 跳过 "uint[N](" 里的数组长度 N，从 '(' 后才是数据
    if (lp == std::string::npos || lp >= end) return false;
    const char* s = src.c_str() + lp + 1;
    const char* e = src.c_str() + end;
    while (s < e) {
        while (s < e && (*s < '0' || *s > '9')) ++s;   // 跳过非数字（逗号/换行/u 后缀/括号）
        if (s >= e) break;
        unsigned v = 0;
        while (s < e && *s >= '0' && *s <= '9') { v = v * 10u + (unsigned)(*s - '0'); ++s; }
        out.push_back(v);
    }
    return out.size() >= 4;
}

// 解析 "const float kMXS = 81.0;" 一类的浮点常量（embed 位置量化系数，按画布自适应；
// 旧版文件无此常量，调用方用默认值 128/64 兼容）
static bool ParseFloatConst(const std::string& src, const char* name, float& v)
{
    std::string pat = name;
    pat += " =";
    size_t p = src.find(pat);
    if (p == std::string::npos) return false;
    v = (float)atof(src.c_str() + p + pat.size());
    return v > 0.f;
}

// ---- GLSL ----
static const char* kCommonVS = R"(
#version 330 core
uniform vec2 uRes;
uniform vec2 uPlateSz;
uniform vec2 uOff;
vec2 qToClip(vec2 q){
    return vec2((uOff.x + q.x*uPlateSz.x) / uRes.x * 2.0 - 1.0,
                1.0 - 2.0*(uOff.y + q.y*uPlateSz.y) / uRes.y);
}
)";

static const char* kCommonFS = R"(
#version 330 core
uniform vec2 uRes;
uniform vec2 uPlateSz;
uniform vec2 uOff;
vec2 fragQ(){
    vec2 fc = gl_FragCoord.xy;
    return (vec2(fc.x, uRes.y - fc.y) - uOff) / uPlateSz;
}
vec2 qToClip(vec2 q){
    return vec2((uOff.x + q.x*uPlateSz.x) / uRes.x * 2.0 - 1.0,
                1.0 - 2.0*(uOff.y + q.y*uPlateSz.y) / uRes.y);
}
)";

static const char* kVSFull = R"(
layout(location = 0) in vec2 aPos;
void main(){ gl_Position = vec4(aPos, 0.0, 1.0); }
)";

// 背景：纯黑（图片透明区合成到黑）
static const char* kFSBg = R"(
out vec4 FragColor;
void main(){ FragColor = vec4(0.0, 0.0, 0.0, 1.0); }
)";

// splat 实例化四边形：顶点取参数算包围盒
static const char* kVSSplat = R"(
uniform sampler2D uSpl;   // NS*3 个 texel 展平为 uTexW 宽的二维纹理
uniform int    uTexW;     // 展平宽度（<< GL_MAX_TEXTURE_SIZE，避免大 NS 纹理单维超限）
uniform int    uBase;     // 动画增量模式：实例偏移（默认 0 = 从第 0 个开始）
uniform float uTime;
uniform float uCap;   // 包围盒偏移上限（放大坐标系下 = 长边*scale）
uniform float uScale; // --scale N：高清直渲放大系数（画布坐标 ×N，GPU 直接在高分辨率下栅格化）
out vec2 vMu; out vec2 vSC; out vec2 vS; out vec3 vCol; out float vA;
vec4 fetchParam(int ti){ return texelFetch(uSpl, ivec2(ti % uTexW, ti / uTexW), 0); }
void main(){
    int id = gl_InstanceID + uBase;
    vec4 A = fetchParam(id * 3 + 0);   // (mx,my,sx,sy)
    vec4 B = fetchParam(id * 3 + 1);   // (r,g,b,a)
    vec4 C = fetchParam(id * 3 + 2);   // (cos,sin,0,0)

    // 包围盒：3σ 旋转椭圆外接四边形（与训练 RenderPixel 的 3σ 剔除一致；
    // 4σ 会把 3σ~4σ 的高斯尾部也渲染出来，导致画面变灰、PSNR 下降）
    const float T = 3.0;
    vec2 c = vec2(float((gl_VertexID<<1)&2), float(gl_VertexID&2)) * 0.5; // 0..1
    vec2 f = c * 2.0 - 1.0;                       // -1..1
    float ex = T * f.x, ey = T * f.y;
    // [ex*sx, ey*sy] = R [dx, dy]  ->  [dx,dy] = R^T [ex*sx, ey*sy]
    float dx = C.x * ex * A.z - C.y * ey * A.w;
    float dy = C.y * ex * A.z + C.x * ey * A.w;
    // 偏移上限保护（对齐 playpiano 的 ex/ey clamp：仅防异常 splat，正常范围不受影响）
    vec2 off = vec2(dx, dy);
    float lo = length(off);
    if (lo > uCap) off *= uCap / lo;
    // 高清直渲：位置/尺寸/包围盒整体乘 uScale，在放大坐标系下栅格化（连续函数的高密度采样）
    vec2 q = uScale * (A.xy + off);

    vMu = uScale * A.xy; vSC = C.xy; vS = uScale * A.zw; vCol = B.rgb; vA = B.w;
    gl_Position = vec4(qToClip(q), 0.0, 1.0);
}
)";

// splat 片元：四边形内算高斯，输出 premultiplied 颜色 + 不透明度
static const char* kFSSplat = R"(
out vec4 FragColor;
in vec2 vMu; in vec2 vSC; in vec2 vS; in vec3 vCol; in float vA;
void main(){
    vec2 d = fragQ() - vMu;
    float ex = (vSC.x*d.x + vSC.y*d.y) / vS.x;
    float ey = (-vSC.y*d.x + vSC.x*d.y) / vS.y;
    float g = exp(-0.5*(ex*ex + ey*ey));
    FragColor = vec4(vCol, vA * g);   // 标准 alpha 合成（SRC_ALPHA / ONE_MINUS_SRC_ALPHA）
}
)";

// 最终输出：clamp 到 [0,1]（合成可能有轻微超界）
// pass2：缓存 FBO（画布尺寸）-> 窗口，letterbox 采样（uRes=窗口尺寸，
// fragQ 返回画布坐标；uImgSz=画布尺寸；纹理行序是"底部优先"，需翻转 y）
static const char* kFSPresent = R"(
out vec4 FragColor;
uniform sampler2D uTex;
uniform sampler2D uResTex;   // 残差修正层（普通 RGBA8，存 v*16，v=4bit 量化值 0-15）
uniform int uHasRes;         // 1=有残差层，0=无
uniform vec2 uImgSz;
void main(){
    vec2 q = fragQ();
    vec2 uv = vec2(q.x / uImgSz.x, (uImgSz.y - q.y) / uImgSz.y);
    vec3 col = clamp(texture(uTex, uv).rgb, 0.0, 1.0);
    if (uHasRes == 1) {
        // 残差纹理与缓存 FBO 同序（row0=图像顶部，上传不翻转），uv 采样与 uTex 相同；
        // 纹理存 t=v*16，texture 返回 t/255；解码 res = (v-8)*(128/15)/255
        //   = (t/16 - 8)*(128/15)/255 = (t*255/16 - 8)*(128/3825) = t*0.5333 - 0.2677
        vec3 res = texture(uResTex, uv).rgb * (128.0 / 240.0) - (1024.0 / 3825.0);
        col = clamp(col + res, 0.0, 1.0);
    }
    FragColor = vec4(col, 1.0);
}
)";

// ---- GL 辅助 ----
struct Prog
{
    GLuint id = 0;
    GLint res = -1, sz = -1, off = -1, time = -1, cap = -1, spl = -1, tex = -1, texW = -1, imgSz = -1;
};

static bool CheckShader(GLuint sh, const char* name)
{
    GLint ok = 0;
    glGetShaderiv(sh, GL_COMPILE_STATUS, &ok);
    if (!ok)
    {
        char log[4096];
        glGetShaderInfoLog(sh, sizeof(log), nullptr, log);
        printf("=== %s COMPILE FAIL ===\n%s\n", name, log);
        return false;
    }
    return true;
}

static Prog MakeProg(const char* vsSrc, const char* fsSrc, const char* name)
{
    Prog P;
    GLuint vs = glCreateShader(GL_VERTEX_SHADER);
    const char* vparts[2] = { kCommonVS, vsSrc };
    glShaderSource(vs, 2, vparts, nullptr);
    glCompileShader(vs);
    if (!CheckShader(vs, name)) { P.id = 0; return P; }

    GLuint fs = glCreateShader(GL_FRAGMENT_SHADER);
    const char* fparts[2] = { kCommonFS, fsSrc };
    glShaderSource(fs, 2, fparts, nullptr);
    glCompileShader(fs);
    if (!CheckShader(fs, name)) { P.id = 0; return P; }

    P.id = glCreateProgram();
    glAttachShader(P.id, vs);
    glAttachShader(P.id, fs);
    glLinkProgram(P.id);
    GLint ok = 0;
    glGetProgramiv(P.id, GL_LINK_STATUS, &ok);
    if (!ok)
    {
        char log[4096];
        glGetProgramInfoLog(P.id, sizeof(log), nullptr, log);
        printf("=== %s LINK FAIL ===\n%s\n", name, log);
        P.id = 0; return P;
    }
    glDeleteShader(vs);
    glDeleteShader(fs);

    P.res  = glGetUniformLocation(P.id, "uRes");
    P.sz   = glGetUniformLocation(P.id, "uPlateSz");
    P.off  = glGetUniformLocation(P.id, "uOff");
    P.time = glGetUniformLocation(P.id, "uTime");
    P.cap  = glGetUniformLocation(P.id, "uCap");
    P.spl  = glGetUniformLocation(P.id, "uSpl");
    P.tex  = glGetUniformLocation(P.id, "uTex");
    P.texW = glGetUniformLocation(P.id, "uTexW");
    P.imgSz = glGetUniformLocation(P.id, "uImgSz");
    return P;
}

static void FramebufferSizeCallback(GLFWwindow* win, int w, int h)
{
    if (w <= 0 || h <= 0) return;
    g_winW = w;
    g_winH = h;
}

// ---- 渲染目标 ----
static GLuint tSpl = 0;                       // splat 参数纹理（NS x 3 RGBA32F）
static GLuint tA = 0, fboA = 0, fbW = 0, fbH = 0;  // 静态底图缓存 FBO（RGBA16F，splat 只渲一次）
static GLuint vaoTri = 0, vbo = 0, vaoQuad = 0;
static const float tri[] = { -1.f, -1.f,  3.f, -1.f,  -1.f, 3.f };

static void CreateTargets(int w, int h)
{
    if (tA) { glDeleteFramebuffers(1, &fboA); glDeleteTextures(1, &tA); }
    fbW = w; fbH = h;
    glGenTextures(1, &tA);
    glBindTexture(GL_TEXTURE_2D, tA);
    // RGBA32F 全精度缓存：splat 叠加的 alpha 合成中间值在半精度（16F）下
    // 会累积量化误差（实测 PSNR 损失 ~1.9dB），用 32F 保持与训练一致的精度
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, w, h, 0, GL_RGBA, GL_FLOAT, nullptr);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glGenFramebuffers(1, &fboA);
    glBindFramebuffer(GL_FRAMEBUFFER, fboA);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, tA, 0);
}

static void MakeFetchTex(GLuint* t, int w, int h, const void* data)
{
    glGenTextures(1, t);
    glBindTexture(GL_TEXTURE_2D, *t);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, w, h, 0, GL_RGBA, GL_FLOAT, data);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
}

// 调试：pass1 后 dump FBO 内容（RGBA float32）到 raw 文件
static void DumpFBO(const char* path, int w, int h)
{
    std::vector<float> px((size_t)w * h * 4);
    glReadPixels(0, 0, w, h, GL_RGBA, GL_FLOAT, px.data());
    FILE* f = nullptr;
    if (fopen_s(&f, path, "wb") == 0 && f) { fwrite(px.data(), 4, px.size(), f); fclose(f); }
    printf("dumped %s (%dx%d)\n", path, w, h);
}

// ---------- 离线导出（GPU 渲染一帧 -> 写图）----------
// 尽量无损：16bit PNG（量化损失 ~0.001dB）+ 8bit BMP（显示用）。PNG 用 zlib "stored" 未压缩块手写，无外部依赖。
static unsigned int g_crc32[256];
static bool g_crc32Ready = false;
static void InitCrc32()
{
    for (unsigned int i = 0; i < 256; ++i) {
        unsigned int c = i;
        for (int k = 0; k < 8; ++k) c = (c & 1) ? 0xEDB88320u ^ (c >> 1) : c >> 1;
        g_crc32[i] = c;
    }
    g_crc32Ready = true;
}
static unsigned int Crc32(const unsigned char* d, size_t n, unsigned int c = 0xFFFFFFFFu)
{
    for (size_t i = 0; i < n; ++i) c = g_crc32[(c ^ d[i]) & 0xFF] ^ (c >> 8);
    return c;
}
static unsigned int Adler32(const unsigned char* d, size_t n)
{
    unsigned int a = 1, b = 0;
    for (size_t i = 0; i < n; ++i) { a = (a + d[i]) % 65521u; b = (b + a) % 65521u; }
    return (b << 16) | a;
}
static void PNGChunk(FILE* f, const char* type, const unsigned char* data, size_t n)
{
    unsigned char len[4] = { (unsigned char)(n >> 24), (unsigned char)((n >> 16) & 0xFF),
                             (unsigned char)((n >> 8) & 0xFF), (unsigned char)(n & 0xFF) };
    unsigned int c = Crc32((const unsigned char*)type, 4);
    if (n) c = Crc32(data, n, c);
    c ^= 0xFFFFFFFFu;
    unsigned char crc[4] = { (unsigned char)(c >> 24), (unsigned char)((c >> 16) & 0xFF),
                             (unsigned char)((c >> 8) & 0xFF), (unsigned char)(c & 0xFF) };
    fwrite(len, 1, 4, f); fwrite(type, 1, 4, f);
    if (n) fwrite(data, 1, n, f);
    fwrite(crc, 1, 4, f);
}

// rgba = w*h*4 float（glReadPixels 行序：自底向上，FBO 底部为第 0 行）
static bool WritePNG16(const char* path, int w, int h, const float* rgba)
{
    if (!g_crc32Ready) { InitCrc32(); g_crc32Ready = true; }
    FILE* f = nullptr;
    if (fopen_s(&f, path, "wb") != 0 || !f) return false;
    static const unsigned char sig[8] = { 0x89, 'P', 'N', 'G', 0x0D, 0x0A, 0x1A, 0x0A };
    fwrite(sig, 1, 8, f);
    unsigned char ihdr[13] = { 0 };
    ihdr[0] = (unsigned char)(w >> 24); ihdr[1] = (unsigned char)((w >> 16) & 0xFF);
    ihdr[2] = (unsigned char)((w >> 8) & 0xFF); ihdr[3] = (unsigned char)(w & 0xFF);
    ihdr[4] = (unsigned char)(h >> 24); ihdr[5] = (unsigned char)((h >> 16) & 0xFF);
    ihdr[6] = (unsigned char)((h >> 8) & 0xFF); ihdr[7] = (unsigned char)(h & 0xFF);
    ihdr[8] = 16; ihdr[9] = 2;   // bit depth 16, color type 2 (RGB)
    PNGChunk(f, "IHDR", ihdr, 13);
    // 原始数据：每行 filter 字节(0) + W*6（R16BE,G16BE,B16BE）
    size_t rowBytes = (size_t)w * 6;
    std::vector<unsigned char> raw((size_t)h * (rowBytes + 1));
    for (int jj = 0; jj < h; ++jj) {
        int j = h - 1 - jj;   // PNG 是 top-down，而 glReadPixels 返回 bottom-up（第 0 行在底部），须翻转行序
        unsigned char* row = raw.data() + (size_t)jj * (rowBytes + 1);
        row[0] = 0;
        const float* px = rgba + (size_t)j * w * 4;
        for (int i = 0; i < w; ++i) {
            unsigned r = (unsigned)(ClampF(px[i * 4], 0, 1) * 65535.0f + 0.5f);
            unsigned g = (unsigned)(ClampF(px[i * 4 + 1], 0, 1) * 65535.0f + 0.5f);
            unsigned b = (unsigned)(ClampF(px[i * 4 + 2], 0, 1) * 65535.0f + 0.5f);
            row[1 + i * 6] = (unsigned char)(r >> 8); row[2 + i * 6] = (unsigned char)(r & 0xFF);
            row[3 + i * 6] = (unsigned char)(g >> 8); row[4 + i * 6] = (unsigned char)(g & 0xFF);
            row[5 + i * 6] = (unsigned char)(b >> 8); row[6 + i * 6] = (unsigned char)(b & 0xFF);
        }
    }
    // zlib 流：header 0x78 0x01 + deflate stored 块（<=64KB/块，末块 final=1）+ adler32
    std::vector<unsigned char> zlib;
    zlib.push_back(0x78); zlib.push_back(0x01);
    size_t off = 0;
    while (off < raw.size()) {
        size_t n = std::min((size_t)65535, raw.size() - off);
        bool last = (off + n >= raw.size());
        zlib.push_back(last ? 0x01 : 0x00);
        zlib.push_back((unsigned char)(n & 0xFF)); zlib.push_back((unsigned char)((n >> 8) & 0xFF));
        zlib.push_back((unsigned char)((~n) & 0xFF)); zlib.push_back((unsigned char)(((~n) >> 8) & 0xFF));
        zlib.insert(zlib.end(), raw.begin() + off, raw.begin() + off + n);
        off += n;
    }
    unsigned int ad = Adler32(raw.data(), raw.size());
    zlib.push_back((unsigned char)(ad >> 24)); zlib.push_back((unsigned char)((ad >> 16) & 0xFF));
    zlib.push_back((unsigned char)((ad >> 8) & 0xFF)); zlib.push_back((unsigned char)(ad & 0xFF));
    PNGChunk(f, "IDAT", zlib.data(), zlib.size());
    PNGChunk(f, "IEND", nullptr, 0);
    fclose(f);
    return true;
}

static bool WriteBMP24(const char* path, int w, int h, const float* rgba)
{
    int rowSize = (w * 3 + 3) & ~3;
    int dataSize = rowSize * h;
    int fileSize = 54 + dataSize;
    FILE* f = nullptr;
    if (fopen_s(&f, path, "wb") != 0 || !f) return false;
    unsigned char hdr[54] = { 0 };
    hdr[0] = 'B'; hdr[1] = 'M';
    hdr[2] = (unsigned char)(fileSize & 0xFF); hdr[3] = (unsigned char)((fileSize >> 8) & 0xFF);
    hdr[4] = (unsigned char)((fileSize >> 16) & 0xFF); hdr[5] = (unsigned char)((fileSize >> 24) & 0xFF);
    hdr[10] = 54; hdr[14] = 40;
    hdr[18] = (unsigned char)(w & 0xFF); hdr[19] = (unsigned char)((w >> 8) & 0xFF);
    hdr[20] = (unsigned char)((w >> 16) & 0xFF); hdr[21] = (unsigned char)((w >> 24) & 0xFF);
    hdr[22] = (unsigned char)(h & 0xFF); hdr[23] = (unsigned char)((h >> 8) & 0xFF);
    hdr[24] = (unsigned char)((h >> 16) & 0xFF); hdr[25] = (unsigned char)((h >> 24) & 0xFF);
    hdr[26] = 1; hdr[28] = 24;
    fwrite(hdr, 1, 54, f);
    std::vector<unsigned char> row((size_t)rowSize, 0);
    for (int j = 0; j < h; ++j) {   // GL 读回自底向上 = BMP 文件行序（自底向上），直接顺序写
        const float* px = rgba + (size_t)j * w * 4;
        for (int i = 0; i < w; ++i) {
            row[i * 3]     = (unsigned char)(ClampF(px[i * 4 + 2], 0, 1) * 255.0f + 0.5f);   // B
            row[i * 3 + 1] = (unsigned char)(ClampF(px[i * 4 + 1], 0, 1) * 255.0f + 0.5f);   // G
            row[i * 3 + 2] = (unsigned char)(ClampF(px[i * 4], 0, 1) * 255.0f + 0.5f);       // R
        }
        fwrite(row.data(), 1, (size_t)rowSize, f);
    }
    fclose(f);
    return true;
}

// 导出：GPU 渲染结果（pass1 已写入 FBO）读回 float32 -> 16bit PNG + 8bit BMP
static void ExportAll(const char* base, int w, int h)
{
    std::vector<float> px((size_t)w * h * 4);
    glReadPixels(0, 0, w, h, GL_RGBA, GL_FLOAT, px.data());
    // 残差修正层叠加（与 pass2 相同逻辑）。注意：glReadPixels 行序底部优先（row0=图像底部），
    // 而 kRes 顶部优先（row0=图像顶部），叠加时必须行翻转对齐。
    // scale>1（高清直渲导出）时按比例映射回画布分辨率（最近邻取整 + clamp）。
    if (!g_resVals.empty()) {
        int cw = (int)g_imgW, ch = (int)g_imgH;
        for (int j = 0; j < h; ++j)
            for (int i = 0; i < w; ++i) {
                size_t pr = (size_t)j * w + i;             // px 行 j（底部优先）
                int cx = std::max(0, std::min((int)((double)i * cw / w), cw - 1));
                int cy = std::max(0, std::min((int)((double)j * ch / h), ch - 1));
                size_t rr = (size_t)(ch - 1 - cy) * cw + cx;   // kRes 行（顶部优先）
                unsigned v = g_resVals[rr];
                for (int c = 0; c < 3; ++c) {
                    int vc = (v >> (c * 4)) & 15;
                    float r01 = ((float)(vc - 8)) * (128.0f / 15.0f) / 255.0f;   // res = (v-8)*(128/15)/255
                    px[pr * 4 + c] = ClampF(px[pr * 4 + c] + r01, 0.0f, 1.0f);
                }
            }
    }
    std::string p16 = std::string(base) + ".png";
    std::string b24 = std::string(base) + ".bmp";
    long long t0 = NowMs();
    bool ok16 = WritePNG16(p16.c_str(), w, h, px.data());
    bool ok24 = WriteBMP24(b24.c_str(), w, h, px.data());
    printf("导出完成: %s (16bit PNG) + %s (8bit BMP), %.0fms\n",
           p16.c_str(), b24.c_str(), (double)(NowMs() - t0));
    if (!ok16) printf("  16bit PNG 写入失败!\n");
    if (!ok24) printf("  8bit BMP 写入失败!\n");
}

int main(int argc, char** argv)
{
    setvbuf(stdout, nullptr, _IONBF, 0);
    bool dump = false, dumpwin = false;
    std::string exportBase, inputPath;   // inputPath: 显式指定 GLSL 文件（--input <file> 或位置参数）
    int animMode = 0;      // --anim：动画模式，按 kData 数组顺序渐进绘制（混沌浮现效果）
    int animRate = 10000;  // --anim-rate N：动画每秒新增 splat 数（默认 10000）
    int animFps = 60;      // --anim-fps 30|60：动画节奏（默认 60；30 时每帧推进量翻倍，总时长一致）
    float scale = 1.0f;    // --scale N：高清直渲放大系数（pass1 渲染目标 = 画布×N，splat 坐标×N）
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--dump") == 0) dump = true;
        else if (strcmp(argv[i], "--dumpwin") == 0) dumpwin = true;
        else if (strcmp(argv[i], "--input") == 0 && i + 1 < argc) inputPath = argv[++i];   // 显式指定 GLSL 文件
        else if (strcmp(argv[i], "--export") == 0 && i + 1 < argc) exportBase = argv[++i]; // 离线导出：base.png(16bit)+base.bmp
        else if (strcmp(argv[i], "--anim") == 0) animMode = 1;                             // 启动动画模式
        else if (strcmp(argv[i], "--anim-rate") == 0 && i + 1 < argc) animRate = atoi(argv[++i]);
        else if (strcmp(argv[i], "--anim-fps") == 0 && i + 1 < argc) animFps = atoi(argv[++i]);
        else if (strcmp(argv[i], "--scale") == 0 && i + 1 < argc) scale = (float)atof(argv[++i]);
        else if (argv[i][0] != '-') inputPath = argv[i];   // 位置参数：直接给 GLSL 文件路径
    }
    if (animFps != 30 && animFps != 60) animFps = 60;   // 只允许 30/60，非法值回落 60
    if (animRate <= 0) animRate = 10000;
    if (scale <= 0.f) scale = 1.0f;   // --scale 非法值回落 1（1:1 画布渲染）
    if (!glfwInit()) { printf("glfwInit 失败\n"); return 1; }

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    if (!exportBase.empty()) glfwWindowHint(GLFW_VISIBLE, GLFW_FALSE);   // 离线导出：隐藏窗口，渲染一帧后退出

    GLFWwindow* win = glfwCreateWindow(g_winW, g_winH, "fitsplat_gl (ESC 退出)", nullptr, nullptr);
    if (!win) { printf("glfwCreateWindow 失败\n"); return 1; }
    glfwMakeContextCurrent(win);
    glfwSwapInterval(1);
    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) { printf("glad 失败\n"); return 1; }
    glfwSetFramebufferSizeCallback(win, FramebufferSizeCallback);
    // 窗口客户区（framebuffer）≠ 请求的窗口外部尺寸（含标题栏/边框）。
    // 必须以实际 framebuffer 尺寸为准，否则 FBO/视口与窗口不匹配会导致显示错位。
    glfwGetFramebufferSize(win, &g_winW, &g_winH);
    printf("窗口 framebuffer: %dx%d\n", g_winW, g_winH);

    // ---- 加载 GLSL 解析画布尺寸 ----
    // 优先用命令行指定的文件（--input <file> 或位置参数），否则回退 exe 同目录 input.glsl
    std::vector<unsigned char> glslBuf;
    std::string glslPath;
    if (!inputPath.empty()) {
        if (ReadFileBin(inputPath, glslBuf)) glslPath = inputPath;
    } else {
        std::string exeDir = ExeDir();
        for (const auto& p : { exeDir + "/input.glsl", std::string("input.glsl") }) {
            if (ReadFileBin(p, glslBuf)) { glslPath = p; break; }
        }
    }
    if (glslPath.empty()) { printf("未找到 GLSL 文件（用 --input <file> 或位置参数指定，或放到 exe 同目录 input.glsl）\n"); return 1; }
    std::string glsl((const char*)glslBuf.data(), glslBuf.size());
    if (!ParseImgSize(glsl, g_imgW, g_imgH)) {
        printf("input.glsl 中未找到 IMG_W/IMG_H 常量\n"); return 1;
    }
    // 高清直渲：pass1 渲染目标 = 画布 × scale（splat 坐标在 VS 里乘 uScale 对齐）
    int rW = (int)std::max(1, (int)llroundf(g_imgW * scale));
    int rH = (int)std::max(1, (int)llroundf(g_imgH * scale));
    if (scale != 1.0f)
        printf("高清直渲: --scale %.2f -> pass1 渲染目标 %dx%d\n", scale, rW, rH);
    printf("已加载: %s (画布 %.0fx%.0f)\n", glslPath.c_str(), g_imgW, g_imgH);
    // 窗口按渲染目标尺寸创建（逻辑像素 1:1；framebuffer 受系统缩放影响，pass2 letterbox 自动适配）
    glfwSetWindowSize(win, rW, rH);
    glfwGetFramebufferSize(win, &g_winW, &g_winH);
    printf("窗口 framebuffer: %dx%d\n", g_winW, g_winH);

    // ---- 加载 splat 参数 ----
    // 优先读 *_splatA/B/C.raw（RGBA32F，每行一个 splat）；若缺失则从
    // 无纹理纯 GLSL（--embed）的 kSA/kSB/kSC const 数组解析 —— 单文件即可渲染。
    std::string baseDir = glslPath.substr(0, glslPath.find_last_of("\\/"));
    std::vector<unsigned char> rawA, rawB, rawC;
    std::vector<float> vA, vB, vC;
    bool haveRaw = ReadFileBin(baseDir + "/input_splatA.raw", rawA) &&
                   ReadFileBin(baseDir + "/input_splatB.raw", rawB) &&
                   ReadFileBin(baseDir + "/input_splatC.raw", rawC);
    if (haveRaw) {
        printf("参数来源: *_splatA/B/C.raw\n");
    } else if (ParseVec4Array(glsl, "kSA", vA) && ParseVec4Array(glsl, "kSB", vB) &&
               ParseVec4Array(glsl, "kSC", vC)) {
        printf("参数来源: input.glsl 内嵌 kSA/kSB/kSC const 数组（无纹理模式）\n");
        rawA.resize(vA.size() * 4); memcpy(rawA.data(), vA.data(), vA.size() * sizeof(float));
        rawB.resize(vB.size() * 4); memcpy(rawB.data(), vB.data(), vB.size() * sizeof(float));
        rawC.resize(vC.size() * 4); memcpy(rawC.data(), vC.data(), vC.size() * sizeof(float));
    } else {
        // 新版 --embed：uint 打包压缩（每 splat 4 个 uint），CPU 解码器还原参数
        std::vector<unsigned> uData;
        if (!ParseUintArray(glsl, "kData", uData) || uData.size() < 4 || uData.size() % 4 != 0) {
            printf("未找到 *_splatA/B/C.raw，也无法解析 kSA/kSB/kSC/kData 数组\n"); return 1;
        }
        size_t ns = uData.size() / 4;
        printf("参数来源: input.glsl 内嵌 kData uint 打包数组（CPU 解码 %d splat）\n", (int)ns);
        float smx = 128.f, smy = 64.f;   // 位置量化系数：旧版 embed 固定 128/64；新版按画布自适应写入 kMXS/kMYS
        if (ParseFloatConst(glsl, "kMXS", smx) && ParseFloatConst(glsl, "kMYS", smy))
            printf("位置量化系数: kMXS=%.0f kMYS=%.0f\n", smx, smy);
        rawA.resize(ns * 16); rawB.resize(ns * 16); rawC.resize(ns * 16);
        long long tDec = NowMs();
        for (size_t i = 0; i < ns; ++i) {
            unsigned u0 = uData[i * 4], u1 = uData[i * 4 + 1], u2 = uData[i * 4 + 2], u3 = uData[i * 4 + 3];
            float* pA = (float*)rawA.data() + i * 4;
            float* pB = (float*)rawB.data() + i * 4;
            float* pC = (float*)rawC.data() + i * 4;
            // u0: mx(16bit,1/smx px) | my(16bit,1/smy px)；u1: r,g,b,a 各 8bit
            // u2: sx,sy log2 量化各 12bit（[2^-5,2^9]）；u3: cos,sin 各 16bit（[0,65535] -> [-1,1]）
            pA[0] = (float)(u0 & 0xFFFFu) / smx;
            pA[1] = (float)(u0 >> 16) / smy;
            pA[2] = exp2f((float)(u2 & 0xFFFu) * (14.0f / 4095.0f) - 5.0f);
            pA[3] = exp2f((float)((u2 >> 12) & 0xFFFu) * (14.0f / 4095.0f) - 5.0f);
            pB[0] = (float)(u1 & 0xFFu) / 255.0f;
            pB[1] = (float)((u1 >> 8) & 0xFFu) / 255.0f;
            pB[2] = (float)((u1 >> 16) & 0xFFu) / 255.0f;
            pB[3] = (float)(u1 >> 24) / 255.0f;
            pC[0] = (float)(u3 & 0xFFFFu) / 32767.0f - 1.0f;
            pC[1] = (float)((u3 >> 16) & 0xFFFFu) / 32767.0f - 1.0f;
            pC[2] = 0.f; pC[3] = 0.f;
        }
        printf("CPU 解码完成: %lldms\n", NowMs() - tDec);
    }
    size_t NS = rawA.size() / 16;
    if (NS == 0 || rawB.size() != rawA.size() || rawC.size() != rawA.size()) {
        printf("splat 数据尺寸不一致\n"); return 1;
    }
    // 注意合成顺序：训练是"从前往后"（splat 0 在最前），而 GL 的
    // SRC_ALPHA/ONE_MINUS_SRC_ALPHA 混合是"从后往前"（先画的在最底层），
    // 因此上传纹理时把 splat 顺序反转，使 GL 先画训练列表里最靠后的 splat。
    std::vector<float> spl(NS * 3 * 4);
    for (size_t i = 0; i < NS; ++i) {
        const float* pA = (const float*)rawA.data() + i * 4;
        const float* pB = (const float*)rawB.data() + i * 4;
        const float* pC = (const float*)rawC.data() + i * 4;
        size_t r = NS - 1 - i;                       // 反转后的实例列（先画最底层）
        float* base = spl.data() + r * 4;            // 纹理 NS 宽 x 3 高，行主序
        memcpy(base + 0 * NS * 4, pA, 16);           // row0 = SA
        memcpy(base + 1 * NS * 4, pB, 16);           // row1 = SB
        memcpy(base + 2 * NS * 4, pC, 16);           // row2 = SC
    }
    printf("splats: %d\n", (int)NS);

    // ---- 残差修正层（可选）：解析 kRes -> 普通 RGBA8 纹理 ----
    // 行序：必须与缓存 FBO 纹理一致（row0=图像底部），pass2 用同一 uv 采样才能对齐。
    // 因此上传时把 kRes（图像顶部=row0）行翻转。
    GLuint resTex = 0;
    {
        std::vector<unsigned> uResArr;
        if (ParseUintArray(glsl, "kRes", uResArr) && uResArr.size() == (size_t)g_imgW * g_imgH) {
            g_resVals = uResArr;
            std::vector<unsigned char> px((size_t)g_imgW * g_imgH * 4);
            for (int j = 0; j < (int)g_imgH; ++j)
                for (int i = 0; i < (int)g_imgW; ++i) {
                    unsigned v = uResArr[(size_t)((int)g_imgH - 1 - j) * (int)g_imgW + i];
                    unsigned char* d = &px[((size_t)j * (int)g_imgW + i) * 4];
                    d[0] = (v & 15) * 16; d[1] = ((v >> 4) & 15) * 16; d[2] = ((v >> 8) & 15) * 16; d[3] = 255;
                }
            glGenTextures(1, &resTex);
            glBindTexture(GL_TEXTURE_2D, resTex);
            // LINEAR：scale>1 高清直渲时残差纹理被放大采样，线性插值更平滑；
            // 1:1 时采样点恰在纹素中心，与 NEAREST 等价。
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, (GLsizei)g_imgW, (GLsizei)g_imgH, 0, GL_RGBA, GL_UNSIGNED_BYTE, px.data());
            printf("残差修正层: 已加载 kRes (%dx%d, RGBA8)\n", (int)g_imgW, (int)g_imgH);
        } else {
            printf("残差修正层: 未找到 kRes，跳过\n");
        }
    }

    // ---- GL 对象 ----
    glGenVertexArrays(1, &vaoTri);
    glGenBuffers(1, &vbo);
    glBindVertexArray(vaoTri);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof(tri), tri, GL_STATIC_DRAW);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, nullptr);
    glGenVertexArrays(1, &vaoQuad);

    GLint maxTex = 0;
    glGetIntegerv(GL_MAX_TEXTURE_SIZE, &maxTex);
    printf("GL_MAX_TEXTURE_SIZE: %d\n", maxTex);

    // 参数纹理：NS*3 个 texel 展平为 kTexW 宽二维纹理，
    // 规避 GL_MAX_TEXTURE_SIZE 单维限制（大 NS 时 31012 宽会超 16384 导致纹理创建失败 -> 全黑）。
    // 注意：spl 是 (row, id) 行主序（row0=SA...），须重排为 ti = id*3+row 连续布局，
    // 与 VS 的 fetchParam(id*3+row) 交错索引一一对应。
    static const int kTexW = 8192;
    int texH = (int)((NS * 3 + kTexW - 1) / kTexW);
    std::vector<float> flat((size_t)texH * kTexW * 4, 0.f);
    for (size_t i = 0; i < NS; ++i)
        for (int row = 0; row < 3; ++row)
            memcpy(flat.data() + (i * 3 + row) * 4, spl.data() + ((size_t)row * NS + i) * 4, 16);
    MakeFetchTex(&tSpl, kTexW, texH, flat.data());
    printf("参数纹理: %dx%d (NS*3=%d texels)\n", kTexW, texH, (int)(NS * 3));

    Prog pSplat = MakeProg(kVSSplat, kFSSplat, "splat");
    Prog pPresent = MakeProg(kVSFull, kFSPresent, "present");
    if (!pSplat.id || !pPresent.id) return 1;
    GLint uBaseLoc = glGetUniformLocation(pSplat.id, "uBase");     // 动画增量：实例偏移
    GLint uScaleLoc = glGetUniformLocation(pSplat.id, "uScale");   // 高清直渲：坐标放大系数

    glDisable(GL_DEPTH_TEST);
    glDisable(GL_CULL_FACE);

    // 缓存 FBO 固定为渲染目标尺寸 rW×rH（= 画布×scale，不随窗口变化），
    // pass2 再做 letterbox 采样，这样窗口 framebuffer 尺寸（受屏幕/边框影响）不影响正确性。
    CreateTargets(rW, rH);

    // ---- 渲染循环 ----
    bool cacheReady = false;        // 非动画：FBO 是否已渲染完成
    double lastT = glfwGetTime();
    double fpsAcc = 0.0; int fpsN = 0;
    size_t drawN = 0;               // 动画模式：已绘制的 splat 数（按 kData 顺序累积）
    double beatAcc = 0.0;           // 动画节拍累计器（animFps 节拍驱动）
    bool fboInited = false;         // 动画模式：FBO 是否已 clear

    // 增量绘制 [base, base+cnt) 段。over 算子满足结合律：段内顺序保持，段间在
    // 持久 FBO 上叠加（RGBA32F 全精度）≡ 一次性按序全量渲染。动画每帧只画新增段，
    // 渲染开销恒定 O(animRate)，不会随累计量增长（重绘方案是 ΣN ≈ N²/2 的二次方）。
    auto DrawSeg = [&](int base, int cnt, bool doClear) {
        glViewport(0, 0, rW, rH);
        glBindFramebuffer(GL_FRAMEBUFFER, fboA);
        if (doClear) { glClearColor(0.f, 0.f, 0.f, 0.f); glClear(GL_COLOR_BUFFER_BIT); }
        glEnable(GL_BLEND);
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
        glUseProgram(pSplat.id);
        glUniform2f(pSplat.res, (float)rW, (float)rH);
        glUniform2f(pSplat.sz, 1.f, 1.f);          // 画布像素 -> 画布像素（1:1）
        glUniform2f(pSplat.off, 0.f, 0.f);
        // 上限按放大坐标系长边（覆盖正常 splat 的 3σ 包围盒；此前硬编码 1080*0.35=378，
        // 在大画布（800x1165）上会截断大 splat 的包围盒导致渲染错位/缺失）
        glUniform1f(pSplat.cap, (float)std::max(rW, rH));
        glUniform1f(uScaleLoc, scale);
        glUniform1i(pSplat.spl, 0);
        glUniform1i(pSplat.texW, kTexW);
        glUniform1i(uBaseLoc, base);
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, tSpl);
        glBindVertexArray(vaoQuad);
        glDrawArraysInstanced(GL_TRIANGLE_STRIP, 0, 4, cnt);
        glDisable(GL_BLEND);
        GLenum err = glGetError();
        if (err) printf("splat GL error: 0x%x\n", err);
    };

    while (!glfwWindowShouldClose(win))
    {
        double t = glfwGetTime();
        double dt = t - lastT;
        lastT = t;
        fpsAcc += dt; fpsN++;

        // 动画模式：每节拍增量画 [drawN, drawN+rate) 段到持久 FBO（不重绘、不清屏，
        // 首段前 clear 一次；30fps 节拍 = 每秒 30*animRate，总时长是 60fps 的 2 倍）
        if (animMode && drawN < NS) {
            beatAcc += dt;
            double beat = 1.0 / animFps;
            while (beatAcc >= beat && drawN < NS) {
                beatAcc -= beat;
                int cnt = (int)std::min((size_t)animRate, NS - drawN);
                DrawSeg((int)drawN, cnt, !fboInited);
                fboInited = true;
                drawN += (size_t)cnt;
            }
        }

        if (fpsAcc >= 0.5) {
            char title[128];
            if (animMode)
                snprintf(title, sizeof(title), "fitsplat_gl  %zd/%zd (%.1f%%)  %.1f fps",
                         drawN, NS, 100.0 * (double)drawN / (double)NS, fpsN / fpsAcc);
            else
                snprintf(title, sizeof(title), "fitsplat_gl  %d splats  %.1f fps", (int)NS, fpsN / fpsAcc);
            glfwSetWindowTitle(win, title);
            fpsAcc = 0.0; fpsN = 0;
        }

        int W = g_winW, H = g_winH;
        // 注意：缓存 FBO 固定为画布尺寸（CreateTargets(g_imgW, g_imgH)），
        // 窗口尺寸变化只影响 pass2 的 letterbox 采样——绝不能按窗口尺寸重建 FBO，
        // 否则 pass1 的 1:1 viewport 与 FBO 不匹配，画面会被拉伸。

        // 画板适配
        float sW = std::fmin(W / g_imgW, H / g_imgH);
        float szxW = g_imgW * sW, szyW = g_imgH * sW;
        float offXW = (W - szxW) * 0.5f, offYW = (H - szyW) * 0.5f;

        // ---- pass1（非动画：全量一次渲染缓存 FBO，之后每帧仅 pass2 输出）----
        if (!cacheReady)
        {
            cacheReady = true;
            DrawSeg(0, (int)NS, true);
            if (dump) DumpFBO("fbodump.raw", rW, rH);
            if (!exportBase.empty()) {
                ExportAll(exportBase.c_str(), rW, rH);   // 离线导出：读回 float32 -> PNG16/BMP24（scale>1 即高清直渲导出）
                glfwSetWindowShouldClose(win, GLFW_TRUE);
            }
        }

        // ---- pass2（每帧）：缓存 FBO -> 窗口输出（letterbox 缩放，clamp）----
        glBindFramebuffer(GL_FRAMEBUFFER, 0);
        glViewport(0, 0, W, H);
        glUseProgram(pPresent.id);
        glUniform2f(pPresent.res, (float)W, (float)H);
        glUniform2f(pPresent.sz, sW, sW);
        glUniform2f(pPresent.off, offXW, offYW);
        // uv 分母用画布尺寸（非渲染目标 rW×rH）：letterbox 后 q 是画布坐标（0..W画布），
        // FBO 内容 = 画布坐标×scale，归一化 uv 与画布 0-1 线性对应。若用 rW×rH 只采样纹理一部分。
        glUniform2f(pPresent.imgSz, g_imgW, g_imgH);
        glUniform1i(pPresent.tex, 0);
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, tA);
        GLint uResLoc = glGetUniformLocation(pPresent.id, "uResTex");
        GLint uHasLoc = glGetUniformLocation(pPresent.id, "uHasRes");
        // 动画模式：绘制完成前不叠残差（保留"混沌浮现"过程），画完后才叠加锐化细节
        bool resActive = resTex && (!animMode || drawN >= NS);
        if (resActive) {
            glUniform1i(uResLoc, 1);
            glUniform1i(uHasLoc, 1);
            glActiveTexture(GL_TEXTURE1);
            glBindTexture(GL_TEXTURE_2D, resTex);
        } else {
            glUniform1i(uHasLoc, 0);   // 无残差层或动画未完成：shader 跳过叠加
        }
        glBindVertexArray(vaoTri);
        glDrawArrays(GL_TRIANGLES, 0, 3);
        if (dumpwin) { DumpFBO("windump.raw", W, H); dumpwin = false; }

        glfwSwapBuffers(win);
        glfwPollEvents();
        if (glfwGetKey(win, GLFW_KEY_ESCAPE) == GLFW_PRESS)
            glfwSetWindowShouldClose(win, GLFW_TRUE);
    }

    glfwDestroyWindow(win);
    glfwTerminate();
    return 0;
}
