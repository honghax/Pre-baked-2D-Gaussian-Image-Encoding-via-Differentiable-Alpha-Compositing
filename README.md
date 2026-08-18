# Pre-baked 2D Gaussian Image Encoding via Differentiable Alpha Compositing

Image representation as a superposition of parametric primitives offers a compelling alternative to conventional pixel-based formats, enabling resolution-independent synthesis and efficient real-time playback. While 3D Gaussian Splatting has recently demonstrated the power of differentiable rendering for novel view synthesis, its underlying optimization and alpha-compositing pipeline can be repurposed for 2D image encoding without any camera geometry.

This project formulates the task of fitting a single raster image as the minimization of a reconstruction loss between the target and a render composed of multiple anisotropic 2D Gaussian kernels, each carrying spatial, color, and opacity attributes. The final parameters are exported either as constant arrays embedded in GLSL shaders or as texture-based lookup tables, allowing real-time image reconstruction on commodity GPUs (including integrated graphics).

## How it works

- **Primitives**: anisotropic 2D Gaussian kernels, each defined by position `(mx, my)`, axis lengths `(sx, sy)`, orientation `rot`, color `(r, g, b)`, and opacity `a`.
- **Rendering**: per-pixel alpha compositing (ordered front-to-back blending) `acc += c * α * T; T *= (1 - α)`, operating entirely in screen space — no camera model, depth, or multi-view data.
- **Optimization**: L1 + SSIM reconstruction loss, Adam with learning-rate decay, tile-based culling, stochastic sampling, and periodic density control (splitting under-fit splats, pruning low-opacity ones).
- **Export**: parameters are quantized into compact `uint` arrays embedded in a single self-contained GLSL file (or raw texture lookups), with an optional **residual correction layer** that stores the quantized per-pixel difference between target and Gaussian render, recovering high-frequency detail.

## Repository layout

```
.
├── fitsplat/            # CPU trainer (offline optimization)
│   ├── fitsplat.cpp
│   └── CMakeLists.txt
├── fitsplat_gl/         # OpenGL real-time renderer
│   ├── main.cpp
│   └── CMakeLists.txt
├── third_party/         # vendored dependencies
│   ├── stb/stb_image.h
│   ├── glad.c, include/glad/glad.h, include/KHR/khrplatform.h
│   └── glfw-3.4.bin.WIN64/   (headers + prebuilt lib-vc2022)
└── images/              # example renders
```

## Dependencies

| Dependency | Used by | Notes |
|---|---|---|
| [stb_image](https://github.com/nothings/stb) | `fitsplat` | image loading (PNG/JPEG/WebP) |
| [glad](https://github.com/Dav1dde/glad) | `fitsplat_gl` | OpenGL 3.3 core loader |
| [GLFW 3.4](https://www.glfw.org/) | `fitsplat_gl` | windowing, prebuilt Win64 in `third_party/` |
| CMake ≥ 3.16 / 3.20 | both | build system |

All dependencies are vendored under `third_party/`, so no package installation is required.

## Building

Open the repository folder in **Visual Studio** (or VS Code with the CMake extension) — the IDE auto-detects the MSVC toolchain and configures CMake. Or build from a Developer PowerShell:

```powershell
# fitsplat (trainer)
cmake -S fitsplat -B fitsplat/build
cmake --build fitsplat/build --config Release

# fitsplat_gl (renderer)
cmake -S fitsplat_gl -B fitsplat_gl/build
cmake --build fitsplat_gl/build --config Release
```

The prebuilt GLFW binaries in `third_party/glfw-3.4.bin.WIN64` target MSVC 2022 (`lib-vc2022`); for other toolchains, download the matching [GLFW release](https://www.glfw.org/download.html) and update `GLFW_ROOT` in `fitsplat_gl/CMakeLists.txt`.

## Usage

### 1. Train — `fitsplat`

```powershell
fitsplat input.png --splats 350000 --iters 15000 --width 800 --embed 1 --out out.glsl
```

| Option | Default | Description |
|---|---|---|
| `input.png` | — | source image (PNG/JPEG/WebP via stb_image) |
| `--splats N` | 600 | initial number of splats (density control grows/shrinks this) |
| `--iters N` | 1000 | optimization iterations |
| `--width N` | 320 | output canvas width (aspect-preserving height) |
| `--height N` | 0 | force canvas height (0 = auto from aspect ratio) |
| `--batch N` | 2048 | stochastic pixel batch size per step |
| `--threads N` | auto | worker threads (default ≤ 8) |
| `--bg F` | 0.0 | background color |
| `--seed N` | 42 | RNG seed for reproducibility |
| `--preview N` | 10 | save a progress BMP every N% of iterations |
| `--out FILE` | fitsplat.glsl | output GLSL path |
| `--embed N` | 0 | `1` = quantize params into `uint` arrays embedded in one self-contained GLSL (no runtime texture files); `0` = texture-based raw output |
| `--residual N` | 0 | `1` = append residual correction layer (4-bit/channel, ±64) to the embedded GLSL — recovers high-frequency detail, typically **+9 dB PSNR** at +2% size |
| `--l1only N` | 0 | `1` = use L1 loss only (default is L1 + SSIM) |
| `--l2 N` | 0 | `1` = use L2 loss |
| `--ckpt FILE` | — | save a checkpoint every `--ckpt-every` iterations |
| `--ckpt-every N` | 2000 | checkpoint interval |
| `--resume FILE` | — | resume training from a checkpoint |
| `--tilecap N` | 1024 | cap per-tile splat index count in the export (0 = unlimited) |

Example (embed + residual, as shown in the figures below):

```powershell
fitsplat input.png --splats 350000 --iters 15000 --width 800 --embed 1 --residual 1 --out input.glsl
```

### 2. Render — `fitsplat_gl`

The renderer loads `input.glsl` from the directory of the executable. Drop the exported GLSL next to `fitsplat_gl.exe` and run:

```powershell
fitsplat_gl.exe                 # real-time window (ESC to quit)
fitsplat_gl.exe --export out    # headless: write out.png (16-bit) + out.bmp (8-bit), then exit
```

| Option | Description |
|---|---|
| *(none)* | open the real-time preview window, letterboxed to the canvas aspect ratio |
| `--export BASE` | render one frame headlessly, save `BASE.png` (16-bit) + `BASE.bmp`, then exit |
| `--dump` | debug: dump the FBO to `fbodump.raw` |
| `--dumpwin` | debug: dump the window frame to `windump.raw` |

The renderer acts as a **decoder**: it parses the embedded `uint` arrays from the GLSL on the CPU (~74 ms for 1M splats), uploads them as parameter textures, and runs a small shader for per-splat instanced quads — avoiding GPU driver compilation of a multi-hundred-MB constant array.

## Results

800×1165 canvas, 1,017,194 splats, 15k iterations (quantized embedded export).

| | Original | Gaussian only | Gaussian + residual layer |
|---|---|---|---|
| Render | ![original](images/original.png) | ![without residual](images/without_residual.png) | ![with residual](images/with_residual.png) |
| PSNR | — | 27.05 dB | **36.30 dB** |

The residual correction layer stores `target − gaussian_render` quantized to 4-bit per channel (±64 range) in a single `uint` per pixel (~1.4 MB, ≈2% of the file size). At render time it is applied as one texture fetch plus a small arithmetic correction — zero impact on frame rate.

## Output format

`--embed 1` produces a single self-contained GLSL file containing:

- `kMXS`/`kMYS` — adaptive position quantization scale (avoids 16-bit overflow on large canvases)
- `kData[]` — quantized splat parameters (position, size, rotation, color, opacity)
- `kRes[]` — optional residual layer (`--residual 1`)
- `sampleImage()` — the full alpha-compositing render function

This file is portable: copy it next to any `fitsplat_gl` binary and it will render without any external data.
