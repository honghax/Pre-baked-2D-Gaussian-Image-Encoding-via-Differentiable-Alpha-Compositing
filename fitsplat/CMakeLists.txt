cmake_minimum_required(VERSION 3.16)
project(fitsplat LANGUAGES C CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# --gpu 前向基准需要 OpenGL 4.3 compute shader：
#   glad43 = 训练器专用 4.3 loader（与播放器 fitsplat_gl 的 3.3 glad 独立，两个 exe 各自链接）
#   glfw   = 创建隐藏窗口上下文（静态库 glfw3.lib，无需 dll）
set(GLFW_ROOT "${CMAKE_CURRENT_LIST_DIR}/../third_party/glfw-3.4.bin.WIN64")

add_executable(fitsplat
    fitsplat.cpp
    ${CMAKE_CURRENT_LIST_DIR}/../third_party/glad43/src/glad.c
)

target_include_directories(fitsplat PRIVATE
    ${CMAKE_CURRENT_LIST_DIR}/../third_party/stb
    ${CMAKE_CURRENT_LIST_DIR}/../third_party/glad43/include
    ${GLFW_ROOT}/include
)

target_link_libraries(fitsplat PRIVATE
    ${GLFW_ROOT}/lib-vc2022/glfw3.lib
    opengl32
)

if(MSVC)
  # /O2 仅用于非 Debug 配置：VS Debug 默认带 /Od /RTC1，与 /O2 互斥（D8016）
  target_compile_options(fitsplat PRIVATE $<$<NOT:$<CONFIG:Debug>>:/O2> /fp:fast /utf-8)
  # 显式指定系统库路径（链接器环境的 LIB 变量可能被沙箱过滤，
  # 裸库名 opengl32/kernel32/msvcprt 等会因找不到 .lib 而链接失败）
  get_filename_component(_MSVC_ROOT "${CMAKE_CXX_COMPILER}" DIRECTORY)
  get_filename_component(_MSVC_ROOT "${_MSVC_ROOT}/../../.." ABSOLUTE)   # .../MSVC/<ver>
  target_link_options(fitsplat PRIVATE "/LIBPATH:${_MSVC_ROOT}/lib/x64")
  file(GLOB _WIN_KITS "C:/Program Files (x86)/Windows Kits/10/Lib/*")
  foreach(_kit ${_WIN_KITS})
    if(EXISTS "${_kit}/um/x64")
      target_link_options(fitsplat PRIVATE
        "/LIBPATH:${_kit}/um/x64"
        "/LIBPATH:${_kit}/ucrt/x64")
    endif()
  endforeach()
endif()
