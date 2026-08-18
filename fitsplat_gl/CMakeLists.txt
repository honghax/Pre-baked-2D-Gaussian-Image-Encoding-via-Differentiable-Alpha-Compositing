cmake_minimum_required(VERSION 3.20)
project(fitsplat_gl LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(GLFW_ROOT "${CMAKE_CURRENT_LIST_DIR}/../third_party/glfw-3.4.bin.WIN64")

add_executable(fitsplat_gl
    main.cpp
    ${CMAKE_CURRENT_LIST_DIR}/../third_party/glad.c
)

target_include_directories(fitsplat_gl PRIVATE
    ${CMAKE_CURRENT_LIST_DIR}/../third_party/include
    ${GLFW_ROOT}/include
)

target_link_libraries(fitsplat_gl PRIVATE
    ${GLFW_ROOT}/lib-vc2022/glfw3.lib
    opengl32
)

if(MSVC)
    target_compile_options(fitsplat_gl PRIVATE /W4 /utf-8)
    # 显式指定系统库路径（链接器环境的 LIB 变量可能被沙箱过滤，
    # 裸库名 opengl32/kernel32/msvcprt 等会因找不到 .lib 而链接失败）
    get_filename_component(_MSVC_ROOT "${CMAKE_CXX_COMPILER}" DIRECTORY)
    get_filename_component(_MSVC_ROOT "${_MSVC_ROOT}/../../.." ABSOLUTE)   # .../MSVC/<ver>
    target_link_options(fitsplat_gl PRIVATE "/LIBPATH:${_MSVC_ROOT}/lib/x64")
    file(GLOB _WIN_KITS "C:/Program Files (x86)/Windows Kits/10/Lib/*")
    foreach(_kit ${_WIN_KITS})
        if(EXISTS "${_kit}/um/x64")
            target_link_options(fitsplat_gl PRIVATE
                "/LIBPATH:${_kit}/um/x64"
                "/LIBPATH:${_kit}/ucrt/x64")
        endif()
    endforeach()
endif()
