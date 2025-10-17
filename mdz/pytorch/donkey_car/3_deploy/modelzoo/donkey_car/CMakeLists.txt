cmake_minimum_required(VERSION 3.10)
# set sln name
set(TARGET_NAME main)

project(${TARGET_NAME} LANGUAGES CXX C)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED YES)

set(LIBRARY_OUTPUT_PATH ${CMAKE_BINARY_DIR})
set(EXECUTABLE_OUTPUT_PATH ${CMAKE_BINARY_DIR})

add_compile_options("$<$<C_COMPILER_ID:MSVC>:/utf-8>")
add_compile_options("$<$<CXX_COMPILER_ID:MSVC>:/utf-8>")
#/Zi 是 MSVC 编译器的一个选项，用于生成调试信息
add_compile_options("$<$<C_COMPILER_ID:MSVC>:/Zi>")
add_compile_options("$<$<CXX_COMPILER_ID:MSVC>:/Zi>")
#/permissive- 用于启用更严格的语法检查和符合性检查
add_compile_options("$<$<C_COMPILER_ID:MSVC>:/permissive->")
add_compile_options("$<$<CXX_COMPILER_ID:MSVC>:/permissive->")
#/wd4244 是 MSVC 编译器的一个选项，用于禁止特定的警告
add_compile_options("$<$<C_COMPILER_ID:MSVC>:/wd4244>")
add_compile_options("$<$<CXX_COMPILER_ID:MSVC>:/wd4244>")
#/INCREMENTAL:NO 是 MSVC 链接器的一个选项，用于禁用增量链接
add_link_options("$<$<C_COMPILER_ID:MSVC>:/INCREMENTAL:NO>")
add_link_options("$<$<CXX_COMPILER_ID:MSVC>:/INCREMENTAL:NO>")
add_link_options("$<$<C_COMPILER_ID:MSVC>:/DEBUG>")
add_link_options("$<$<CXX_COMPILER_ID:MSVC>:/DEBUG>")
#/OPT:REF 是 MSVC 链接器的一个选项，用于进行引用消除优化
add_link_options("$<$<AND:$<C_COMPILER_ID:MSVC>,$<CONFIG:Release>>:/OPT:REF>")
add_link_options("$<$<AND:$<CXX_COMPILER_ID:MSVC>,$<CONFIG:Release>>:/OPT:REF>")
#/OPT:ICF 是 MSVC 链接器的一个选项，用于进行函数节省优化（Identical Code Folding）
add_link_options("$<$<AND:$<C_COMPILER_ID:MSVC>,$<CONFIG:Release>>:/OPT:ICF>")
add_link_options("$<$<AND:$<CXX_COMPILER_ID:MSVC>,$<CONFIG:Release>>:/OPT:ICF>")

include_directories(
	../../Deps/thirdparty/include/Eigen-3.4.0
	../../Deps/thirdparty/include
    ../../Deps/modelzoo
	${CMAKE_CURRENT_LIST_DIR}/onnxruntime/include
	)

# Unix/Linux 操作系统
SET(CMAKE_SYSTEM_NAME Linux)
SET(CMAKE_SYSTEM_PROCESSOR aarch64)
SET(CMAKE_C_COMPILER /usr/bin/aarch64-linux-gnu-gcc)
SET(CMAKE_CXX_COMPILER /usr/bin/aarch64-linux-gnu-g++)
# SET(CMAKE_BUILD_TYPE Debug)
SET(CMAKE_BUILD_TYPE Release)


link_directories(
	../../Deps/thirdparty/a
)

link_libraries(
	opencv_imgcodecs opencv_imgproc opencv_core tegra_hal libjpeg-turbo ittnotify libpng libwebp libtiff libopenjp2 zlib 
	fmt pthread dl
	yaml-cpp
	${CMAKE_CURRENT_LIST_DIR}/onnxruntime/lib/aarch64/libonnxruntime.so
	rt
)
find_package(Icraft-BuyiBackend REQUIRED)
find_package(Icraft-HostBackend REQUIRED)


add_executable(${TARGET_NAME} source/main.cpp)
# find_package 会找指定包使用的所有依赖包
target_link_libraries(${TARGET_NAME} PRIVATE Icraft::HostBackend Icraft::BuyiBackend)