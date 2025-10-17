cmake_minimum_required (VERSION 3.20)

set(CMAKE_PROJECT_TOP_LEVEL_INCLUDES cmake/conan_provider.cmake)

project(ncnn_test VERSION 1.0 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)

set(CMAKE_EXPORT_COMPILE_COMMANDS TRUE)
set(CMAKE_COLOR_DIAGNOSTICS TRUE)

set(LIBRARY_OUTPUT_PATH ${CMAKE_BINARY_DIR})
set(EXECUTABLE_OUTPUT_PATH ${CMAKE_BINARY_DIR})

add_subdirectory(${CMAKE_CURRENT_LIST_DIR}/source/)

target_include_directories(drl_run PUBLIC
	$<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/onnxruntime/include>
)

# build_arm_gpu build_arm build_x64
if(CMAKE_SYSTEM_PROCESSOR MATCHES "(x86_64)|(X86_64)|(x64)|(X64)|(amd64)|(AMD64)|(i686)")
set(ORT_LIB_DIR x64) 
else()
set(ORT_LIB_DIR aarch64)
endif()

target_link_libraries(drl_run PRIVATE ${CMAKE_CURRENT_LIST_DIR}/onnxruntime/lib/${ORT_LIB_DIR}/libonnxruntime.so)

target_link_libraries(drl_run 
	PRIVATE
	pthread
	rt
    dl 
)
