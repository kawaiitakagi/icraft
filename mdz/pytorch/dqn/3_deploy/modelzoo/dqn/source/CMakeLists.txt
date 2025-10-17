cmake_minimum_required (VERSION 3.20)
# find_package(opencv REQUIRED)
# find_package(mnn REQUIRED)

add_executable(ort_run "")

target_sources(ort_run PRIVATE
	main.cpp
)