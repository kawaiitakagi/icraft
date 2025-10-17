cmake_minimum_required (VERSION 3.20)
# find_package(opencv REQUIRED)
# find_package(mnn REQUIRED)

add_executable(drl_run "")

target_sources(drl_run PRIVATE
	main.cpp
)