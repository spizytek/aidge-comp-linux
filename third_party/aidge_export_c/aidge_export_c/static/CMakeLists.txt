cmake_minimum_required(VERSION 3.22)
project(run_export C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

file(GLOB_RECURSE PROJECT_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/data/*.c
    ${CMAKE_CURRENT_SOURCE_DIR}/dnn/src/*.c
)

add_executable(run_export main.c ${PROJECT_SOURCES})

target_include_directories(run_export PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/dnn
    ${CMAKE_CURRENT_SOURCE_DIR}/dnn/include
)

if(MSVC)
    target_compile_options(run_export PRIVATE /W4)
else()
    target_compile_options(run_export PRIVATE -Wall -Wextra -O2)
endif()
