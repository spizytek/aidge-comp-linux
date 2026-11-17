cmake_minimum_required(VERSION 3.22)
project(run_export CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(AIDGE_ENABLE_OPENMP "Enable OpenMP in exported kernels" OFF)

if(AIDGE_ENABLE_OPENMP)
    find_package(OpenMP)

    if(OpenMP_CXX_FOUND)
        message(STATUS "OpenMP found")
    else()
        message(STATUS "OpenMP not found; building without OpenMP")
    endif()
endif()

# This file will be dynamically generated in the export folder by export_scheduler() call
# via the cmake_libs.jinja template
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/libs.cmake)

# Project sources
file(GLOB_RECURSE PROJECT_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/data/*.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/dnn/src/*.cpp
)

# Sources and includes shared by the regular and strict executables.
set(AIDGE_EXPORT_SOURCES main.cpp ${PROJECT_SOURCES})
set(AIDGE_EXPORT_INCLUDE_DIRS
    ${CMAKE_CURRENT_SOURCE_DIR}/dnn
    ${CMAKE_CURRENT_SOURCE_DIR}/dnn/include
    ${CMAKE_CURRENT_SOURCE_DIR}
)

function(aidge_configure_export_target target)
    if(AIDGE_ENABLE_OPENMP AND OpenMP_CXX_FOUND)
        target_link_libraries(${target} PRIVATE OpenMP::OpenMP_CXX)
    endif()

    target_include_directories(${target} PRIVATE ${AIDGE_EXPORT_INCLUDE_DIRS})
    target_link_libraries(${target} PRIVATE ${AIDGE_EXPORT_LIBS})

    if(MSVC)
        target_compile_options(${target} PRIVATE /W4 /O2 /fp:fast)
    elseif(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        target_compile_options(${target} PRIVATE -Wall -Wextra -O3 -ffast-math -march=native)
    endif()
endfunction()

# Default export executable.
add_executable(run_export ${AIDGE_EXPORT_SOURCES})
aidge_configure_export_target(run_export)

# Opt-in strict build. It has its own objects and output directory so running
# this target never changes the regular build artifact.
add_executable(strict EXCLUDE_FROM_ALL ${AIDGE_EXPORT_SOURCES})
aidge_configure_export_target(strict)
set_target_properties(strict PROPERTIES
    OUTPUT_NAME run_export
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/strict
)
if(MSVC)
    target_compile_options(strict PRIVATE /WX /permissive-)
elseif(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(strict PRIVATE
        -Werror
        -Wpedantic
        -Wconversion
        -Wsign-conversion
    )
endif()

# Static-analysis targets. Standard profiles focus on likely defects, while
# strict profiles deliberately enable broader and potentially noisier checks.
set(AIDGE_ANALYSIS_INCLUDES)
foreach(include_dir IN LISTS AIDGE_EXPORT_INCLUDE_DIRS)
    list(APPEND AIDGE_ANALYSIS_INCLUDES -I${include_dir})
endforeach()

find_program(CPPCHECK_EXECUTABLE NAMES cppcheck)
if(CPPCHECK_EXECUTABLE)
    add_custom_target(cppcheck
        COMMAND ${CPPCHECK_EXECUTABLE}
            --enable=warning
            --error-exitcode=1
            --std=c++17
            --suppress=missingIncludeSystem
            ${AIDGE_ANALYSIS_INCLUDES}
            ${AIDGE_EXPORT_SOURCES}
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        COMMAND_EXPAND_LISTS
        VERBATIM
    )
    add_custom_target(cppcheck-strict
        COMMAND ${CPPCHECK_EXECUTABLE}
            --enable=warning,style,performance,portability
            --inconclusive
            --error-exitcode=1
            --std=c++17
            --suppress=missingIncludeSystem
            ${AIDGE_ANALYSIS_INCLUDES}
            ${AIDGE_EXPORT_SOURCES}
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        COMMAND_EXPAND_LISTS
        VERBATIM
    )
else()
    add_custom_target(cppcheck
        COMMAND ${CMAKE_COMMAND} -E echo "cppcheck is required for this target"
        COMMAND ${CMAKE_COMMAND} -E false
    )
    add_custom_target(cppcheck-strict
        COMMAND ${CMAKE_COMMAND} -E echo "cppcheck is required for this target"
        COMMAND ${CMAKE_COMMAND} -E false
    )
endif()

find_program(CLANG_TIDY_EXECUTABLE NAMES clang-tidy)
if(CLANG_TIDY_EXECUTABLE)
    add_custom_target(clang-tidy
        COMMAND ${CLANG_TIDY_EXECUTABLE}
            --config={}
            --checks=-*,clang-analyzer-*
            --warnings-as-errors=clang-analyzer-*
            -p=${CMAKE_BINARY_DIR}
            ${AIDGE_EXPORT_SOURCES}
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        COMMAND_EXPAND_LISTS
        VERBATIM
    )
    add_custom_target(clang-tidy-strict
        COMMAND ${CLANG_TIDY_EXECUTABLE}
            --config={}
            --checks=-*,clang-analyzer-*,bugprone-*,performance-*,portability-*
            --warnings-as-errors=clang-analyzer-*,bugprone-*,performance-*,portability-*
            -p=${CMAKE_BINARY_DIR}
            ${AIDGE_EXPORT_SOURCES}
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        COMMAND_EXPAND_LISTS
        VERBATIM
    )
else()
    add_custom_target(clang-tidy
        COMMAND ${CMAKE_COMMAND} -E echo "clang-tidy is required for this target"
        COMMAND ${CMAKE_COMMAND} -E false
    )
    add_custom_target(clang-tidy-strict
        COMMAND ${CMAKE_COMMAND} -E echo "clang-tidy is required for this target"
        COMMAND ${CMAKE_COMMAND} -E false
    )
endif()

add_custom_target(static-analysis DEPENDS cppcheck clang-tidy)
add_custom_target(static-analysis-strict
    DEPENDS cppcheck-strict clang-tidy-strict
)
