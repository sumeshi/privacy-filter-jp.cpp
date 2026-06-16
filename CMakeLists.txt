cmake_minimum_required(VERSION 3.21)
project(privacy-filter C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(PF_VULKAN "Enable the ggml Vulkan backend" OFF)
option(PF_CUDA   "Enable the ggml CUDA backend"   OFF)
option(PF_FUZZ   "Build libFuzzer targets (clang only)" OFF)
# OFF lets a consumer (e.g. the LocalAI gRPC backend) add_subdirectory() this
# project and link the pf lib without pulling in the CLI/bench/test tree.
option(PF_BUILD_TOOLS "Build pf-cli and pf-bench" ON)
option(PF_BUILD_TESTS "Build the test suite"      ON)
set(PF_FUZZ_SANITIZERS "fuzzer,address,undefined" CACHE STRING
    "Sanitizers for fuzz builds")

if (PF_VULKAN)
    set(GGML_VULKAN ON)
endif()
if (PF_CUDA)
    set(GGML_CUDA ON)
    # ggml's VMM pool uses the CUDA *driver* API (cuMem*), which links
    # libcuda.so.1 — present only at runtime via the GPU driver. A GPU-less
    # build container then can't link the final binary against the shared
    # libggml-cuda.so. This classifier is tiny and single-GPU, so the legacy
    # cudaMalloc pool is equivalent; disabling VMM drops the driver dependency.
    set(GGML_CUDA_NO_VMM ON)
endif()
set(GGML_BUILD_TESTS    OFF)
set(GGML_BUILD_EXAMPLES OFF)
add_subdirectory(ggml)

if (PF_FUZZ)
    if (NOT CMAKE_CXX_COMPILER_ID MATCHES "Clang")
        message(FATAL_ERROR "PF_FUZZ requires clang (libFuzzer)")
    endif()
    add_compile_options(-fsanitize=${PF_FUZZ_SANITIZERS} -fno-omit-frame-pointer -g)
    add_link_options(-fsanitize=${PF_FUZZ_SANITIZERS})
endif()

add_library(pf STATIC
    src/backend.cpp
    src/gguf_loader.cpp
    src/model.cpp
    src/ner.cpp
    src/pf.cpp
    src/tokenizer.cpp
)
target_include_directories(pf PUBLIC include)
target_link_libraries(pf PUBLIC ggml)
target_compile_options(pf PRIVATE -Wall -Wextra)

if (PF_BUILD_TOOLS)
    add_executable(pf-cli tools/pf-cli.cpp)
    target_link_libraries(pf-cli PRIVATE pf)
    target_include_directories(pf-cli PRIVATE src)
    target_compile_options(pf-cli PRIVATE -Wall -Wextra)

    add_executable(pf-bench bench/pf-bench.cpp)
    target_link_libraries(pf-bench PRIVATE pf)
    target_include_directories(pf-bench PRIVATE src)

    # GFLOP/s by dtype/shape -- diagnostic for the CPU matmul analysis
    # (docs/cpu-perf.md). Links ggml directly; no pf/model deps.
    add_executable(pf-gemm-bench bench/gemm_microbench.cpp)
    target_link_libraries(pf-gemm-bench PRIVATE ggml)
endif()

if (PF_FUZZ)
    foreach(target fuzz_tokenizer fuzz_gguf)
        add_executable(${target} fuzz/${target}.cpp)
        target_link_libraries(${target} PRIVATE pf)
        target_include_directories(${target} PRIVATE src)
    endforeach()
endif()

if (PF_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()
