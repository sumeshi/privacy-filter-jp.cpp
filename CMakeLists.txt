cmake_minimum_required(VERSION 3.21)
project(privacy-filter C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(PF_VULKAN "Enable the ggml Vulkan backend" OFF)
option(PF_FUZZ   "Build libFuzzer targets (clang only)" OFF)
set(PF_FUZZ_SANITIZERS "fuzzer,address,undefined" CACHE STRING
    "Sanitizers for fuzz builds")

if (PF_VULKAN)
    set(GGML_VULKAN ON)
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

add_executable(pf-cli tools/pf-cli.cpp)
target_link_libraries(pf-cli PRIVATE pf)
target_include_directories(pf-cli PRIVATE src)
target_compile_options(pf-cli PRIVATE -Wall -Wextra)

enable_testing()
add_subdirectory(tests)
