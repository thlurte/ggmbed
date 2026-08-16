cmake_minimum_required(VERSION 3.19...3.30)
project(ggmbed_backend LANGUAGES C CXX)

if(POLICY CMP0169)
    cmake_policy(SET CMP0169 OLD)
endif()

# FORCE PIC FOR ALL DEPENDENCIES (Fixes the R_X86_64_PC32 linker error)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Use standard C++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Fix manylinux_2_28 aarch64 compatibility: newer GCC generates outline atomic
# calls (__aarch64_ldadd4_acq_rel, etc.) that require glibc 2.31+ symbols.
# Using -mno-outline-atomics forces inline atomics, keeping glibc <= 2.28.
if(CMAKE_SYSTEM_PROCESSOR STREQUAL "aarch64" AND UNIX AND NOT APPLE)
    add_compile_options(-mno-outline-atomics)
endif()

include(FetchContent)

# 2. Fetch llama.cpp
# Disable examples/tests to speed up build
set(BUILD_SHARED_LIBS OFF CACHE BOOL "Build shared libraries" FORCE)
set(LLAMA_BUILD_TESTS OFF CACHE BOOL "Disable tests" FORCE)
set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "Disable examples" FORCE)
set(LLAMA_BUILD_SERVER OFF CACHE BOOL "Disable server" FORCE)
set(GGML_OPENMP OFF CACHE BOOL "Disable OpenMP" FORCE)
set(GGML_NATIVE ON CACHE BOOL "Enable native optimization")
set(GGML_METAL OFF CACHE BOOL "Disable Metal" FORCE)
set(LLAMA_METAL OFF CACHE BOOL "Disable Metal" FORCE)

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/build/_deps/llama_cpp-src/CMakeLists.txt")
    set(FETCHCONTENT_SOURCE_DIR_LLAMA_CPP "${CMAKE_CURRENT_SOURCE_DIR}/build/_deps/llama_cpp-src")
endif()

FetchContent_Declare(
    llama_cpp
    GIT_REPOSITORY https://github.com/ggml-org/llama.cpp.git
    GIT_TAG        b8500 # Stable release tag
    GIT_SHALLOW    TRUE
)
FetchContent_MakeAvailable(llama_cpp)

# 3. Locate nanobind automatically using your current Python environment
find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)
execute_process(
    COMMAND "${Python_EXECUTABLE}" -c "import nanobind; print(nanobind.cmake_dir())"
    OUTPUT_VARIABLE nanobind_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
find_package(nanobind REQUIRED)

# 4. Define our C++ extension module
nanobind_add_module(
    _core                         # The name of the module
    src/ggmbed/_core.cpp          # Where the C++ source code is
)

# 5. Link libraries
target_link_libraries(_core PRIVATE llama)

# Note: We do NOT use -static-libgcc or -static-libstdc++ because statically linking
# standard C++ libraries into a Python extension module causes segmentation faults
# (e.g. std::codecvt / std::locale collisions) when dynamically loaded alongside
# Python and NumPy. cibuildwheel's auditwheel automatically handles copying and
# renaming libstdc++.so for broad manylinux compatibility.

# Configure RPATH for the extension module to find libraries in the same directory
if(APPLE)
    set_target_properties(_core PROPERTIES
        INSTALL_RPATH "@loader_path;@loader_path/../lib"
        BUILD_WITH_INSTALL_RPATH TRUE
    )
elseif(UNIX)
    set_target_properties(_core PROPERTIES
        INSTALL_RPATH "$ORIGIN;$ORIGIN/../lib"
        BUILD_WITH_INSTALL_RPATH TRUE
    )
endif()

# 6. Tell the build system to drop the finished binary inside your python package folder
install(TARGETS _core DESTINATION ggmbed)