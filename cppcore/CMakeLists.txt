cmake_minimum_required(VERSION 3.5)
project(pybinding_cppcore CXX)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/cmake")

option(PB_WERROR "Make all warnings into errors" OFF)
option(PB_TESTS "Enable testing" ON)
option(PB_NATIVE_SIMD "Enable all instruction sets supported by the local machine" ON)
option(PB_MKL "Use Intel's Math Kernel Library" OFF)
option(PB_CUDA "Enable compilation of components written in CUDA" OFF)
option(PB_CARTESIAN_FLOAT "Use floats for positions" OFF)

message(STATUS "PB_WERROR: ${PB_WERROR}")
message(STATUS "PB_TESTS: ${PB_TESTS}")
message(STATUS "PB_NATIVE_SIMD: ${PB_NATIVE_SIMD}")
message(STATUS "PB_MKL: ${PB_MKL}")
message(STATUS "PB_CUDA: ${PB_CUDA}")
message(STATUS "PB_CARTESIAN_FLOAT: ${PB_CARTESIAN_FLOAT}")

add_library(cppcore
    include/compute/eigen3/lanczos.hpp
    include/compute/mkl/lanczos.hpp
    include/compute/mkl/wrapper.hpp
    include/compute/detail.hpp
    include/compute/kernel_polynomial.hpp
    include/compute/lanczos.hpp
    include/detail/algorithm.hpp
    include/detail/config.hpp
    include/detail/macros.hpp
    include/detail/opaque_alias.hpp
    include/detail/slice.hpp
    include/detail/strategy.hpp
    include/detail/sugar.hpp
    include/detail/thread.hpp
    include/detail/typelist.hpp
    include/hamiltonian/Hamiltonian.hpp
    include/hamiltonian/HamiltonianModifiers.hpp
    include/kpm/default/collectors.hpp
    include/kpm/default/Compute.hpp
    include/kpm/Bounds.hpp
    include/kpm/calc_moments.hpp
    include/kpm/Config.hpp
    include/kpm/Core.hpp
    include/kpm/Kernel.hpp
    include/kpm/Moments.hpp
    include/kpm/OptimizedHamiltonian.hpp
    include/kpm/reconstruct.hpp
    include/kpm/Starter.hpp
    include/kpm/Stats.hpp
    include/leads/HamiltonianPair.hpp
    include/leads/Leads.hpp
    include/leads/Spec.hpp
    include/leads/Structure.hpp
    include/numeric/arrayref.hpp
    include/numeric/constant.hpp
    include/numeric/dense.hpp
    include/numeric/ellmatrix.hpp
    include/numeric/random.hpp
    include/numeric/sparse.hpp
    include/numeric/sparseref.hpp
    include/numeric/traits.hpp
    include/solver/FEAST.hpp
    include/solver/Solver.hpp
    include/support/format.hpp
    include/support/simd.hpp
    include/system/CompressedSublattices.hpp
    include/system/Foundation.hpp
    include/system/HoppingBlocks.hpp
    include/system/Registry.hpp
    include/system/Shape.hpp
    include/system/StructureModifiers.hpp
    include/system/Symmetry.hpp
    include/system/System.hpp
    include/utils/Chrono.hpp
    include/KPM.hpp
    include/Lattice.hpp
    include/Model.hpp
    src/hamiltonian/Hamiltonian.cpp
    src/hamiltonian/HamiltonianModifiers.cpp
    src/kpm/default/collectors.cpp
    src/kpm/default/Compute.cpp
    src/kpm/Bounds.cpp
    src/kpm/Core.cpp
    src/kpm/Kernel.cpp
    src/kpm/Moments.cpp
    src/kpm/OptimizedHamiltonian.cpp
    src/kpm/Starter.cpp
    src/kpm/Stats.cpp
    src/leads/Leads.cpp
    src/leads/Spec.cpp
    src/leads/Structure.cpp
    src/solver/FEAST.cpp
    src/solver/Solver.cpp
    src/system/CompressedSublattices.cpp
    src/system/Foundation.cpp
    src/system/HoppingBlocks.cpp
    src/system/Registry.cpp
    src/system/Shape.cpp
    src/system/StructureModifiers.cpp
    src/system/Symmetry.cpp
    src/system/System.cpp
    src/utils/Chrono.cpp
    src/KPM.cpp
    src/Lattice.cpp
    src/Model.cpp
)
add_library(pybinding::cppcore ALIAS cppcore)
target_include_directories(cppcore PUBLIC include)

find_package(Threads REQUIRED)
target_link_libraries(cppcore PUBLIC Threads::Threads)

include(warnings)
enable_warnings(cppcore)

include(download)
download_dependency(eigen3 3.4.0
                    https://gitlab.com/libeigen/eigen/-/archive/\${VERSION}
                    eigen-\${VERSION}.tar.gz */Eigen)
target_include_directories(cppcore SYSTEM PUBLIC ${EIGEN3_INCLUDE_DIR})
download_dependency(simdpp dev # dangerous, but we need the latest version. No release since 2017.
                    https://github.com/p12tic/libsimdpp/archive/refs/heads
                    /\${VERSION}.zip */simdpp)
target_include_directories(cppcore SYSTEM PUBLIC ${SIMDPP_INCLUDE_DIR})

include(fmt)
target_link_libraries(cppcore PUBLIC fmt)

if(PB_NATIVE_SIMD AND NOT MSVC) # MSVC does not have anything like a /arch:native flag
    include(CheckCXXCompilerFlag)
    check_cxx_compiler_flag(-march=native PB_HAS_ARCH_NATIVE)
    if(PB_HAS_ARCH_NATIVE)
        MESSAGE(STATUS "march=native option added")
        target_compile_options(cppcore PUBLIC -march=native)
    endif()
endif()

if(PB_MKL)
    include(mkl)
    target_link_mkl(cppcore PUBLIC)
    target_compile_definitions(cppcore PUBLIC CPB_USE_MKL)
endif()

if(PB_CUDA)
    add_subdirectory(cuda)
    target_link_libraries(cppcore PUBLIC pybinding_cuda)
endif()

if(PB_TESTS)
    set(catch_url https://raw.githubusercontent.com/catchorg/Catch2/v\${VERSION}/extras)
    download_dependency(catch 3.8.0 ${catch_url} catch_amalgamated.hpp catch_amalgamated.cpp)
    if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/deps/catch/catch_amalgamated.hpp)
        file(RENAME ${CMAKE_CURRENT_SOURCE_DIR}/deps/catch/catch_amalgamated.hpp
                ${CMAKE_CURRENT_SOURCE_DIR}/deps/catch/catch.hpp)
    endif()
    set(INPUT_FILE "${CMAKE_CURRENT_SOURCE_DIR}/deps/catch/catch_amalgamated.cpp")
    set(OUTPUT_FILE "${CMAKE_CURRENT_SOURCE_DIR}/deps/catch/catch.cpp")
    if(EXISTS ${INPUT_FILE})
        file(RENAME ${INPUT_FILE} ${OUTPUT_FILE})
        file(READ ${OUTPUT_FILE} FILE_CONTENTS)
        string(REPLACE "#include \"catch_amalgamated.hpp\"" "#include \"catch.hpp\"" FILE_CONTENTS "${FILE_CONTENTS}")
        file(WRITE ${OUTPUT_FILE} "${FILE_CONTENTS}")
    endif()
    add_library(catch2 OBJECT ${OUTPUT_FILE})
    add_subdirectory(tests)
endif()

if(PB_CARTESIAN_FLOAT)
    target_compile_definitions(cppcore PUBLIC CPB_CARTESIAN_FLOAT)
endif()
