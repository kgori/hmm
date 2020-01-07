cmake_minimum_required(VERSION 3.15)
project(hmm)

set(CMAKE_CXX_STANDARD 17)

include(ExternalProject)

ExternalProject_Add(
        xtl
        GIT_REPOSITORY https://github.com/xtensor-stack/xtl.git
        GIT_TAG 0.6.9
        PREFIX ${CMAKE_CURRENT_BINARY_DIR}/xtl
        CMAKE_ARGS -DCMAKE_INSTALL_PREFIX:PATH=<INSTALL_DIR>
)
ExternalProject_Get_Property(xtl install_dir)
set(XTL_INCLUDE_PATH "${install_dir}/include")
message("xtl include dir " ${XTL_INCLUDE_PATH})

ExternalProject_Add(
        xtensor
        GIT_REPOSITORY https://github.com/xtensor-stack/xtensor.git
        GIT_TAG 0.20.10
        PREFIX ${CMAKE_CURRENT_BINARY_DIR}/xtensor
        CMAKE_ARGS -DCMAKE_INSTALL_PREFIX:PATH=<INSTALL_DIR>
)
ExternalProject_Get_Property(xtensor install_dir)
set(XTENSOR_INCLUDE_PATH "${install_dir}/include")
message("xtensor include dir " ${XTENSOR_INCLUDE_PATH})

ExternalProject_Add(
        xsimd
        GIT_REPOSITORY https://github.com/xtensor-stack/xsimd.git
        GIT_TAG 7.4.2
        PREFIX ${CMAKE_CURRENT_BINARY_DIR}/xsimd
        CMAKE_ARGS -DCMAKE_INSTALL_PREFIX:PATH=<INSTALL_DIR>
)
ExternalProject_Get_Property(xsimd install_dir)
set(XSIMD_INCLUDE_PATH "${install_dir}/include")
message("xsimd include dir " ${XSIMD_INCLUDE_PATH})

ExternalProject_Add(
        xtensor-blas
        GIT_REPOSITORY https://github.com/xtensor-stack/xtensor-blas.git
        GIT_TAG 0.16.1
        PREFIX ${CMAKE_CURRENT_BINARY_DIR}/xtensor-blas
        CMAKE_ARGS -DCMAKE_INSTALL_PREFIX:PATH=<INSTALL_DIR>
)
ExternalProject_Get_Property(xtensor-blas install_dir)
set(XTENSORBLAS_INCLUDE_PATH "${install_dir}/include")
message("xtensor-blas include dir " ${XTENSORBLAS_INCLUDE_PATH})

list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake/Modules)

find_package(LAPACK REQUIRED)
set(BLA_VENDOR OpenBLAS)
find_package(OpenBLAS REQUIRED)

add_executable(hmm main.cpp hmm.cpp)
add_dependencies(hmm xtl)
add_dependencies(hmm xtensor)
add_dependencies(hmm xsimd)
add_dependencies(hmm xtensor-blas)

target_include_directories(hmm PUBLIC "${XTL_INCLUDE_PATH}")
target_include_directories(hmm PUBLIC "${XTENSOR_INCLUDE_PATH}")
target_include_directories(hmm PUBLIC "${XSIMD_INCLUDE_PATH}")
target_include_directories(hmm PUBLIC "${XTENSORBLAS_INCLUDE_PATH}")
target_link_libraries(hmm ${CONAN_LIBS} ${BLAS_LIBRARIES} ${LAPACK_LIBRARIES})

#if(MSVC)
#    target_compile_options(hmm PRIVATE /W4 /WX)
#else()
#    target_compile_options(hmm PRIVATE -Wall -Wextra -pedantic)
#endif()