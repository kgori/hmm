cmake_minimum_required(VERSION 3.15)
project(hmm)

set(CMAKE_CXX_STANDARD 14)
include(${CMAKE_BINARY_DIR}/conanbuildinfo.cmake)
conan_basic_setup()

include_directories(
        /Users/kg8/homebrew/include
)

list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake/Modules)

find_package(LAPACK REQUIRED)
set(BLA_VENDOR OpenBLAS)
find_package(OpenBLAS REQUIRED)

add_executable(hmm main.cpp hmm.cpp)
target_link_libraries(hmm ${CONAN_LIBS} ${BLAS_LIBRARIES} ${LAPACK_LIBRARIES})

#if(MSVC)
#    target_compile_options(hmm PRIVATE /W4 /WX)
#else()
#    target_compile_options(hmm PRIVATE -Wall -Wextra -pedantic)
#endif()