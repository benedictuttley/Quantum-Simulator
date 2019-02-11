cmake_minimum_required(VERSION 3.13)
project(matrix_expo C)

set(CMAKE_C_STANDARD 99)

add_executable(matrix_expo main.c)

include_directories(/opt/OpenBLAS/include)
link_directories(/opt/OpenBLAS/lib)
target_link_libraries(matrix_expo /opt/OpenBLAS/lib/libopenblas.a)
# Change "Test" to your project name, this is for linking static lib.

target_link_libraries(matrix_expo m)


