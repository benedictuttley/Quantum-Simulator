cmake_minimum_required(VERSION 3.13)
project(matrix_expo C)

set(CMAKE_C_STANDARD 99)

add_executable(matrix_expo main.c)




target_link_libraries(matrix_expo m)