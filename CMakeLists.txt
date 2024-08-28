cmake_minimum_required(VERSION 3.29)
project(LeNet_source_code)

set(CMAKE_CXX_STANDARD 14)

include_directories(.)

add_executable(LeNet_source_code
    core01.cc
    core01.h
    core01lut.h
    core02.cc
    core02.h
    core02lut.h
    core03.cc
    core03.h
    core03lut.h
    core04.cc
    core04.h
    core04lut.h
    core05.cc
    core05.h
    core05lut.h
        core06.cc
        core06.h
    graph.cpp
    graph.h
        core06lut.h)
