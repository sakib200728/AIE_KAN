cmake_minimum_required(VERSION 3.29)
project(LeNet_source_code)

set(CMAKE_CXX_STANDARD 14)

include_directories(.)

add_executable(LeNet_source_code
        src/core01.cc
        src/core01.h
        src/core01lut.h
        src/core02.cc
        src/core02.h
        src/core02lut.h
        src/core03.cc
        src/core03.h
        src/core03lut.h
        src/core04.cc
        src/core04.h
        src/core04lut.h
        src/core05.cc
        src/core05.h
        src/core05lut.h
        src/core06.cc
        src/core06.h
        src/graph.cpp
        src/graph.h
        src/core06lut.h)
