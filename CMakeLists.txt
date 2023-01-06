# Set CMake minimum version to 3.8
cmake_minimum_required(VERSION 3.8)
# Create project with VERSION and LANGUAGE flags
project(MLDD VERSION 1.0.0 LANGUAGES CXX)

# Add our main.cpp as a library to CMake target machine-learning-differential-diagnoses
add_library(MLDD src/main.cpp)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_SOURCE_DIR}/cmake-build-debug")

target_include_directories(MLDD
        PUBLIC
            $<INSTALL_INTERFACE:include>
            $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/src
        )
find_package(mlpack 3.4.2 REQUIRED MODULE)

target_compile_options(MLDD PRIVATE -Wall)
target_compile_features(MLDD PRIVATE cxx_std_17)


