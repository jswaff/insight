cmake_minimum_required(VERSION 3.17)
project(insight VERSION 0.1)

#include(FetchContent)
#FetchContent_Declare(
#  googletest
#  URL https://github.com/google/googletest/archive/5376968f6948923e2411081fd9372e71a59d8e77.zip
#)

# For Windows: Prevent overriding the parent project's compiler/linker settings
#set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
#FetchContent_MakeAvailable(googletest)

# Add torch and torchvision libraries
list(APPEND CMAKE_PREFIX_PATH "/opt/libtorch")
find_package(TorchVision REQUIRED)

# OpenCV libs
set(OpenCV_DIR /home/james/opencv/install/lib/cmake/opencv4)
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# build
add_executable(insight main.cpp)
target_compile_features(insight PUBLIC cxx_range_for)
target_link_libraries(insight ${OpenCV_LIBS})
target_link_libraries(insight TorchVision::TorchVision)
set_property(TARGET insight PROPERTY CXX_STANDARD 17)

# link against gtest or gtest_main as needed. Eg
#add_executable(example example.cpp)
##target_link_libraries(insight gtest_main)
##add_test(NAME example_test COMMAND example)
