cmake_minimum_required(VERSION 3.17)
project(insight VERSION 0.1)

# Add torch and torchvision libraries
list(APPEND CMAKE_PREFIX_PATH "/opt/libtorch")
find_package(TorchVision REQUIRED)

# OpenCV libs
set(OpenCV_DIR /home/james/opencv/install/lib/cmake/opencv4)
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# Python libs
find_package(Python COMPONENTS Development REQUIRED)
include_directories(${PYTHON_INCLUDE_DIRS})

# Google Test libs
enable_testing()
#set(GTest_DIR /opt/googletest)
#find_package(GTest CONFIG REQUIRED) 
#include_directories(${GTEST_INCLUDE_DIR})
include(FetchContent)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/5376968f6948923e2411081fd9372e71a59d8e77.zip
)
# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)


# build
file(GLOB_RECURSE sources CONFIGURE_DEPENDS "src/*.cpp")
add_executable(insight ${sources})
target_compile_features(insight PUBLIC cxx_range_for)
target_link_libraries(insight ${OpenCV_LIBS})
target_link_libraries(insight ${PYTHON_INCLUDE_DIRS})
target_link_libraries(insight TorchVision::TorchVision)
set_property(TARGET insight PROPERTY CXX_STANDARD 17)

# build test executable
file(GLOB_RECURSE test_sources CONFIGURE_DEPENDS "test/*.cpp")
add_executable(insight_test src/conv2.cpp src/conv3.cpp src/imfilter.cpp ${test_sources})
target_compile_features(insight_test PUBLIC cxx_range_for)
target_link_libraries(insight_test ${OpenCV_LIBS})
target_link_libraries(insight_test ${PYTHON_INCLUDE_DIRS})
target_link_libraries(insight_test TorchVision::TorchVision)
target_link_libraries(insight_test gtest_main)
set_property(TARGET insight_test PROPERTY CXX_STANDARD 17)
add_test(NAME matrix_test COMMAND insight_test)
