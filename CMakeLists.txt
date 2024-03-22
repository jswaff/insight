cmake_minimum_required(VERSION 3.10 FATAL_ERROR)
project(insight VERSION 0.1)
#set(CMAKE_OSX_ARCHITECTURES "x86_64")

list(APPEND CMAKE_PREFIX_PATH "/opt/libtorch")
list(APPEND CMAKE_PREFIX_PATH "/home/james/vision/build")

set(OpenCV_DIR, "/home/james/opencv_build/build")

# Tell compiler to use C++ 17 features
#set(CMAKE_CXX_STANDARD 17)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(insight main.cpp)
target_link_libraries(insight "${OpenCV_LIBS}")


#list(APPEND CMAKE_PREFIX_PATH "/home/james/vision/build")
find_package(Torch REQUIRED)
find_package(TorchVision REQUIRED)

#target_link_libraries(insight "${TORCH_LIBRARIES}")
target_compile_features(insight PUBLIC cxx_range_for)
target_link_libraries(insight TorchVision::TorchVision)
set_property(TARGET insight PROPERTY CXX_STANDARD 17)
