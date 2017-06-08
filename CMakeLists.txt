cmake_minimum_required(VERSION 3.7)
project(GrabCut)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

set(CMAKE_CXX_STANDARD 14)

set(SOURCE_FILES main.cpp GaussDistribution.cpp GaussDistribution.h Hiddens.h GrabCut.cpp GrabCut.h MyUtility.h GCApplication.cpp GCApplication.h GMM.cpp GMM.h)
add_executable(GrabCut ${SOURCE_FILES})

target_link_libraries(GrabCut ${OpenCV_LIBS} )