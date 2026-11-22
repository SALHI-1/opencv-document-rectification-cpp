# ============================================================================
# Projet : Transformation de Perspective sous OpenCV (C++17)
# Auteur : Mohammed SALHI
# ============================================================================

cmake_minimum_required(VERSION 3.10)
project(PerspectiveDocumentSalhi LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenCV REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(perspective_transform main.cpp)
target_link_libraries(perspective_transform ${OpenCV_LIBS})
