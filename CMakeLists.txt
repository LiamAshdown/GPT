cmake_minimum_required(VERSION 4.1)
project(bye_pair_encoder)

set(CMAKE_CXX_STANDARD 17)

set(CMAKE_PREFIX_PATH "${CMAKE_CURRENT_SOURCE_DIR}/third-party/libtorch")
find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

file(GLOB_RECURSE HEADERS CONFIGURE_DEPENDS llm/*.h nn/*.h util/*.h)

add_executable(bye_pair_encoder main.cpp ${HEADERS})

target_link_libraries(bye_pair_encoder "${TORCH_LIBRARIES}")
target_include_directories(bye_pair_encoder PRIVATE
    "${TORCH_INCLUDE_DIRS}"
    "${CMAKE_CURRENT_SOURCE_DIR}/third-party")