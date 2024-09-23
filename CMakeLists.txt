cmake_minimum_required(VERSION 3.15)
project(onnx_cpp_lib)
set(CMAKE_CXX_STANDARD 17)

find_package(Protobuf REQUIRED)
include_directories(${Protobuf_INCLUDE_DIRS})

# Compile your .proto files (if needed)
# file(GLOB PROTO_FILES "src/onnx/*.proto3")
# protobuf_generate_cpp(PROTO_SRCS PROTO_HDRS ${PROTO_FILES})

file(GLOB_RECURSE LIB_SRC_FILES src/*.cpp src/*.cc)
add_library(onnx_cpp_lib STATIC ${LIB_SRC_FILES} ${PROTO_SRCS})
target_link_libraries(onnx_cpp_lib ${Protobuf_LIBRARIES})
target_include_directories(onnx_cpp_lib PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)

# GTest
enable_testing()
include(FetchContent)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/releases/download/v1.15.2/googletest-1.15.2.tar.gz
  DOWNLOAD_EXTRACT_TIMESTAMP TRUE
)

FetchContent_MakeAvailable(googletest)
include_directories(${gtest_SOURCE_DIR}/include ${gtest_SOURCE_DIR})

file(GLOB TEST_SOURCES "${PROJECT_SOURCE_DIR}/tests/*.cpp")
foreach(TEST_SOURCE ${TEST_SOURCES})
    get_filename_component(TEST_NAME ${TEST_SOURCE} NAME_WE)
    add_executable(${TEST_NAME} ${TEST_SOURCE})
    target_link_libraries(${TEST_NAME} gtest gtest_main onnx_cpp_lib ${Protobuf_LIBRARIES})
    add_test(NAME ${TEST_NAME} COMMAND ${TEST_NAME})
endforeach()