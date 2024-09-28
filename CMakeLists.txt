cmake_minimum_required(VERSION 3.20)
if(USE_HIP)
    set(CMAKE_CXX_COMPILER /opt/rocm/bin/hipcc) # Adjust the path if needed
    project(onnx_cpp_lib LANGUAGES CXX HIP)
else()
    project(onnx_cpp_lib LANGUAGES CXX)
endif()

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD 17)

# HIP configuration
if(USE_HIP)
    message(STATUS "USE_HIP flag is set, using HIP for compilation")
    find_package(HIP REQUIRED)
else()
    message(STATUS "USE_HIP flag is not set, using default compiler")
endif()

# ProtoBuf configuration
find_package(Protobuf REQUIRED)
include_directories(${Protobuf_INCLUDE_DIRS})
file(GLOB PROTO_FILES proto/*.proto)
set(PROTO_INCLUDE_DIRS "proto/")
set(PROTO_OUTPUT_DIR ${CMAKE_BINARY_DIR}/generated_protobuf)
file(MAKE_DIRECTORY ${PROTO_OUTPUT_DIR})
PROTOBUF_GENERATE_CPP(PROTO_SRCS PROTO_HDRS ${PROTO_FILES}
    IMPORT_DIRS ${PROTO_INCLUDE_DIRS}
    PROTOC_OUT_DIR ${PROTO_OUTPUT_DIR}
)

file(GLOB_RECURSE LIB_SRC_FILES src/*.cpp src/*.cc)

add_library(onnx_cpp_lib STATIC ${LIB_SRC_FILES} ${PROTO_SRCS})

target_link_libraries(onnx_cpp_lib PRIVATE ${Protobuf_LIBRARIES})

target_include_directories(onnx_cpp_lib PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${PROTO_OUTPUT_DIR}
    ${CMAKE_CURRENT_BINARY_DIR}
)

if(USE_HIP)
    target_link_libraries(onnx_cpp_lib PRIVATE hip::device)
    target_include_directories(onnx_cpp_lib PRIVATE ${HIP_INCLUDE_DIRS})
    target_compile_definitions(onnx_cpp_lib PRIVATE USE_HIP)
endif()

# GTest configuration
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
    target_link_libraries(${TEST_NAME} PRIVATE gtest gtest_main onnx_cpp_lib ${Protobuf_LIBRARIES})

    if(USE_HIP)
        target_link_libraries(${TEST_NAME} PRIVATE hip::device)
        target_compile_definitions(${TEST_NAME} PRIVATE USE_HIP)
    endif()

    add_test(NAME ${TEST_NAME} COMMAND ${TEST_NAME})
endforeach()
