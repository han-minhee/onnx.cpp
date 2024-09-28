cmake_minimum_required(VERSION 3.20)
if(USEHIP)
    set(CMAKE_CXX_COMPILER /opt/rocm/bin/hipcc) # Adjust the path if needed
endif()


project(onnx_cpp_lib LANGUAGES CXX HIP)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD 17)

# HIP configuration
if(USEHIP)
    message(STATUS "USEHIP flag is set, using HIP for compilation")
    find_package(HIP REQUIRED)

    # Define USE_HIP globally
    add_compile_definitions(USE_HIP)
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

if(USEHIP)
    add_library(onnx_cpp_lib STATIC ${LIB_SRC_FILES} ${PROTO_SRCS})
    target_link_libraries(onnx_cpp_lib ${Protobuf_LIBRARIES} hip::device)

    target_include_directories(onnx_cpp_lib PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${PROTO_OUTPUT_DIR}
        ${CMAKE_CURRENT_BINARY_DIR}
        ${HIP_INCLUDE_DIRS}
    )

    target_compile_definitions(onnx_cpp_lib PRIVATE USE_HIP)
else()
    add_library(onnx_cpp_lib STATIC ${LIB_SRC_FILES} ${PROTO_SRCS})

    target_link_libraries(onnx_cpp_lib ${Protobuf_LIBRARIES})

    target_include_directories(onnx_cpp_lib PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${PROTO_OUTPUT_DIR}
        ${CMAKE_CURRENT_BINARY_DIR}
    )
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
    target_link_libraries(${TEST_NAME} gtest gtest_main onnx_cpp_lib ${Protobuf_LIBRARIES})
    target_compile_definitions(${TEST_NAME} PRIVATE USE_HIP)

    add_test(NAME ${TEST_NAME} COMMAND ${TEST_NAME})
endforeach()
