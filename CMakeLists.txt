cmake_minimum_required(VERSION 3.15)
project(onnx_cpp_lib)
set(CMAKE_CXX_STANDARD 17)

find_package(Protobuf REQUIRED)
include_directories(${Protobuf_INCLUDE_DIRS})

# Add the proto files
file(GLOB PROTO_FILES proto/*.proto)
set(PROTO_INCLUDE_DIRS "proto/")

# Specify the output directory for the generated Protobuf files
set(PROTO_OUTPUT_DIR ${CMAKE_BINARY_DIR}/generated_protobuf)

# Ensure the output directory exists
file(MAKE_DIRECTORY ${PROTO_OUTPUT_DIR})

# Generate C++ source files from proto files into the output directory
PROTOBUF_GENERATE_CPP(PROTO_SRCS PROTO_HDRS ${PROTO_FILES}
    IMPORT_DIRS ${PROTO_INCLUDE_DIRS}
    PROTOC_OUT_DIR ${PROTO_OUTPUT_DIR}
)

# Add the source files for your library
file(GLOB_RECURSE LIB_SRC_FILES src/*.cpp src/*.cc)

# Add the library target, including the generated proto sources
add_library(onnx_cpp_lib STATIC ${LIB_SRC_FILES} ${PROTO_SRCS})

# Link protobuf libraries
target_link_libraries(onnx_cpp_lib ${Protobuf_LIBRARIES})

# Include directories for the library, including the generated Protobuf files
target_include_directories(onnx_cpp_lib PUBLIC 
    ${CMAKE_CURRENT_SOURCE_DIR}/include 
    ${PROTO_OUTPUT_DIR}  # Include the directory where the generated proto headers are located
    ${CMAKE_CURRENT_BINARY_DIR}  # Additional include if needed
)

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

# Locate the test files
file(GLOB TEST_SOURCES "${PROJECT_SOURCE_DIR}/tests/*.cpp")
foreach(TEST_SOURCE ${TEST_SOURCES})
    get_filename_component(TEST_NAME ${TEST_SOURCE} NAME_WE)
    add_executable(${TEST_NAME} ${TEST_SOURCE})
    target_link_libraries(${TEST_NAME} gtest gtest_main onnx_cpp_lib ${Protobuf_LIBRARIES})
    add_test(NAME ${TEST_NAME} COMMAND ${TEST_NAME})
endforeach()
