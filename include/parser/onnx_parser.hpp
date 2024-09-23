#ifndef ONNX_PARSER_HPP
#define ONNX_PARSER_HPP

#include <string>
#include "tensor/tensor.hpp"
#include "graph/graph.hpp"
#include "onnx/onnx.proto3.pb.h" // Include the generated protobuf header

TensorDataType convertONNXDataType(int onnx_data_type);
Tensor parseONNXTensor(const onnx::TensorProto &onnx_tensor);
Graph parseONNX(const std::string &file_path);

#endif // ONNX_PARSER_HPP
