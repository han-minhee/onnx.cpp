#ifndef OPERATOR_HPP
#define OPERATOR_HPP

#include <vector>
#include <string>
#include <unordered_map>
#include <memory>

#include "types/half_t.hpp"
#include "graph/node.hpp"
#include "tensor/tensor.hpp"
#include "device/device.hpp"
#include "enums.hpp"

namespace OperatorUtils
{
    std::string OperatorExecuteResultToString(OperatorExecuteResult result);
    std::string OperatorTypeToString(OperatorType type);
    OperatorType StringToOperatorType(const std::string &str);
}

/// FIXME: These auxiliary functions should be moved to a different file
std::vector<size_t> compute_broadcast_shape(const std::vector<std::vector<size_t>> &shapes);
std::vector<size_t> compute_broadcast_strides(const std::vector<size_t> &input_shape, const std::vector<size_t> &output_shape);

#endif // OPERATOR_HPP
