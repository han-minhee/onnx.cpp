#ifndef OPERATORS_HPP
#define OPERATORS_HPP

#include "operator/operator.hpp"
#include "device/device.hpp"

#define DEFINE_OPERATOR_STRUCT(BaseName)                                                                                                                    \
  struct BaseName##Operator                                                                                                                                 \
  {                                                                                                                                                         \
    static OperatorExecuteResult execute(const std::vector<Tensor> &inputs,                                                                                 \
                                         std::vector<Tensor *> &outputs,                                                                                    \
                                         const std::unordered_map<std::string, Node::AttributeValue> &attributes, DeviceType deviceType = DeviceType::CPU); \
    static std::vector<std::vector<size_t>> inferOutputShapes(const std::vector<Tensor> &inputs,                                                            \
                                                              const std::unordered_map<std::string, Node::AttributeValue> &attributes);                     \
    static std::vector<TensorDataType> inferOutputDataTypes(const std::vector<Tensor> &inputs,                                                              \
                                                            const std::unordered_map<std::string, Node::AttributeValue> &attributes);                       \
  };

#define DEFINE_CPU_OPERATOR_IMPL(BaseName)                                                                           \
  namespace CPU_OP                                                                                                   \
  {                                                                                                                  \
    struct BaseName##OperatorImpl                                                                                    \
    {                                                                                                                \
      static OperatorExecuteResult execute(const std::vector<Tensor> &inputs,                                        \
                                           std::vector<Tensor *> &outputs,                                           \
                                           const std::unordered_map<std::string, Node::AttributeValue> &attributes); \
    };                                                                                                               \
  }

#define DEFINE_HIP_OPERATOR_IMPL(BaseName)                                                                           \
  namespace HIP_OP                                                                                                   \
  {                                                                                                                  \
    struct BaseName##OperatorImpl                                                                                    \
    {                                                                                                                \
      static OperatorExecuteResult execute(const std::vector<Tensor> &inputs,                                        \
                                           std::vector<Tensor *> &outputs,                                           \
                                           const std::unordered_map<std::string, Node::AttributeValue> &attributes); \
    };                                                                                                               \
  }

DEFINE_OPERATOR_STRUCT(Add)
DEFINE_OPERATOR_STRUCT(Conv)
DEFINE_OPERATOR_STRUCT(Constant)
DEFINE_OPERATOR_STRUCT(Sub)
DEFINE_OPERATOR_STRUCT(Reshape)
DEFINE_OPERATOR_STRUCT(Split)
DEFINE_OPERATOR_STRUCT(Concat)
DEFINE_OPERATOR_STRUCT(MatMul)
DEFINE_OPERATOR_STRUCT(Div)
DEFINE_OPERATOR_STRUCT(Mul)
DEFINE_OPERATOR_STRUCT(Sigmoid)
DEFINE_OPERATOR_STRUCT(Slice)
DEFINE_OPERATOR_STRUCT(Gather)
DEFINE_OPERATOR_STRUCT(Shape)
DEFINE_OPERATOR_STRUCT(Softmax)
DEFINE_OPERATOR_STRUCT(Transpose)
DEFINE_OPERATOR_STRUCT(Resize)
DEFINE_OPERATOR_STRUCT(MaxPool)

DEFINE_CPU_OPERATOR_IMPL(Conv)
DEFINE_CPU_OPERATOR_IMPL(Constant)
DEFINE_CPU_OPERATOR_IMPL(Add)
DEFINE_CPU_OPERATOR_IMPL(Sub)
DEFINE_CPU_OPERATOR_IMPL(Reshape)
DEFINE_CPU_OPERATOR_IMPL(Split)
DEFINE_CPU_OPERATOR_IMPL(Concat)
DEFINE_CPU_OPERATOR_IMPL(MatMul)
DEFINE_CPU_OPERATOR_IMPL(Div)
DEFINE_CPU_OPERATOR_IMPL(Mul)
DEFINE_CPU_OPERATOR_IMPL(Sigmoid)
DEFINE_CPU_OPERATOR_IMPL(Slice)
DEFINE_CPU_OPERATOR_IMPL(Gather)
DEFINE_CPU_OPERATOR_IMPL(Shape)
DEFINE_CPU_OPERATOR_IMPL(Softmax)
DEFINE_CPU_OPERATOR_IMPL(Transpose)
DEFINE_CPU_OPERATOR_IMPL(Resize)
DEFINE_CPU_OPERATOR_IMPL(MaxPool)

#ifdef USE_HIP
DEFINE_HIP_OPERATOR_IMPL(Conv)
DEFINE_HIP_OPERATOR_IMPL(Constant)
DEFINE_HIP_OPERATOR_IMPL(Add)
DEFINE_HIP_OPERATOR_IMPL(Sub)
DEFINE_HIP_OPERATOR_IMPL(Reshape)
DEFINE_HIP_OPERATOR_IMPL(Split)
DEFINE_HIP_OPERATOR_IMPL(Concat)
DEFINE_HIP_OPERATOR_IMPL(MatMul)
DEFINE_HIP_OPERATOR_IMPL(Div)
DEFINE_HIP_OPERATOR_IMPL(Mul)
DEFINE_HIP_OPERATOR_IMPL(Sigmoid)
DEFINE_HIP_OPERATOR_IMPL(Slice)
DEFINE_HIP_OPERATOR_IMPL(Gather)
DEFINE_HIP_OPERATOR_IMPL(Shape)
DEFINE_HIP_OPERATOR_IMPL(Softmax)
DEFINE_HIP_OPERATOR_IMPL(Transpose)
DEFINE_HIP_OPERATOR_IMPL(Resize)
DEFINE_HIP_OPERATOR_IMPL(MaxPool)
#endif

#endif
