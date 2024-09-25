#ifndef OPERATORS_HPP
#define OPERATORS_HPP

#include "operator/operator.hpp"
#include "operator/operator_impl.hpp"
#include "device/device.hpp"

// Macro to define the operator class and CPU implementation
#define DEFINE_OPERATOR_CLASS(BaseName)                                                                                                                       \
  class BaseName##Operator : public Operator                                                                                                                  \
  {                                                                                                                                                           \
  public:                                                                                                                                                     \
    BaseName##Operator() : Operator(OperatorType::BaseName) {}                                                                                                \
    static OperatorExecuteResult execute(const std::vector<Tensor> &inputs,                                                                                          \
                                  std::vector<Tensor *> &outputs,                                                                                             \
                                  const std::unordered_map<std::string, Node::AttributeValue> &attributes, DeviceType deviceType = DeviceType::CPU); \
    static std::vector<std::vector<size_t>> inferOutputShapes(const std::vector<Tensor> &inputs,                                                                     \
                                                       const std::unordered_map<std::string, Node::AttributeValue> &attributes);                     \
    static std::vector<TensorDataType> inferOutputDataTypes(const std::vector<Tensor> &inputs,                                                                       \
                                                     const std::unordered_map<std::string, Node::AttributeValue> &attributes);                       \
  };                                                                                                                                                          \
                                                                                                                                                              \
  namespace CPU_OP                                                                                                                                            \
  {                                                                                                                                                           \
    class BaseName##OperatorImpl : public OperatorImpl                                                                                                        \
    {                                                                                                                                                         \
    public:                                                                                                                                                   \
      static OperatorExecuteResult execute(const std::vector<Tensor> &inputs,                                                                                 \
                                           std::vector<Tensor *> &outputs,                                                                                    \
                                           const std::unordered_map<std::string, Node::AttributeValue> &attributes);                                          \
    };                                                                                                                                                        \
  };

// Macro to define the HIP implementation
#define DEFINE_HIP_OPERATOR_IMPL(BaseName)                                                                           \
  namespace HIP_OP                                                                                                   \
  {                                                                                                                  \
    class BaseName##OperatorImpl : public OperatorImpl                                                               \
    {                                                                                                                \
    public:                                                                                                          \
      static OperatorExecuteResult execute(const std::vector<Tensor> &inputs,                                        \
                                           std::vector<Tensor *> &outputs,                                           \
                                           const std::unordered_map<std::string, Node::AttributeValue> &attributes); \
    };                                                                                                               \
  }

// Define the operators using the updated macro
DEFINE_OPERATOR_CLASS(Conv)
DEFINE_OPERATOR_CLASS(Constant)
DEFINE_OPERATOR_CLASS(Add)
DEFINE_OPERATOR_CLASS(Sub)
DEFINE_OPERATOR_CLASS(Reshape)
DEFINE_OPERATOR_CLASS(Split)
DEFINE_OPERATOR_CLASS(Concat)
DEFINE_OPERATOR_CLASS(MatMul)
DEFINE_OPERATOR_CLASS(Div)
DEFINE_OPERATOR_CLASS(Mul)
DEFINE_OPERATOR_CLASS(Sigmoid)
DEFINE_OPERATOR_CLASS(Slice)
DEFINE_OPERATOR_CLASS(Gather)
DEFINE_OPERATOR_CLASS(Shape)
DEFINE_OPERATOR_CLASS(Softmax)
DEFINE_OPERATOR_CLASS(Transpose)
DEFINE_OPERATOR_CLASS(Resize)
DEFINE_OPERATOR_CLASS(MaxPool)

// Conditionally define HIP operator implementations if USE_HIP is enabled
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
