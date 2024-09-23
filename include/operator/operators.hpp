#ifndef OPERATORS_HPP
#define OPERATORS_HPP

#include "operator/operator.hpp"


/// FIXME: Implement output shape / type inference for all operators

#define  DEFINE_OPERATOR_CLASS(ClassName, OperatorTypeEnum) \
 class ClassName : public Operator \
 { \
 public: \
 ClassName() : Operator(OperatorTypeEnum) {} \
 OperatorExecuteResult execute(const std::vector<Tensor> &inputs, \
 std::vector<Tensor *> &outputs, \
 const std::unordered_map<std::string, Node::AttributeValue> &attributes) override; \
    std::vector<std::vector<size_t>> inferOutputShapes(const std::vector<Tensor> &inputs, \
    const std::unordered_map<std::string, Node::AttributeValue> &attributes) override; \
      std::vector<TensorDataType> inferOutputDataTypes(const std::vector<Tensor> &inputs, \
      const std::unordered_map<std::string, Node::AttributeValue> &attributes) override; \
 };


#define  DEFINE_OPERATOR_CLASS_ONLY_EXCUTE(ClassName, OperatorTypeEnum) \
 class ClassName : public Operator \
 { \
 public: \
 ClassName() : Operator(OperatorTypeEnum) {} \
 OperatorExecuteResult execute(const std::vector<Tensor> &inputs, \
 std::vector<Tensor *> &outputs, \
 const std::unordered_map<std::string, Node::AttributeValue> &attributes) override; \
 };

#define DEFINE_OPERATOR_CLASS_NO_EXECUTE(ClassName, OperatorTypeEnum) \
 class ClassName : public Operator \
 { \
 public: \
 ClassName() : Operator(OperatorTypeEnum) {} \
 };

 DEFINE_OPERATOR_CLASS(ConvOperator, OperatorType::Conv)
 DEFINE_OPERATOR_CLASS(ConstantOperator, OperatorType::Constant)
 DEFINE_OPERATOR_CLASS(AddOperator, OperatorType::Add)
 DEFINE_OPERATOR_CLASS(SubOperator, OperatorType::Sub)
 DEFINE_OPERATOR_CLASS(ReshapeOperator, OperatorType::Reshape)
 DEFINE_OPERATOR_CLASS(SplitOperator, OperatorType::Split)
 DEFINE_OPERATOR_CLASS(ConcatOperator, OperatorType::Concat)
 DEFINE_OPERATOR_CLASS(MatMulOperator, OperatorType::MatMul)
 DEFINE_OPERATOR_CLASS(DivOperator, OperatorType::Div)
 DEFINE_OPERATOR_CLASS(MulOperator, OperatorType::Mul)
 DEFINE_OPERATOR_CLASS(SigmoidOperator, OperatorType::Sigmoid)
 DEFINE_OPERATOR_CLASS(SliceOperator, OperatorType::Slice)
 DEFINE_OPERATOR_CLASS(GatherOperator, OperatorType::Gather)
 DEFINE_OPERATOR_CLASS(ShapeOperator, OperatorType::Shape)
 DEFINE_OPERATOR_CLASS(SoftmaxOperator, OperatorType::Softmax)
 DEFINE_OPERATOR_CLASS(TransposeOperator, OperatorType::Transpose)
 DEFINE_OPERATOR_CLASS(ResizeOperator, OperatorType::Resize)
 DEFINE_OPERATOR_CLASS(MaxPoolOperator, OperatorType::MaxPool)

#endif
