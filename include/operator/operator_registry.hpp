// operator_registry.hpp
#ifndef OPERATOR_REGISTRY_HPP
#define OPERATOR_REGISTRY_HPP

#include <unordered_map>
#include <functional>
#include "enums.hpp"
#include "tensor/tensor.hpp"
#include "graph/node.hpp"

class OperatorRegistry
{
public:
    using ExecuteFunction = std::function<OperatorExecuteResult(const std::vector<Tensor> &,
                                                                std::vector<Tensor *> &,
                                                                const std::unordered_map<std::string, Node::AttributeValue> &,
                                                                Device &)>;

    using InferOutputShapesFunction = std::function<std::vector<std::vector<size_t>>(const std::vector<Tensor> &,
                                                                                     const std::unordered_map<std::string, Node::AttributeValue> &)>;

    using InferOutputDataTypesFunction = std::function<std::vector<TensorDataType>(const std::vector<Tensor> &,
                                                                                   const std::unordered_map<std::string, Node::AttributeValue> &)>;

    struct OperatorFunctions
    {
        ExecuteFunction execute;
        InferOutputShapesFunction inferOutputShapes;
        InferOutputDataTypesFunction inferOutputDataTypes;
    };

    static void registerOperator(OperatorType type, const OperatorFunctions &functions);

    static const OperatorFunctions *getOperatorFunctions(OperatorType type);

private:
    static std::unordered_map<OperatorType, OperatorFunctions> registry;
};

#endif // OPERATOR_REGISTRY_HPP
