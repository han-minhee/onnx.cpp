// operator_registry.cpp
#include "operator/operator_registry.hpp"
#include "operator/operator.hpp"
#include "operator/operators.hpp"

std::unordered_map<OperatorType, OperatorRegistry::OperatorFunctions> OperatorRegistry::registry;

void OperatorRegistry::registerOperator(OperatorType type, const OperatorFunctions &functions)
{
    registry[type] = functions;
}

const OperatorRegistry::OperatorFunctions *OperatorRegistry::getOperatorFunctions(OperatorType type)
{
    auto it = registry.find(type);
    if (it != registry.end())
    {
        return &it->second;
    }
    return nullptr;
}

#define RegisterOperatorRegistrar(BaseName) \
    struct BaseName##OperatorRegistrar \
    { \
        BaseName##OperatorRegistrar() \
        { \
            OperatorRegistry::OperatorFunctions functions = { \
                &BaseName##Operator::execute, \
                &BaseName##Operator::inferOutputShapes, \
                &BaseName##Operator::inferOutputDataTypes \
            }; \
            OperatorRegistry::registerOperator(OperatorType::BaseName, functions); \
        } \
    }; \
    static BaseName##OperatorRegistrar BaseName##OperatorRegistrarInstance;

// Register all operators
RegisterOperatorRegistrar(Add)
RegisterOperatorRegistrar(Conv)
RegisterOperatorRegistrar(Constant)
RegisterOperatorRegistrar(Sub)
RegisterOperatorRegistrar(Reshape)
RegisterOperatorRegistrar(Split)
RegisterOperatorRegistrar(Concat)
RegisterOperatorRegistrar(MatMul)
RegisterOperatorRegistrar(Div)
RegisterOperatorRegistrar(Mul)
RegisterOperatorRegistrar(Sigmoid)
RegisterOperatorRegistrar(Slice)
RegisterOperatorRegistrar(Gather)
RegisterOperatorRegistrar(Shape)
RegisterOperatorRegistrar(Softmax)
RegisterOperatorRegistrar(Transpose)
RegisterOperatorRegistrar(Resize)
RegisterOperatorRegistrar(MaxPool)

RegisterOperatorRegistrar(MatMulNBits)

#undef RegisterOperatorRegistrar