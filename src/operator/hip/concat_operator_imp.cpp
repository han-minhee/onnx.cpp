#ifdef USE_HIP
#include "operator/operators.hpp"
namespace HIP_OP
{
    OperatorExecuteResult ConcatOperatorImpl::execute(
        const std::vector<Tensor> &inputs,
        std::vector<Tensor *> &outputs,
        const std::unordered_map<std::string, Node::AttributeValue> &attributes, Device *device)
    {

        return OperatorExecuteResult::NOT_IMPLEMENTED;
        }

}
#endif // USE_HIP