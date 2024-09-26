#include "graph/node.hpp"
#include "operator/operator.hpp"
#include <sstream>

Node::Node(const std::string &name, const std::string &op_type)
    : name(name), op_type(op_type) {}

void Node::addInput(const std::string &input_name)
{
    inputNames.push_back(input_name);
}

void Node::addOutput(const std::string &output_name)
{
    outputNames.push_back(output_name);
}

void Node::addAttribute(const std::string &key, const Node::AttributeValue &value)
{
    attributes.emplace(key, value);
}

template <typename T>
std::optional<T> Node::getAttribute(const std::string &key) const
{
    auto it = attributes.find(key);
    if (it != attributes.end())
    {

        if (std::holds_alternative<T>(it->second))
        {
            return std::get<T>(it->second);
        }
    }
    return std::nullopt;
}

const std::string &Node::getName() const
{
    return name;
}

const std::string &Node::getOpTypeString() const
{
    return op_type;
}

const OperatorType Node::getOpType() const
{
    return OperatorUtils::StringToOperatorType(op_type);
}

const std::vector<std::string> &Node::getInputNames() const
{
    return inputNames;
}

const std::vector<std::string> &Node::getOutputNames() const
{
    return outputNames;
}

const std::unordered_map<std::string, Node::AttributeValue> &Node::getAttributes() const
{
    return attributes;
}

std::string Node::toString() const
{
    std::ostringstream oss;
    oss << "Node: " << name << ", OpType: " << op_type << "\n";

    oss << "  Inputs: ";
    for (const auto &input : inputNames)
    {
        oss << input << " ";
    }
    oss << "\n";

    oss << "  Outputs: ";
    for (const auto &output : outputNames)
    {
        oss << output << " ";
    }
    oss << "\n";

    oss << "  Attributes: ";
    for (const auto &attr : attributes)
    {
        oss << attr.first << ": ";

        std::visit([&oss](auto &&arg)
                   {
                       using T = std::decay_t<decltype(arg)>;
                       if constexpr (std::is_same_v<T, int>)
                           oss << arg;
                       else if constexpr (std::is_same_v<T, float>)
                           oss << arg;
                       else if constexpr (std::is_same_v<T, std::vector<int64_t>>)
                       {
                           oss << "[";
                           for (const auto &val : arg)
                               oss << val << " ";
                           oss << "]";
                       }
                       else if constexpr (std::is_same_v<T, std::vector<float>>)
                       {
                           oss << "[";
                           for (const auto &val : arg)
                               oss << val << " ";
                           oss << "]";
                       }
                       else if constexpr (std::is_same_v<T, Tensor>)
                           oss << "Tensor(...)"; },
                   attr.second);

        oss << " ";
    }
    oss << "\n";

    return oss.str();
}
