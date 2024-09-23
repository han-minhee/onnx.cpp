#include "graph/node.hpp"
#include <sstream>

Node::Node(const std::string &name, const std::string &op_type)
    : name(name), op_type(op_type) {}

void Node::addInput(const std::string &input_name)
{
    inputs.push_back(input_name);
}

void Node::addOutput(const std::string &output_name)
{
    outputs.push_back(output_name);
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

const std::string &Node::getOpType() const
{
    return op_type;
}

const std::vector<std::string> &Node::getInputs() const
{
    return inputs;
}

const std::vector<std::string> &Node::getOutputs() const
{
    return outputs;
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
    for (const auto &input : inputs)
    {
        oss << input << " ";
    }
    oss << "\n";

    oss << "  Outputs: ";
    for (const auto &output : outputs)
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
