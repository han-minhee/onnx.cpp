#include "graph/graph.hpp"
#include <sstream>
#include <queue>

void Graph::addNode(const Node &node)
{
    nodes.push_back(node);
}

const std::vector<Node> &Graph::getNodes() const
{
    return nodes;
}

std::string Graph::toString() const
{
    std::ostringstream oss;
    for (const auto &node : nodes)
    {
        oss << node.toString() << "\n";
    }
    return oss.str();
}

void Graph::topologicalSort()
{
    size_t num_nodes = nodes.size();
    std::vector<Node> sortedNodes;
    sortedNodes.reserve(num_nodes);

    // Step 1: Build mappings from tensor names to producers and consumers
    std::unordered_map<std::string, size_t> tensor_producers;              // tensor name -> node index
    std::unordered_map<std::string, std::vector<size_t>> tensor_consumers; // tensor name -> list of node indices

    for (size_t i = 0; i < num_nodes; ++i)
    {
        const Node &node = nodes[i];

        // Map outputs to this node (producer)
        for (const auto &output : node.getOutputNames())
        {
            tensor_producers[output] = i;
        }

        // Map inputs to this node (consumer)
        for (const auto &input : node.getInputNames())
        {
            tensor_consumers[input].push_back(i);
        }
    }

    // Step 2: Compute in-degree for each node
    std::vector<int> in_degree(num_nodes, 0);

    for (size_t i = 0; i < num_nodes; ++i)
    {
        const Node &node = nodes[i];
        for (const auto &input : node.getInputNames())
        {
            auto it = tensor_producers.find(input);
            if (it != tensor_producers.end())
            {
                // This input is produced by another node in the graph
                in_degree[i]++;
            }
        }
    }

    // Step 3: Kahn's Algorithm
    std::queue<size_t> zero_in_degree_nodes;
    for (size_t i = 0; i < num_nodes; ++i)
    {
        if (in_degree[i] == 0)
        {
            zero_in_degree_nodes.push(i);
        }
    }

    while (!zero_in_degree_nodes.empty())
    {
        size_t node_index = zero_in_degree_nodes.front();
        zero_in_degree_nodes.pop();

        // Add node to sorted list
        sortedNodes.push_back(nodes[node_index]);
        const Node &node = nodes[node_index];

        // For each output tensor, reduce in-degree of consumer nodes
        for (const auto &output : node.getOutputNames())
        {
            auto consumers_it = tensor_consumers.find(output);
            if (consumers_it != tensor_consumers.end())
            {
                for (size_t consumer_index : consumers_it->second)
                {
                    in_degree[consumer_index]--;
                    if (in_degree[consumer_index] == 0)
                    {
                        zero_in_degree_nodes.push(consumer_index);
                    }
                }
            }
        }
    }

    // Check for cycles in the graph
    if (sortedNodes.size() != num_nodes)
    {
        throw std::runtime_error("Graph has cycles; topological sort not possible.");
    }

    // for sortedNodes, move nodes with op_type (string) "Constant" to the front
    // Separate "Constant" nodes from other nodes
    std::vector<Node> constantNodes;
    std::vector<Node> otherNodes;

    for (const auto &node : sortedNodes)
    {
        if (node.getOpTypeString() == "Constant")
        {
            constantNodes.push_back(node);
        }
        else
        {
            otherNodes.push_back(node);
        }
    }

    // Combine constant nodes with other nodes, putting constants first
    sortedNodes.clear();
    sortedNodes.reserve(num_nodes);
    sortedNodes.insert(sortedNodes.end(), constantNodes.begin(), constantNodes.end());
    sortedNodes.insert(sortedNodes.end(), otherNodes.begin(), otherNodes.end());

    // Replace nodes with sorted nodes
    nodes = std::move(sortedNodes);
}