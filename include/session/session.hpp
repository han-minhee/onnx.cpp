#ifndef SESSION_HPP
#define SESSION_HPP

#include <unordered_map>
#include <string>
#include <memory>

#include "device/device.hpp"
#include "graph/graph.hpp"
#include "tensor/tensor.hpp"
#include "device/device.hpp"
#include "device/device_cpu.hpp"
#include "enums.hpp"

struct SessionConfig
{
    Device *device = new CpuDevice();
    bool enable_optimizations = false;
    std::unordered_map<std::string, std::string> custom_args = {};
};

class Session
{
public:
    Session(const std::string &onnx_file_path, SessionConfig config);
    std::unordered_map<std::string, Tensor> run(const std::unordered_map<std::string, Tensor> &inputs);
    Tensor &getTensorByName(const std::string &name);
    Tensor &getOrAllocateIntermediateTensor(const std::string &name, const std::vector<size_t> &dims, TensorDataType dtype);

    // currently bool is used as a return type to indicate success or failure
    bool addDevice(const Device *device);
    bool selectDevice(Device *device);
    bool selectDeviceByName(const std::string &name, size_t deviceIndex = 0);
    bool selectDeviceByType(DeviceType type, size_t deviceIndex = 0);

    Device *getDeviceByName(const std::string &name, size_t deviceIndex = 0);
    Device *getDeviceByType(DeviceType type, size_t deviceIndex = 0);

    // methods for testing
    std::unordered_map<std::string, Tensor> runWithValidation(const std::unordered_map<std::string, Tensor> &inputs);
    void executeAndValidateNode(const Node &node);
    void compareOutputToReference(const Node &node, const std::string &sanitized_output_name, const Tensor &output_tensor);

private:
    SessionConfig sessionConfig;
    CpuDevice* hostDevice; // it is used as a default device

    /// XXX: For now, sessionConfig.device is used for all the operations
    std::vector<Device *> devices; // It's a list of all devices available, but not used now.
    
    Graph graph;
    void prepareExecution(const std::unordered_map<std::string, Tensor> &inputs);
    void executeNode(const Node &node);

    std::unordered_map<std::string, Tensor> tensorMap;
    std::unordered_map<std::string, Tensor> intermediateTensorPool;

    std::vector<std::string> graphInputNames;
    std::vector<std::string> graphOutputNames;
};

std::string sanitizeFileName(const std::string &name);

#endif // SESSION_HPP
