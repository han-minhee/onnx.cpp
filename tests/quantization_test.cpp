#include <gtest/gtest.h>

#include "tensor/tensor.hpp"
#include "operator/operators.hpp"
#include "operator/operator_registry.hpp"

#include "device/device.hpp"
#include "device/device_cpu.hpp"

void PrintTo(OperatorExecuteResult result, std::ostream *os)
{
    *os << OperatorUtils::OperatorExecuteResultToString(result);
}

void run_and_check_operator(const OperatorRegistry::OperatorFunctions *op,
                            const std::vector<Tensor> &inputs,
                            std::vector<Tensor *> outputs,
                            std::vector<Tensor> &expected,
                            std::unordered_map<std::string, Node::AttributeValue> attributes,
                            OperatorExecuteResult expected_execute_result,
                            Device *device)
{
    OperatorExecuteResult result_code = op->execute(inputs, outputs, attributes, device);

    ASSERT_EQ(result_code, expected_execute_result);

    if (result_code != OperatorExecuteResult::SUCCESS)
    {
        return;
    }

    ASSERT_EQ(outputs.size(), expected.size());
    for (size_t i = 0; i < outputs.size(); i++)
    {
        ASSERT_EQ(outputs[i]->getDims(), expected[i].getDims());
        ASSERT_EQ(outputs[i]->getDataType(), expected[i].getDataType());

        // move tensors to host
        outputs[i]->to(new CpuDevice());
        expected[i].to(new CpuDevice());

        switch (outputs[i]->getDataType())
        {
        case TensorDataType::FLOAT32:
        {
            const float *output_data = outputs[i]->data<float>();
            const float *expected_data = expected[i].data<float>();

            for (size_t j = 0; j < expected[i].getNumElements(); j++)
            {
                ASSERT_NEAR(output_data[j], expected_data[j], 1e-4);
            }
            break;
        }
        case TensorDataType::INT32:
        {
            const int32_t *output_data = outputs[i]->data<int32_t>();
            const int32_t *expected_data = expected[i].data<int32_t>();

            for (size_t j = 0; j < expected[i].getNumElements(); j++)
            {
                ASSERT_EQ(output_data[j], expected_data[j]);
            }
            break;
        }
        case TensorDataType::INT64:
        {
            const int64_t *output_data = outputs[i]->data<int64_t>();
            const int64_t *expected_data = expected[i].data<int64_t>();

            for (size_t j = 0; j < expected[i].getNumElements(); j++)
            {
                ASSERT_EQ(output_data[j], expected_data[j]);
            }
            break;
        }

        case TensorDataType::FLOAT16:
        {
            const half_t *output_data = outputs[i]->data<half_t>();
            const half_t *expected_data = expected[i].data<half_t>();

            for (size_t j = 0; j < expected[i].getNumElements(); j++)
            {
                ASSERT_NEAR(output_data[j], expected_data[j], 1e-4);
            }
            break;
        }

        default:
            throw std::runtime_error("Unsupported data type.");
        }
    }
}

#define RUN_TEST_CASE(operator_type, input_tensors, expected_tensors, attributes, expectedResult, device)      \
    do                                                                                                         \
    {                                                                                                          \
        const OperatorRegistry::OperatorFunctions *op = OperatorRegistry::getOperatorFunctions(operator_type); \
        std::vector<std::vector<size_t>> output_shapes = op->inferOutputShapes(input_tensors, attributes);     \
        std::vector<TensorDataType> output_data_types = op->inferOutputDataTypes(input_tensors, attributes);   \
                                                                                                               \
        std::vector<Tensor *> outputs;                                                                         \
        for (size_t i = 0; i < output_shapes.size(); i++)                                                      \
        {                                                                                                      \
            outputs.push_back(new Tensor(output_data_types[i], output_shapes[i], device));                     \
        }                                                                                                      \
                                                                                                               \
        run_and_check_operator(op, input_tensors, outputs, expected_tensors, attributes,                       \
                               expectedResult, device);                                                        \
                                                                                                               \
        for (auto &output : outputs)                                                                           \
        {                                                                                                      \
            delete output;                                                                                     \
        }                                                                                                      \
    } while (0)

// ----------------------- AddOperator Tests -----------------------
TEST(QuantizationTestCPU, AddOperatorBasic)
{
    CpuDevice cpuDevice = CpuDevice();

    Tensor t1(TensorDataType::FLOAT16, {2, 3}, std::vector<half_t>{1.0f, 2.0f, 3.0f, 4.0f, 3.0f, 4.0f}, &cpuDevice);
    Tensor t2(TensorDataType::FLOAT16, {2, 3}, std::vector<half_t>{5.0f, 6.0f, 7.0f, 8.0f, 3.0f, 4.0f}, &cpuDevice);
    Tensor expected(TensorDataType::FLOAT16, {2, 3}, std::vector<half_t>{6.0f, 8.0f, 10.0f, 12.0f, 6.0f, 8.0f}, &cpuDevice);
    std::unordered_map<std::string, Node::AttributeValue> attributes;

    std::vector<Tensor> inputs = {t1, t2};
    std::vector<Tensor> expected_tensors = {expected};

    RUN_TEST_CASE(OperatorType::Add, inputs, expected_tensors, attributes, OperatorExecuteResult::SUCCESS, &cpuDevice);
}

// TEST(OperatorTestHIP, AddOperatorBroadcastScalar)
// {
//     // Broadcasting scalar addition
//     HipDevice hipDevice = HipDevice(0);
//     Tensor t1(TensorDataType::FLOAT32, {2, 2}, std::vector<float>{1.0f, 2.0f, 3.0f, 4.0f}, &hipDevice);
//     Tensor t2(TensorDataType::FLOAT32, {}, std::vector<float>{10.0f}, &hipDevice); // Scalar tensor
//     Tensor expected(TensorDataType::FLOAT32, {2, 2}, std::vector<float>{11.0f, 12.0f, 13.0f, 14.0f}, &hipDevice);
//     std::unordered_map<std::string, Node::AttributeValue> attributes;

//     std::vector<Tensor> inputs = {t1, t2};
//     std::vector<Tensor> expected_tensors = {expected};

//     RUN_TEST_CASE(OperatorType::Add, inputs, expected_tensors, attributes, OperatorExecuteResult::SUCCESS, &hipDevice);
// }

// TEST(OperatorTestHIP, AddOperatorShapeMismatchError)
// {
//     // Shape mismatch error
//     HipDevice hipDevice = HipDevice(0);
//     Tensor t1(TensorDataType::FLOAT32, {2, 2}, std::vector<float>{1.0f, 2.0f, 3.0f, 4.0f}, &hipDevice);
//     Tensor t2(TensorDataType::FLOAT32, {3, 2}, std::vector<float>{5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f}, &hipDevice);
//     std::unordered_map<std::string, Node::AttributeValue> attributes;

//     std::vector<Tensor> inputs = {t1, t2};
//     std::vector<Tensor> expected_tensors;

//     RUN_TEST_CASE(OperatorType::Add, inputs, expected_tensors, attributes, OperatorExecuteResult::SHAPE_MISMATCH_ERROR, &hipDevice);
// }

// TEST(OperatorTestHIP, AddOperatorBroadcastVector)
// {
//     // Broadcasting vector addition
//     HipDevice hipDevice = HipDevice(0);
//     Tensor t1(TensorDataType::FLOAT32, {2, 2}, std::vector<float>{1.0f, 2.0f, 3.0f, 4.0f}, &hipDevice);
//     Tensor t2(TensorDataType::FLOAT32, {2}, std::vector<float>{10.0f, 20.0f}, &hipDevice); // Vector tensor
//     Tensor expected(TensorDataType::FLOAT32, {2, 2}, std::vector<float>{11.0f, 22.0f, 13.0f, 24.0f}, &hipDevice);
//     std::unordered_map<std::string, Node::AttributeValue> attributes;

//     std::vector<Tensor> inputs = {t1, t2};
//     std::vector<Tensor> expected_tensors = {expected};

//     RUN_TEST_CASE(OperatorType::Add, inputs, expected_tensors, attributes, OperatorExecuteResult::SUCCESS, &hipDevice);
// }
