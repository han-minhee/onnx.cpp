#ifdef USE_HIP
#include <gtest/gtest.h>
#include <hip/hip_runtime.h>

#include "tensor/tensor.hpp"
#include "operator/operators.hpp"
#include "operator/operator_registry.hpp"

#include "device/device.hpp"
#include "device/device_hip.hpp"

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
                            HipDevice *device)
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

        // std::cout << inputs[i].toString() << std::endl;
        // std::cout << outputs[i]->toString() << std::endl;

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
TEST(OperatorTestHIP, AddOperatorBasic)
{
    HipDevice hipDevice = HipDevice(0);

    Tensor t1 = create_tensor(TensorDataType::FLOAT32, {2, 3}, {1.0f, 2.0f, 3.0f, 4.0f, 3.0f, 4.0f}, &hipDevice);
    Tensor t2 = create_tensor(TensorDataType::FLOAT32, {2, 3}, {5.0f, 6.0f, 7.0f, 8.0f, 3.0f, 4.0f}, &hipDevice);
    Tensor expected = create_tensor(TensorDataType::FLOAT32, {2, 3}, {6.0f, 8.0f, 10.0f, 12.0f, 6.0f, 8.0f}, &hipDevice);
    std::unordered_map<std::string, Node::AttributeValue> attributes;

    std::vector<Tensor> inputs = {t1, t2};
    std::vector<Tensor> expected_tensors = {expected};

    RUN_TEST_CASE(OperatorType::Add, inputs, expected_tensors, attributes, OperatorExecuteResult::SUCCESS, &hipDevice);
}

TEST(OperatorTestHIP, AddOperatorBroadcastScalar)
{
    // Broadcasting scalar addition
    HipDevice hipDevice = HipDevice(0);
    Tensor t1 = create_tensor(TensorDataType::FLOAT32, {2, 2}, {1.0f, 2.0f, 3.0f, 4.0f}, &hipDevice);
    Tensor t2 = create_tensor(TensorDataType::FLOAT32, {}, {10.0f}, &hipDevice); // Scalar tensor
    Tensor expected = create_tensor(TensorDataType::FLOAT32, {2, 2}, {11.0f, 12.0f, 13.0f, 14.0f}, &hipDevice);
    std::unordered_map<std::string, Node::AttributeValue> attributes;

    std::vector<Tensor> inputs = {t1, t2};
    std::vector<Tensor> expected_tensors = {expected};

    RUN_TEST_CASE(OperatorType::Add, inputs, expected_tensors, attributes, OperatorExecuteResult::SUCCESS, &hipDevice);
}

TEST(OperatorTestHIP, AddOperatorShapeMismatchError)
{
    // Shape mismatch error
    HipDevice hipDevice = HipDevice(0);
    Tensor t1 = create_tensor(TensorDataType::FLOAT32, {2, 2}, {1.0f, 2.0f, 3.0f, 4.0f}, &hipDevice);
    Tensor t2 = create_tensor(TensorDataType::FLOAT32, {3, 2}, {5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f}, &hipDevice);
    std::unordered_map<std::string, Node::AttributeValue> attributes;

    std::vector<Tensor> inputs = {t1, t2};
    std::vector<Tensor> expected_tensors;

    RUN_TEST_CASE(OperatorType::Add, inputs, expected_tensors, attributes, OperatorExecuteResult::SHAPE_MISMATCH_ERROR, &hipDevice);
}

TEST(OperatorTestHIP, AddOperatorBroadcastVector)
{
    // Broadcasting vector addition
    HipDevice hipDevice = HipDevice(0);
    Tensor t1 = create_tensor(TensorDataType::FLOAT32, {2, 2}, {1.0f, 2.0f, 3.0f, 4.0f}, &hipDevice);
    Tensor t2 = create_tensor(TensorDataType::FLOAT32, {2}, {10.0f, 20.0f}, &hipDevice); // Vector tensor
    Tensor expected = create_tensor(TensorDataType::FLOAT32, {2, 2}, {11.0f, 22.0f, 13.0f, 24.0f}, &hipDevice);
    std::unordered_map<std::string, Node::AttributeValue> attributes;

    std::vector<Tensor> inputs = {t1, t2};
    std::vector<Tensor> expected_tensors = {expected};

    RUN_TEST_CASE(OperatorType::Add, inputs, expected_tensors, attributes, OperatorExecuteResult::SUCCESS, &hipDevice);
}

// ----------------------- SubOperator Tests -----------------------
TEST(OperatorTestHIP, SubOperatorBasic)
{
    // Basic subtraction test
    HipDevice hipDevice = HipDevice(0);
    Tensor t1 = create_tensor(TensorDataType::FLOAT32, {2, 2}, {5.0f, 7.0f, 9.0f, 11.0f}, &hipDevice);
    Tensor t2 = create_tensor(TensorDataType::FLOAT32, {2, 2}, {1.0f, 2.0f, 3.0f, 4.0f}, &hipDevice);
    Tensor expected = create_tensor(TensorDataType::FLOAT32, {2, 2}, {4.0f, 5.0f, 6.0f, 7.0f}, &hipDevice);

    std::unordered_map<std::string, Node::AttributeValue> attributes;

    std::vector<Tensor> inputs = {t1, t2};
    std::vector<Tensor> expected_tensors = {expected};

    RUN_TEST_CASE(OperatorType::Sub, inputs, expected_tensors, attributes, OperatorExecuteResult::SUCCESS, &hipDevice);
}

TEST(OperatorTestHIP, SubOperatorShapeMismatchError)
{
    HipDevice hipDevice = HipDevice(0);
    Tensor t1 = create_tensor(TensorDataType::FLOAT32, {2, 2}, {1.0f, 2.0f, 3.0f, 4.0f}, &hipDevice);
    Tensor t2 = create_tensor(TensorDataType::FLOAT32, {3, 2}, {5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f}, &hipDevice);
    std::unordered_map<std::string, Node::AttributeValue> attributes;

    std::vector<Tensor> inputs = {t1, t2};
    std::vector<Tensor> expected_tensors;

    RUN_TEST_CASE(OperatorType::Sub, inputs, expected_tensors, attributes, OperatorExecuteResult::SHAPE_MISMATCH_ERROR, &hipDevice);
}

TEST(OperatorTestHIP, SubOperatorBroadcastScalar)
{
    // Broadcasting scalar subtraction
    HipDevice hipDevice = HipDevice(0);
    Tensor t1 = create_tensor(TensorDataType::FLOAT32, {2, 2}, {10.0f, 20.0f, 30.0f, 40.0f}, &hipDevice);
    Tensor t2 = create_tensor(TensorDataType::FLOAT32, {}, {2.0f}, &hipDevice); // Scalar tensor
    Tensor expected = create_tensor(TensorDataType::FLOAT32, {2, 2}, {8.0f, 18.0f, 28.0f, 38.0f}, &hipDevice);
    std::unordered_map<std::string, Node::AttributeValue> attributes;

    std::vector<Tensor> inputs = {t1, t2};
    std::vector<Tensor> expected_tensors = {expected};

    RUN_TEST_CASE(OperatorType::Sub, inputs, expected_tensors, attributes, OperatorExecuteResult::SUCCESS, &hipDevice);
}

/// FIXME: Broadcasting is likely broken
TEST(OperatorTestHIP, SubOperatorBroadcastVector0)
{
    // Broadcasting vector subtraction
    HipDevice hipDevice = HipDevice(0);
    Tensor t1 = create_tensor(TensorDataType::FLOAT32, {2, 2}, {1.0f, 2.0f, 3.0f, 4.0f}, &hipDevice);
    Tensor t2 = create_tensor(TensorDataType::FLOAT32, {2}, {10.0f, 20.0f}, &hipDevice); // Vector tensor
    Tensor expected = create_tensor(TensorDataType::FLOAT32, {2, 2}, {-9.0f, -18.0f, -7.0f, -16.0f}, &hipDevice);
    std::unordered_map<std::string, Node::AttributeValue> attributes;

    std::vector<Tensor> inputs = {t1, t2};
    std::vector<Tensor> expected_tensors = {expected};

    RUN_TEST_CASE(OperatorType::Sub, inputs, expected_tensors, attributes, OperatorExecuteResult::SUCCESS, &hipDevice);
}

TEST(OperatorTestHIP, SubOperatorBroadcastVector1)
{
    // Broadcasting vector subtraction
    HipDevice hipDevice = HipDevice(0);
    Tensor t1 = create_tensor(TensorDataType::FLOAT32, {2, 3}, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f}, &hipDevice);
    Tensor t2 = create_tensor(TensorDataType::FLOAT32, {3}, {10.0f, 20.0f, 30.0f}, &hipDevice); // Vector tensor
    Tensor expected = create_tensor(TensorDataType::FLOAT32, {2, 3}, {-9.0f, -18.0f, -27.0f, -6.0f, -15.0f, -24.0f}, &hipDevice);
    std::unordered_map<std::string, Node::AttributeValue> attributes;

    std::vector<Tensor> inputs = {t1, t2};
    std::vector<Tensor> expected_tensors = {expected};

    RUN_TEST_CASE(OperatorType::Sub, inputs, expected_tensors, attributes, OperatorExecuteResult::SUCCESS, &hipDevice);
}
TEST(OperatorTestHIP, SubOperatorBroadcastVector2)
{
    // Broadcasting vector subtraction
    HipDevice hipDevice = HipDevice(0);
    Tensor t1 = create_tensor(TensorDataType::FLOAT32, {1, 2, 3}, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f}, &hipDevice);
    Tensor t2 = create_tensor(TensorDataType::FLOAT32, {2, 3}, {10.0f, 20.0f, 30.0f, 40.0f, 50.0f, 60.0f}, &hipDevice); // Vector tensor
    Tensor expected = create_tensor(TensorDataType::FLOAT32, {1, 2, 3}, {-9.0f, -18.0f, -27.0f, -36.0f, -45.0f, -54.0f}, &hipDevice);

    std::unordered_map<std::string, Node::AttributeValue> attributes;

    std::vector<Tensor> inputs = {t1, t2};
    std::vector<Tensor> expected_tensors = {expected};

    RUN_TEST_CASE(OperatorType::Sub, inputs, expected_tensors, attributes, OperatorExecuteResult::SUCCESS, &hipDevice);
}

// ----------------------- ConcatOperator Tests -----------------------
TEST(OperatorTestHIP, ConcatOperatorBasic)
{
    // Basic concatenation along axis 0
    HipDevice hipDevice = HipDevice(0);
    Tensor t1 = create_tensor(TensorDataType::FLOAT32, {2, 3}, {1, 2, 3, 4, 5, 6}, &hipDevice);
    Tensor t2 = create_tensor(TensorDataType::FLOAT32, {2, 3}, {7, 8, 9, 10, 11, 12}, &hipDevice);
    Tensor expected = create_tensor(TensorDataType::FLOAT32, {4, 3},
                                    {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}, &hipDevice);
    std::unordered_map<std::string, Node::AttributeValue> attributes;
    attributes["axis"] = 0;

    std::vector<Tensor> inputs = {t1, t2};
    std::vector<Tensor> expected_tensors = {expected};

    RUN_TEST_CASE(OperatorType::Concat, inputs, expected_tensors, attributes, OperatorExecuteResult::SUCCESS, &hipDevice);
}

TEST(OperatorTestHIP, ConcatOperatorShapeMismatchError)
{
    // Shape mismatch error
    HipDevice hipDevice = HipDevice(0);
    Tensor t1 = create_tensor(TensorDataType::FLOAT32, {2, 2}, {1, 2, 3, 4}, &hipDevice);
    Tensor t2 = create_tensor(TensorDataType::FLOAT32, {2, 3}, {5, 6, 7, 8, 9, 10}, &hipDevice);
    std::unordered_map<std::string, Node::AttributeValue> attributes;
    attributes["axis"] = 0;

    std::vector<Tensor> inputs = {t1, t2};
    std::vector<Tensor> expected_tensors;

    RUN_TEST_CASE(OperatorType::Concat, inputs, expected_tensors, attributes, OperatorExecuteResult::SHAPE_MISMATCH_ERROR, &hipDevice);
}

// // ----------------------- ConstantOperator Tests -----------------------
// TEST(OperatorTestHIP, ConstantOperatorBasic)
// {
//     // Create a constant tensor
//     Tensor value_tensor = create_tensor(TensorDataType::FLOAT32, {2, 2}, {1.0f, 2.0f, 3.0f, 4.0f});
//     std::unordered_map<std::string, Node::AttributeValue> attributes;
//     attributes["value"] = value_tensor;
//     Tensor expected = create_tensor(TensorDataType::FLOAT32, {2, 2}, {1.0f, 2.0f, 3.0f, 4.0f});

//     std::vector<Tensor> inputs;
//     std::vector<Tensor> expected_tensors = {expected};

//     RUN_TEST_CASE(OperatorType::Constant, inputs, expected_tensors, attributes, DeviceType::CPU, OperatorExecuteResult::SUCCESS);
// }

// // ----------------------- ConvOperator Tests -----------------------
// TEST(OperatorTestHIP, ConvOperatorBasic)
// {
//     // Simple convolution test
//     Tensor X = create_tensor(TensorDataType::FLOAT32, {1, 1, 4, 4},
//                              {1, 2, 3, 4,
//                               5, 6, 7, 8,
//                               9, 10, 11, 12,
//                               13, 14, 15, 16});
//     Tensor W = create_tensor(TensorDataType::FLOAT32, {1, 1, 3, 3},
//                              {1, 0, -1,
//                               1, 0, -1,
//                               1, 0, -1});
//     Tensor B = create_tensor(TensorDataType::FLOAT32, {1}, {0});
//     Tensor expected = create_tensor(TensorDataType::FLOAT32, {1, 1, 2, 2}, {-6, -6, -6, -6});

//     std::unordered_map<std::string, Node::AttributeValue> attributes;
//     attributes["dilations"] = std::vector<int64_t>{1, 1};
//     attributes["group"] = 1;
//     attributes["kernel_shape"] = std::vector<int64_t>{3, 3};
//     attributes["pads"] = std::vector<int64_t>{0, 0, 0, 0};
//     attributes["strides"] = std::vector<int64_t>{1, 1};

//     std::vector<Tensor> inputs = {X, W, B};
//     std::vector<Tensor> expected_tensors = {expected};

//     RUN_TEST_CASE(OperatorType::Conv, inputs, expected_tensors, attributes, DeviceType::CPU, OperatorExecuteResult::SUCCESS);
// }

// // ----------------------- DivOperator Tests -----------------------
// TEST(OperatorTestHIP, DivOperatorBasic)
// {
//     // Basic division test
//     Tensor t1 = create_tensor(TensorDataType::FLOAT32, {2, 2}, {10.0f, 20.0f, 30.0f, 40.0f});
//     Tensor t2 = create_tensor(TensorDataType::FLOAT32, {2, 2}, {2.0f, 4.0f, 5.0f, 8.0f});
//     Tensor expected = create_tensor(TensorDataType::FLOAT32, {2, 2}, {5.0f, 5.0f, 6.0f, 5.0f});
//     std::unordered_map<std::string, Node::AttributeValue> attributes;

//     std::vector<Tensor> inputs = {t1, t2};
//     std::vector<Tensor> expected_tensors = {expected};

//     RUN_TEST_CASE(OperatorType::Div, inputs, expected_tensors, attributes, DeviceType::CPU, OperatorExecuteResult::SUCCESS);
// }

// // ----------------------- GatherOperator Tests -----------------------
// TEST(OperatorTestHIP, GatherOperatorBasic)
// {
//     // Basic gather test
//     Tensor data = create_tensor(TensorDataType::FLOAT32, {3, 4},
//                                 {0, 1, 2, 3,
//                                  4, 5, 6, 7,
//                                  8, 9, 10, 11});
//     Tensor indices = create_tensor(TensorDataType::INT64, {2}, {0, 2});
//     Tensor expected = create_tensor(TensorDataType::FLOAT32, {2, 4},
//                                     {0, 1, 2, 3,
//                                      8, 9, 10, 11});
//     std::unordered_map<std::string, Node::AttributeValue> attributes;
//     attributes["axis"] = 0;

//     std::vector<Tensor> inputs = {data, indices};
//     std::vector<Tensor> expected_tensors = {expected};

//     RUN_TEST_CASE(OperatorType::Gather, inputs, expected_tensors, attributes, DeviceType::CPU, OperatorExecuteResult::SUCCESS);
// }

// // ----------------------- MatMulOperator Tests -----------------------
// TEST(OperatorTestHIP, MatMulOperatorBasic)
// {
//     // Basic matrix multiplication
//     Tensor A = create_tensor(TensorDataType::FLOAT32, {2, 3},
//                              {1, 2, 3,
//                               4, 5, 6});
//     Tensor B = create_tensor(TensorDataType::FLOAT32, {3, 2},
//                              {7, 8,
//                               9, 10,
//                               11, 12});
//     Tensor expected = create_tensor(TensorDataType::FLOAT32, {2, 2},
//                                     {58, 64,
//                                      139, 154});

//     std::unordered_map<std::string, Node::AttributeValue> attributes;

//     std::vector<Tensor> inputs = {A, B};
//     std::vector<Tensor> expected_tensors = {expected};

//     RUN_TEST_CASE(OperatorType::MatMul, inputs, expected_tensors, attributes, DeviceType::CPU, OperatorExecuteResult::SUCCESS);
// }

// // ----------------------- MaxPoolOperator Tests -----------------------
// TEST(OperatorTestHIP, MaxPoolOperatorBasic)
// {
//     // Basic MaxPool test
//     Tensor X = create_tensor(TensorDataType::FLOAT32, {1, 1, 4, 4},
//                              {1, 2, 3, 4,
//                               5, 6, 7, 8,
//                               9, 10, 11, 12,
//                               13, 14, 15, 16});
//     Tensor expected = create_tensor(TensorDataType::FLOAT32, {1, 1, 2, 2},
//                                     {6, 8,
//                                      14, 16});
//     std::unordered_map<std::string, Node::AttributeValue> attributes;
//     attributes["kernel_shape"] = std::vector<int64_t>{2, 2};
//     attributes["pads"] = std::vector<int64_t>{0, 0, 0, 0};
//     attributes["strides"] = std::vector<int64_t>{2, 2};

//     std::vector<Tensor> inputs = {X};
//     std::vector<Tensor> expected_tensors = {expected};

//     RUN_TEST_CASE(OperatorType::MaxPool, inputs, expected_tensors, attributes, DeviceType::CPU, OperatorExecuteResult::SUCCESS);
// }

// // ----------------------- MulOperator Tests -----------------------
// TEST(OperatorTestHIP, MulOperatorBasic)
// {
//     // Basic multiplication test
//     Tensor t1 = create_tensor(TensorDataType::FLOAT32, {2, 2}, {1.0f, 2.0f, 3.0f, 4.0f});
//     Tensor t2 = create_tensor(TensorDataType::FLOAT32, {2, 2}, {5.0f, 6.0f, 7.0f, 8.0f});
//     Tensor expected = create_tensor(TensorDataType::FLOAT32, {2, 2}, {5.0f, 12.0f, 21.0f, 32.0f});
//     std::unordered_map<std::string, Node::AttributeValue> attributes;

//     std::vector<Tensor> inputs = {t1, t2};
//     std::vector<Tensor> expected_tensors = {expected};

//     RUN_TEST_CASE(OperatorType::Mul, inputs, expected_tensors, attributes, DeviceType::CPU, OperatorExecuteResult::SUCCESS);
// }

// // ----------------------- SigmoidOperator Tests -----------------------
// TEST(OperatorTestHIP, SigmoidOperatorBasic)
// {
//     // Basic sigmoid test
//     Tensor data = create_tensor(TensorDataType::FLOAT32, {2, 2}, {0.0f, -1.0f, 1.0f, 2.0f});
//     Tensor expected = create_tensor(TensorDataType::FLOAT32, {2, 2},
//                                     {0.5f, 0.26894f, 0.73106f, 0.880797f});

//     std::unordered_map<std::string, Node::AttributeValue> attributes;

//     std::vector<Tensor> inputs = {data};
//     std::vector<Tensor> expected_tensors = {expected};

//     RUN_TEST_CASE(OperatorType::Sigmoid, inputs, expected_tensors, attributes, DeviceType::CPU, OperatorExecuteResult::SUCCESS);
// }

// // ----------------------- SliceOperator Tests -----------------------
// TEST(OperatorTestHIP, SliceOperatorBasic)
// {
//     // Basic slice test
//     Tensor data = create_tensor(TensorDataType::FLOAT32, {3, 4},
//                                 {0, 1, 2, 3,
//                                  4, 5, 6, 7,
//                                  8, 9, 10, 11});
//     Tensor starts = create_tensor(TensorDataType::INT64, {1}, {1});
//     Tensor ends = create_tensor(TensorDataType::INT64, {1}, {3});
//     Tensor axes = create_tensor(TensorDataType::INT64, {1}, {0});
//     Tensor expected = create_tensor(TensorDataType::FLOAT32, {2, 4},
//                                     {4, 5, 6, 7,
//                                      8, 9, 10, 11});

//     std::unordered_map<std::string, Node::AttributeValue> attributes;

//     std::vector<Tensor> inputs = {data, starts, ends, axes};
//     std::vector<Tensor> expected_tensors = {expected};

//     RUN_TEST_CASE(OperatorType::Slice, inputs, expected_tensors, attributes, DeviceType::CPU, OperatorExecuteResult::SUCCESS);
// }

// // ----------------------- SoftmaxOperator Tests -----------------------
// TEST(OperatorTestHIP, SoftmaxOperatorBasic)
// {
//     // Basic softmax test
//     Tensor data = create_tensor(TensorDataType::FLOAT32, {2, 2}, {1.0f, 2.0f, 3.0f, 4.0f});
//     Tensor expected = create_tensor(TensorDataType::FLOAT32, {2, 2},
//                                     {0.26894f, 0.73106f, 0.26894f, 0.73106f});
//     std::unordered_map<std::string, Node::AttributeValue> attributes;
//     attributes["axis"] = 1;

//     std::vector<Tensor> inputs = {data};
//     std::vector<Tensor> expected_tensors = {expected};

//     RUN_TEST_CASE(OperatorType::Softmax, inputs, expected_tensors, attributes, DeviceType::CPU, OperatorExecuteResult::SUCCESS);
// }

// // ----------------------- SplitOperator Tests -----------------------
// TEST(OperatorTestHIP, SplitOperatorBasic)
// {
//     // Basic split test
//     Tensor data = create_tensor(TensorDataType::FLOAT32, {2, 4},
//                                 {1, 2, 3, 4,
//                                  5, 6, 7, 8});
//     Tensor split = create_tensor(TensorDataType::INT64, {2}, {2, 2});
//     Tensor expected1 = create_tensor(TensorDataType::FLOAT32, {2, 2},
//                                      {1, 2,
//                                       5, 6});
//     Tensor expected2 = create_tensor(TensorDataType::FLOAT32, {2, 2},
//                                      {3, 4,
//                                       7, 8});
//     std::unordered_map<std::string, Node::AttributeValue> attributes;
//     attributes["axis"] = 1;

//     std::vector<Tensor> inputs = {data, split};
//     std::vector<Tensor> expected_tensors = {expected1, expected2};

//     RUN_TEST_CASE(OperatorType::Split, inputs, expected_tensors, attributes, DeviceType::CPU, OperatorExecuteResult::SUCCESS);
// }

// /// FIXME: Shape mismatch should be handled during the shape inference phase
// // TEST(OperatorTest1, SplitOperatorShapeMismatchError)
// // {
// //     Tensor data = create_tensor(TensorDataType::FLOAT32, {2, 4}, {1, 2, 3, 4, 5, 6, 7, 8});
// //     Tensor split = create_tensor(TensorDataType::INT64, {2}, {3, 2});
// //     std::unordered_map<std::string, Node::AttributeValue> attributes;
// //     attributes["axis"] = 1;

// //     std::vector<Tensor> inputs = {data, split};
// //     std::vector<Tensor> expected_tensors;

// //     RUN_TEST_CASE(OperatorType::Split, inputs, expected_tensors, attributes, DeviceType::CPU, OperatorExecuteResult::SHAPE_MISMATCH_ERROR);
// // }

// // ----------------------- TransposeOperator Tests -----------------------
// TEST(OperatorTestHIP, TransposeOperatorBasic)
// {
//     // Basic transpose test
//     Tensor data = create_tensor(TensorDataType::FLOAT32, {2, 3}, {1, 2, 3, 4, 5, 6});
//     Tensor expected = create_tensor(TensorDataType::FLOAT32, {3, 2}, {1, 4, 2, 5, 3, 6});
//     std::unordered_map<std::string, Node::AttributeValue> attributes;
//     attributes["perm"] = std::vector<int64_t>{1, 0};

//     std::vector<Tensor> inputs = {data};
//     std::vector<Tensor> expected_tensors = {expected};

//     RUN_TEST_CASE(OperatorType::Transpose, inputs, expected_tensors, attributes, DeviceType::CPU, OperatorExecuteResult::SUCCESS);
// }

// // ----------------------- ReshapeOperator Tests -----------------------
// TEST(OperatorTestHIP, ReshapeOperatorBasic)
// {
//     // Basic reshape test
//     Tensor data = create_tensor(TensorDataType::FLOAT32, {2, 3}, {1, 2, 3, 4, 5, 6});
//     Tensor shape = create_tensor(TensorDataType::INT64, {3}, {3, 2, 1});
//     Tensor expected = create_tensor(TensorDataType::FLOAT32, {3, 2, 1}, {1, 2, 3, 4, 5, 6});

//     std::unordered_map<std::string, Node::AttributeValue> attributes;

//     std::vector<Tensor> inputs = {data, shape};
//     std::vector<Tensor> expected_tensors = {expected};

//     RUN_TEST_CASE(OperatorType::Reshape, inputs, expected_tensors, attributes, DeviceType::CPU, OperatorExecuteResult::SUCCESS);
// }

// // ----------------------- ResizeOperator Tests -----------------------
// TEST(OperatorTestHIP, ResizeOperatorBasic)
// {
//     // Basic resize test (nearest neighbor)
//     Tensor data = create_tensor(TensorDataType::FLOAT32, {1, 1, 2, 2}, {1, 2, 3, 4});
//     Tensor scales = create_tensor(TensorDataType::FLOAT32, {4}, {1.0f, 1.0f, 2.0f, 2.0f});
//     Tensor expected = create_tensor(TensorDataType::FLOAT32, {1, 1, 4, 4},
//                                     {1, 1, 2, 2,
//                                      1, 1, 2, 2,
//                                      3, 3, 4, 4,
//                                      3, 3, 4, 4});
//     std::unordered_map<std::string, Node::AttributeValue> attributes;
//     attributes["mode"] = std::string("nearest");

//     std::vector<Tensor> inputs = {data, Tensor(), scales};
//     std::vector<Tensor> expected_tensors = {expected};

//     RUN_TEST_CASE(OperatorType::Resize, inputs, expected_tensors, attributes, DeviceType::CPU, OperatorExecuteResult::SUCCESS);
// }

// // ----------------------- ShapeOperator Tests -----------------------
// TEST(OperatorTestHIP, ShapeOperatorBasic)
// {
//     // Basic shape operator test
//     Tensor data = create_tensor(TensorDataType::FLOAT32, {2, 3, 4}, std::vector<float>(24, 1.0f));
//     Tensor expected = create_tensor(TensorDataType::INT64, {3}, {2, 3, 4});

//     std::unordered_map<std::string, Node::AttributeValue> attributes;

//     std::vector<Tensor> inputs = {data};
//     std::vector<Tensor> expected_tensors = {expected};

//     RUN_TEST_CASE(OperatorType::Shape, inputs, expected_tensors, attributes, DeviceType::CPU, OperatorExecuteResult::SUCCESS);
// }

#endif // USE_HIP