#include <gtest/gtest.h>

#include "tensor/tensor.hpp"
#include "operator/operators.hpp"
#include "operator/operator_registry.hpp"

#include "device/device.hpp"
#include "device/device_cpu.hpp"

#ifdef USE_HIP
#include "device/device_hip.hpp"
#endif

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
                ASSERT_NEAR(output_data[j], expected_data[j], 1e-3);
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

#ifdef USE_HIP

// ----------------------- AddOperator Tests -----------------------
TEST(QuantizationTestHIP, AddOperatorBasic)
{
    HipDevice hipDevice = HipDevice(0);

    Tensor t1(TensorDataType::FLOAT16, {2, 3}, std::vector<half_t>{1.0f, 2.0f, 3.0f, 4.0f, 3.0f, 4.0f}, &hipDevice);
    Tensor t2(TensorDataType::FLOAT16, {2, 3}, std::vector<half_t>{5.0f, 6.0f, 7.0f, 8.0f, 3.0f, 4.0f}, &hipDevice);
    Tensor expected(TensorDataType::FLOAT16, {2, 3}, std::vector<half_t>{6.0f, 8.0f, 10.0f, 12.0f, 6.0f, 8.0f}, &hipDevice);
    std::unordered_map<std::string, Node::AttributeValue> attributes;

    std::vector<Tensor> inputs = {t1, t2};
    std::vector<Tensor> expected_tensors = {expected};

    RUN_TEST_CASE(OperatorType::Add, inputs, expected_tensors, attributes, OperatorExecuteResult::SUCCESS, &hipDevice);
}
TEST(QuantizationTestHIP, AddOperatorBroadcastVector)
{
    // Broadcasting vector addition
    HipDevice hipDevice = HipDevice(0);
    Tensor t1(TensorDataType::FLOAT16, {2, 2}, std::vector<half_t>{1.0f, 2.0f, 3.0f, 4.0f}, &hipDevice);
    Tensor t2(TensorDataType::FLOAT16, {2}, std::vector<half_t>{10.0f, 20.0f}, &hipDevice); // Vector tensor
    Tensor expected(TensorDataType::FLOAT16, {2, 2}, std::vector<half_t>{11.0f, 22.0f, 13.0f, 24.0f}, &hipDevice);
    std::unordered_map<std::string, Node::AttributeValue> attributes;

    std::vector<Tensor> inputs = {t1, t2};
    std::vector<Tensor> expected_tensors = {expected};

    RUN_TEST_CASE(OperatorType::Add, inputs, expected_tensors, attributes, OperatorExecuteResult::SUCCESS, &hipDevice);
}

// ----------------------- SubOperator Tests -----------------------
TEST(QuantizationTestHIP, SubOperatorBasic)
{
    // Basic subtraction test
    HipDevice hipDevice = HipDevice(0);
    Tensor t1(TensorDataType::FLOAT16, {2, 2}, std::vector<half_t>{5.0f, 7.0f, 9.0f, 11.0f}, &hipDevice);
    Tensor t2(TensorDataType::FLOAT16, {2, 2}, std::vector<half_t>{1.0f, 2.0f, 3.0f, 4.0f}, &hipDevice);
    Tensor expected(TensorDataType::FLOAT16, {2, 2}, std::vector<half_t>{4.0f, 5.0f, 6.0f, 7.0f}, &hipDevice);

    std::unordered_map<std::string, Node::AttributeValue> attributes;

    std::vector<Tensor> inputs = {t1, t2};
    std::vector<Tensor> expected_tensors = {expected};

    RUN_TEST_CASE(OperatorType::Sub, inputs, expected_tensors, attributes, OperatorExecuteResult::SUCCESS, &hipDevice);
}

TEST(QuantizationTestHIP, SubOperatorShapeMismatchError)
{
    HipDevice hipDevice = HipDevice(0);
    Tensor t1(TensorDataType::FLOAT16, {2, 2}, std::vector<half_t>{1.0f, 2.0f, 3.0f, 4.0f}, &hipDevice);
    Tensor t2(TensorDataType::FLOAT16, {3, 2}, std::vector<half_t>{5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f}, &hipDevice);
    std::unordered_map<std::string, Node::AttributeValue> attributes;

    std::vector<Tensor> inputs = {t1, t2};
    std::vector<Tensor> expected_tensors;

    RUN_TEST_CASE(OperatorType::Sub, inputs, expected_tensors, attributes, OperatorExecuteResult::SHAPE_MISMATCH_ERROR, &hipDevice);
}

TEST(QuantizationTestHIP, SubOperatorBroadcastScalar)
{
    // Broadcasting scalar subtraction
    HipDevice hipDevice = HipDevice(0);
    Tensor t1(TensorDataType::FLOAT16, {2, 2}, std::vector<half_t>{10.0f, 20.0f, 30.0f, 40.0f}, &hipDevice);
    Tensor t2(TensorDataType::FLOAT16, {}, std::vector<half_t>{2.0f}, &hipDevice); // Scalar tensor
    Tensor expected(TensorDataType::FLOAT16, {2, 2}, std::vector<half_t>{8.0f, 18.0f, 28.0f, 38.0f}, &hipDevice);
    std::unordered_map<std::string, Node::AttributeValue> attributes;

    std::vector<Tensor> inputs = {t1, t2};
    std::vector<Tensor> expected_tensors = {expected};

    RUN_TEST_CASE(OperatorType::Sub, inputs, expected_tensors, attributes, OperatorExecuteResult::SUCCESS, &hipDevice);
}

/// FIXME: Broadcasting is likely broken
TEST(QuantizationTestHIP, SubOperatorBroadcastVector0)
{
    // Broadcasting vector subtraction
    HipDevice hipDevice = HipDevice(0);
    Tensor t1(TensorDataType::FLOAT16, {2, 2}, std::vector<half_t>{1.0f, 2.0f, 3.0f, 4.0f}, &hipDevice);
    Tensor t2(TensorDataType::FLOAT16, {2}, std::vector<half_t>{10.0f, 20.0f}, &hipDevice); // Vector tensor
    Tensor expected(TensorDataType::FLOAT16, {2, 2}, std::vector<half_t>{-9.0f, -18.0f, -7.0f, -16.0f}, &hipDevice);
    std::unordered_map<std::string, Node::AttributeValue> attributes;

    std::vector<Tensor> inputs = {t1, t2};
    std::vector<Tensor> expected_tensors = {expected};

    RUN_TEST_CASE(OperatorType::Sub, inputs, expected_tensors, attributes, OperatorExecuteResult::SUCCESS, &hipDevice);
}

TEST(QuantizationTestHIP, SubOperatorBroadcastVector1)
{
    // Broadcasting vector subtraction
    HipDevice hipDevice = HipDevice(0);
    Tensor t1(TensorDataType::FLOAT16, {2, 3}, std::vector<half_t>{1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f}, &hipDevice);
    Tensor t2(TensorDataType::FLOAT16, {3}, std::vector<half_t>{10.0f, 20.0f, 30.0f}, &hipDevice); // Vector tensor
    Tensor expected(TensorDataType::FLOAT16, {2, 3}, std::vector<half_t>{-9.0f, -18.0f, -27.0f, -6.0f, -15.0f, -24.0f}, &hipDevice);
    std::unordered_map<std::string, Node::AttributeValue> attributes;

    std::vector<Tensor> inputs = {t1, t2};
    std::vector<Tensor> expected_tensors = {expected};

    RUN_TEST_CASE(OperatorType::Sub, inputs, expected_tensors, attributes, OperatorExecuteResult::SUCCESS, &hipDevice);
}
TEST(QuantizationTestHIP, SubOperatorBroadcastVector2)
{
    // Broadcasting vector subtraction
    HipDevice hipDevice = HipDevice(0);
    Tensor t1(TensorDataType::FLOAT16, {1, 2, 3}, std::vector<half_t>{1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f}, &hipDevice);
    Tensor t2(TensorDataType::FLOAT16, {2, 3}, std::vector<half_t>{10.0f, 20.0f, 30.0f, 40.0f, 50.0f, 60.0f}, &hipDevice); // Vector tensor
    Tensor expected(TensorDataType::FLOAT16, {1, 2, 3}, std::vector<half_t>{-9.0f, -18.0f, -27.0f, -36.0f, -45.0f, -54.0f}, &hipDevice);

    std::unordered_map<std::string, Node::AttributeValue> attributes;

    std::vector<Tensor> inputs = {t1, t2};
    std::vector<Tensor> expected_tensors = {expected};

    RUN_TEST_CASE(OperatorType::Sub, inputs, expected_tensors, attributes, OperatorExecuteResult::SUCCESS, &hipDevice);
}

// ----------------------- ConcatOperator Tests -----------------------
TEST(QuantizationTestHIP, ConcatOperatorBasic)
{
    // Basic concatenation along axis 0
    HipDevice hipDevice = HipDevice(0);
    Tensor t1(TensorDataType::FLOAT16, {2, 3}, std::vector<half_t>{1, 2, 3, 4, 5, 6}, &hipDevice);
    Tensor t2(TensorDataType::FLOAT16, {2, 3}, std::vector<half_t>{7, 8, 9, 10, 11, 12}, &hipDevice);
    Tensor expected(TensorDataType::FLOAT16, {4, 3},
                    std::vector<half_t>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}, &hipDevice);
    std::unordered_map<std::string, Node::AttributeValue> attributes;
    attributes["axis"] = 0;

    std::vector<Tensor> inputs = {t1, t2};
    std::vector<Tensor> expected_tensors = {expected};

    RUN_TEST_CASE(OperatorType::Concat, inputs, expected_tensors, attributes, OperatorExecuteResult::SUCCESS, &hipDevice);
}

TEST(QuantizationTestHIP, ConcatOperatorShapeMismatchError)
{
    // Shape mismatch error
    HipDevice hipDevice = HipDevice(0);
    Tensor t1(TensorDataType::FLOAT16, {2, 2}, std::vector<half_t>{1, 2, 3, 4}, &hipDevice);
    Tensor t2(TensorDataType::FLOAT16, {2, 3}, std::vector<half_t>{5, 6, 7, 8, 9, 10}, &hipDevice);
    std::unordered_map<std::string, Node::AttributeValue> attributes;
    attributes["axis"] = 0;

    std::vector<Tensor> inputs = {t1, t2};
    std::vector<Tensor> expected_tensors;

    RUN_TEST_CASE(OperatorType::Concat, inputs, expected_tensors, attributes, OperatorExecuteResult::SHAPE_MISMATCH_ERROR, &hipDevice);
}

// ----------------------- ConstantOperator Tests -----------------------
TEST(QuantizationTestHIP, ConstantOperatorBasic)
{
    // Create a constant tensor
    HipDevice hipDevice = HipDevice(0);
    Tensor value_tensor(TensorDataType::FLOAT16, {2, 2}, std::vector<half_t>{1.0f, 2.0f, 3.0f, 4.0f}, &hipDevice);
    std::unordered_map<std::string, Node::AttributeValue> attributes;
    attributes["value"] = value_tensor;
    Tensor expected(TensorDataType::FLOAT16, {2, 2}, std::vector<half_t>{1.0f, 2.0f, 3.0f, 4.0f}, &hipDevice);

    std::vector<Tensor> inputs;
    std::vector<Tensor> expected_tensors = {expected};

    RUN_TEST_CASE(OperatorType::Constant, inputs, expected_tensors, attributes, OperatorExecuteResult::SUCCESS, &hipDevice);
}

// ----------------------- ConvOperator Tests -----------------------
TEST(QuantizationTestHIP, ConvOperatorBasic)
{
    // Simple convolution test
    HipDevice hipDevice = HipDevice(0);
    Tensor X(TensorDataType::FLOAT16, {1, 1, 4, 4},
             std::vector<half_t>{1, 2, 3, 4,
                                5, 6, 7, 8,
                                9, 10, 11, 12,
                                13, 14, 15, 16},
             &hipDevice);
    Tensor W(TensorDataType::FLOAT16, {1, 1, 3, 3},
             std::vector<half_t>{1, 0, -1,
                                1, 0, -1,
                                1, 0, -1},
             &hipDevice);
    Tensor B(TensorDataType::FLOAT16, {1}, std::vector<half_t>{0}, &hipDevice);
    Tensor expected(TensorDataType::FLOAT16, {1, 1, 2, 2}, std::vector<half_t>{-6, -6, -6, -6}, &hipDevice);

    std::unordered_map<std::string, Node::AttributeValue> attributes;
    attributes["dilations"] = std::vector<int64_t>{1, 1};
    attributes["group"] = 1;
    attributes["kernel_shape"] = std::vector<int64_t>{3, 3};
    attributes["pads"] = std::vector<int64_t>{0, 0, 0, 0};
    attributes["strides"] = std::vector<int64_t>{1, 1};

    std::vector<Tensor> inputs = {X, W, B};
    std::vector<Tensor> expected_tensors = {expected};

    RUN_TEST_CASE(OperatorType::Conv, inputs, expected_tensors, attributes, OperatorExecuteResult::SUCCESS, &hipDevice);
}

// ----------------------- DivOperator Tests -----------------------
TEST(QuantizationTestHIP, DivOperatorBasic)
{
    // Basic division test
    HipDevice hipDevice = HipDevice(0);
    Tensor t1(TensorDataType::FLOAT16, {2, 2}, std::vector<half_t>{10.0f, 20.0f, 30.0f, 40.0f}, &hipDevice);
    Tensor t2(TensorDataType::FLOAT16, {2, 2}, std::vector<half_t>{2.0f, 4.0f, 5.0f, 8.0f}, &hipDevice);
    Tensor expected(TensorDataType::FLOAT16, {2, 2}, std::vector<half_t>{5.0f, 5.0f, 6.0f, 5.0f}, &hipDevice);
    std::unordered_map<std::string, Node::AttributeValue> attributes;

    std::vector<Tensor> inputs = {t1, t2};
    std::vector<Tensor> expected_tensors = {expected};

    RUN_TEST_CASE(OperatorType::Div, inputs, expected_tensors, attributes, OperatorExecuteResult::SUCCESS, &hipDevice);
}

// ----------------------- GatherOperator Tests -----------------------
TEST(QuantizationTestHIP, GatherOperatorBasic)
{
    // Basic gather test
    HipDevice hipDevice = HipDevice(0);
    Tensor data(TensorDataType::FLOAT16, {3, 4},
                std::vector<half_t>{0, 1, 2, 3,
                                   4, 5, 6, 7,
                                   8, 9, 10, 11},
                &hipDevice);
    Tensor indices(TensorDataType::INT64, {2}, std::vector<int64_t>{0, 2}, &hipDevice);
    Tensor expected(TensorDataType::FLOAT16, {2, 4},
                    std::vector<half_t>{0, 1, 2, 3,
                                       8, 9, 10, 11},
                    &hipDevice);
    std::unordered_map<std::string, Node::AttributeValue> attributes;
    attributes["axis"] = 0;

    std::vector<Tensor> inputs = {data, indices};
    std::vector<Tensor> expected_tensors = {expected};

    RUN_TEST_CASE(OperatorType::Gather, inputs, expected_tensors, attributes, OperatorExecuteResult::SUCCESS, &hipDevice);
}

// ----------------------- MatMulOperator Tests -----------------------
// TEST(QuantizationTestHIP, MatMulOperatorBasic)
// {
//     // Basic matrix multiplication
//     HipDevice hipDevice = HipDevice(0);
//     Tensor A(TensorDataType::FLOAT16, {2, 3},
//              std::vector<half_t>{1, 2, 3,
//                                 4, 5, 6},
//              &hipDevice);
//     Tensor B(TensorDataType::FLOAT16, {3, 2},
//              std::vector<half_t>{7, 8,
//                                 9, 10,
//                                 11, 12},
//              &hipDevice);
//     Tensor expected(TensorDataType::FLOAT16, {2, 2},
//                     std::vector<half_t>{58, 64,
//                                        139, 154},
//                     &hipDevice);

//     std::unordered_map<std::string, Node::AttributeValue> attributes;

//     std::vector<Tensor> inputs = {A, B};
//     std::vector<Tensor> expected_tensors = {expected};

//     RUN_TEST_CASE(OperatorType::MatMul, inputs, expected_tensors, attributes, OperatorExecuteResult::SUCCESS, &hipDevice);
// }

// ----------------------- MaxPoolOperator Tests -----------------------
TEST(QuantizationTestHIP, MaxPoolOperatorBasic)
{
    // Basic MaxPool test
    HipDevice hipDevice = HipDevice(0);
    Tensor X(TensorDataType::FLOAT16, {1, 1, 4, 4},
             std::vector<half_t>{1, 2, 3, 4,
                                5, 6, 7, 8,
                                9, 10, 11, 12,
                                13, 14, 15, 16},
             &hipDevice);
    Tensor expected(TensorDataType::FLOAT16, {1, 1, 2, 2},
                    std::vector<half_t>{6, 8,
                                       14, 16},
                    &hipDevice);
    std::unordered_map<std::string, Node::AttributeValue> attributes;
    attributes["kernel_shape"] = std::vector<int64_t>{2, 2};
    attributes["pads"] = std::vector<int64_t>{0, 0, 0, 0};
    attributes["strides"] = std::vector<int64_t>{2, 2};

    std::vector<Tensor> inputs = {X};
    std::vector<Tensor> expected_tensors = {expected};

    RUN_TEST_CASE(OperatorType::MaxPool, inputs, expected_tensors, attributes, OperatorExecuteResult::SUCCESS, &hipDevice);
}

// ----------------------- MulOperator Tests -----------------------
TEST(QuantizationTestHIP, MulOperatorBasic)
{
    // Basic multiplication test
    HipDevice hipDevice = HipDevice(0);
    Tensor t1(TensorDataType::FLOAT16, {2, 2}, std::vector<half_t>{1.0f, 2.0f, 3.0f, 4.0f}, &hipDevice);
    Tensor t2(TensorDataType::FLOAT16, {2, 2}, std::vector<half_t>{5.0f, 6.0f, 7.0f, 8.0f}, &hipDevice);
    Tensor expected(TensorDataType::FLOAT16, {2, 2}, std::vector<half_t>{5.0f, 12.0f, 21.0f, 32.0f}, &hipDevice);
    std::unordered_map<std::string, Node::AttributeValue> attributes;

    std::vector<Tensor> inputs = {t1, t2};
    std::vector<Tensor> expected_tensors = {expected};

    RUN_TEST_CASE(OperatorType::Mul, inputs, expected_tensors, attributes, OperatorExecuteResult::SUCCESS, &hipDevice);
}

// ----------------------- SigmoidOperator Tests -----------------------
TEST(QuantizationTestHIP, SigmoidOperatorBasic)
{
    // Basic sigmoid test
    HipDevice hipDevice = HipDevice(0);
    Tensor data(TensorDataType::FLOAT16, {2, 2}, std::vector<half_t>{0.0f, -1.0f, 1.0f, 2.0f}, &hipDevice);
    Tensor expected(TensorDataType::FLOAT16, {2, 2},
                    std::vector<half_t>{0.5f, 0.26894f, 0.73106f, 0.880797f}, &hipDevice);

    std::unordered_map<std::string, Node::AttributeValue> attributes;

    std::vector<Tensor> inputs = {data};
    std::vector<Tensor> expected_tensors = {expected};

    RUN_TEST_CASE(OperatorType::Sigmoid, inputs, expected_tensors, attributes, OperatorExecuteResult::SUCCESS, &hipDevice);
}

// ----------------------- SliceOperator Tests -----------------------
TEST(QuantizationTestHIP, SliceOperatorBasic)
{
    // Basic slice test
    HipDevice hipDevice = HipDevice(0);
    Tensor data(TensorDataType::FLOAT16, {3, 4},
                std::vector<half_t>{0, 1, 2, 3,
                                   4, 5, 6, 7,
                                   8, 9, 10, 11},
                &hipDevice);
    Tensor starts(TensorDataType::INT64, {1}, std::vector<int64_t>{1}, &hipDevice);
    Tensor ends(TensorDataType::INT64, {1}, std::vector<int64_t>{3}, &hipDevice);
    Tensor axes(TensorDataType::INT64, {1}, std::vector<int64_t>{0}, &hipDevice);
    Tensor expected(TensorDataType::FLOAT16, {2, 4},
                    std::vector<half_t>{4, 5, 6, 7,
                                       8, 9, 10, 11},
                    &hipDevice);

    std::unordered_map<std::string, Node::AttributeValue> attributes;

    std::vector<Tensor> inputs = {data, starts, ends, axes};
    std::vector<Tensor> expected_tensors = {expected};

    RUN_TEST_CASE(OperatorType::Slice, inputs, expected_tensors, attributes, OperatorExecuteResult::SUCCESS, &hipDevice);
}

// ----------------------- SoftmaxOperator Tests -----------------------
TEST(QuantizationTestHIP, SoftmaxOperatorBasic)
{
    // Basic softmax test
    HipDevice hipDevice = HipDevice(0);
    Tensor data(TensorDataType::FLOAT16, {2, 2}, std::vector<half_t>{1.0f, 2.0f, 3.0f, 4.0f}, &hipDevice);
    Tensor expected(TensorDataType::FLOAT16, {2, 2},
                    std::vector<half_t>{0.26894f, 0.73106f, 0.26894f, 0.73106f}, &hipDevice);
    std::unordered_map<std::string, Node::AttributeValue> attributes;
    attributes["axis"] = 1;

    std::vector<Tensor> inputs = {data};
    std::vector<Tensor> expected_tensors = {expected};

    RUN_TEST_CASE(OperatorType::Softmax, inputs, expected_tensors, attributes, OperatorExecuteResult::SUCCESS, &hipDevice);
}

// ----------------------- SplitOperator Tests -----------------------
TEST(QuantizationTestHIP, SplitOperatorBasic)
{
    // Basic split test
    HipDevice hipDevice = HipDevice(0);
    Tensor data(TensorDataType::FLOAT16, {2, 4},
                std::vector<half_t>{1, 2, 3, 4,
                                   5, 6, 7, 8},
                &hipDevice);
    Tensor split(TensorDataType::INT64, {2}, std::vector<int64_t>{2, 2}, &hipDevice);
    Tensor expected1(TensorDataType::FLOAT16, {2, 2},
                     std::vector<half_t>{1, 2,
                                        5, 6},
                     &hipDevice);
    Tensor expected2(TensorDataType::FLOAT16, {2, 2},
                     std::vector<half_t>{3, 4,
                                        7, 8},
                     &hipDevice);
    std::unordered_map<std::string, Node::AttributeValue> attributes;
    attributes["axis"] = 1;

    std::vector<Tensor> inputs = {data, split};
    std::vector<Tensor> expected_tensors = {expected1, expected2};

    RUN_TEST_CASE(OperatorType::Split, inputs, expected_tensors, attributes, OperatorExecuteResult::SUCCESS, &hipDevice);
}

// ----------------------- TransposeOperator Tests -----------------------
TEST(QuantizationTestHIP, TransposeOperatorBasic)
{
    // Basic transpose test
    HipDevice hipDevice = HipDevice(0);
    Tensor input(TensorDataType::FLOAT16, {2, 3}, std::vector<half_t>{1, 2, 3, 4, 5, 6}, &hipDevice);
    Tensor expected(TensorDataType::FLOAT16, {3, 2}, std::vector<half_t>{1, 4, 2, 5, 3, 6}, &hipDevice);
    std::unordered_map<std::string, Node::AttributeValue> attributes;
    attributes["perm"] = std::vector<int64_t>{1, 0};

    std::vector<Tensor> inputs = {input};
    std::vector<Tensor> expected_tensors = {expected};

    RUN_TEST_CASE(OperatorType::Transpose, inputs, expected_tensors, attributes, OperatorExecuteResult::SUCCESS, &hipDevice);
}

// ----------------------- ReshapeOperator Tests -----------------------
TEST(QuantizationTestHIP, ReshapeOperatorBasic)
{
    // Basic reshape test
    HipDevice hipDevice = HipDevice(0);
    Tensor data(TensorDataType::FLOAT16, {2, 3}, std::vector<half_t>{1, 2, 3, 4, 5, 6}, &hipDevice);
    Tensor shape(TensorDataType::INT64, {3}, std::vector<int64_t>{3, 2, 1}, &hipDevice);
    Tensor expected(TensorDataType::FLOAT16, {3, 2, 1}, std::vector<half_t>{1, 2, 3, 4, 5, 6}, &hipDevice);

    std::unordered_map<std::string, Node::AttributeValue> attributes;

    std::vector<Tensor> inputs = {data, shape};
    std::vector<Tensor> expected_tensors = {expected};

    RUN_TEST_CASE(OperatorType::Reshape, inputs, expected_tensors, attributes, OperatorExecuteResult::SUCCESS, &hipDevice);
}

// ----------------------- ResizeOperator Tests -----------------------
TEST(QuantizationTestHIP, ResizeOperatorBasic)
{
    // Basic resize test (nearest neighbor)
    HipDevice hipDevice = HipDevice(0);
    Tensor data(TensorDataType::FLOAT16, {1, 1, 2, 2}, std::vector<half_t>{1, 2, 3, 4}, &hipDevice);
    Tensor scales(TensorDataType::FLOAT32, {4}, std::vector<float>{1.0f, 1.0f, 2.0f, 2.0f}, &hipDevice);
    Tensor expected(TensorDataType::FLOAT16, {1, 1, 4, 4},
                    std::vector<half_t>{1, 1, 2, 2,
                                       1, 1, 2, 2,
                                       3, 3, 4, 4,
                                       3, 3, 4, 4},
                    &hipDevice);
    std::unordered_map<std::string, Node::AttributeValue> attributes;
    attributes["mode"] = std::string("nearest");

    std::vector<Tensor> inputs = {data, Tensor(), scales};
    std::vector<Tensor> expected_tensors = {expected};

    RUN_TEST_CASE(OperatorType::Resize, inputs, expected_tensors, attributes, OperatorExecuteResult::SUCCESS, &hipDevice);
}

// ----------------------- ShapeOperator Tests -----------------------
TEST(QuantizationTestHIP, ShapeOperatorBasic)
{
    // Basic shape operator test
    HipDevice hipDevice = HipDevice(0);
    Tensor data(TensorDataType::FLOAT16, {2, 3, 4}, std::vector<half_t>(24, 1.0f), &hipDevice);
    Tensor expected(TensorDataType::INT64, {3}, std::vector<int64_t>{2, 3, 4}, &hipDevice);

    std::unordered_map<std::string, Node::AttributeValue> attributes;

    std::vector<Tensor> inputs = {data};
    std::vector<Tensor> expected_tensors = {expected};

    RUN_TEST_CASE(OperatorType::Shape, inputs, expected_tensors, attributes, OperatorExecuteResult::SUCCESS, &hipDevice);
}

#endif // USE_HIP