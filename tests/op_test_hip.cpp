#ifdef USE_HIP
#include <gtest/gtest.h>
#include <hip/hip_runtime.h>

#include "tensor/tensor.hpp"
#include "operator/operators.hpp"
#include "operator/operator_registry.hpp"

void PrintTo(OperatorExecuteResult result, std::ostream *os)
{
    *os << OperatorUtils::OperatorExecuteResultToString(result);
}

void run_and_check_operator(const OperatorRegistry::OperatorFunctions *op,
                            const std::vector<Tensor> &inputs,
                            std::vector<Tensor *> outputs,
                            const std::vector<Tensor> &expected,
                            std::unordered_map<std::string, Node::AttributeValue> attributes = {},
                            OperatorExecuteResult expected_execute_result = OperatorExecuteResult::SUCCESS,
                            DeviceType deviceType = DeviceType::HIP)
{
    OperatorExecuteResult result_code = op->execute(inputs, outputs, attributes, deviceType);

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

#define RUN_TEST_CASE(operator_type, input_tensors, expected_tensors, attributes, deviceType, expectedResult)  \
    do                                                                                                         \
    {                                                                                                          \
        const OperatorRegistry::OperatorFunctions *op = OperatorRegistry::getOperatorFunctions(operator_type); \
        std::vector<std::vector<size_t>> output_shapes = op->inferOutputShapes(input_tensors, attributes);     \
        std::vector<TensorDataType> output_data_types = op->inferOutputDataTypes(input_tensors, attributes);   \
                                                                                                               \
        std::vector<Tensor *> outputs;                                                                         \
        for (size_t i = 0; i < output_shapes.size(); i++)                                                      \
        {                                                                                                      \
            outputs.push_back(new Tensor(output_data_types[i], output_shapes[i]));                             \
        }                                                                                                      \
                                                                                                               \
        run_and_check_operator(op, input_tensors, outputs, expected_tensors, attributes,                       \
                               expectedResult, deviceType);                                                    \
                                                                                                               \
        for (auto &output : outputs)                                                                           \
        {                                                                                                      \
            delete output;                                                                                     \
        }                                                                                                      \
    } while (0)

// ----------------------- AddOperator Tests -----------------------
TEST(OperaotrTestCPU, AddOperatorBasic)
{
    Tensor t1 = create_tensor(TensorDataType::FLOAT32, {2, 2}, {1.0f, 2.0f, 3.0f, 4.0f}, DeviceType::HIP);
    Tensor t2 = create_tensor(TensorDataType::FLOAT32, {2, 2}, {5.0f, 6.0f, 7.0f, 8.0f}, DeviceType::HIP);
    Tensor expected = create_tensor(TensorDataType::FLOAT32, {2, 2}, {6.0f, 8.0f, 10.0f, 12.0f}, DeviceType::HIP);
    std::unordered_map<std::string, Node::AttributeValue> attributes;

    std::vector<Tensor> inputs = {t1, t2};
    std::vector<Tensor> expected_tensors = {expected};

    RUN_TEST_CASE(OperatorType::Add, inputs, expected_tensors, attributes, DeviceType::HIP, OperatorExecuteResult::SUCCESS);
}

// TEST(OperaotrTestCPU, AddOperatorBroadcastScalar)
// {
//     // Broadcasting scalar addition
//     Tensor t1 = create_tensor(TensorDataType::FLOAT32, {2, 2}, {1.0f, 2.0f, 3.0f, 4.0f});
//     Tensor t2 = create_tensor(TensorDataType::FLOAT32, {}, {10.0f}); // Scalar tensor
//     Tensor expected = create_tensor(TensorDataType::FLOAT32, {2, 2}, {11.0f, 12.0f, 13.0f, 14.0f});
//     std::unordered_map<std::string, Node::AttributeValue> attributes;

//     std::vector<Tensor> inputs = {t1, t2};
//     std::vector<Tensor> expected_tensors = {expected};

//     RUN_TEST_CASE(OperatorType::Add, inputs, expected_tensors, attributes, DeviceType::CPU, OperatorExecuteResult::SUCCESS);
// }

// TEST(OperaotrTestCPU, AddOperatorShapeMismatchError)
// {
//     Tensor t1 = create_tensor(TensorDataType::FLOAT32, {2, 2}, {1.0f, 2.0f, 3.0f, 4.0f});
//     Tensor t2 = create_tensor(TensorDataType::FLOAT32, {3, 2}, {5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f});
//     std::unordered_map<std::string, Node::AttributeValue> attributes;

//     std::vector<Tensor> inputs = {t1, t2};
//     std::vector<Tensor> expected_tensors;

//     RUN_TEST_CASE(OperatorType::Add, inputs, expected_tensors, attributes, DeviceType::CPU, OperatorExecuteResult::SHAPE_MISMATCH_ERROR);
// }

// // ----------------------- ConcatOperator Tests -----------------------
// TEST(OperaotrTestCPU, ConcatOperatorBasic)
// {
//     // Basic concatenation along axis 0
//     Tensor t1 = create_tensor(TensorDataType::FLOAT32, {2, 3}, {1, 2, 3, 4, 5, 6});
//     Tensor t2 = create_tensor(TensorDataType::FLOAT32, {2, 3}, {7, 8, 9, 10, 11, 12});
//     Tensor expected = create_tensor(TensorDataType::FLOAT32, {4, 3},
//                                     {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12});
//     std::unordered_map<std::string, Node::AttributeValue> attributes;
//     attributes["axis"] = 0;

//     std::vector<Tensor> inputs = {t1, t2};
//     std::vector<Tensor> expected_tensors = {expected};

//     RUN_TEST_CASE(OperatorType::Concat, inputs, expected_tensors, attributes, DeviceType::CPU, OperatorExecuteResult::SUCCESS);
// }

// TEST(OperaotrTestCPU, ConcatOperatorShapeMismatchError)
// {
//     Tensor t1 = create_tensor(TensorDataType::FLOAT32, {2, 2}, {1, 2, 3, 4});
//     Tensor t2 = create_tensor(TensorDataType::FLOAT32, {2, 3}, {5, 6, 7, 8, 9, 10});
//     std::unordered_map<std::string, Node::AttributeValue> attributes;
//     attributes["axis"] = 0;

//     std::vector<Tensor> inputs = {t1, t2};
//     std::vector<Tensor> expected_tensors;

//     RUN_TEST_CASE(OperatorType::Concat, inputs, expected_tensors, attributes, DeviceType::CPU, OperatorExecuteResult::SHAPE_MISMATCH_ERROR);
// }

// // ----------------------- ConstantOperator Tests -----------------------
// TEST(OperaotrTestCPU, ConstantOperatorBasic)
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
// TEST(OperaotrTestCPU, ConvOperatorBasic)
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
// TEST(OperaotrTestCPU, DivOperatorBasic)
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
// TEST(OperaotrTestCPU, GatherOperatorBasic)
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
// TEST(OperaotrTestCPU, MatMulOperatorBasic)
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
// TEST(OperaotrTestCPU, MaxPoolOperatorBasic)
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
// TEST(OperaotrTestCPU, MulOperatorBasic)
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
// TEST(OperaotrTestCPU, SigmoidOperatorBasic)
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
// TEST(OperaotrTestCPU, SliceOperatorBasic)
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
// TEST(OperaotrTestCPU, SoftmaxOperatorBasic)
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
// TEST(OperaotrTestCPU, SplitOperatorBasic)
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

// // ----------------------- SubOperator Tests -----------------------
// TEST(OperaotrTestCPU, SubOperatorBasic)
// {
//     // Basic subtraction test
//     Tensor t1 = create_tensor(TensorDataType::FLOAT32, {2, 2}, {5.0f, 7.0f, 9.0f, 11.0f});
//     Tensor t2 = create_tensor(TensorDataType::FLOAT32, {2, 2}, {1.0f, 2.0f, 3.0f, 4.0f});
//     Tensor expected = create_tensor(TensorDataType::FLOAT32, {2, 2}, {4.0f, 5.0f, 6.0f, 7.0f});

//     std::unordered_map<std::string, Node::AttributeValue> attributes;

//     std::vector<Tensor> inputs = {t1, t2};
//     std::vector<Tensor> expected_tensors = {expected};

//     RUN_TEST_CASE(OperatorType::Sub, inputs, expected_tensors, attributes, DeviceType::CPU, OperatorExecuteResult::SUCCESS);
// }

// // ----------------------- TransposeOperator Tests -----------------------
// TEST(OperaotrTestCPU, TransposeOperatorBasic)
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
// TEST(OperaotrTestCPU, ReshapeOperatorBasic)
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
// TEST(OperaotrTestCPU, ResizeOperatorBasic)
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
// TEST(OperaotrTestCPU, ShapeOperatorBasic)
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