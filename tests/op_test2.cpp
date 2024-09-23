#include <gtest/gtest.h>
#include "tensor/tensor.hpp"
#include "operator/operators.hpp"

void PrintTo(OperatorExecuteResult result, std::ostream *os)
{
    *os << OperatorUtils::OperatorExecuteResultToString(result);
}

void run_and_check_operator(Operator &op,
                            const std::vector<Tensor> &inputs,
                            std::vector<Tensor *> outputs,
                            const std::vector<Tensor> &expected,
                            std::unordered_map<std::string, Node::AttributeValue> attributes = {},
                            OperatorExecuteResult expected_execute_result = OperatorExecuteResult::SUCCESS)
{
    OperatorExecuteResult result_code = op.execute(inputs, outputs, attributes);
    ASSERT_EQ(result_code, expected_execute_result);

    if (result_code != OperatorExecuteResult::SUCCESS)
        return;

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

// -------------------- SliceOperator Tests --------------------
TEST(OperatorTest2, SliceOperatorBasic)
{
    // Basic slice test
    Tensor data = create_tensor(TensorDataType::FLOAT32, {3, 4},
                                {0, 1, 2, 3,
                                 4, 5, 6, 7,
                                 8, 9, 10, 11});
    Tensor starts = create_tensor(TensorDataType::INT64, {1}, {1});
    Tensor ends = create_tensor(TensorDataType::INT64, {1}, {3});
    Tensor axes = create_tensor(TensorDataType::INT64, {1}, {0});
    Tensor expected = create_tensor(TensorDataType::FLOAT32, {2, 4},
                                    {4, 5, 6, 7,
                                     8, 9, 10, 11});
    Tensor output;

    SliceOperator slice_op;
    run_and_check_operator(slice_op, {data, starts, ends, axes}, {&output}, {expected});
}

TEST(OperatorTest2, SliceOperatorShapeMismatchError)
{
    // Error case: shape mismatch in the start and end tensors
    Tensor data = create_tensor(TensorDataType::FLOAT32, {2, 4}, {1, 2, 3, 4, 5, 6, 7, 8});
    Tensor starts = create_tensor(TensorDataType::INT64, {2}, {0, 0});
    Tensor ends = create_tensor(TensorDataType::INT64, {1}, {2});
    Tensor axes = create_tensor(TensorDataType::INT64, {1}, {0});
    Tensor output;

    SliceOperator slice_op;
    run_and_check_operator(slice_op, {data, starts, ends, axes}, {&output}, {}, {}, OperatorExecuteResult::SHAPE_MISMATCH_ERROR);
}

// -------------------- GatherOperator Tests --------------------
TEST(OperatorTest2, GatherOperatorBasic)
{
    // Basic gather test
    Tensor data = create_tensor(TensorDataType::FLOAT32, {3, 4},
                                {0, 1, 2, 3,
                                 4, 5, 6, 7,
                                 8, 9, 10, 11});
    Tensor indices = create_tensor(TensorDataType::INT64, {2}, {0, 2});
    Tensor expected = create_tensor(TensorDataType::FLOAT32, {2, 4},
                                    {0, 1, 2, 3,
                                     8, 9, 10, 11});
    Tensor output;

    GatherOperator gather_op;
    std::unordered_map<std::string, Node::AttributeValue> attributes;
    attributes["axis"] = 0;

    run_and_check_operator(gather_op, {data, indices}, {&output}, {expected}, attributes);
}

// -------------------- ShapeOperator Tests --------------------
TEST(OperatorTest2, ShapeOperatorBasic)
{
    // Basic shape operator test
    Tensor data = create_tensor(TensorDataType::FLOAT32, {2, 3, 4}, std::vector<float>(24, 1.0f));
    Tensor expected = create_tensor(TensorDataType::INT64, {3}, {2, 3, 4});
    Tensor output;

    ShapeOperator shape_op;
    run_and_check_operator(shape_op, {data}, {&output}, {expected});
}

// -------------------- ReshapeOperator Tests --------------------
TEST(OperatorTest2, ReshapeOperatorBasic)
{
    // Basic reshape test
    Tensor data = create_tensor(TensorDataType::FLOAT32, {2, 3}, {1, 2, 3, 4, 5, 6});
    Tensor shape = create_tensor(TensorDataType::INT64, {3}, {3, 2, 1});
    Tensor expected = create_tensor(TensorDataType::FLOAT32, {3, 2, 1}, {1, 2, 3, 4, 5, 6});
    Tensor output;

    ReshapeOperator reshape_op;
    run_and_check_operator(reshape_op, {data, shape}, {&output}, {expected});
}

// -------------------- SoftmaxOperator Tests --------------------
TEST(OperatorTest2, SoftmaxOperatorBasic)
{
    // Basic softmax test
    Tensor data = create_tensor(TensorDataType::FLOAT32, {2, 2}, {1.0f, 2.0f, 3.0f, 4.0f});
    Tensor expected = create_tensor(TensorDataType::FLOAT32, {2, 2},
                                    {0.26894f, 0.73106f, 0.26894f, 0.73106f});
    Tensor output;

    SoftmaxOperator softmax_op;
    std::unordered_map<std::string, Node::AttributeValue> attributes;
    attributes["axis"] = 1;

    run_and_check_operator(softmax_op, {data}, {&output}, {expected}, attributes);
}

// -------------------- TransposeOperator Tests --------------------
TEST(OperatorTest2, TransposeOperatorBasic)
{
    // Basic transpose test
    Tensor data = create_tensor(TensorDataType::FLOAT32, {2, 3}, {1, 2, 3, 4, 5, 6});
    Tensor expected = create_tensor(TensorDataType::FLOAT32, {3, 2}, {1, 4, 2, 5, 3, 6});
    Tensor output;

    TransposeOperator transpose_op;
    std::unordered_map<std::string, Node::AttributeValue> attributes;
    attributes["perm"] = std::vector<int64_t>{1, 0};

    run_and_check_operator(transpose_op, {data}, {&output}, {expected}, attributes);
}

// -------------------- ResizeOperator Tests --------------------
TEST(OperatorTest2, ResizeOperatorBasic)
{
    // Basic resize test (nearest neighbor)
    Tensor data = create_tensor(TensorDataType::FLOAT32, {1, 1, 2, 2}, {1, 2, 3, 4});
    Tensor scales = create_tensor(TensorDataType::FLOAT32, {4}, {1.0f, 1.0f, 2.0f, 2.0f});
    Tensor expected = create_tensor(TensorDataType::FLOAT32, {1, 1, 4, 4},
                                    {1, 1, 2, 2,
                                     1, 1, 2, 2,
                                     3, 3, 4, 4,
                                     3, 3, 4, 4});
    Tensor output;

    ResizeOperator resize_op;
    std::unordered_map<std::string, Node::AttributeValue> attributes;
    attributes["mode"] = std::string("nearest");

    run_and_check_operator(resize_op, {data, Tensor(), scales}, {&output}, {expected}, attributes);
}

// -------------------- MaxPoolOperator Tests --------------------
TEST(OperatorTest2, MaxPoolOperatorBasic)
{
    // Basic MaxPool test
    Tensor X = create_tensor(TensorDataType::FLOAT32, {1, 1, 4, 4},
                             {1, 2, 3, 4,
                              5, 6, 7, 8,
                              9, 10, 11, 12,
                              13, 14, 15, 16});
    Tensor expected = create_tensor(TensorDataType::FLOAT32, {1, 1, 2, 2},
                                    {6, 8,
                                     14, 16});

    MaxPoolOperator maxpool_op;
    std::unordered_map<std::string, Node::AttributeValue> attributes;
    attributes["kernel_shape"] = std::vector<int64_t>{2, 2};
    attributes["pads"] = std::vector<int64_t>{0, 0, 0, 0};
    attributes["strides"] = std::vector<int64_t>{2, 2};

    Tensor output;

    run_and_check_operator(maxpool_op, {X}, {&output}, {expected}, attributes);
}
