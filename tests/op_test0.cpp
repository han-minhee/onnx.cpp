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

// ----------------------- AddOperator Tests -----------------------
TEST(OperatorTest0, AddOperatorBasic)
{
    // Basic addition test
    Tensor t1 = create_tensor(TensorDataType::FLOAT32, {2, 2}, {1.0f, 2.0f, 3.0f, 4.0f});
    Tensor t2 = create_tensor(TensorDataType::FLOAT32, {2, 2}, {5.0f, 6.0f, 7.0f, 8.0f});
    Tensor expected = create_tensor(TensorDataType::FLOAT32, {2, 2}, {6.0f, 8.0f, 10.0f, 12.0f});

    AddOperator add_op;

    // get the output shapes and data types, initialize the output
    std::vector<std::vector<size_t>> output_shapes = add_op.inferOutputShapes({t1, t2}, {});
    std::vector<TensorDataType> output_data_types = add_op.inferOutputDataTypes({t1, t2}, {});
    
    std::vector<Tensor*> outputs;
    for (size_t i = 0; i < output_shapes.size(); i++)
    {
        outputs.push_back(new Tensor(output_data_types[i], output_shapes[i]));
    }

    run_and_check_operator(add_op, {t1, t2}, outputs, {expected});
}

TEST(OperatorTest0, AddOperatorBroadcastScalar)
{
    // Broadcasting scalar addition
    Tensor t1 = create_tensor(TensorDataType::FLOAT32, {2, 2}, {1.0f, 2.0f, 3.0f, 4.0f});
    Tensor t2 = create_tensor(TensorDataType::FLOAT32, {}, {10.0f}); // Scalar tensor
    Tensor expected = create_tensor(TensorDataType::FLOAT32, {2, 2}, {11.0f, 12.0f, 13.0f, 14.0f});
    Tensor output;

    AddOperator add_op;
    run_and_check_operator(add_op, {t1, t2}, {&output}, {expected});
}

// ----------------------- ConcatOperator Tests -----------------------
TEST(OperatorTest0, ConcatOperatorBasic)
{
    // Basic concatenation along axis 0
    Tensor t1 = create_tensor(TensorDataType::FLOAT32, {2, 3}, {1, 2, 3, 4, 5, 6});
    Tensor t2 = create_tensor(TensorDataType::FLOAT32, {2, 3}, {7, 8, 9, 10, 11, 12});
    Tensor expected = create_tensor(TensorDataType::FLOAT32, {4, 3},
                                    {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12});

    ConcatOperator concat_op;
    std::unordered_map<std::string, Node::AttributeValue> attributes;
    attributes["axis"] = 0;
    Tensor output;

    run_and_check_operator(concat_op, {t1, t2}, {&output}, {expected}, attributes);
}

// ----------------------- ConstantOperator Tests -----------------------
TEST(OperatorTest0, ConstantOperatorBasic)
{
    // Create a constant tensor
    Tensor value_tensor = create_tensor(TensorDataType::FLOAT32, {2, 2}, {1.0f, 2.0f, 3.0f, 4.0f});
    ConstantOperator const_op;
    std::unordered_map<std::string, Node::AttributeValue> attributes;
    attributes["value"] = value_tensor;

    Tensor output;
    run_and_check_operator(const_op, {}, {&output}, {value_tensor}, attributes);
}

// ----------------------- ConvOperator Tests -----------------------
TEST(OperatorTest0, ConvOperatorBasic)
{
    // Simple convolution test
    Tensor X = create_tensor(TensorDataType::FLOAT32, {1, 1, 4, 4},
                             {1, 2, 3, 4,
                              5, 6, 7, 8,
                              9, 10, 11, 12,
                              13, 14, 15, 16});
    Tensor W = create_tensor(TensorDataType::FLOAT32, {1, 1, 3, 3},
                             {1, 0, -1,
                              1, 0, -1,
                              1, 0, -1});
    Tensor B = create_tensor(TensorDataType::FLOAT32, {1}, {0});
    Tensor expected = create_tensor(TensorDataType::FLOAT32, {1, 1, 2, 2}, {-6, -6, -6, -6});

    ConvOperator conv_op;
    std::unordered_map<std::string, Node::AttributeValue> attributes;
    attributes["dilations"] = std::vector<int64_t>{1, 1};
    attributes["group"] = 1;
    attributes["kernel_shape"] = std::vector<int64_t>{3, 3};
    attributes["pads"] = std::vector<int64_t>{0, 0, 0, 0};
    attributes["strides"] = std::vector<int64_t>{1, 1};

    Tensor output;

    run_and_check_operator(conv_op, {X, W, B}, {&output}, {expected}, attributes);
}

// ----------------------- DivOperator Tests -----------------------
TEST(OperatorTest0, DivOperatorBasic)
{
    // Basic division test
    Tensor t1 = create_tensor(TensorDataType::FLOAT32, {2, 2}, {10.0f, 20.0f, 30.0f, 40.0f});
    Tensor t2 = create_tensor(TensorDataType::FLOAT32, {2, 2}, {2.0f, 4.0f, 5.0f, 8.0f});
    Tensor expected = create_tensor(TensorDataType::FLOAT32, {2, 2}, {5.0f, 5.0f, 6.0f, 5.0f});
    Tensor output;

    DivOperator div_op;
    run_and_check_operator(div_op, {t1, t2}, {&output}, {expected});
}

// ----------------------- GatherOperator Tests -----------------------
TEST(OperatorTest0, GatherOperatorBasic)
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

    GatherOperator gather_op;
    std::unordered_map<std::string, Node::AttributeValue> attributes;
    attributes["axis"] = 0;
    Tensor output;

    run_and_check_operator(gather_op, {data, indices}, {&output}, {expected}, attributes);
}

// ----------------------- MatMulOperator Tests -----------------------
TEST(OperatorTest0, MatMulOperatorBasic)
{
    // Basic matrix multiplication
    Tensor A = create_tensor(TensorDataType::FLOAT32, {2, 3},
                             {1, 2, 3,
                              4, 5, 6});
    Tensor B = create_tensor(TensorDataType::FLOAT32, {3, 2},
                             {7, 8,
                              9, 10,
                              11, 12});
    Tensor expected = create_tensor(TensorDataType::FLOAT32, {2, 2},
                                    {58, 64,
                                     139, 154});

    MatMulOperator matmul_op;
    Tensor output;

    run_and_check_operator(matmul_op, {A, B}, {&output}, {expected});
}

// ----------------------- MaxPoolOperator Tests -----------------------
TEST(OperatorTest0, MaxPoolOperatorBasic)
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

// ----------------------- MulOperator Tests -----------------------
TEST(OperatorTest0, MulOperatorBasic)
{
    // Basic multiplication test
    Tensor t1 = create_tensor(TensorDataType::FLOAT32, {2, 2}, {1.0f, 2.0f, 3.0f, 4.0f});
    Tensor t2 = create_tensor(TensorDataType::FLOAT32, {2, 2}, {5.0f, 6.0f, 7.0f, 8.0f});
    Tensor expected = create_tensor(TensorDataType::FLOAT32, {2, 2}, {5.0f, 12.0f, 21.0f, 32.0f});
    Tensor output;

    MulOperator mul_op;
    run_and_check_operator(mul_op, {t1, t2}, {&output}, {expected});
}

// ----------------------- ReshapeOperator Tests -----------------------
TEST(OperatorTest0, ReshapeOperatorBasic)
{
    // Basic reshape test
    Tensor data = create_tensor(TensorDataType::FLOAT32, {2, 3}, {1, 2, 3, 4, 5, 6});
    Tensor shape = create_tensor(TensorDataType::INT64, {3}, {3, 2, 1});
    Tensor expected = create_tensor(TensorDataType::FLOAT32, {3, 2, 1}, {1, 2, 3, 4, 5, 6});
    Tensor output;

    ReshapeOperator reshape_op;
    run_and_check_operator(reshape_op, {data, shape}, {&output}, {expected});
}

// ----------------------- ResizeOperator Tests -----------------------
TEST(OperatorTest0, ResizeOperatorBasic)
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

// ----------------------- ShapeOperator Tests -----------------------
TEST(OperatorTest0, ShapeOperatorBasic)
{
    // Basic shape operator test
    Tensor data = create_tensor(TensorDataType::FLOAT32, {2, 3, 4}, std::vector<float>(24, 1.0f));
    Tensor expected = create_tensor(TensorDataType::INT64, {3}, {2, 3, 4});
    Tensor output;

    ShapeOperator shape_op;
    run_and_check_operator(shape_op, {data}, {&output}, {expected});
}

// ----------------------- SigmoidOperator Tests -----------------------
TEST(OperatorTest0, SigmoidOperatorBasic)
{
    // Basic sigmoid test
    Tensor data = create_tensor(TensorDataType::FLOAT32, {2, 2}, {0.0f, -1.0f, 1.0f, 2.0f});
    Tensor expected = create_tensor(TensorDataType::FLOAT32, {2, 2},
                                    {0.5f, 0.26894f, 0.73106f, 0.880797f});
    Tensor output;

    SigmoidOperator sigmoid_op;
    run_and_check_operator(sigmoid_op, {data}, {&output}, {expected});
}

// ----------------------- SliceOperator Tests -----------------------
TEST(OperatorTest0, SliceOperatorBasic)
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

// ----------------------- SoftmaxOperator Tests -----------------------
TEST(OperatorTest0, SoftmaxOperatorBasic)
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

// ----------------------- SplitOperator Tests -----------------------
TEST(OperatorTest0, SplitOperatorBasic)
{
    // Basic split test
    Tensor data = create_tensor(TensorDataType::FLOAT32, {2, 4},
                                {1, 2, 3, 4,
                                 5, 6, 7, 8});
    Tensor split = create_tensor(TensorDataType::INT64, {2}, {2, 2});
    Tensor expected1 = create_tensor(TensorDataType::FLOAT32, {2, 2},
                                     {1, 2,
                                      5, 6});
    Tensor expected2 = create_tensor(TensorDataType::FLOAT32, {2, 2},
                                     {3, 4,
                                      7, 8});
    Tensor output1, output2;

    SplitOperator split_op;
    std::unordered_map<std::string, Node::AttributeValue> attributes;
    attributes["axis"] = 1;

    run_and_check_operator(split_op, {data, split}, {&output1, &output2}, {expected1, expected2}, attributes);
}

// ----------------------- SubOperator Tests -----------------------
TEST(OperatorTest0, SubOperatorBasic)
{
    // Basic subtraction test
    Tensor t1 = create_tensor(TensorDataType::FLOAT32, {2, 2}, {5.0f, 7.0f, 9.0f, 11.0f});
    Tensor t2 = create_tensor(TensorDataType::FLOAT32, {2, 2}, {1.0f, 2.0f, 3.0f, 4.0f});
    Tensor expected = create_tensor(TensorDataType::FLOAT32, {2, 2}, {4.0f, 5.0f, 6.0f, 7.0f});
    Tensor output;

    SubOperator sub_op;
    run_and_check_operator(sub_op, {t1, t2}, {&output}, {expected});
}

// ----------------------- TransposeOperator Tests -----------------------
TEST(OperatorTest0, TransposeOperatorBasic)
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
