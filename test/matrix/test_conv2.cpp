#include <ATen/ATen.h>
#include <torch/torch.h>
#include <gtest/gtest.h>

#include "../../src/matrix/matrix.h"

static bool float_array_equals(float* arr1, float* arr2, int len) {
	for (int i=0;i<len;i++) {
		float v1 = arr1[i];
		float v2 = arr2[i];
		if (std::abs(v1 - v2) > 0.0001) return false;
	}
	return true;
}

TEST(matrix_test, conv2)
{
    float input_arr[16] = { 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15 };
    auto conv_input = torch::from_blob(input_arr, { 1,1,4,4 }); // batch size, in channels, height, width

    auto kernel = at::ones({1,1,3,3}); // out channels, in channels, kernel height, kernel width

    auto conv = conv2(conv_input, kernel, SAME, true);

	float expected[16] = { 10, 18, 24, 18,
	                       27, 45, 54, 39,
                           51, 81, 90, 63,
                           42, 66, 72, 50 };

    ASSERT_EQ(conv.sizes()[2], 4);
    ASSERT_EQ(conv.sizes()[3], 4);
    ASSERT_TRUE(float_array_equals(conv.data_ptr<float>(), expected, 16));
}

TEST(matrix_test, conv2_2)
{
    float input_arr[16] = { 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15 };
    auto conv_input = torch::from_blob(input_arr, { 1,1,4,4 });

    float kernel_arr[9] = { 2,1,2,2,1,2,2,1,2 };
    auto kernel = torch::from_blob(kernel_arr, {1,1,3,3});

    auto conv = conv2(conv_input, kernel, SAME, true);

	float expected[16] = { 16, 30, 40, 26,
                           42, 75, 90, 57,
                           78,135,150, 93,
	                       64,110,120, 74 };

    ASSERT_EQ(conv.sizes()[2], 4);
    ASSERT_EQ(conv.sizes()[3], 4);
    ASSERT_TRUE(float_array_equals(conv.data_ptr<float>(), expected, 16));
}

TEST(matrix_test, conv2_3)
{
    float input_arr[16] = { 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15 };
    auto conv_input = torch::from_blob(input_arr, { 1,1,4,4 });

    float kernel_arr[9] = { 2, 1, 6,
                            2, 5, 1,
                            0, 4, 3 };
    auto kernel = torch::from_blob(kernel_arr, {1,1,3,3});

    auto conv = conv2(conv_input, kernel, SAME, true);

	float expected[16] = { 16, 50, 67, 60,
                           56,122,146,130,
                          112,218,242,210,
                          118,165,180,163 };

    ASSERT_EQ(conv.sizes()[2], 4);
    ASSERT_EQ(conv.sizes()[3], 4);
    ASSERT_TRUE(float_array_equals(conv.data_ptr<float>(), expected, 16));
}

TEST(matrix_test, conv2_4_correlation)
{
    float input_arr[16] = { 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15 };
    auto conv_input = torch::from_blob(input_arr, { 1,1,4,4 });

    float kernel_arr[9] = { 2, 1, 6,
                            2, 5, 1,
                            0, 4, 3 };
    auto kernel = torch::from_blob(kernel_arr, {1,1,3,3});

    auto conv = conv2(conv_input, kernel, SAME, false);

	float expected[16] = {  32,    45,    60,    47,
                            90,   118,   142,    98,
	                       170,   214,   238,   154,
	                       135,   188,   205,   134 };

    ASSERT_EQ(conv.sizes()[2], 4);
    ASSERT_EQ(conv.sizes()[3], 4);
    ASSERT_TRUE(float_array_equals(conv.data_ptr<float>(), expected, 16));
}

TEST(matrix_test, conv2_valid_1)
{
    float input_arr[16] = { 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15 };
    auto conv_input = torch::from_blob(input_arr, { 1,1,4,4 });

    float kernel_arr[9] = { 2, 1, 6,
                            2, 5, 1,
                            0, 4, 3 };
    auto kernel = torch::from_blob(kernel_arr, {1,1,3,3});

    auto conv = conv2(conv_input, kernel, VALID, true);

	float expected[4] = { 122, 146, 
                          218, 242 };

    ASSERT_EQ(conv.sizes()[2], 2);
    ASSERT_EQ(conv.sizes()[3], 2);
    ASSERT_TRUE(float_array_equals(conv.data_ptr<float>(), expected, 4));
}

TEST(matrix_test, conv2_valid_2) 
{
    float input_arr[16] = { 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15 };
    auto conv_input = torch::from_blob(input_arr, { 1,1,4,4 });

    float kernel_arr[6] = { 2, 1,
                            2, 5,
                            0, 4 };
    auto kernel = torch::from_blob(kernel_arr, {1,1,3,2});

    auto conv = conv2(conv_input, kernel, VALID, true);

	float expected[6] = {  56,  70,  84,
                          112, 126, 140 };

    ASSERT_EQ(conv.sizes()[2], 2);
    ASSERT_EQ(conv.sizes()[3], 3);
    ASSERT_TRUE(float_array_equals(conv.data_ptr<float>(), expected, 6));
}

TEST(matrix_test, conv2_valid_3)
{
    float input_arr[16] = { 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15 };
    auto conv_input = torch::from_blob(input_arr, { 1,1,4,4 });

    float kernel_arr[6] = { 2, 1,
                            2, 5,
                            0, 4 };
    auto kernel = torch::from_blob(kernel_arr, {1,1,3,2});

    auto conv = conv2(conv_input, kernel, VALID, false);

	float expected[6] = {  70,  84,  98,
                          126, 140, 154 };

    ASSERT_EQ(conv.sizes()[2], 2);
    ASSERT_EQ(conv.sizes()[3], 3);
    ASSERT_TRUE(float_array_equals(conv.data_ptr<float>(), expected, 6));
}
