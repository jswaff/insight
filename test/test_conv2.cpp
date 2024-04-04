#include <ATen/ATen.h>
#include <torch/torch.h>
#include <gtest/gtest.h>

#include "../src/matrix.h"

bool float_array_equals(float* arr1, float* arr2, int len) {
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

    auto conv = conv2(conv_input, kernel);

	float expected[16] = { 10, 18, 24, 18,
	                       27, 45, 54, 39,
						   51, 81, 90, 63,
						   42, 66, 72, 50 };

    ASSERT_TRUE(float_array_equals(conv.data_ptr<float>(), expected, 16));
}

