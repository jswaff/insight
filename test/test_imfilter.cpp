#include <ATen/ATen.h>
#include <torch/torch.h>
#include <gtest/gtest.h>

#include "../src/matrix/matrix.h"

static bool float_array_equals(float* arr1, float* arr2, int len) {
	for (int i=0;i<len;i++) {
		float v1 = arr1[i];
		float v2 = arr2[i];
		if (std::abs(v1 - v2) > 0.0001) return false;
	}
	return true;
}

// 2D image with 2D kernel
TEST(matrix_test, imfilter_1)
{
	float input_arr[25] = {
            17,    24,     1,     8,    15,
            23,     5,     7,    14,    16,
             4,     6,    13,    20,    22,
            10,    12,    19,    21,     3,
            11,    18,    25,     2,     9
    };
    auto input = torch::from_blob(input_arr, { 1,1,5,5 }); // batch size, in channels, iH, iW

	float h_arr[3] = { -1.0, 0, 1.0 };
    auto h = torch::from_blob(h_arr, { 1,1,1,3 });

    auto mat = imfilter(input, h);

    float expected[25] = {
            24,   -16,   -16,    14,    -8,
             5,   -16,     9,     9,   -14,
             6,     9,    14,     9,   -20,
            12,     9,     9,   -16,   -21,
            18,    14,   -16,   -16,    -2        
    };

    ASSERT_EQ(mat.sizes()[2], 5);
    ASSERT_EQ(mat.sizes()[3], 5);
    ASSERT_TRUE(float_array_equals(mat.data_ptr<float>(), expected, 25));
}


// 3D image with 3D kernel
// only producing 1x1x1 with value 36.  suspect the issue is the same as with
// conv3d when using an even sized kernel
TEST(matrix_test, DISABLED_imfilter_2)
{
	float input_arr[8] = { 1, 2, 3, 4, 5, 6, 7, 8 };
    auto input = torch::from_blob(input_arr, { 1,1,2,2,2 });

    auto h = at::ones({1,1,2,2,2});

    auto mat = imfilter(input, h);
    std::cout << "mat: " << mat << "\n";
    std::cout << "sizes: " << mat.sizes() << "\n";

    float expected[8] = {
            36,    20,
            22,    12,

            26,    14,
            15,     8        
    };

    ASSERT_EQ(mat.sizes()[2], 2);
    ASSERT_EQ(mat.sizes()[3], 2);
    ASSERT_EQ(mat.sizes()[4], 2);
    ASSERT_TRUE(float_array_equals(mat.data_ptr<float>(), expected, 8));
}

// only producing 1x1x1 with value 160.  suspect the issue is the same as with
// conv3d when using an even sized kernel
TEST(matrix_test, DISABLED_imfilter_3)
{
	float input_arr[8] = { 1, 2, 3, 4, 5, 6, 7, 8 };
    auto input = torch::from_blob(input_arr, { 1,1,2,2,2 });

	float h_arr[8] = { 4, 1, 4, 0, 7, 2, 9, 4 };
    auto h = torch::from_blob(h_arr, { 1,1,2,2,2 });

    auto mat = imfilter(input, h);
    std::cout << "mat: " << mat << "\n";
    std::cout << "sizes: " << mat.sizes() << "\n";

    float expected[8] = {
            160,   138,
             81,    72,

             54,    56,
             36,    32,   
    };

    ASSERT_EQ(mat.sizes()[2], 2);
    ASSERT_EQ(mat.sizes()[3], 2);
    ASSERT_EQ(mat.sizes()[4], 2);
    ASSERT_TRUE(float_array_equals(mat.data_ptr<float>(), expected, 8));
}
