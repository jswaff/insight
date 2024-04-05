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

TEST(matrix_test, conv3_same_1)
{
    float input_arr[64] = { 1, 2, 3, 4,
                            5, 6, 7, 8, 
                            9,10,11,12,
                            13,14,15,16,
                            // page 2
                            17,18,19,20,
                            21,22,23,24,
                            25,26,27,28,
                            29,30,31,32,
                            // page 3
                            33,34,35,36,
                            37,38,39,40,
                            41,42,43,44,
                            45,46,47,48,
                            // page 4
                            49,50,51,52,
                            53,54,55,56,
                            57,58,59,60,
                            61,62,63,64 };
    auto conv_input = torch::from_blob(input_arr, { 1,1,4,4,4 }); // batch size, in channels, iT, iH, iW

    auto kernel = at::ones({1,1,3,3,3}); // out channels, in channels, kT, kH, kW

    auto conv = conv3(conv_input, kernel, SAME, true);

    float expected[64] = {
                            92,   144,   156,   108,
                           162,   252,   270,   186,
                           210,   324,   342,   234,
                           156,   240,   252,   172,

                           234,   360,   378,   258,
                           387,   594,   621,   423,
                           459,   702,   729,   495,
                           330,   504,   522,   354,

                           426,   648,   666,   450,
                           675,  1026,  1053,   711,
                           747,  1134,  1161,   783,
                           522,   792,   810,   546,

                           348,   528,   540,   364,
                           546,   828,   846,   570,
                           594,   900,   918,   618,
                           412,   624,   636,   428
    };

    ASSERT_EQ(conv.sizes()[2], 4);
    ASSERT_EQ(conv.sizes()[3], 4);
    ASSERT_EQ(conv.sizes()[4], 4);
    ASSERT_TRUE(float_array_equals(conv.data_ptr<float>(), expected, 64));
}

TEST(matrix_test, conv3_same_2)
{
    float input_arr[64] = { 1, 2, 3, 4,
                            5, 6, 7, 8, 
                            9,10,11,12,
                            13,14,15,16,
                            // page 2
                            17,18,19,20,
                            21,22,23,24,
                            25,26,27,28,
                            29,30,31,32,
                            // page 3
                            33,34,35,36,
                            37,38,39,40,
                            41,42,43,44,
                            45,46,47,48,
                            // page 4
                            49,50,51,52,
                            53,54,55,56,
                            57,58,59,60,
                            61,62,63,64 };
    auto conv_input = torch::from_blob(input_arr, { 1,1,4,4,4 });

	float kernel_arr[27] = {  1,  2,  3,  4,  5,  6,  7,  8,  9,
                             11, 12, 13, 14, 15, 16, 17, 18, 19,
                             21, 22, 23, 24, 25, 26, 27, 28, 29 };

    auto kernel = torch::from_blob(kernel_arr, { 1,1,3,3,3 });

    auto conv = conv3(conv_input, kernel, SAME, true);

    float expected[64] = {
         390,         700,         802,         626,
         960,        1644,        1824,        1374,
        1416,        2364,        2544,        1878,
        1370,        2236,        2374,        1718,

        1723,        2874,        3117,        2293,
        3543,        5796,        6201,        4488,
        4587,        7416,        7821,        5604,
        3961,        6330,        6627,        4699,

        4219,        6762,        7005,        4981,
        7719,       12276,       12681,        8952,
        8763,       13896,       14301,       10068,
        7033,       11082,       11379,        7963,

        5918,        9244,        9466,        6570,
       10068,       15684,       16044,       11106,
       11004,       17124,       17484,       12090,
        8306,       12892,       13150,        9070 
    };

    ASSERT_EQ(conv.sizes()[2], 4);
    ASSERT_EQ(conv.sizes()[3], 4);
    ASSERT_EQ(conv.sizes()[4], 4);
    ASSERT_TRUE(float_array_equals(conv.data_ptr<float>(), expected, 64));
}

// use correlation
TEST(matrix_test, conv3_same_3)
{
    float input_arr[64] = { 1, 2, 3, 4,
                            5, 6, 7, 8, 
                            9,10,11,12,
                            13,14,15,16,
                            // page 2
                            17,18,19,20,
                            21,22,23,24,
                            25,26,27,28,
                            29,30,31,32,
                            // page 3
                            33,34,35,36,
                            37,38,39,40,
                            41,42,43,44,
                            45,46,47,48,
                            // page 4
                            49,50,51,52,
                            53,54,55,56,
                            57,58,59,60,
                            61,62,63,64 };
    auto conv_input = torch::from_blob(input_arr, { 1,1,4,4,4 });

	float kernel_arr[27] = {  1,  2,  3,  4,  5,  6,  7,  8,  9,
                             11, 12, 13, 14, 15, 16, 17, 18, 19,
                             21, 22, 23, 24, 25, 26, 27, 28, 29 };

    auto kernel = torch::from_blob(kernel_arr, { 1,1,3,3,3 });

    auto conv = conv3(conv_input, kernel, SAME, false);

    float expected[64] = {
        2370,        3620,        3878,        2614,
        3900,        5916,        6276,        4206,
        4884,        7356,        7716,        5142,
        3310,        4964,        5186,        3442,

        5297,        7926,        8223,        5447,
        8067,       12024,       12429,        8202,
        9183,       13644,       14049,        9246,
        5939,        8790,        9033,        5921,

        8561,       12678,       12975,        8519,
       12531,       18504,       18909,       12378,
       13647,       20124,       20529,       13422,
        8627,       12678,       12921,        8417,

        4522,        6596,        6734,        4350,
        6312,        9156,        9336,        5994,
        6816,        9876,       10056,        6450,
        4054,        5828,        5930,        3770    
    };

    ASSERT_EQ(conv.sizes()[2], 4);
    ASSERT_EQ(conv.sizes()[3], 4);
    ASSERT_EQ(conv.sizes()[4], 4);
    ASSERT_TRUE(float_array_equals(conv.data_ptr<float>(), expected, 64));
}

// 3D matrix with 2D kernel
// TODO: the resulting matrix isn't right, due to padding.  it could be
// that it would match with an odd sized kernel.
TEST(matrix_test, DISABLED_conv3_same_4)
{
    float input_arr[64] = { 1, 2, 3, 4,
                            5, 6, 7, 8, 
                            9,10,11,12,
                            13,14,15,16,
                            // page 2
                            17,18,19,20,
                            21,22,23,24,
                            25,26,27,28,
                            29,30,31,32,
                            // page 3
                            33,34,35,36,
                            37,38,39,40,
                            41,42,43,44,
                            45,46,47,48,
                            // page 4
                            49,50,51,52,
                            53,54,55,56,
                            57,58,59,60,
                            61,62,63,64 };
    auto conv_input = torch::from_blob(input_arr, { 1,1,4,4,4 });

	float kernel_arr[4] = {  1,  2,  3,  4 };

    auto kernel = torch::from_blob(kernel_arr, { 1,1,1,2,2 });

    auto conv = conv3(conv_input, kernel, SAME, true);
    std::cout << "conv: " << conv << "\n";
    std::cout << "sizes: " << conv.sizes() << "\n";

    float expected[64] = {
             26,    36,    46,    32,
             66,    76,    86,    56,
            106,   116,   126,    80,
             94,   101,   108,    64,

            186,   196,   206,   128,
            226,   236,   246,   152,
            266,   276,   286,   176,
            206,   213,   220,   128,

            346,   356,   366,   224,
            386,   396,   406,   248,
            426,   436,   446,   272,
            318,   325,   332,   192,

            506,   516,   526,   320,
            546,   556,   566,   344,
            586,   596,   606,   368,
            430,   437,   444,   256    };

    ASSERT_EQ(conv.sizes()[2], 4);
    ASSERT_EQ(conv.sizes()[3], 4);
    ASSERT_EQ(conv.sizes()[4], 4);
    ASSERT_TRUE(float_array_equals(conv.data_ptr<float>(), expected, 64));
}

TEST(matrix_test, conv3_valid_1)
{
    float input_arr[64] = { 1, 2, 3, 4,
                            5, 6, 7, 8, 
                            9,10,11,12,
                            13,14,15,16,
                            // page 2
                            17,18,19,20,
                            21,22,23,24,
                            25,26,27,28,
                            29,30,31,32,
                            // page 3
                            33,34,35,36,
                            37,38,39,40,
                            41,42,43,44,
                            45,46,47,48,
                            // page 4
                            49,50,51,52,
                            53,54,55,56,
                            57,58,59,60,
                            61,62,63,64 };
    auto conv_input = torch::from_blob(input_arr, { 1,1,4,4,4 });

	float kernel_arr[27] = {  1,  2,  3,  4,  5,  6,  7,  8,  9,
                             11, 12, 13, 14, 15, 16, 17, 18, 19,
                             21, 22, 23, 24, 25, 26, 27, 28, 29 };

    auto kernel = torch::from_blob(kernel_arr, { 1,1,3,3,3 });

    auto conv = conv3(conv_input, kernel, VALID, true);

    float expected[8] = {
        5796,        6201,
        7416,        7821,

       12276,       12681,
       13896,       14301   };

    ASSERT_EQ(conv.sizes()[2], 2);
    ASSERT_EQ(conv.sizes()[3], 2);
    ASSERT_EQ(conv.sizes()[4], 2);
    ASSERT_TRUE(float_array_equals(conv.data_ptr<float>(), expected, 8));
}

// with correlation
TEST(matrix_test, conv3_valid_2)
{
    float input_arr[64] = { 1, 2, 3, 4,
                            5, 6, 7, 8, 
                            9,10,11,12,
                            13,14,15,16,
                            // page 2
                            17,18,19,20,
                            21,22,23,24,
                            25,26,27,28,
                            29,30,31,32,
                            // page 3
                            33,34,35,36,
                            37,38,39,40,
                            41,42,43,44,
                            45,46,47,48,
                            // page 4
                            49,50,51,52,
                            53,54,55,56,
                            57,58,59,60,
                            61,62,63,64 };
    auto conv_input = torch::from_blob(input_arr, { 1,1,4,4,4 });

	float kernel_arr[27] = {  1,  2,  3,  4,  5,  6,  7,  8,  9,
                             11, 12, 13, 14, 15, 16, 17, 18, 19,
                             21, 22, 23, 24, 25, 26, 27, 28, 29 };

    auto kernel = torch::from_blob(kernel_arr, { 1,1,3,3,3 });

    auto conv = conv3(conv_input, kernel, VALID, false);

    float expected[8] = {
       12024,       12429,
       13644,       14049,

       18504,       18909,
       20124,       20529
    };

    ASSERT_EQ(conv.sizes()[2], 2);
    ASSERT_EQ(conv.sizes()[3], 2);
    ASSERT_EQ(conv.sizes()[4], 2);
    ASSERT_TRUE(float_array_equals(conv.data_ptr<float>(), expected, 8));
}

// 3D matrix with 2D kernel
TEST(matrix_test, conv3_valid_3)
{
    float input_arr[64] = { 1, 2, 3, 4,
                            5, 6, 7, 8, 
                            9,10,11,12,
                            13,14,15,16,
                            // page 2
                            17,18,19,20,
                            21,22,23,24,
                            25,26,27,28,
                            29,30,31,32,
                            // page 3
                            33,34,35,36,
                            37,38,39,40,
                            41,42,43,44,
                            45,46,47,48,
                            // page 4
                            49,50,51,52,
                            53,54,55,56,
                            57,58,59,60,
                            61,62,63,64 };
    auto conv_input = torch::from_blob(input_arr, { 1,1,4,4,4 });

	float kernel_arr[4] = {  1,  2,  3,  4 };

    auto kernel = torch::from_blob(kernel_arr, { 1,1,1,2,2 });

    auto conv = conv3(conv_input, kernel, SAME, true);

    float expected[36] = {
             26,    36,    46,
             66,    76,    86,
            106,   116,   126,

            186,   196,   206,
            226,   236,   246,
            266,   276,   286,

            346,   356,   366,
            386,   396,   406,
            426,   436,   446,

            506,   516,   526,
            546,   556,   566,
            586,   596,   606
    };

    ASSERT_EQ(conv.sizes()[2], 4);
    ASSERT_EQ(conv.sizes()[3], 3);
    ASSERT_EQ(conv.sizes()[4], 3);
    ASSERT_TRUE(float_array_equals(conv.data_ptr<float>(), expected, 36));
}
