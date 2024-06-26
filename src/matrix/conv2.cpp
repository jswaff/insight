#include <ATen/ATen.h>
#include <torch/torch.h>

#include "matrix.h"

at::Tensor conv2(at::Tensor &in, at::Tensor &kernel, ConvType conv_type, bool do_conv) {
    int kh = kernel.sizes()[2]; // out channels, in channels, kernel height, kernel width
    int kw = kernel.sizes()[3];
    
    int padh = conv_type==SAME ? (kh-1)/2 : 0;
    int padw = conv_type==SAME ? (kw-1)/2 : 0;

    // Pytorch does correlation by default.  For convolution, rotate the kernel
    if (do_conv) kernel = kernel.flip(2).flip(3);

    return torch::nn::functional::conv2d(in, kernel, 
        torch::nn::functional::Conv2dFuncOptions().padding({padh, padw}));
}
