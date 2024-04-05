#include <ATen/ATen.h>
#include <torch/torch.h>

#include "matrix.h"

at::Tensor conv3(at::Tensor &in, at::Tensor &kernel, ConvType conv_type, bool do_conv) {
    int kt = kernel.sizes()[2]; // out channels, in channels, kT, kH, kW
    int kh = kernel.sizes()[3];
    int kw = kernel.sizes()[4];
    
    int padt = conv_type==SAME ? (kt-1)/2 : 0;
    int padh = conv_type==SAME ? (kh-1)/2 : 0;
    int padw = conv_type==SAME ? (kw-1)/2 : 0;
 
    // Pytorch does correlation by default.  For convolution, rotate the kernel
    if (do_conv) kernel = kernel.flip(2).flip(3).flip(4);

    return torch::nn::functional::conv3d(in, kernel, 
        torch::nn::functional::Conv3dFuncOptions().padding({padt, padh, padw}));
}
