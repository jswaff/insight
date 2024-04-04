#include <ATen/ATen.h>
#include <torch/torch.h>

at::Tensor conv2(at::Tensor &in, at::Tensor &kernel) {
    int kh = kernel.sizes()[2]; // out channels, in channels, kernel height, kernel width
    int padding = (kh-1)/2; // for SAME convolution (assumes symmetric kernel)
    return torch::nn::functional::conv2d(in, kernel, 
        torch::nn::functional::Conv2dFuncOptions().padding(padding));
}
