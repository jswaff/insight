#include <ATen/ATen.h>
#include <torch/torch.h>

#include "matrix.h"

at::Tensor imfilter(at::Tensor &in, at::Tensor &filter)
{
    std::cout << "dim: " << in.dim() << "\n";
    if (in.dim() == 4)
        return conv2(in, filter, SAME, false);
    else
        return conv3(in, filter, SAME, false);
}
