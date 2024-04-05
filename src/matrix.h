#include <ATen/ATen.h>

enum ConvType { VALID, SAME };

at::Tensor conv2(at::Tensor &in, at::Tensor &kernel, ConvType conv_type, bool do_conv);
at::Tensor conv3(at::Tensor &in, at::Tensor &kernel, ConvType conv_type, bool do_conv);

