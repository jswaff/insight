//#include <torch/extension.h>
#include <torch/script.h>
#include <torch/torch.h>
#include <opencv2/opencv.hpp>

#include <iostream>
#include <memory>

#include "matrix.h"

static const int WIDTH = 128;
static const int HEIGHT = 128;

int main(int argc, const char* argv[]) {
    std::cout << "insight 0.1\n\n";

    if (argc != 3) {
        std::cerr << "usage: insight <path-to-script-module> <path-to-image>\n";
        return -1;
    }

    torch::jit::script::Module module;
    try {
        // Deserialize the ScriptModule
        module = torch::jit::load(argv[1] /*, at::kCUDA*/);
        //module.to(at::kCUDA);
    } 
    catch (const c10::Error& e) {
        std::cerr << "error loading the model: " << e.msg() << "\n";
        return -1;
    }

    // get an input image
    cv::Mat image = cv::imread(argv[2], cv::IMREAD_COLOR);
    if (image.empty()) {
        std::cerr << argv[2] << " not found\n";
        return -1;
    }
    cv::imshow("Input Image", image);
    cv::waitKey(0);

    // apply color transformation
    cv::Size orig_size = image.size();
    cv::cvtColor(image, image, cv::COLOR_BGR2RGB);

    // resize
    cv::resize(image, image, cv::Size(WIDTH, HEIGHT));

    // convert to tensor
    auto tensor_image = torch::from_blob(
        image.data, 
        { image.rows, image.cols, image.channels() }, 
        at::kByte);
    tensor_image = tensor_image.permute({ 2,0,1 });
    std::cout << "tensor_image sizes: " << tensor_image.sizes() << "\n";

    // convert to float and scale to range [0.0, 1.0]
    tensor_image = tensor_image.to(torch::kFloat);
    tensor_image = tensor_image.div(255.0);

    // create a vector of inputs
    tensor_image.unsqueeze_(0);
    std::vector<torch::jit::IValue> inputs;
    inputs.push_back(tensor_image.to(torch::kFloat));

    // execute the model
    auto model_output = module.forward(inputs).toTensor();
    auto pred = model_output.argmax(1).squeeze(0);
    std::cout << "pred sizes: " << pred.sizes() << "\n";

    // overlay segmentation
    auto tensor_image_out = tensor_image.squeeze(0);
    std::cout << "tensor_image_out sizes: " << tensor_image_out.sizes() << "\n";
    for (int r=0;r<tensor_image_out.sizes()[1];r++) {
        for (int c=0;c<tensor_image_out.sizes()[2];c++) {
            if (pred[r][c].item<float>() > 0) {
                float alpha = 0.75;
                tensor_image_out[0][r][c] *= alpha;
                tensor_image_out[1][r][c] *= alpha;
                tensor_image_out[2][r][c] *= alpha;
            }
        }
    }
    
    // convert to image
    tensor_image_out = tensor_image_out.mul(255.0);
    tensor_image_out = tensor_image_out.to(torch::kU8);
    tensor_image_out = tensor_image_out.permute({1,2,0});
    cv::Mat image_out = cv::Mat(tensor_image_out.size(1), tensor_image_out.size(0), CV_8UC3, tensor_image_out.data_ptr());
    cv::cvtColor(image_out, image_out, cv::COLOR_RGB2BGR); // OpenCV uses BGR order
    cv::resize(image_out, image_out, orig_size);
    cv::imshow("Output Image", image_out);
    cv::waitKey(0);

    // just a test
    float input_arr[16] = { 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15 };
    auto conv_input = torch::from_blob(input_arr, { 1,1,4,4 }); // batch size, in channels, height, width
    std::cout << "conv_input: " << conv_input << "\n";

    auto kernel = at::ones({1,1,3,3}); // out channels, in channels, kernel height, kernel width
    std::cout << "kernel: " << kernel << "\n";

    auto conv = conv2(conv_input, kernel);
    std::cout << "conv out: " << conv << "\n";

    return 0;
}
