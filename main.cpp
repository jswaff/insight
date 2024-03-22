#include <torch/script.h>
#include <opencv2/opencv.hpp>

#include <iostream>
#include <memory>

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
        module.to(at::kCUDA);
    } 
    catch (const c10::Error& e) {
        std::cerr << "error loading the model\n";
        return -1;
    }

    // get an input image
    cv::Mat image = cv::imread("/home/james/insight/img3.jpg", cv::IMREAD_COLOR);
    if (image.empty()) {
        std::cerr << "img3.jpg not found\n";
        return -1;
    }
    cv::imshow("Input Image", image);
    cv::waitKey(0);

    cv::Mat image2;
    cv::cvtColor(image, image2, cv::COLOR_BGR2RGB);

    cv::imshow("Image2", image2);
    cv::waitKey(0);

    std::cout << "rows: " << image2.rows << " cols: " << image2.cols << " channels: " << image2.channels() << "\n";
    //std::cout << "image data: " << image2.row(128) << "\n";

    // TODO: transform to 128 x 128, center cropped
    // resize 128 x 128
    cv::Mat resized_image;
    cv::resize(image2, resized_image, cv::Size(128, 128));
    std::cout << "rows: " << resized_image.rows << " cols: " << resized_image.cols << " channels: " << 
        resized_image.channels() << "\n";
    //std::cout << "resized image data: " << resized_image.row(0) << "\n";

    // convert to tensor
    auto tensor_image = torch::from_blob(
        image2.data, { image2.rows, image2.cols, image2.channels() }, at::kByte).permute({2, 0, 1});
    
    // auto tensor_image = torch::from_blob(
    //     resized_image.data, { resized_image.rows, resized_image.cols, resized_image.channels() })
    //     .permute({2, 0, 1})
    //     .to(torch::kCUDA);
    

    std::cout << "dim 0: " << tensor_image.sizes()[0] << "\n";
    std::cout << "dim 1: " << tensor_image.sizes()[1] << "\n";
    std::cout << "dim 2: " << tensor_image.sizes()[2] << "\n";
    std::cout << tensor_image.slice(0, 0, 0) << "\n";

    // create a vector of inputs
    // std::vector<torch::jit::IValue> inputs;
    // inputs.push_back(tensor_image.unsqueeze(0));
    //inputs.push_back(torch::ones({1, 3, 128, 128}, torch::kCUDA));

    // // execute the model and turn its output into a tensor
//    at::Tensor output = module.forward(inputs).toTensor();
    //std::cout << output.slice(/*dim=*/1, /*start=*/0, /*end=*/5) << '\n';

    return 0;
}
