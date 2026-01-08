// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

/**
 * @brief Vision Encoder Sample
 * 
 * This sample demonstrates the usage of the VisionEncoder public API
 * to encode images into embeddings that can be used for visual language models.
 * 
 * Usage: vision_encoder_sample <MODEL_DIR> <IMAGE_FILE>
 */

#include <openvino/genai/visual_language/vision_encoder.hpp>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <filesystem>

/**
 * @brief Load an image and convert it to OpenVINO tensor format
 * @param image_path Path to the image file
 * @return ov::Tensor containing the image in NHWC format (uint8)
 */
ov::Tensor load_image(const std::string& image_path) {
    cv::Mat image = cv::imread(image_path, cv::IMREAD_COLOR);
    if (image.empty()) {
        throw std::runtime_error("Failed to load image: " + image_path);
    }
    
    // Convert BGR to RGB
    cv::cvtColor(image, image, cv::COLOR_BGR2RGB);
    
    // Create tensor with shape [1, H, W, C] (NHWC format)
    ov::Tensor tensor(ov::element::u8, {1, static_cast<size_t>(image.rows), 
                                         static_cast<size_t>(image.cols), 3});
    
    // Copy image data to tensor
    std::memcpy(tensor.data<uint8_t>(), image.data, tensor.get_byte_size());
    
    return tensor;
}

int main(int argc, char* argv[]) try {
    if (argc != 3) {
        std::cout << "Usage: " << argv[0] << " <MODEL_DIR> <IMAGE_FILE>\n";
        std::cout << "\n";
        std::cout << "Arguments:\n";
        std::cout << "  MODEL_DIR   Path to VLM model directory containing:\n";
        std::cout << "              - config.json\n";
        std::cout << "              - openvino_vision_embeddings_model.xml\n";
        std::cout << "              - preprocessor_config.json\n";
        std::cout << "  IMAGE_FILE  Path to input image file (jpg, png, etc.)\n";
        return EXIT_FAILURE;
    }

    std::filesystem::path model_dir = argv[1];
    std::string image_path = argv[2];

    std::cout << "Loading VisionEncoder from: " << model_dir << "\n";
    
    // Create VisionEncoder instance
    // The encoder automatically detects the model type from config.json
    ov::genai::VisionEncoder encoder(model_dir, "CPU");
    
    std::cout << "Loading image: " << image_path << "\n";
    
    // Load and preprocess the image
    ov::Tensor image = load_image(image_path);
    std::cout << "Image shape: [" << image.get_shape()[0] << ", " 
              << image.get_shape()[1] << ", "
              << image.get_shape()[2] << ", "
              << image.get_shape()[3] << "]\n";
    
    std::cout << "Encoding image...\n";
    
    // Encode the image to get embeddings
    ov::genai::EncodedImage encoded = encoder.encode(image);
    
    // Print information about the encoded image
    std::cout << "\nEncoded image information:\n";
    std::cout << "  - Embeddings shape: [";
    for (size_t i = 0; i < encoded.resized_source.get_shape().size(); ++i) {
        if (i > 0) std::cout << ", ";
        std::cout << encoded.resized_source.get_shape()[i];
    }
    std::cout << "]\n";
    
    std::cout << "  - Resized source size: " 
              << encoded.resized_source_size.height << " x " 
              << encoded.resized_source_size.width << "\n";
    
    std::cout << "  - Original image size: "
              << encoded.original_image_size.height << " x "
              << encoded.original_image_size.width << "\n";
    
    std::cout << "  - Number of image tokens: " << encoded.num_image_tokens << "\n";
    
    if (encoded.patches_grid.first > 0 || encoded.patches_grid.second > 0) {
        std::cout << "  - Patches grid: " 
                  << encoded.patches_grid.first << " x " 
                  << encoded.patches_grid.second << "\n";
    }
    
    std::cout << "\nImage successfully encoded!\n";
    std::cout << "The embeddings can now be used with a language model for visual QA.\n";
    
    return EXIT_SUCCESS;
    
} catch (const std::exception& error) {
    std::cerr << "Error: " << error.what() << '\n';
    return EXIT_FAILURE;
} catch (...) {
    std::cerr << "Unknown error occurred\n";
    return EXIT_FAILURE;
}
