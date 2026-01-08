// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <iostream>
#include <string>
#include <vector>

#include "openvino/genai/visual_language/inputs_embedder.hpp"
#include "openvino/openvino.hpp"

// Helper function to load an image from file
ov::Tensor load_image(const std::string& image_path) {
    // This is a placeholder. In real usage, you would use OpenCV or similar
    // to load an image and convert it to an ov::Tensor.
    // For demonstration, we create a dummy tensor
    std::cout << "Note: This sample creates a dummy image tensor." << std::endl;
    std::cout << "In real usage, load actual image using OpenCV or similar library." << std::endl;
    
    // Create a dummy RGB image tensor with shape [1, H, W, C]
    const size_t height = 224;
    const size_t width = 224;
    const size_t channels = 3;
    
    ov::Tensor image(ov::element::u8, {1, height, width, channels});
    auto* data = image.data<uint8_t>();
    
    // Fill with some pattern
    for (size_t i = 0; i < height * width * channels; ++i) {
        data[i] = static_cast<uint8_t>(i % 256);
    }
    
    return image;
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <model_dir> [device]" << std::endl;
        std::cerr << std::endl;
        std::cerr << "  model_dir - Directory containing the VLM model files" << std::endl;
        std::cerr << "              (config.json, openvino_text_embeddings_model.xml," << std::endl;
        std::cerr << "              openvino_vision_embeddings_model.xml, etc.)" << std::endl;
        std::cerr << "  device    - (optional) Target device (default: CPU)" << std::endl;
        std::cerr << std::endl;
        std::cerr << "Example: " << argv[0] << " ./vlm_model CPU" << std::endl;
        return 1;
    }

    std::string model_dir = argv[1];
    std::string device = argc > 2 ? argv[2] : "CPU";

    std::cout << "Loading InputsEmbedder from: " << model_dir << std::endl;
    std::cout << "Device: " << device << std::endl;

    try {
        // Create the InputsEmbedder from model directory
        // This loads the text embeddings model and vision encoder
        ov::genai::InputsEmbedder inputs_embedder(model_dir, device);

        // Get the tokenizer for text processing
        auto tokenizer = inputs_embedder.get_tokenizer();
        std::cout << "Tokenizer loaded successfully." << std::endl;

        // Load a sample image
        std::vector<ov::Tensor> images;
        images.push_back(load_image("sample_image.png"));
        std::cout << "Image tensor created with shape: [";
        for (size_t i = 0; i < images[0].get_shape().size(); ++i) {
            std::cout << images[0].get_shape()[i];
            if (i < images[0].get_shape().size() - 1) std::cout << ", ";
        }
        std::cout << "]" << std::endl;

        // Encode images using the vision encoder
        std::cout << "\nEncoding images..." << std::endl;
        auto encoded_images = inputs_embedder.encode_images(images);
        std::cout << "Encoded " << encoded_images.size() << " image(s)." << std::endl;
        
        if (!encoded_images.empty()) {
            const auto& encoded = encoded_images[0];
            std::cout << "First encoded image shape: [";
            for (size_t i = 0; i < encoded.resized_source.get_shape().size(); ++i) {
                std::cout << encoded.resized_source.get_shape()[i];
                if (i < encoded.resized_source.get_shape().size() - 1) std::cout << ", ";
            }
            std::cout << "]" << std::endl;
        }

        // Create a prompt with image placeholder
        std::string prompt = "<image>\nDescribe this image in detail.";
        std::cout << "\nPrompt: " << prompt << std::endl;

        // Get combined input embeddings for the prompt and images
        std::cout << "\nComputing input embeddings..." << std::endl;
        ov::Tensor inputs_embeds = inputs_embedder.get_inputs_embeds(prompt, encoded_images);

        // Display output shape
        const auto& output_shape = inputs_embeds.get_shape();
        std::cout << "Output embeddings shape: [";
        for (size_t i = 0; i < output_shape.size(); ++i) {
            std::cout << output_shape[i];
            if (i < output_shape.size() - 1) std::cout << ", ";
        }
        std::cout << "]" << std::endl;

        // Display first few embedding values
        std::cout << "\nFirst token embedding (first 5 values): [";
        const auto* embedding_data = inputs_embeds.data<float>();
        const size_t hidden_size = output_shape.back();
        const size_t display_count = std::min(static_cast<size_t>(5), hidden_size);
        
        for (size_t i = 0; i < display_count; ++i) {
            std::cout << embedding_data[i];
            if (i < display_count - 1) std::cout << ", ";
        }
        if (hidden_size > display_count) {
            std::cout << ", ...";
        }
        std::cout << "]" << std::endl;

        std::cout << "\nInputs embeddings computed successfully!" << std::endl;
        std::cout << "These embeddings can now be fed to a language model for generation." << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
