// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <iostream>
#include <string>
#include <vector>

#include "openvino/genai/tokenizer.hpp"
#include "openvino/genai/visual_language/vision_encoder.hpp"
#include "openvino/genai/visual_language/embeddings_model.hpp"
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

void print_tensor_shape(const ov::Tensor& tensor, const std::string& name) {
    std::cout << name << " shape: [";
    for (size_t i = 0; i < tensor.get_shape().size(); ++i) {
        std::cout << tensor.get_shape()[i];
        if (i < tensor.get_shape().size() - 1) std::cout << ", ";
    }
    std::cout << "]" << std::endl;
}

// Demonstrates creating InputsEmbedder from model directory (simple approach)
void demo_from_model_dir(const std::string& model_dir, const std::string& device) {
    std::cout << "\n=== Demo 1: Create InputsEmbedder from model directory ===" << std::endl;
    
    // Create the InputsEmbedder directly from model directory
    // This is the simplest approach - all components are loaded internally
    ov::genai::InputsEmbedder inputs_embedder(model_dir, device);

    // Get the tokenizer for text processing
    auto tokenizer = inputs_embedder.get_tokenizer();
    std::cout << "InputsEmbedder created with internal tokenizer." << std::endl;

    // Load and encode a sample image
    std::vector<ov::Tensor> images;
    images.push_back(load_image("sample_image.png"));
    
    std::cout << "\nEncoding images..." << std::endl;
    auto encoded_images = inputs_embedder.encode_images(images);
    std::cout << "Encoded " << encoded_images.size() << " image(s)." << std::endl;

    // Create a prompt and get embeddings
    std::string prompt = "<image>\nDescribe this image in detail.";
    std::cout << "Prompt: " << prompt << std::endl;

    std::cout << "\nComputing input embeddings..." << std::endl;
    ov::Tensor inputs_embeds = inputs_embedder.get_inputs_embeds(prompt, encoded_images);
    print_tensor_shape(inputs_embeds, "Output embeddings");
}

// Demonstrates creating InputsEmbedder from pre-loaded components (advanced approach)
void demo_from_components(const std::string& model_dir, const std::string& device) {
    std::cout << "\n=== Demo 2: Create InputsEmbedder from pre-loaded components ===" << std::endl;
    
    // Step 1: Create the individual components separately
    // This allows for more control over initialization and potential reuse
    
    std::cout << "\nStep 1: Loading individual components..." << std::endl;
    
    // Create the tokenizer
    ov::genai::Tokenizer tokenizer(model_dir);
    std::cout << "  - Tokenizer loaded" << std::endl;
    
    // Create the VisionEncoder as a shared pointer
    // The VisionEncoder handles image preprocessing and encoding
    auto vision_encoder = std::make_shared<ov::genai::VisionEncoder>(model_dir, device);
    std::cout << "  - VisionEncoder loaded" << std::endl;
    
    // Create the EmbeddingsModel as a shared pointer
    // The EmbeddingsModel converts token IDs to embeddings
    auto embeddings_model = std::make_shared<ov::genai::EmbeddingsModel>(model_dir, device);
    std::cout << "  - EmbeddingsModel loaded" << std::endl;
    
    // Step 2: Create InputsEmbedder from the pre-loaded components
    // This constructor takes ownership via shared_ptr, allowing the components
    // to be shared with other parts of your application if needed
    std::cout << "\nStep 2: Creating InputsEmbedder from components..." << std::endl;
    ov::genai::InputsEmbedder inputs_embedder(
        tokenizer,
        vision_encoder,      // VisionEncoder::Ptr
        embeddings_model,    // EmbeddingsModel::Ptr
        model_dir            // Config directory path
    );
    std::cout << "InputsEmbedder created from pre-loaded components." << std::endl;

    // Step 3: Use the InputsEmbedder as usual
    std::cout << "\nStep 3: Using InputsEmbedder..." << std::endl;
    
    // Load and encode a sample image
    std::vector<ov::Tensor> images;
    images.push_back(load_image("sample_image.png"));
    
    std::cout << "\nEncoding images using the shared VisionEncoder..." << std::endl;
    auto encoded_images = inputs_embedder.encode_images(images);
    std::cout << "Encoded " << encoded_images.size() << " image(s)." << std::endl;

    // You can also use the VisionEncoder directly if needed
    std::cout << "\nAlternatively, encode directly with VisionEncoder::Ptr..." << std::endl;
    auto directly_encoded = vision_encoder->encode(images[0]);
    print_tensor_shape(directly_encoded.resized_source, "Directly encoded image");

    // Create a prompt and get embeddings
    std::string prompt = "<image>\nWhat do you see in this image?";
    std::cout << "\nPrompt: " << prompt << std::endl;

    std::cout << "\nComputing input embeddings..." << std::endl;
    ov::Tensor inputs_embeds = inputs_embedder.get_inputs_embeds(prompt, encoded_images);
    print_tensor_shape(inputs_embeds, "Output embeddings");
    
    // Demonstrate that the shared components can still be used independently
    std::cout << "\nNote: The VisionEncoder and EmbeddingsModel can be reused" << std::endl;
    std::cout << "      in other parts of your application since they're shared_ptr." << std::endl;
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

    std::cout << "Model directory: " << model_dir << std::endl;
    std::cout << "Device: " << device << std::endl;

    try {
        // Demo 1: Simple approach - create from model directory
        demo_from_model_dir(model_dir, device);
        
        // Demo 2: Advanced approach - create from pre-loaded components
        demo_from_components(model_dir, device);

        std::cout << "\n=== All demos completed successfully! ===" << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
