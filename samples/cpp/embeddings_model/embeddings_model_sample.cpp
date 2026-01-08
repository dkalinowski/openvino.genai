// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <iostream>
#include <string>

#include "openvino/genai/visual_language/embeddings_model.hpp"
#include "openvino/openvino.hpp"

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <model_dir> [device]" << std::endl;
        std::cerr << std::endl;
        std::cerr << "  model_dir - Directory containing the text embeddings model" << std::endl;
        std::cerr << "              (e.g., from a VLM model that includes openvino_text_embeddings_model.xml)" << std::endl;
        std::cerr << "  device    - (optional) Target device (default: CPU)" << std::endl;
        std::cerr << std::endl;
        std::cerr << "Example: " << argv[0] << " ./model_dir CPU" << std::endl;
        return 1;
    }

    std::string model_dir = argv[1];
    std::string device = argc > 2 ? argv[2] : "CPU";

    std::cout << "Loading EmbeddingsModel from: " << model_dir << std::endl;
    std::cout << "Device: " << device << std::endl;

    try {
        // Create the embeddings model
        // The EmbeddingsModel class uses PIMPL pattern for ABI stability.
        // It loads the openvino_text_embeddings_model.xml from the specified directory.
        ov::genai::EmbeddingsModel embeddings_model(model_dir, device);

        // Create sample input token IDs
        // In a real use case, these would come from a tokenizer
        // Shape: [batch_size, sequence_length]
        const size_t batch_size = 1;
        const size_t sequence_length = 5;

        ov::Tensor input_ids(ov::element::i64, {batch_size, sequence_length});
        auto input_data = input_ids.data<int64_t>();
        
        // Fill with sample token IDs (e.g., [1, 100, 200, 300, 2])
        // These would typically be obtained from tokenizing text
        input_data[0] = 1;    // BOS token
        input_data[1] = 100;  // Sample token
        input_data[2] = 200;  // Sample token
        input_data[3] = 300;  // Sample token
        input_data[4] = 2;    // EOS token

        std::cout << "\nInput token IDs: [";
        for (size_t i = 0; i < sequence_length; ++i) {
            std::cout << input_data[i];
            if (i < sequence_length - 1) std::cout << ", ";
        }
        std::cout << "]" << std::endl;

        // Compute embeddings
        std::cout << "\nComputing embeddings..." << std::endl;
        ov::Tensor embeddings = embeddings_model.infer(input_ids);

        // Display output shape
        const auto& output_shape = embeddings.get_shape();
        std::cout << "Output embeddings shape: [";
        for (size_t i = 0; i < output_shape.size(); ++i) {
            std::cout << output_shape[i];
            if (i < output_shape.size() - 1) std::cout << ", ";
        }
        std::cout << "]" << std::endl;

        // Display first few embedding values for the first token
        std::cout << "\nFirst token embedding (first 10 values): [";
        const auto* embedding_data = embeddings.data<float>();
        const size_t hidden_size = output_shape.back();
        const size_t display_count = std::min(static_cast<size_t>(10), hidden_size);
        
        for (size_t i = 0; i < display_count; ++i) {
            std::cout << embedding_data[i];
            if (i < display_count - 1) std::cout << ", ";
        }
        if (hidden_size > display_count) {
            std::cout << ", ...";
        }
        std::cout << "]" << std::endl;

        std::cout << "\nEmbeddings computed successfully!" << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
