// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <memory>
#include <string>
#include <filesystem>
#include <vector>

#include "openvino/genai/visibility.hpp"
#include "openvino/genai/common_types.hpp"
#include "openvino/genai/tokenizer.hpp"
#include "openvino/genai/visual_language/vision_encoder.hpp"
#include "openvino/genai/visual_language/embeddings_model.hpp"
#include "openvino/runtime/tensor.hpp"

namespace ov::genai {

/// @brief A class for computing input embeddings for Visual Language Models.
/// This class combines text tokenization with vision encoding to produce
/// embeddings that can be fed to a language model.
/// This is the public API class that uses PIMPL pattern to hide implementation details.
class OPENVINO_GENAI_EXPORTS InputsEmbedder {
public:
    /// @brief Constructs the InputsEmbedder from model_dir.
    /// @param model_dir A folder containing VLM model files including
    /// openvino_text_embeddings_model.xml, openvino_vision_embeddings_model.xml,
    /// and configuration files (config.json, preprocessor_config.json).
    /// @param device A device to compile the models for.
    /// @param device_config A config to be passed to ov::Core::compile_model().
    InputsEmbedder(
        const std::filesystem::path& model_dir,
        const std::string& device,
        const ov::AnyMap& device_config = {});

    /// @brief Constructs the InputsEmbedder with a pre-loaded tokenizer.
    /// @param tokenizer A pre-loaded tokenizer.
    /// @param config_dir_path A path to the directory containing config.json
    /// and model files for VLM configuration.
    InputsEmbedder(
        const Tokenizer& tokenizer,
        const std::filesystem::path& config_dir_path);

    /// @brief Constructs the InputsEmbedder from pre-loaded components.
    /// @param tokenizer A pre-loaded tokenizer.
    /// @param vision_encoder A pre-loaded VisionEncoder.
    /// @param embeddings_model A pre-loaded EmbeddingsModel.
    /// @param config_dir_path A path to the directory containing config.json
    /// for VLM configuration.
    InputsEmbedder(
        const Tokenizer& tokenizer,
        VisionEncoder vision_encoder,
        EmbeddingsModel embeddings_model,
        const std::filesystem::path& config_dir_path);

    /// @brief Constructs the InputsEmbedder with variadic properties.
    /// @param model_dir A folder containing VLM model files.
    /// @param device A device to compile the models for.
    /// @param properties Variadic properties to pass to ov::Core::compile_model().
    template <typename... Properties, typename std::enable_if<ov::util::StringAny<Properties...>::value, bool>::type = true>
    InputsEmbedder(
        const std::filesystem::path& model_dir,
        const std::string& device,
        Properties&&... properties)
        : InputsEmbedder(model_dir, device, ov::AnyMap{std::forward<Properties>(properties)...}) { }

    /// @brief Default destructor.
    ~InputsEmbedder();

    /// @brief Encode images into image embeddings.
    /// @param images A vector of image tensors with shape [1, H, W, C] or [H, W, C]
    /// in NHWC layout with RGB channel order.
    /// @return A vector of EncodedImage structures containing the image embeddings.
    std::vector<EncodedImage> encode_images(const std::vector<ov::Tensor>& images);

    /// @brief Compute input embeddings for a text prompt with images.
    /// This method tokenizes the prompt, encodes the images, and combines them
    /// into a single embeddings tensor suitable for language model input.
    /// @param prompt The text prompt, which may contain image placeholders.
    /// @param encoded_images A vector of pre-encoded images from encode_images().
    /// @return An embeddings tensor with shape [batch_size, sequence_length, hidden_size].
    ov::Tensor get_inputs_embeds(
        const std::string& prompt,
        const std::vector<EncodedImage>& encoded_images);

    /// @brief Get the tokenizer used by this InputsEmbedder.
    /// @return The tokenizer instance.
    Tokenizer get_tokenizer() const;

private:
    class InputsEmbedderImpl;
    std::unique_ptr<InputsEmbedderImpl> m_pimpl;
};

} // namespace ov::genai
