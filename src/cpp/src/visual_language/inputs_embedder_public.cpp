// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "openvino/genai/visual_language/inputs_embedder.hpp"
#include "visual_language/inputs_embedder_impl.hpp"
#include "visual_language/vlm_config.hpp"
#include "visual_language/vision_encoder_impl.hpp"
#include "visual_language/embedding_model_impl.hpp"
#include "openvino/genai/visual_language/perf_metrics.hpp"
#include "utils.hpp"

namespace ov::genai {

/// @brief The implementation class that wraps the internal InputsEmbedderImpl for the public API.
class InputsEmbedder::InputsEmbedderImpl {
public:
    InputsEmbedderImpl(
        const std::filesystem::path& model_dir,
        const std::string& device,
        const ov::AnyMap& device_config)
        : m_internal_impl(std::make_shared<ov::genai::InputsEmbedderImpl>(model_dir, device, device_config)),
          m_model_dir(model_dir),
          m_device(device),
          m_device_config(device_config) {}

    InputsEmbedderImpl(
        const Tokenizer& tokenizer,
        const std::filesystem::path& config_dir_path)
        : m_tokenizer_ptr(std::make_shared<Tokenizer>(tokenizer)),
          m_config_dir_path(config_dir_path),
          m_use_external_components(true)
    {
        // Read the VLM config to get model type information
        m_vlm_config = utils::from_config_json_if_exists<VLMConfig>(config_dir_path, "config.json");
    }

    InputsEmbedderImpl(
        const Tokenizer& tokenizer,
        VisionEncoder vision_encoder,
        EmbeddingsModel embeddings_model,
        const std::filesystem::path& config_dir_path)
        : m_tokenizer_ptr(std::make_shared<Tokenizer>(tokenizer)),
          m_vision_encoder_ptr(std::make_shared<VisionEncoder>(std::move(vision_encoder))),
          m_embeddings_model_ptr(std::make_shared<EmbeddingsModel>(std::move(embeddings_model))),
          m_config_dir_path(config_dir_path),
          m_use_external_components(true)
    {
        // Read the VLM config to get model type information
        m_vlm_config = utils::from_config_json_if_exists<VLMConfig>(config_dir_path, "config.json");
    }

    std::vector<EncodedImage> encode_images(const std::vector<ov::Tensor>& images) {
        if (m_use_external_components && m_vision_encoder_ptr) {
            std::vector<EncodedImage> encoded;
            encoded.reserve(images.size());
            for (const auto& image : images) {
                encoded.push_back(m_vision_encoder_ptr->encode(image));
            }
            return encoded;
        }
        return m_internal_impl->encode_images(images);
    }

    ov::Tensor get_inputs_embeds(
        const std::string& prompt,
        const std::vector<EncodedImage>& encoded_images)
    {
        VLMPerfMetrics metrics;
        return m_internal_impl->get_inputs_embeds(prompt, encoded_images, metrics, true, {});
    }

    Tokenizer get_tokenizer() const {
        if (m_use_external_components && m_tokenizer_ptr) {
            return *m_tokenizer_ptr;
        }
        return m_internal_impl->get_tokenizer();
    }

private:
    std::shared_ptr<ov::genai::InputsEmbedderImpl> m_internal_impl;
    
    // Store construction parameters for later use
    std::filesystem::path m_model_dir;
    std::string m_device;
    ov::AnyMap m_device_config;
    
    // For external components mode
    std::shared_ptr<Tokenizer> m_tokenizer_ptr;
    std::shared_ptr<VisionEncoder> m_vision_encoder_ptr;
    std::shared_ptr<EmbeddingsModel> m_embeddings_model_ptr;
    std::filesystem::path m_config_dir_path;
    VLMConfig m_vlm_config;
    bool m_use_external_components = false;
};

InputsEmbedder::InputsEmbedder(
    const std::filesystem::path& model_dir,
    const std::string& device,
    const ov::AnyMap& device_config)
    : m_pimpl(std::make_unique<InputsEmbedderImpl>(model_dir, device, device_config)) {}

InputsEmbedder::InputsEmbedder(
    const Tokenizer& tokenizer,
    const std::filesystem::path& config_dir_path)
    : m_pimpl(std::make_unique<InputsEmbedderImpl>(tokenizer, config_dir_path)) {}

InputsEmbedder::InputsEmbedder(
    const Tokenizer& tokenizer,
    VisionEncoder vision_encoder,
    EmbeddingsModel embeddings_model,
    const std::filesystem::path& config_dir_path)
    : m_pimpl(std::make_unique<InputsEmbedderImpl>(tokenizer, std::move(vision_encoder), std::move(embeddings_model), config_dir_path)) {}

InputsEmbedder::~InputsEmbedder() = default;

std::vector<EncodedImage> InputsEmbedder::encode_images(const std::vector<ov::Tensor>& images) {
    return m_pimpl->encode_images(images);
}

ov::Tensor InputsEmbedder::get_inputs_embeds(
    const std::string& prompt,
    const std::vector<EncodedImage>& encoded_images)
{
    return m_pimpl->get_inputs_embeds(prompt, encoded_images);
}

Tokenizer InputsEmbedder::get_tokenizer() const {
    return m_pimpl->get_tokenizer();
}

} // namespace ov::genai
