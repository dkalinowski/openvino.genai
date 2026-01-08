// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "openvino/genai/visual_language/vision_encoder.hpp"
#include "visual_language/vision_encoder_impl.hpp"
#include "visual_language/vlm_config.hpp"
#include "utils.hpp"

namespace ov::genai {

/// @brief The implementation class that wraps VisionEncoderImpl for the public API.
class VisionEncoder::VisionEncoderImpl {
public:
    VisionEncoderImpl(
        const std::filesystem::path& model_dir,
        const std::string& device,
        const ov::AnyMap& properties)
    {
        // Read the VLM config to determine the model type
        VLMConfig vlm_config = utils::from_config_json_if_exists<VLMConfig>(model_dir, "config.json");
        m_impl = ov::genai::VisionEncoderImpl::create(model_dir, vlm_config.model_type, device, properties);
    }

    VisionEncoderImpl(
        const ModelsMap& models_map,
        const std::filesystem::path& config_dir_path,
        const std::string& device,
        const ov::AnyMap& properties)
    {
        // Read the VLM config to determine the model type
        VLMConfig vlm_config = utils::from_config_json_if_exists<VLMConfig>(config_dir_path, "config.json");
        m_impl = ov::genai::VisionEncoderImpl::create(models_map, config_dir_path, vlm_config.model_type, device, properties);
    }

    EncodedImage encode(const ov::Tensor& image, const ov::AnyMap& config_map) {
        return m_impl->encode(image, config_map);
    }

private:
    ov::genai::VisionEncoderImpl::Ptr m_impl;
};

VisionEncoder::VisionEncoder(
    const std::filesystem::path& model_dir,
    const std::string& device,
    const ov::AnyMap& properties)
    : m_pimpl(std::make_unique<VisionEncoderImpl>(model_dir, device, properties)) {}

VisionEncoder::VisionEncoder(
    const ModelsMap& models_map,
    const std::filesystem::path& config_dir_path,
    const std::string& device,
    const ov::AnyMap& properties)
    : m_pimpl(std::make_unique<VisionEncoderImpl>(models_map, config_dir_path, device, properties)) {}

VisionEncoder::~VisionEncoder() = default;

EncodedImage VisionEncoder::encode(const ov::Tensor& image, const ov::AnyMap& config_map) {
    return m_pimpl->encode(image, config_map);
}

} // namespace ov::genai
