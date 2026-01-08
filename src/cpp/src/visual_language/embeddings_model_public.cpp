// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "openvino/genai/visual_language/embeddings_model.hpp"
#include "visual_language/embedding_model_impl.hpp"
#include "visual_language/vlm_config.hpp"
#include "utils.hpp"

namespace ov::genai {

/// @brief The implementation class that wraps EmbeddingsModelImpl for the public API.
class EmbeddingsModel::EmbeddingsModelImpl {
public:
    EmbeddingsModelImpl(
        const std::filesystem::path& model_dir,
        const std::string& device,
        const ov::AnyMap& properties)
    {
        // Read the VLM config to get scale_emb value
        VLMConfig vlm_config = utils::from_config_json_if_exists<VLMConfig>(model_dir, "config.json");
        m_impl = ov::genai::EmbeddingsModelImpl::create(model_dir, vlm_config.scale_emb, device, properties);
    }

    EmbeddingsModelImpl(
        const std::string& model,
        const ov::Tensor& weights,
        const std::string& device,
        const ov::AnyMap& properties)
    {
        // Default scale_emb value of 1.0 when config is not available
        m_impl = ov::genai::EmbeddingsModelImpl::create(model, weights, 1.0f, device, properties);
    }

    ov::Tensor infer(const ov::Tensor& input_ids) {
        auto& queue = m_impl->get_request_queue();
        CircularBufferQueueElementGuard<EmbeddingsRequest> req_guard(queue.get());
        return m_impl->infer(req_guard.get(), input_ids, false);
    }

private:
    ov::genai::EmbeddingsModelImpl::Ptr m_impl;
};

EmbeddingsModel::EmbeddingsModel(
    const std::filesystem::path& model_dir,
    const std::string& device,
    const ov::AnyMap& properties)
    : m_pimpl(std::make_unique<EmbeddingsModelImpl>(model_dir, device, properties)) {}

EmbeddingsModel::EmbeddingsModel(
    const std::string& model,
    const ov::Tensor& weights,
    const std::string& device,
    const ov::AnyMap& properties)
    : m_pimpl(std::make_unique<EmbeddingsModelImpl>(model, weights, device, properties)) {}

EmbeddingsModel::~EmbeddingsModel() = default;

ov::Tensor EmbeddingsModel::infer(const ov::Tensor& input_ids) {
    return m_pimpl->infer(input_ids);
}

} // namespace ov::genai
