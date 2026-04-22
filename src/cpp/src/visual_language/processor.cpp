// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "openvino/genai/visual_language/processor.hpp"

#include "openvino/genai/visual_language/perf_metrics.hpp"
#include "visual_language/inputs_embedder.hpp"
#include "visual_language/processor_impl.hpp"

namespace ov::genai {

VLMProcessor::VLMProcessor(
    const std::filesystem::path& models_path,
    const std::string& device,
    const ov::AnyMap& properties)
    : m_impl{std::make_shared<Impl>()} {
    m_impl->inputs_embedder = std::make_shared<InputsEmbedder>(models_path, device, properties);
    m_impl->tokenizer = m_impl->inputs_embedder->get_tokenizer();
}

VLMProcessor::~VLMProcessor() = default;

Embeddings VLMProcessor::embed(
    const std::string& prompt,
    const std::vector<ov::Tensor>& images,
    const std::vector<ov::Tensor>& videos) {
    VLMPerfMetrics perf_metrics;
    const auto encoded_images = m_impl->inputs_embedder->encode_images(images);
    const auto encoded_videos = m_impl->inputs_embedder->encode_videos(videos);
    const auto normalized = m_impl->inputs_embedder->normalize_prompt(
        prompt, 0, 0, encoded_images, encoded_videos);

    Embeddings result;
    result.inputs_embeds = m_impl->inputs_embedder->get_inputs_embeds(
        normalized.unified_prompt,
        encoded_images,
        encoded_videos,
        perf_metrics,
        /*recalculate_merged_embeddings=*/true,
        normalized.images_sequence,
        normalized.videos_sequence);
    return result;
}

Tokenizer VLMProcessor::get_tokenizer() const {
    return m_impl->tokenizer;
}

}  // namespace ov::genai
