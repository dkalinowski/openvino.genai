// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "openvino/genai/visual_language/processor.hpp"
#include "openvino/genai/visual_language/perf_metrics.hpp"

#include "visual_language/inputs_embedder.hpp"
#include "visual_language/vlm_config.hpp"

#include "utils.hpp"

namespace ov::genai {

class VLMProcessor::Impl {
public:
    std::shared_ptr<InputsEmbedder> m_inputs_embedder;
    bool m_apply_chat_template = true;

    Impl(const std::filesystem::path& models_path,
         const std::string& device,
         const ov::AnyMap& properties) {
        auto properties_copy = properties;

        // Extract per-device properties if provided
        auto device_properties = utils::pop_or_default<ov::AnyMap>(
            properties_copy, ov::device::properties.name(), {});

        auto embedder_device = device;
        // NPU uses AUTO for embedder
        if (device.find("NPU") != std::string::npos) {
            embedder_device = "AUTO";
        }

        auto embedder_properties = device_properties.empty()
            ? properties_copy
            : utils::pop_or_default<ov::AnyMap>(device_properties, embedder_device, {});

        m_inputs_embedder = std::make_shared<InputsEmbedder>(
            models_path, embedder_device, embedder_properties);
    }

    Impl(const ModelsMap& models_map,
         const Tokenizer& tokenizer,
         const std::filesystem::path& config_dir_path,
         const std::string& device,
         const ov::AnyMap& properties) {
        m_inputs_embedder = std::make_shared<InputsEmbedder>(
            models_map, tokenizer, config_dir_path, device, properties);
    }

    VLMInputs prepare(
        const std::string& prompt,
        const std::vector<ov::Tensor>& images,
        const std::vector<ov::Tensor>& videos
    ) {
        VLMPerfMetrics perf_metrics;

        // Step 1: Encode images and videos
        const auto encoded_images = m_inputs_embedder->encode_images(images);
        const auto encoded_videos = m_inputs_embedder->encode_videos(videos);

        // Step 2: Normalize prompt (replace universal tags with model-native tags)
        auto [unified_prompt, image_sequence, video_sequence] =
            m_inputs_embedder->normalize_prompt(prompt, 0, 0, encoded_images, encoded_videos);

        // Step 3: Set chat template application mode
        m_inputs_embedder->set_apply_chat_template_status(m_apply_chat_template);

        // Step 4: Get merged embeddings
        VLMInputs result;
        bool has_vision = !encoded_images.empty() || !encoded_videos.empty();

        if (m_inputs_embedder->has_token_type_ids()) {
            auto [embeds, type_ids] =
                m_inputs_embedder->get_inputs_embeds_with_token_type_ids(
                    unified_prompt, encoded_images, encoded_videos, perf_metrics,
                    has_vision, image_sequence, video_sequence);
            result.inputs_embeds = std::move(embeds);
            result.token_type_ids = std::move(type_ids);
        } else {
            result.inputs_embeds = m_inputs_embedder->get_inputs_embeds(
                unified_prompt, encoded_images, encoded_videos, perf_metrics,
                has_vision, image_sequence, video_sequence);
        }

        const size_t inputs_embeds_size = result.inputs_embeds.get_shape().at(1);

        // Step 5: Compute attention mask
        result.attention_mask = ov::Tensor{ov::element::i64, {1, inputs_embeds_size}};
        std::fill_n(result.attention_mask.data<int64_t>(), inputs_embeds_size, 1);

        // Step 6: Compute position IDs
        auto [position_ids, rope_delta] = m_inputs_embedder->get_position_ids(inputs_embeds_size, 0);
        result.position_ids = std::move(position_ids);
        result.rope_delta = rope_delta;

        return result;
    }

    std::string apply_chat_template(
        const std::string& prompt,
        const std::string& system_message,
        bool add_generation_prompt
    ) const {
        Tokenizer tokenizer = m_inputs_embedder->get_tokenizer();

        ChatHistory history;
        if (!system_message.empty()) {
            history.push_back({{"role", "system"}, {"content", system_message}});
        }
        history.push_back({{"role", "user"}, {"content", prompt}});

        return tokenizer.apply_chat_template(history, add_generation_prompt);
    }

    std::vector<ov::Tensor> get_vision_embeddings(
        const std::vector<ov::Tensor>& images
    ) {
        const auto encoded_images = m_inputs_embedder->encode_images(images);

        std::vector<ov::Tensor> embeddings;
        embeddings.reserve(encoded_images.size());
        for (const auto& enc : encoded_images) {
            embeddings.push_back(enc.resized_source);
        }
        return embeddings;
    }

    std::vector<ov::Tensor> get_video_embeddings(
        const std::vector<ov::Tensor>& videos
    ) {
        const auto encoded_videos = m_inputs_embedder->encode_videos(videos);

        std::vector<ov::Tensor> embeddings;
        embeddings.reserve(encoded_videos.size());
        for (const auto& enc : encoded_videos) {
            embeddings.push_back(enc.video_features);
        }
        return embeddings;
    }

    Tokenizer get_tokenizer() const {
        return m_inputs_embedder->get_tokenizer();
    }
};

// ---- VLMProcessor public methods ----

VLMProcessor::VLMProcessor(
    const std::filesystem::path& models_path,
    const std::string& device,
    const ov::AnyMap& properties
) : m_impl(std::make_unique<Impl>(models_path, device, properties)) { }

VLMProcessor::VLMProcessor(
    const ModelsMap& models_map,
    const Tokenizer& tokenizer,
    const std::filesystem::path& config_dir_path,
    const std::string& device,
    const ov::AnyMap& properties
) : m_impl(std::make_unique<Impl>(models_map, tokenizer, config_dir_path, device, properties)) { }

VLMProcessor::~VLMProcessor() = default;

VLMProcessor::VLMProcessor(VLMProcessor&&) noexcept = default;
VLMProcessor& VLMProcessor::operator=(VLMProcessor&&) noexcept = default;

VLMInputs VLMProcessor::prepare(
    const std::string& prompt,
    const std::vector<ov::Tensor>& images,
    const std::vector<ov::Tensor>& videos
) {
    return m_impl->prepare(prompt, images, videos);
}

std::string VLMProcessor::apply_chat_template(
    const std::string& prompt,
    const std::string& system_message,
    bool add_generation_prompt
) const {
    return m_impl->apply_chat_template(prompt, system_message, add_generation_prompt);
}

std::vector<ov::Tensor> VLMProcessor::get_vision_embeddings(
    const std::vector<ov::Tensor>& images
) {
    return m_impl->get_vision_embeddings(images);
}

std::vector<ov::Tensor> VLMProcessor::get_video_embeddings(
    const std::vector<ov::Tensor>& videos
) {
    return m_impl->get_video_embeddings(videos);
}

Tokenizer VLMProcessor::get_tokenizer() const {
    return m_impl->get_tokenizer();
}

void VLMProcessor::set_apply_chat_template(bool apply) {
    m_impl->m_apply_chat_template = apply;
}

void VLMProcessor::set_chat_template(const std::string& new_template) {
    m_impl->get_tokenizer().set_chat_template(new_template);
}

} // namespace ov::genai
