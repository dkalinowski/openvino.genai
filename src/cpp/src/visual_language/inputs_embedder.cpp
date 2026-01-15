// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "openvino/genai/visual_language/inputs_embedder.hpp"
#include "openvino/genai/visual_language/perf_metrics.hpp"
#include "visual_language/inputs_embedder_impl.hpp"
#include "visual_language/vlm_config.hpp"

#include "visual_language/clip.hpp"
#include "visual_language/vision_encoder_impl.hpp"
#include "visual_language/embedding_model_impl.hpp"

#include "visual_language/qwen2vl/classes.hpp"
#include "visual_language/qwen2_5_vl/classes.hpp"
#include "visual_language/phi3_vision/classes.hpp"
#include "visual_language/phi4mm/classes.hpp"
#include "visual_language/minicpm/classes.hpp"
#include "visual_language/llava/classes.hpp"
#include "visual_language/nanollava/classes.hpp"
#include "visual_language/llava_next/classes.hpp"
#include "visual_language/llava_next_video/classes.hpp"
#include "visual_language/internvl_chat/classes.hpp"
#include "visual_language/gemma3/classes.hpp"

#include "utils.hpp"

namespace ov::genai {

// ======================== InputsEmbedderImpl (internal) ========================

std::pair<ov::Tensor, std::optional<int64_t>> InputsEmbedderImpl::IInputsEmbedder::get_position_ids(const size_t inputs_embeds_size, const size_t history_size) {
    ov::Tensor position_ids = ov::Tensor{ov::element::i64, { 1, inputs_embeds_size }};
    std::iota(position_ids.data<int64_t>(), position_ids.data<int64_t>() + position_ids.get_size(), history_size);
    return {position_ids, std::nullopt};
}

std::pair<ov::Tensor, std::optional<int64_t>> InputsEmbedderImpl::IInputsEmbedder::get_generation_phase_position_ids(const size_t inputs_embeds_size, const size_t history_size, int64_t rope_delta) {
    return get_position_ids(inputs_embeds_size, history_size);
}

void InputsEmbedderImpl::IInputsEmbedder::start_chat(const std::string& system_message) {
    m_is_chat_conversation = true;
    if (!m_kv_cache_state.get_state().empty()) {
        m_kv_cache_state.reset_state();
    }
    if (system_message.empty()) {
        return;
    }
}

void InputsEmbedderImpl::IInputsEmbedder::update_chat_history(const std::string& decoded_results, const ov::genai::GenerationStatus generation_finish_status) {
    m_kv_cache_state.num_tokens_to_trim = 0;
    if (generation_finish_status == ov::genai::GenerationStatus::CANCEL) {
        // If chat generation process was cancelled by user, let's rollback to previous state of kv cache
        std::vector<int64_t>& state = m_kv_cache_state.get_state();

        m_kv_cache_state.num_tokens_to_trim = state.size() - m_prev_hist_length;
        state.resize(m_prev_hist_length);
        m_kv_cache_state.reset_mem_state = state.empty();
    }
}

void InputsEmbedderImpl::IInputsEmbedder::finish_chat() {
    m_is_chat_conversation = false;
    m_kv_cache_state.reset_state();
}

InputsEmbedderImpl::IInputsEmbedder::IInputsEmbedder(
        const VLMConfig& vlm_config,
        const std::filesystem::path& model_dir,
        const std::string& device,
        const ov::AnyMap device_config) :
    m_vlm_config{vlm_config},
    m_vision_encoder(VisionEncoderImpl::create(model_dir, m_vlm_config.model_type, device, device_config)),
    m_embedding(EmbeddingsModelImpl::create(model_dir, m_vlm_config.scale_emb, device, device_config)),
    m_tokenizer{model_dir, device_config} { }

InputsEmbedderImpl::IInputsEmbedder::IInputsEmbedder(
        const VLMConfig& vlm_config,
        const ModelsMap& models_map,
        const Tokenizer& tokenizer,
        const std::filesystem::path& config_dir_path,
        const std::string& device,
        const ov::AnyMap device_config) :
    m_vlm_config{vlm_config},
    m_vision_encoder(VisionEncoderImpl::create(
        models_map,
        config_dir_path,
        m_vlm_config.model_type,
        device,
        device_config
    )),
    m_embedding(EmbeddingsModelImpl::create(
        utils::get_model_weights_pair(models_map, "text_embeddings").first,
        utils::get_model_weights_pair(models_map, "text_embeddings").second,
        m_vlm_config.scale_emb,
        device,
        device_config
    )),
    m_tokenizer(tokenizer) { }

InputsEmbedderImpl::IInputsEmbedder::IInputsEmbedder(
        const VLMConfig& vlm_config,
        const Tokenizer& tokenizer,
        VisionEncoderImpl::Ptr vision_encoder_impl,
        EmbeddingsModelImpl::Ptr embeddings_model_impl) :
    m_vlm_config{vlm_config},
    m_vision_encoder(vision_encoder_impl),
    m_embedding(embeddings_model_impl),
    m_tokenizer(tokenizer) { }

ov::Tensor InputsEmbedderImpl::IInputsEmbedder::apply_chat_template_tokenize(const std::string& prompt, ov::genai::VLMPerfMetrics& metrics) {
    bool add_special_tokens = m_add_special_tokens_is_set ? m_add_special_tokens : !(m_is_chat_conversation || m_apply_chat_template);
    if (m_is_chat_conversation) {
        std::string prompt_to_encode = prompt;
        auto start_tokenizer_time = std::chrono::steady_clock::now();
        ov::Tensor new_chat_tokens = m_tokenizer.encode(prompt_to_encode, ov::genai::add_special_tokens(add_special_tokens)).input_ids;
        auto end_tokenizer_time = std::chrono::steady_clock::now();
        metrics.raw_metrics.tokenization_durations.emplace_back(PerfMetrics::get_microsec(end_tokenizer_time - start_tokenizer_time));
        return new_chat_tokens;
    } else {
        ov::Tensor encoded_input_ids;
        auto start_tokenizer_time = std::chrono::steady_clock::now();
        if (m_apply_chat_template) {
            std::string templated_prompt;
            ChatHistory history({{{"role", "user"}, {"content", prompt}}});
            constexpr bool add_generation_prompt = true;

            templated_prompt = m_tokenizer.apply_chat_template(history, add_generation_prompt);
            encoded_input_ids = m_tokenizer.encode(templated_prompt, ov::genai::add_special_tokens(add_special_tokens)).input_ids;
        } else {
            encoded_input_ids = m_tokenizer.encode(prompt, ov::genai::add_special_tokens(add_special_tokens)).input_ids;
        }
        auto end_tokenizer_time = std::chrono::steady_clock::now();
        metrics.raw_metrics.tokenization_durations.emplace_back(PerfMetrics::get_microsec(end_tokenizer_time - start_tokenizer_time));
        return encoded_input_ids;
    }
}

ov::Tensor InputsEmbedderImpl::IInputsEmbedder::update_history(const ov::Tensor& new_chat_tokens) {
    ov::Tensor encoded_inputs;
    if (m_is_chat_conversation) {
        ov::genai::align_kv_cache_and_history(new_chat_tokens, m_kv_cache_state);
        encoded_inputs = get_chat_encoded_input(new_chat_tokens, m_kv_cache_state).input_ids;
    } else {
        encoded_inputs = new_chat_tokens;
    }

    return encoded_inputs;
}

ov::Tensor InputsEmbedderImpl::IInputsEmbedder::get_encoded_input_ids(const std::string& prompt, ov::genai::VLMPerfMetrics& metrics) {
    const auto new_chat_tokens = apply_chat_template_tokenize(prompt, metrics);
    auto new_input_ids = update_history(new_chat_tokens);
    m_prev_hist_length = m_kv_cache_state.get_state().size();
    m_kv_cache_state.add_inputs(new_input_ids);

    return new_input_ids;
}

std::vector<ov::Tensor> InputsEmbedderImpl::IInputsEmbedder::to_single_image_tensors(const std::vector<ov::Tensor>& images) {
    std::vector<ov::Tensor> single_image_tensors;
    for (const auto& image : images) {
        ov::Tensor reshaped_image = image;
        ov::Shape image_shape = image.get_shape();
        switch (image_shape.size()) {
            case 3:
                reshaped_image.set_shape({1, image_shape.at(0), image_shape.at(1), image_shape.at(2)});
                break;
            case 4: break;
            default: OPENVINO_THROW("Input image must have [NHWC] or [HWC] layout, given image shape is ", image_shape);
        }
        ov::Shape reshaped_image_shape = reshaped_image.get_shape();
        for (size_t batch_idx = 0; batch_idx < reshaped_image_shape.at(0); ++batch_idx) {
            ov::Tensor single_image{
                reshaped_image.get_element_type(),
                {1, reshaped_image_shape.at(1), reshaped_image_shape.at(2), reshaped_image_shape.at(3)},
                reshaped_image.data<uint8_t>() + batch_idx * reshaped_image_shape.at(1) * reshaped_image_shape.at(2) * reshaped_image_shape.at(3)
            };
            single_image_tensors.push_back(std::move(single_image));
        }
    }
    return single_image_tensors;
}

std::vector<ov::genai::EncodedImage> InputsEmbedderImpl::IInputsEmbedder::encode_images(const std::vector<ov::Tensor>& images) {
    std::vector<EncodedImage> embeds;
    std::vector<ov::Tensor> single_images = to_single_image_tensors(images);
    for (const ov::Tensor& image : single_images) {
        embeds.emplace_back(m_vision_encoder->encode(image));
    }
    return embeds;
}

ov::Tensor InputsEmbedderImpl::IInputsEmbedder::get_inputs_embeds(
    const std::string& prompt,
    const std::vector<ov::genai::EncodedImage>& images,
    const std::vector<ov::genai::EncodedVideo>& videos,
    ov::genai::VLMPerfMetrics& metrics,
    bool recalculate_merged_embeddings,
    const std::vector<size_t>& images_sequence,
    const std::vector<size_t>& videos_sequence,
    const std::vector<std::pair<std::size_t, std::size_t>>& history_vision_count) {
    if (!videos.size()) {
        return get_inputs_embeds(prompt, images, metrics, recalculate_merged_embeddings, images_sequence);
    }
    OPENVINO_THROW("Current model doesn't support video preprocess currently. Input images are processed as separate images.");
}

std::vector<ov::genai::EncodedVideo> InputsEmbedderImpl::IInputsEmbedder::encode_videos(const std::vector<ov::Tensor>& videos) {
    if (!videos.size()) {
        return {};
    }
    OPENVINO_THROW("Current model doesn't support video preprocess currently. Input images are processed as separate images.");
}

NormalizedPrompt InputsEmbedderImpl::IInputsEmbedder::normalize_prompt(
    const std::string& prompt,
    size_t base_image_id,
    size_t base_video_id,
    const std::vector<EncodedImage>& images,
    const std::vector<EncodedVideo>& videos) const {
    if (!videos.size()) {
        return normalize_prompt(prompt, base_image_id, images);
    }
    OPENVINO_THROW("Current model doesn't support video preprocess currently. Input images are processed as separate images.");
}

std::pair<ov::Tensor, ov::Tensor> InputsEmbedderImpl::IInputsEmbedder::get_inputs_embeds_with_token_type_ids(
    const std::string& prompt,
    const std::vector<EncodedImage>& images,
    VLMPerfMetrics& metrics,
    bool recalculate_merged_embeddings,
    const std::vector<size_t>& image_sequence) {
    OPENVINO_THROW("This model does not support token_type_ids.");
}

std::pair<ov::Tensor, ov::Tensor> InputsEmbedderImpl::IInputsEmbedder::get_inputs_embeds_with_token_type_ids(
    const std::string& prompt,
    const std::vector<EncodedImage>& images,
    const std::vector<EncodedVideo>& videos,
    VLMPerfMetrics& metrics,
    bool recalculate_merged_embeddings,
    const std::vector<size_t>& image_sequence,
    const std::vector<size_t>& videos_sequence,
    const std::vector<std::pair<std::size_t, std::size_t>>& history_vision_count) {
    OPENVINO_ASSERT(videos.size() == 0U, "The model doesn't support 'videos' preprocessing yet. Please use 'images' instead.");

    return get_inputs_embeds_with_token_type_ids(prompt, images, metrics, recalculate_merged_embeddings, image_sequence);
}

bool InputsEmbedderImpl::IInputsEmbedder::has_token_type_ids() const { return false; }

/// Public InputsEmbedder class

InputsEmbedderImpl::InputsEmbedderImpl(const std::filesystem::path& model_dir,
                               const std::string& device,
                               const ov::AnyMap device_config) {
    auto vlm_config = utils::from_config_json_if_exists<VLMConfig>(model_dir, "config.json");

    if (vlm_config.model_type == VLMModelType::MINICPM) {
        m_impl = std::make_shared<InputsEmbedderMiniCPM>(vlm_config, model_dir, device, device_config);
    } else if (vlm_config.model_type == VLMModelType::LLAVA) {
        m_impl = std::make_shared<InputsEmbedderLLaVA>(vlm_config, model_dir, device, device_config);
    } else if (vlm_config.model_type == VLMModelType::NANOLLAVA) {
        m_impl = std::make_shared<InputsEmbedderNanoLLaVA>(vlm_config, model_dir, device, device_config);
    } else if (vlm_config.model_type == VLMModelType::LLAVA_NEXT) {
        m_impl = std::make_shared<InputsEmbedderLLaVANext>(vlm_config, model_dir, device, device_config);
    } else if (vlm_config.model_type == VLMModelType::LLAVA_NEXT_VIDEO) {
        m_impl = std::make_shared<InputsEmbedderLLaVANextVideo>(vlm_config, model_dir, device, device_config);
    } else if (vlm_config.model_type == VLMModelType::INTERNVL_CHAT) {
        m_impl = std::make_shared<InputsEmbedderInternVLChat>(vlm_config, model_dir, device, device_config);
    } else if (vlm_config.model_type == VLMModelType::PHI3_V) {
        m_impl = std::make_shared<InputsEmbedderPhi3V>(vlm_config, model_dir, device, device_config);
    } else if (vlm_config.model_type == VLMModelType::PHI4MM) {
        m_impl = std::make_shared<InputsEmbedderPhi4MM>(vlm_config, model_dir, device, device_config);
    } else if (vlm_config.model_type == VLMModelType::QWEN2_VL) {
        m_impl = std::make_shared<InputsEmbedderQwen2VL>(vlm_config, model_dir, device, device_config);
    } else if (vlm_config.model_type == VLMModelType::QWEN2_5_VL) {
        m_impl = std::make_shared<InputsEmbedderQwen2_5_VL>(vlm_config, model_dir, device, device_config);
    } else if (vlm_config.model_type == VLMModelType::GEMMA3) {
        m_impl = std::make_shared<InputsEmbedderGemma3>(vlm_config, model_dir, device, device_config);
    } else {
        OPENVINO_THROW("Unsupported model type in VLM InputsEmbedder class. Please, create feature request on new model support");
    }
}

InputsEmbedderImpl::InputsEmbedderImpl(const ModelsMap& models_map,
                               const Tokenizer& tokenizer,
                               const std::filesystem::path& config_dir_path,
                               const std::string& device,
                               const ov::AnyMap device_config) {
    auto vlm_config = utils::from_config_json_if_exists<VLMConfig>(config_dir_path, "config.json");

    if (vlm_config.model_type == VLMModelType::MINICPM) {
        m_impl = std::make_shared<InputsEmbedderMiniCPM>(vlm_config, models_map, tokenizer, config_dir_path, device, device_config);
    } else if (vlm_config.model_type == VLMModelType::LLAVA) {
        m_impl = std::make_shared<InputsEmbedderLLaVA>(vlm_config, models_map, tokenizer, config_dir_path, device, device_config);
    } else if (vlm_config.model_type == VLMModelType::NANOLLAVA) {
        m_impl = std::make_shared<InputsEmbedderNanoLLaVA>(vlm_config, models_map, tokenizer, config_dir_path, device, device_config);
    } else if (vlm_config.model_type == VLMModelType::LLAVA_NEXT) {
        m_impl = std::make_shared<InputsEmbedderLLaVANext>(vlm_config, models_map, tokenizer, config_dir_path, device, device_config);
    } else if (vlm_config.model_type == VLMModelType::LLAVA_NEXT_VIDEO) {
        m_impl = std::make_shared<InputsEmbedderLLaVANextVideo>(vlm_config, models_map, tokenizer, config_dir_path, device, device_config);
    } else if (vlm_config.model_type == VLMModelType::INTERNVL_CHAT) {
        m_impl = std::make_shared<InputsEmbedderInternVLChat>(vlm_config, models_map, tokenizer, config_dir_path, device, device_config);
    } else if (vlm_config.model_type == VLMModelType::PHI3_V) {
        m_impl = std::make_shared<InputsEmbedderPhi3V>(vlm_config, models_map, tokenizer, config_dir_path, device, device_config);
    } else if (vlm_config.model_type == VLMModelType::PHI4MM) {
        m_impl = std::make_shared<InputsEmbedderPhi4MM>(vlm_config, models_map, tokenizer, config_dir_path, device, device_config);
    } else if (vlm_config.model_type == VLMModelType::QWEN2_VL) {
        m_impl = std::make_shared<InputsEmbedderQwen2VL>(vlm_config, models_map, tokenizer, config_dir_path, device, device_config);
    } else if (vlm_config.model_type == VLMModelType::QWEN2_5_VL) {
        m_impl = std::make_shared<InputsEmbedderQwen2_5_VL>(vlm_config, models_map, tokenizer, config_dir_path, device, device_config);
    } else if (vlm_config.model_type == VLMModelType::GEMMA3) {
        m_impl = std::make_shared<InputsEmbedderGemma3>(vlm_config, models_map, tokenizer, config_dir_path, device, device_config);
    } else {
        OPENVINO_THROW("Unsupported model type in VLM InputsEmbedder class. Please, create feature request on new model support");
    }
}

InputsEmbedderImpl::InputsEmbedderImpl(const Tokenizer& tokenizer,
                               VisionEncoderImpl::Ptr vision_encoder_impl,
                               EmbeddingsModelImpl::Ptr embeddings_model_impl,
                               const std::filesystem::path& config_dir_path) {
    auto vlm_config = utils::from_config_json_if_exists<VLMConfig>(config_dir_path, "config.json");

    if (vlm_config.model_type == VLMModelType::MINICPM) {
        m_impl = std::make_shared<InputsEmbedderMiniCPM>(vlm_config, tokenizer, vision_encoder_impl, embeddings_model_impl);
    } else if (vlm_config.model_type == VLMModelType::LLAVA) {
        m_impl = std::make_shared<InputsEmbedderLLaVA>(vlm_config, tokenizer, vision_encoder_impl, embeddings_model_impl);
    } else if (vlm_config.model_type == VLMModelType::NANOLLAVA) {
        m_impl = std::make_shared<InputsEmbedderNanoLLaVA>(vlm_config, tokenizer, vision_encoder_impl, embeddings_model_impl);
    } else if (vlm_config.model_type == VLMModelType::LLAVA_NEXT) {
        m_impl = std::make_shared<InputsEmbedderLLaVANext>(vlm_config, tokenizer, vision_encoder_impl, embeddings_model_impl);
    } else if (vlm_config.model_type == VLMModelType::LLAVA_NEXT_VIDEO) {
        m_impl = std::make_shared<InputsEmbedderLLaVANextVideo>(vlm_config, tokenizer, vision_encoder_impl, embeddings_model_impl);
    } else if (vlm_config.model_type == VLMModelType::INTERNVL_CHAT) {
        m_impl = std::make_shared<InputsEmbedderInternVLChat>(vlm_config, tokenizer, vision_encoder_impl, embeddings_model_impl);
    } else if (vlm_config.model_type == VLMModelType::PHI3_V) {
        m_impl = std::make_shared<InputsEmbedderPhi3V>(vlm_config, tokenizer, vision_encoder_impl, embeddings_model_impl);
    } else if (vlm_config.model_type == VLMModelType::PHI4MM) {
        m_impl = std::make_shared<InputsEmbedderPhi4MM>(vlm_config, tokenizer, vision_encoder_impl, embeddings_model_impl);
    } else if (vlm_config.model_type == VLMModelType::QWEN2_VL) {
        m_impl = std::make_shared<InputsEmbedderQwen2VL>(vlm_config, tokenizer, vision_encoder_impl, embeddings_model_impl);
    } else if (vlm_config.model_type == VLMModelType::QWEN2_5_VL) {
        m_impl = std::make_shared<InputsEmbedderQwen2_5_VL>(vlm_config, tokenizer, vision_encoder_impl, embeddings_model_impl);
    } else if (vlm_config.model_type == VLMModelType::GEMMA3) {
        m_impl = std::make_shared<InputsEmbedderGemma3>(vlm_config, tokenizer, vision_encoder_impl, embeddings_model_impl);
    } else {
        OPENVINO_THROW("Unsupported model type in VLM InputsEmbedder class. Please, create feature request on new model support");
    }
}

ov::Tensor InputsEmbedderImpl::get_inputs_embeds(const std::string& prompt, const std::vector<ov::genai::EncodedImage>& images, ov::genai::VLMPerfMetrics& metrics, bool recalculate_merged_embeddings, const std::vector<size_t>& image_sequence) {
    return m_impl->get_inputs_embeds(prompt, images, metrics, recalculate_merged_embeddings, image_sequence);
}

ov::Tensor InputsEmbedderImpl::get_inputs_embeds(const std::string& prompt,
                                             const std::vector<ov::genai::EncodedImage>& images,
                                             const std::vector<ov::genai::EncodedVideo>& videos,
                                             ov::genai::VLMPerfMetrics& metrics,
                                             bool recalculate_merged_embeddings,
                                             const std::vector<size_t>& images_sequence,
                                             const std::vector<size_t>& videos_sequence,
                                             const std::vector<std::pair<std::size_t, std::size_t>>& history_vision_count) {
    return m_impl->get_inputs_embeds(prompt,
                                     images,
                                     videos,
                                     metrics,
                                     recalculate_merged_embeddings,
                                     images_sequence,
                                     videos_sequence,
                                     history_vision_count);
}

std::pair<ov::Tensor, ov::Tensor> InputsEmbedderImpl::get_inputs_embeds_with_token_type_ids(
    const std::string& prompt,
    const std::vector<EncodedImage>& images,
    VLMPerfMetrics& metrics,
    bool recalculate_merged_embeddings,
    const std::vector<size_t>& image_sequence) {
    return m_impl->get_inputs_embeds_with_token_type_ids(
        prompt, images, metrics, recalculate_merged_embeddings, image_sequence);
}

std::pair<ov::Tensor, ov::Tensor> InputsEmbedderImpl::get_inputs_embeds_with_token_type_ids(
    const std::string& prompt,
    const std::vector<EncodedImage>& images,
    const std::vector<EncodedVideo>& videos,
    VLMPerfMetrics& metrics,
    bool recalculate_merged_embeddings,
    const std::vector<size_t>& image_sequence,
    const std::vector<size_t>& videos_sequence,
    const std::vector<std::pair<std::size_t, std::size_t>>& history_vision_count) {
    return m_impl->get_inputs_embeds_with_token_type_ids(prompt,
                                                         images,
                                                         videos,
                                                         metrics,
                                                         recalculate_merged_embeddings,
                                                         image_sequence,
                                                         videos_sequence,
                                                         history_vision_count);
}

bool InputsEmbedderImpl::has_token_type_ids() const {
    return m_impl->has_token_type_ids();
}

std::vector<ov::genai::EncodedImage> InputsEmbedderImpl::encode_images(const std::vector<ov::Tensor>& images) {
    return m_impl->encode_images(images);
}

std::vector<ov::genai::EncodedVideo> InputsEmbedderImpl::encode_videos(const std::vector<ov::Tensor>& videos) {
    return m_impl->encode_videos(videos);
}

std::pair<ov::Tensor, std::optional<int64_t>> InputsEmbedderImpl::get_position_ids(const size_t inputs_embeds_size, const size_t history_size) {
    return m_impl->get_position_ids(inputs_embeds_size, history_size);
}

void InputsEmbedderImpl::set_position_ids(const ov::Tensor& position_ids) {
    m_impl->set_position_ids(position_ids);
}

void InputsEmbedderImpl::set_rope_delta(int64_t rope_delta) {
    m_impl->set_rope_delta(rope_delta);
}

std::pair<ov::Tensor, std::optional<int64_t>> InputsEmbedderImpl::get_generation_phase_position_ids(const size_t inputs_embeds_size, const size_t history_size, int64_t rope_delta) {
    return m_impl->get_generation_phase_position_ids(inputs_embeds_size, history_size, rope_delta);
}

EmbeddingsModelImpl::Ptr InputsEmbedderImpl::get_embedding_model() const {
    return m_impl->get_embedding_model();
}

ov::genai::utils::KVCacheState& InputsEmbedderImpl::get_kv_cache_state() {
    return  m_impl->get_kv_cache_state();
}

Tokenizer InputsEmbedderImpl::get_tokenizer() const {
    return m_impl->get_tokenizer();
}

void InputsEmbedderImpl::start_chat(const std::string& system_message) {
    return m_impl->start_chat(system_message);
}

void InputsEmbedderImpl::update_chat_history(const std::string& decoded_results, const ov::genai::GenerationStatus generation_finish_status) {
    return m_impl->update_chat_history(decoded_results, generation_finish_status);
}

void InputsEmbedderImpl::set_apply_chat_template_status(bool apply_chat_template) {
    return m_impl->set_apply_chat_template_status(apply_chat_template);
}

void InputsEmbedderImpl::finish_chat() {
    return m_impl->finish_chat();
}

NormalizedPrompt InputsEmbedderImpl::normalize_prompt(
    const std::string& prompt,
    size_t base_id,
    const std::vector<EncodedImage>& images
) const {
    auto norm_prompt = m_impl->normalize_prompt(prompt, base_id, 0, images, {});
    return {norm_prompt.unified_prompt, norm_prompt.images_sequence};
}

NormalizedPrompt InputsEmbedderImpl::normalize_prompt(const std::string& prompt,
    size_t base_image_id,
    size_t base_video_id,
    const std::vector<EncodedImage>& images,
    const std::vector<EncodedVideo>& videos
) const {
     return m_impl->normalize_prompt(prompt, base_image_id, base_video_id, images, videos);
}

void verify_ids(const std::vector<size_t>& image_ids, size_t base_id, size_t n_images) {
    for (size_t idx : image_ids) {
        OPENVINO_ASSERT(base_id <= idx, "Referring to older images isn't implemented");
        OPENVINO_ASSERT(idx < base_id + n_images, "Missing image ", idx);
    }
}

std::pair<std::string, std::vector<size_t>> InputsEmbedderImpl::IInputsEmbedder::normalize(
    const std::string& prompt,
    const std::string& native_tag,
    const std::string& automatic_tag,
    size_t base_id,
    size_t n_images
) const {
    size_t pos = prompt.find(native_tag);
    auto [image_prompt, image_sequence] = universal_to_native(prompt, [&](std::ostream& os, size_t) {
        os << automatic_tag;
    });
    if (!image_sequence.empty()) {
        OPENVINO_ASSERT(pos == std::string::npos, "Prompt can contain only one type of image tags.");
        verify_ids(image_sequence, base_id, n_images);
        return {std::move(image_prompt), std::move(image_sequence)};
    }
    // Restore ids from native tags
    while (pos != std::string::npos) {
        image_sequence.push_back(base_id + image_sequence.size());
        pos = prompt.find(native_tag, pos + native_tag.length());
    }
    if (!image_sequence.empty()) {
        OPENVINO_ASSERT(image_sequence.size() == n_images, "The number of native image tags and provided images must match because it's ambiguous which image should be ignored.");
        return {std::move(image_prompt), std::move(image_sequence)};
    }
    // Prepend automatic tags
    std::stringstream stream;
    for (size_t relative_id = 0; relative_id < n_images; relative_id++) {
        image_sequence.push_back(base_id + relative_id);
        stream << automatic_tag;
    }
    stream << prompt;
    return {stream.str(), std::move(image_sequence)};
}

// ======================== InputsEmbedder (public API) ========================

/// @brief The implementation class that wraps the internal InputsEmbedderImpl for the public API.
class InputsEmbedder::InputsEmbedderImplWrapper {
public:
    InputsEmbedderImplWrapper(
        const std::filesystem::path& model_dir,
        const std::string& device,
        const ov::AnyMap& device_config)
        : m_internal_impl(std::make_shared<ov::genai::InputsEmbedderImpl>(model_dir, device, device_config)),
          m_model_dir(model_dir),
          m_device(device),
          m_device_config(device_config) {}

    InputsEmbedderImplWrapper(
        const Tokenizer& tokenizer,
        const std::filesystem::path& config_dir_path)
        : m_tokenizer_ptr(std::make_shared<Tokenizer>(tokenizer)),
          m_config_dir_path(config_dir_path),
          m_use_external_components(true)
    {
        // Read the VLM config to get model type information
        m_vlm_config = utils::from_config_json_if_exists<VLMConfig>(config_dir_path, "config.json");
    }

    InputsEmbedderImplWrapper(
        const Tokenizer& tokenizer,
        VisionEncoder::Ptr vision_encoder,
        EmbeddingsModel::Ptr embeddings_model,
        const std::filesystem::path& config_dir_path)
        : m_tokenizer_ptr(std::make_shared<Tokenizer>(tokenizer)),
          m_vision_encoder_ptr(vision_encoder),
          m_embeddings_model_ptr(embeddings_model),
          m_config_dir_path(config_dir_path),
          m_use_external_components(true)
    {
        // Read the VLM config to get model type information
        m_vlm_config = utils::from_config_json_if_exists<VLMConfig>(config_dir_path, "config.json");
        
        // Create internal impl using the pre-loaded components
        // This allows the InputsEmbedder to work with VLMPipeline
        m_internal_impl = std::make_shared<ov::genai::InputsEmbedderImpl>(
            tokenizer,
            vision_encoder->get_internal_impl(),
            embeddings_model->get_internal_impl(),
            config_dir_path
        );
    }

    std::vector<EncodedImage> encode_images(const std::vector<ov::Tensor>& images) {
        // Use internal impl for encoding to ensure consistency
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

    /// @brief Get the internal InputsEmbedderImpl for use by friend classes.
    /// @note This is only accessible by friend classes (e.g., VLMPipeline).
    std::shared_ptr<ov::genai::InputsEmbedderImpl> get_internal_impl() const {
        return m_internal_impl;
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
    : m_pimpl(std::make_unique<InputsEmbedderImplWrapper>(model_dir, device, device_config)) {}

InputsEmbedder::InputsEmbedder(
    const Tokenizer& tokenizer,
    const std::filesystem::path& config_dir_path)
    : m_pimpl(std::make_unique<InputsEmbedderImplWrapper>(tokenizer, config_dir_path)) {}

InputsEmbedder::InputsEmbedder(
    const Tokenizer& tokenizer,
    VisionEncoder::Ptr vision_encoder,
    EmbeddingsModel::Ptr embeddings_model,
    const std::filesystem::path& config_dir_path)
    : m_pimpl(std::make_unique<InputsEmbedderImplWrapper>(tokenizer, vision_encoder, embeddings_model, config_dir_path)) {}

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

std::shared_ptr<InputsEmbedderImpl> InputsEmbedder::get_internal_impl() const {
    return m_pimpl->get_internal_impl();
}

} // namespace ov::genai
