// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "openvino/genai/visual_language/processor.hpp"

#include <tuple>

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
    m_impl->vision_registry = std::make_shared<VisionRegistry>();
}

VLMProcessor::~VLMProcessor() = default;

namespace {
// Core embed implementation. `populate_prompt_ids` enables the
// KV-reuse opt-in: it snapshots the tokens the embedder built
// `inputs_embeds` from (including vision placeholder tokens at merge
// positions) into `result.prompt_ids`, keeping the two seq-aligned.
Embeddings embed_impl(
    const std::shared_ptr<InputsEmbedder>& inputs_embedder,
    const std::string& prompt,
    const std::vector<ov::Tensor>& images,
    const std::vector<ov::Tensor>& videos,
    bool populate_prompt_ids
) {
    VLMPerfMetrics perf_metrics;
    // The shared InputsEmbedder tracks chat-mode cache history inside
    // get_encoded_input_ids(). VLMProcessor::embed() is supposed to be
    // side-effect-free, so clear any leftover history both before and
    // after the call; otherwise repeated embed() calls (or a subsequent
    // string-API generate() on a pipeline built from this processor)
    // would see stale tokens in the embedder's cache state.
    inputs_embedder->get_cache_state().reset_state();
    const auto encoded_images = inputs_embedder->encode_images(images);
    const auto encoded_videos = inputs_embedder->encode_videos(videos);
    const auto normalized = inputs_embedder->normalize_prompt(
        prompt, 0, 0, encoded_images, encoded_videos);

    Embeddings result;
    if (inputs_embedder->has_token_type_ids()) {
        ov::Tensor token_type_ids;
        std::tie(result.inputs_embeds, token_type_ids) =
            inputs_embedder->get_inputs_embeds_with_token_type_ids(
                normalized.unified_prompt,
                encoded_images,
                encoded_videos,
                perf_metrics,
                /*recalculate_merged_embeddings=*/true,
                normalized.images_sequence,
                normalized.videos_sequence);
        result.token_type_ids = token_type_ids;
    } else {
        result.inputs_embeds = inputs_embedder->get_inputs_embeds(
            normalized.unified_prompt,
            encoded_images,
            encoded_videos,
            perf_metrics,
            /*recalculate_merged_embeddings=*/true,
            normalized.images_sequence,
            normalized.videos_sequence);
    }

    // The embedder still stores per-prompt scratch (position_ids, extras)
    // as mutable members that get reassigned on every embed() call.
    // ov::Tensor is a ref-counted handle, so taking a value copy here keeps
    // the Embeddings independent: a subsequent embed() reassigns the
    // embedder's members without touching this instance's storage.
    const size_t inputs_embeds_size = result.inputs_embeds.get_shape().at(1);
    auto [position_ids, rope_delta] = inputs_embedder->get_position_ids(
        inputs_embeds_size, /*history_size=*/0);
    result.position_ids = position_ids;
    result.rope_delta = rope_delta;
    result.lm_extra_inputs = inputs_embedder->get_lm_extra_inputs();

    if (populate_prompt_ids) {
        // Snapshot the token IDs the embedder used to build
        // `inputs_embeds`. IInputsEmbedder::get_encoded_input_ids() stores
        // them in its cache state as a side effect of tokenisation + chat
        // template application; they include vision placeholder tokens at
        // merge positions, so the result is seq-aligned with
        // `inputs_embeds`. VLMPipeline uses them to diff against its own
        // KV-cache history and prefill only the delta on subsequent
        // generate() calls (multi-turn chat KV-reuse).
        const auto& cached_tokens = inputs_embedder->get_cache_state().get_state();
        OPENVINO_ASSERT(cached_tokens.size() == inputs_embeds_size,
            "Tokenised prompt length does not match inputs_embeds sequence length; "
            "embedder and tokenizer fell out of sync.");
        result.prompt_ids = ov::Tensor{ov::element::i64, {1, cached_tokens.size()}};
        std::copy(cached_tokens.begin(), cached_tokens.end(), result.prompt_ids.data<int64_t>());
    }

    // Restore the embedder to a clean, side-effect-free state.
    inputs_embedder->get_cache_state().reset_state();
    return result;
}
}  // namespace

Embeddings VLMProcessor::embed(
    const std::string& prompt,
    const std::vector<ov::Tensor>& images,
    const std::vector<ov::Tensor>& videos) {
    // String overload is stateless: it does not opt into KV-reuse
    // because two unrelated string prompts may share identical vision
    // placeholder tokens while carrying different image embeddings.
    // The pipeline will full-reset its KV cache for the resulting
    // Embeddings (no prompt_ids).
    return embed_impl(m_impl->inputs_embedder, prompt, images, videos,
                      /*populate_prompt_ids=*/false);
}

Embeddings VLMProcessor::embed(
    const ChatHistory& history,
    const std::vector<ov::Tensor>& images,
    const std::vector<ov::Tensor>& videos) {
    const std::string prompt = m_impl->tokenizer.apply_chat_template(history, /*add_generation_prompt=*/true);
    return embed_impl(m_impl->inputs_embedder, prompt, images, videos,
                      /*populate_prompt_ids=*/true);
}

Tokenizer VLMProcessor::get_tokenizer() const {
    return m_impl->tokenizer;
}

}  // namespace ov::genai
