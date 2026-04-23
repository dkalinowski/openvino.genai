// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <filesystem>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#include "openvino/genai/chat_history.hpp"
#include "openvino/genai/tokenizer.hpp"
#include "openvino/genai/visibility.hpp"
#include "openvino/runtime/properties.hpp"
#include "openvino/runtime/tensor.hpp"

namespace ov::genai {

/// @brief Structured inputs for VLM generation produced by VLMProcessor.
/// Holds the merged text + vision embeddings and everything else the
/// language model needs for the prompt step, ready to be consumed by
/// VLMPipeline::generate().
///
/// Embeddings are self-contained and immutable with respect to the
/// processor: once returned from VLMProcessor::embed(), they do not
/// share mutable state with the processor and can be safely passed to
/// multiple pipelines or used across multiple generate() calls.
///
/// All per-token fields (`inputs_embeds`, `prompt_ids`, `position_ids`,
/// `token_type_ids`, and any seq-aligned entries in `lm_extra_inputs`)
/// are guaranteed to be seq-aligned along the sequence axis, so the
/// pipeline can slice them uniformly when prefilling only a suffix of
/// the prompt for KV-cache reuse.
struct OPENVINO_GENAI_EXPORTS Embeddings {
    /// Merged embeddings of text tokens and vision features.
    /// Shape: [1, sequence_length, hidden_size].
    ov::Tensor inputs_embeds;

    /// Token IDs of the tokenised prompt, including vision placeholder
    /// tokens at every position where vision features were merged into
    /// `inputs_embeds`. Shape: [1, sequence_length].
    ///
    /// Populated only by the `embed(ChatHistory, ...)` overload.
    /// The `embed(std::string, ...)` overload leaves this tensor default
    /// constructed: that path is stateless and the pipeline must
    /// full-reset its KV cache for it, since two independent string
    /// prompts may share identical vision placeholder tokens while
    /// carrying different image embeddings.
    ///
    /// When populated, lets VLMPipeline compare the prompt against its
    /// KV-cache history (longest common prefix) and prefill only the
    /// delta on subsequent `generate()` calls \u2014 the "image once,
    /// follow-ups as text" chat pattern.
    ov::Tensor prompt_ids;

    /// Position ids for the prompt. May be empty if the model does
    /// not use explicit position ids.
    ov::Tensor position_ids;

    /// Optional rope delta used by some VLMs (e.g. Qwen2/3-VL).
    std::optional<int64_t> rope_delta;

    /// Optional token type ids for models that need them (e.g. Gemma3).
    std::optional<ov::Tensor> token_type_ids;

    /// Extra named inputs to be forwarded to the language model
    /// alongside `inputs_embeds` (e.g. `deepstack_visual_embeds`,
    /// `visual_pos_masks` for Qwen3-VL).
    std::unordered_map<std::string, ov::Tensor> lm_extra_inputs;
};

/// @brief A processor for Visual Language Models that handles vision
/// encoding, text tokenization, and embedding merging.
///
/// Analogous to transformers's AutoProcessor — combines an image
/// processor (VisionEncoder) with a tokenizer/embedder to produce
/// Embeddings that a VLMPipeline can consume.
class OPENVINO_GENAI_EXPORTS VLMProcessor {
public:
    /// @brief Construct a processor from a directory with vision encoder,
    /// text embeddings model, tokenizer, and configs.
    /// @param models_path Directory with the exported model files.
    /// @param device Inference device for vision encoder and embeddings model.
    /// @param properties Device configuration properties.
    VLMProcessor(
        const std::filesystem::path& models_path,
        const std::string& device,
        const ov::AnyMap& properties = {}
    );

    ~VLMProcessor();

    /// @brief Prepare Embeddings for VLM generation from a text prompt
    /// and optional images / videos.
    /// @param prompt Pre-formatted text prompt. Chat template is not
    ///               applied here.
    /// @param images RGB image tensors.
    /// @param videos Video frame tensors.
    Embeddings embed(
        const std::string& prompt,
        const std::vector<ov::Tensor>& images = {},
        const std::vector<ov::Tensor>& videos = {}
    );

    /// @brief Prepare Embeddings from a chat history.
    /// Applies the chat template via the internal tokenizer and then
    /// performs the same vision encoding + embedding merging as the
    /// string overload. Image and video tags inside message contents
    /// (e.g. "<ov_genai_image_0>") are resolved against the supplied
    /// images and videos.
    ///
    /// The call is stateless: the processor does not track previous
    /// turns and always returns Embeddings for the full history. The
    /// pipeline that consumes the Embeddings is responsible for KV-cache
    /// reuse \u2014 it matches `Embeddings::prompt_ids` against its own
    /// token-ID history and prefills only the delta on each call.
    ///
    /// For a typical multi-turn chat, supply `images` / `videos` only
    /// on the turn where each item is first referenced; later text-only
    /// turns reuse the vision tokens already in KV.
    /// @param history Chat history with messages.
    /// @param images RGB image tensors.
    /// @param videos Video frame tensors.
    Embeddings embed(
        const ChatHistory& history,
        const std::vector<ov::Tensor>& images = {},
        const std::vector<ov::Tensor>& videos = {}
    );

    /// @brief Get the underlying tokenizer.
    Tokenizer get_tokenizer() const;

private:
    class Impl;
    std::shared_ptr<Impl> m_impl;
    friend class VLMPipeline;
};

}  // namespace ov::genai
