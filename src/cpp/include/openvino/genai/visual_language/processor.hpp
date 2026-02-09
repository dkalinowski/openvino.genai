// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <filesystem>
#include <memory>
#include <string>
#include <vector>

#include "openvino/runtime/tensor.hpp"
#include "openvino/genai/tokenizer.hpp"
#include "openvino/genai/visibility.hpp"
#include "openvino/genai/common_types.hpp"
#include "openvino/genai/visual_language/vlm_inputs.hpp"

namespace ov::genai {

/// @brief A processor for Visual Language Models that handles vision encoding,
/// text tokenization, embedding merging, and chat template application.
///
/// Separates the preprocessing (vision + text → merged embeddings) from the
/// language model generation. The VLMProcessor owns the vision encoder, text
/// embeddings model, and tokenizer. It produces VLMInputs that can be passed
/// to VLMPipeline::generate() or ContinuousBatchingPipeline::add_request().
///
/// This is analogous to HuggingFace's AutoProcessor, which combines an
/// ImageProcessor and a Tokenizer into a single preprocessing pipeline.
///
/// Usage:
/// @code
///   auto processor = VLMProcessor("path/to/models", "GPU");
///   auto inputs = processor.prepare("Describe this image", {image_tensor});
///   // inputs can be inspected, modified, or passed to an LLM pipeline
///   auto result = llm.generate(inputs, generation_config);
/// @endcode
class OPENVINO_GENAI_EXPORTS VLMProcessor {
public:
    /// @brief Construct a processor from a directory containing vision encoder,
    /// text embeddings model, tokenizer, and configuration files.
    /// @param models_path Directory with openvino_vision_embeddings_model.xml,
    ///        openvino_text_embeddings_model.xml, tokenizer files, config.json,
    ///        and preprocessor_config.json.
    /// @param device Inference device for vision encoder and embeddings model.
    /// @param properties Device configuration properties.
    VLMProcessor(
        const std::filesystem::path& models_path,
        const std::string& device,
        const ov::AnyMap& properties = {}
    );

    /// @brief Construct from pre-loaded models.
    /// @param models_map Map of model name to (IR string, weights tensor) pairs.
    ///        Expected keys: "vision_embeddings", "text_embeddings",
    ///        and optionally "resampler".
    /// @param tokenizer Pre-initialized tokenizer.
    /// @param config_dir_path Path to directory containing config.json.
    /// @param device Inference device.
    /// @param properties Device configuration properties.
    VLMProcessor(
        const ModelsMap& models_map,
        const Tokenizer& tokenizer,
        const std::filesystem::path& config_dir_path,
        const std::string& device,
        const ov::AnyMap& properties = {}
    );

    /// @brief Variadic property constructor.
    template <typename... Properties,
              typename std::enable_if<ov::util::StringAny<Properties...>::value, bool>::type = true>
    VLMProcessor(
        const std::filesystem::path& models_path,
        const std::string& device,
        Properties&&... properties)
        : VLMProcessor(models_path, device, ov::AnyMap{std::forward<Properties>(properties)...}) { }

    ~VLMProcessor();

    // Non-copyable, movable
    VLMProcessor(const VLMProcessor&) = delete;
    VLMProcessor& operator=(const VLMProcessor&) = delete;
    VLMProcessor(VLMProcessor&&) noexcept;
    VLMProcessor& operator=(VLMProcessor&&) noexcept;

    /// @brief Prepare inputs for VLM generation from a text prompt
    /// and optional images/videos.
    ///
    /// Performs the full preprocessing pipeline:
    /// 1. Normalizes the prompt (replaces universal image/video tags with model-native tags).
    /// 2. Optionally applies the model's chat template.
    /// 3. Encodes images/videos through the vision encoder.
    /// 4. Tokenizes the text and computes text token embeddings.
    /// 5. Merges projected vision features into the text embedding
    ///    sequence at placeholder positions.
    /// 6. Computes attention mask and position IDs.
    ///
    /// For image/video tag usage, see:
    /// https://openvinotoolkit.github.io/openvino.genai/docs/use-cases/image-processing/
    ///
    /// @param prompt Text prompt, optionally containing image/video placeholder tags.
    /// @param images RGB image tensors with [NHWC] or [HWC] layout.
    /// @param videos Video frame tensors with [NHWC] layout.
    /// @return VLMInputs ready to pass to VLMPipeline::generate()
    ///         or ContinuousBatchingPipeline::add_request().
    VLMInputs prepare(
        const std::string& prompt,
        const std::vector<ov::Tensor>& images = {},
        const std::vector<ov::Tensor>& videos = {}
    );

    /// @brief Apply the model's chat template to a prompt string.
    /// Useful when chat template control is needed separately from prepare().
    ///
    /// @param prompt The raw user message.
    /// @param system_message Optional system prompt prepended to the conversation.
    /// @param add_generation_prompt If true, appends assistant turn-start tokens.
    /// @return Formatted prompt string with model-specific special tokens applied.
    std::string apply_chat_template(
        const std::string& prompt,
        const std::string& system_message = "",
        bool add_generation_prompt = true
    ) const;

    /// @brief Extract vision encoder embeddings without merging with text.
    /// Returns the raw vision encoder outputs (before any projection/resampling
    /// that happens during the merge step in prepare()).
    /// Useful for embedding-only workflows (retrieval, caching, similarity)
    /// that do not require LLM generation.
    ///
    /// @param images RGB image tensors.
    /// @return Vector of embedding tensors, one per image.
    ///         Shape per tensor depends on the model architecture
    ///         (typically [N, H*W, hidden_size]).
    std::vector<ov::Tensor> get_vision_embeddings(
        const std::vector<ov::Tensor>& images
    );

    /// @brief Extract projected video embeddings without merging with text.
    /// @param videos Video frame tensors.
    /// @return Vector of embedding tensors, one per video.
    std::vector<ov::Tensor> get_video_embeddings(
        const std::vector<ov::Tensor>& videos
    );

    /// @brief Get the underlying tokenizer.
    Tokenizer get_tokenizer() const;

    /// @brief Enable or disable automatic chat template application in prepare().
    /// @param apply If false, prepare() will pass the prompt through without
    ///        wrapping it in a chat template. Default is true.
    void set_apply_chat_template(bool apply);

    /// @brief Override the default chat template with a custom one.
    /// @param new_template Jinja2-style template string.
    void set_chat_template(const std::string& new_template);

private:
    class Impl;
    std::unique_ptr<Impl> m_impl;
};

} // namespace ov::genai
