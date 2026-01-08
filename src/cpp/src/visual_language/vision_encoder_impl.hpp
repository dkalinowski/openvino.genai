// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <memory>
#include "openvino/runtime/infer_request.hpp"

#include "openvino/genai/common_types.hpp"
#include "openvino/genai/visual_language/vision_encoder.hpp"
#include "visual_language/vlm_config.hpp"
#include "visual_language/processor_config.hpp"
#include "circular_buffer_queue.hpp"


namespace ov::genai {

/// @brief Embeddings of a given video. 
struct EncodedVideo {
    /// @brief Embeddings of a given video obtained by applying preprocessing to frames and feature extracting models (resampler, mm_projector, etc.)
    ov::Tensor video_features;

    /// @brief Number of video tokens required to append to a normalized prompt
    size_t num_video_tokens;

    /// @brief A size of an image used to compute embeddings for
    /// divided by ProcessorConfig's patch_size.
    ImageSize resized_source_size;

    /// @brief A number of encoded frames.
    size_t frame_num;
};

/// @brief An internal class used to infer embeddings of an image using
/// ov::InferRequest and configured by ProcessorConfig.
/// This is the implementation class used internally by VLM implementations.
class VisionEncoderImpl {
public:
    using Ptr = std::shared_ptr<VisionEncoderImpl>;

    /// @brief Constructs the encoder from model_dir.
    /// @param model_dir A folder containing openvino_vision_embeddings_model.xml and
    /// preprocessor_config.json.
    /// @param model_type A type of VLM model.
    /// @param device A device to compile the encoder for.
    /// @param properties A config to be passed to
    /// ov::Core::compile_model().
    static VisionEncoderImpl::Ptr create(
        const std::filesystem::path& model_dir,
        const VLMModelType model_type,
        const std::string& device,
        const ov::AnyMap properties = {});

    /// @brief Constructs the encoder from models map.
    /// @param models_map Models map
    /// @param config_dir_path A path to directory containing preprocessor_config.json.
    /// @param model_type A type of VLM model.
    /// @param device A device to compile the encoder for.
    /// @param properties A config to be passed to
    /// ov::Core::compile_model().
    static VisionEncoderImpl::Ptr create(
        const ModelsMap& models_map,
        const std::filesystem::path& config_dir_path,
        const VLMModelType model_type,
        const std::string& device,
        const ov::AnyMap properties = {});

    /// @brief Compute embeddings of an image given
    /// ProcessorConfig members.
    /// @param image An image to infer embeddings for. Image shape must be
    /// [1CHW]. Only batch 1 is supported.
    /// @param config_map A config or its members values to follow
    /// instead of the config obtained in constructors.
    /// @return Resulting embeddings for the resized source image and
    /// its slices.
    virtual EncodedImage encode(const ov::Tensor& image, const ov::AnyMap& config_map = {}) = 0;

    /// @brief Compute embeddings of a or multiple video given
    virtual EncodedVideo encode_frames(const std::vector<ov::Tensor>& frames, const ov::AnyMap& config_map = {}) {
        OPENVINO_THROW("The current model does not support 'video' input, please use 'images' instead.");
    }
    /// @brief Reshape the vision encoder model if needed (dynamic shapes)
    /// @param model The model to reshape. By default does nothing.
    virtual void reshape(std::shared_ptr<ov::Model> model) {};


    /// @brief Gets processor config
    /// @return Processor config
    ProcessorConfig get_processor_config() const;

    virtual ~VisionEncoderImpl() = default;

protected:
    /// @brief  Infer requests queue for image encoding model.
    std::unique_ptr<CircularBufferQueue<ov::InferRequest>> m_ireq_queue_vision_encoder;

    /// @brief A config to follow.
    ProcessorConfig m_processor_config;

    /// @brief Protected default constructor for derived classes that need custom initialization
    VisionEncoderImpl() = default;

public:
    VisionEncoderImpl(
        const std::filesystem::path& model_dir,
        const std::string& device,
        const ov::AnyMap properties);

    VisionEncoderImpl(
        const ModelsMap& models_map,
        const std::filesystem::path& config_dir_path,
        const std::string& device,
        const ov::AnyMap properties);
};

} // namespace ov::genai
