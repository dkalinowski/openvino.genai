// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <memory>
#include <string>
#include <filesystem>

#include "openvino/genai/visibility.hpp"
#include "openvino/genai/common_types.hpp"
#include "openvino/runtime/tensor.hpp"

namespace ov::genai {

/// @brief A pair describing image size.
struct OPENVINO_GENAI_EXPORTS ImageSize {
    /// @brief Height of a corresponding image.
    size_t height = 0;
    /// @brief Width of a corresponding image.
    size_t width = 0;
};

/// @brief Resampled image data structure (internal use).
struct OPENVINO_GENAI_EXPORTS ResampledImage {
    ov::Tensor resampled_source;
    std::vector<std::vector<ov::Tensor>> vision_embed_tensors;
};

/// @brief Embeddings of a given image. The number of slices is no
/// greater than ProcessorConfig's max_slice_nums.
struct OPENVINO_GENAI_EXPORTS EncodedImage {
    /// @brief Embeddings of a resized image based on ProcessorConfig's
    /// scale_resolution. The tensor's shape is
    /// [N, H*W, hidden_size]. [N, 1014, 1152] is a possible example for
    /// openbmb/MiniCPM-V-2. Only batch 1 is supported.
    ov::Tensor resized_source;
    /// @brief A size of an image used to compute embeddings for
    /// divided by ProcessorConfig's patch_size.
    ImageSize resized_source_size;

    /// @brief Shape of embeddings of images obtained from a source image by slicing 
    /// at no more than max_slice_nums pieces and resizing,
    /// This shape is [slice_y, slice_x, number_of_embeddings, embedding_size].
    /// Used only by MiniCPM
    ov::Shape slices_shape;

    /// @brief Patches grid after llava_next preprocessing.
    /// Format: [num_patches_height, num_patches_width]
    std::pair<int, int> patches_grid;
    
    /// @brief Original size of the image
    ImageSize original_image_size;

    /// @brief Images features projection, used only by Phi3V and Phi4MM.
    /// Internal use only.
    ov::Tensor images_features_projection;
  
    /// @brief Resampled image, used only by MiniCPM.
    /// Internal use only.
    ResampledImage resampled_image;

    /// @brief Number of image tokens required to append to a normalized prompt
    size_t num_image_tokens = 0;
};

/// @brief A class used to infer embeddings of an image using
/// ov::InferRequest. This is the public API class that uses PIMPL
/// pattern to hide implementation details.
class OPENVINO_GENAI_EXPORTS VisionEncoder {
public:
    /// @brief Constructs the encoder from model_dir.
    /// @param model_dir A folder containing openvino_vision_embeddings_model.xml and
    /// preprocessor_config.json.
    /// @param device A device to compile the encoder for.
    /// @param properties A config to be passed to
    /// ov::Core::compile_model().
    VisionEncoder(
        const std::filesystem::path& model_dir,
        const std::string& device,
        const ov::AnyMap& properties = {});

    /// @brief Constructs the encoder from models map.
    /// @param models_map Models map
    /// @param config_dir_path A path to directory containing preprocessor_config.json.
    /// @param device A device to compile the encoder for.
    /// @param properties A config to be passed to
    /// ov::Core::compile_model().
    VisionEncoder(
        const ModelsMap& models_map,
        const std::filesystem::path& config_dir_path,
        const std::string& device,
        const ov::AnyMap& properties = {});

    /// @brief Constructs the encoder with additional variadic properties.
    /// @param model_dir A folder containing openvino_vision_embeddings_model.xml and
    /// preprocessor_config.json.
    /// @param device A device to compile the encoder for.
    /// @param properties Variadic properties to pass to ov::Core::compile_model().
    template <typename... Properties, typename std::enable_if<ov::util::StringAny<Properties...>::value, bool>::type = true>
    VisionEncoder(
        const std::filesystem::path& model_dir,
        const std::string& device,
        Properties&&... properties)
        : VisionEncoder(model_dir, device, ov::AnyMap{std::forward<Properties>(properties)...}) { }

    /// @brief Default destructor.
    ~VisionEncoder();

    /// @brief Move constructor.
    VisionEncoder(VisionEncoder&& other) noexcept;

    /// @brief Move assignment operator.
    VisionEncoder& operator=(VisionEncoder&& other) noexcept;

    /// @brief Compute embeddings of an image.
    /// @param image An image to infer embeddings for. Image shape must be
    /// [1CHW] or [CHW]. Only batch 1 is supported.
    /// @param config_map A config or its members values to follow
    /// instead of the config obtained in constructors.
    /// @return Resulting embeddings for the resized source image.
    EncodedImage encode(const ov::Tensor& image, const ov::AnyMap& config_map = {});

    /// @brief Compute embeddings of an image with variadic config.
    /// @param image An image to infer embeddings for. Image shape must be
    /// [1CHW] or [CHW]. Only batch 1 is supported.
    /// @param properties Variadic properties for encoding configuration.
    /// @return Resulting embeddings for the resized source image.
    template <typename... Properties>
    util::EnableIfAllStringAny<EncodedImage, Properties...> encode(
        const ov::Tensor& image,
        Properties&&... properties
    ) {
        return encode(image, ov::AnyMap{std::forward<Properties>(properties)...});
    }

private:
    class VisionEncoderImpl;
    std::unique_ptr<VisionEncoderImpl> m_pimpl;
};

} // namespace ov::genai
