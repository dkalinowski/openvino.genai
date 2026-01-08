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

/// @brief A class used to compute text embeddings using a text embeddings model.
/// This is the public API class that uses PIMPL pattern to hide implementation details.
class OPENVINO_GENAI_EXPORTS EmbeddingsModel {
public:
    /// @brief Constructs the embeddings model from model_dir.
    /// @param model_dir A folder containing openvino_text_embeddings_model.xml.
    /// @param device A device to compile the model for.
    /// @param properties A config to be passed to ov::Core::compile_model().
    EmbeddingsModel(
        const std::filesystem::path& model_dir,
        const std::string& device,
        const ov::AnyMap& properties = {});

    /// @brief Constructs the embeddings model from model string and weights.
    /// @param model Model IR as string.
    /// @param weights Model weights as tensor.
    /// @param device A device to compile the model for.
    /// @param properties A config to be passed to ov::Core::compile_model().
    EmbeddingsModel(
        const std::string& model,
        const ov::Tensor& weights,
        const std::string& device,
        const ov::AnyMap& properties = {});

    /// @brief Constructs the embeddings model with additional variadic properties.
    /// @param model_dir A folder containing openvino_text_embeddings_model.xml.
    /// @param device A device to compile the model for.
    /// @param properties Variadic properties to pass to ov::Core::compile_model().
    template <typename... Properties, typename std::enable_if<ov::util::StringAny<Properties...>::value, bool>::type = true>
    EmbeddingsModel(
        const std::filesystem::path& model_dir,
        const std::string& device,
        Properties&&... properties)
        : EmbeddingsModel(model_dir, device, ov::AnyMap{std::forward<Properties>(properties)...}) { }

    /// @brief Default destructor.
    ~EmbeddingsModel();

    /// @brief Move constructor.
    EmbeddingsModel(EmbeddingsModel&& other) noexcept;

    /// @brief Move assignment operator.
    EmbeddingsModel& operator=(EmbeddingsModel&& other) noexcept;

    /// @brief Compute embeddings for the given input token IDs.
    /// @param input_ids Input token IDs tensor with shape [batch_size, sequence_length].
    /// @return Embeddings tensor with shape [batch_size, sequence_length, hidden_size].
    ov::Tensor infer(const ov::Tensor& input_ids);

    /// @brief Compute embeddings for the given input token IDs with variadic config.
    /// @param input_ids Input token IDs tensor.
    /// @param properties Variadic properties for inference configuration.
    /// @return Embeddings tensor.
    template <typename... Properties>
    util::EnableIfAllStringAny<ov::Tensor, Properties...> infer(
        const ov::Tensor& input_ids,
        Properties&&... properties
    ) {
        return infer(input_ids);
    }

private:
    class EmbeddingsModelImpl;
    std::unique_ptr<EmbeddingsModelImpl> m_pimpl;
};

} // namespace ov::genai
