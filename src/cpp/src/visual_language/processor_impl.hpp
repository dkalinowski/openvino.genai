// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <memory>

#include "openvino/genai/tokenizer.hpp"
#include "openvino/genai/visual_language/processor.hpp"
#include "visual_language/inputs_embedder.hpp"
#include "visual_language/vision_registry.hpp"

namespace ov::genai {

class VLMProcessor::Impl {
public:
    std::shared_ptr<InputsEmbedder> inputs_embedder;
    Tokenizer tokenizer;
    std::shared_ptr<VisionRegistry> vision_registry;
};

}  // namespace ov::genai
