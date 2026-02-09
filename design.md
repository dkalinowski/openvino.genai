## Design Rationale
Why the current design needs changing?  
The current VLMPipeline is monolithic — it owns the vision encoder, text embeddings, the tokenizer, and the LLM. The InputsEmbedder (which is the de facto "processor") is an internal class hidden in src, not exposed to users. This causes several problems:

* No embedding reuse — you can't get vision embeddings without running the LLM.
* Tight coupling — `ContinuousBatchingPipeline` duplicates VLM logic via `VLMContinuousBatchingAdapter` instead of sharing a common Processor.
* It is not possible to define target device and plugin config separately for the processor models and the LLM model. (**CVS-162621**)
* Combinatorial overload explosion — `VLMPipeline::generate()` has ~12 overloads for every combination of (string/ChatHistory) × (image/images/videos/none) × (config/properties).
* Not aligned with other libraries, for example HF — HuggingFace's pattern is Processor + Model, which allows users to inspect/modify embeddings between the two steps.
HuggingFace alignment
HF's canonical VLM flow:
```
processor = AutoProcessor.from_pretrained("model_id")
model = AutoModelForVision2Seq.from_pretrained("model_id")

inputs = processor(images=images, text=prompt, return_tensors="pt")  
# → BatchFeature { input_ids, attention_mask, pixel_values } 
# OR after internal vision encoding:
# → { inputs_embeds, attention_mask, position_ids }

output_ids = model.generate(**inputs)
text = processor.batch_decode(output_ids)
```

The key insight: the Processor produces a structured input; the model consumes it. The Processor can also be used standalone (e.g., get_image_features()).

## Proposed API (C++ Headers)
### 1. VLMInputs — The structured intermediate
```cpp
// openvino/genai/visual_language/vlm_inputs.hpp

#pragma once
#include <optional>
#include <vector>
#include "openvino/runtime/tensor.hpp"

namespace ov::genai {

/// @brief Structured inputs for VLM generation, produced by VLMProcessor.
/// Analogous to HuggingFace's BatchFeature returned by a Processor.
struct VLMInputs {
    /// @brief Merged embeddings of text tokens and vision features.
    /// Shape: [1, sequence_length, hidden_size].
    /// The vision encoder outputs have already been projected into
    /// the LLM's embedding space and merged at <image>/<video> positions.
    ov::Tensor inputs_embeds;

    /// @brief Attention mask for the merged sequence.
    /// Shape: [1, sequence_length]. 1 = attend, 0 = ignore.
    ov::Tensor attention_mask;

    /// @brief Position IDs for models that require explicit positioning
    /// (e.g., Qwen2-VL uses 3D MROPE position IDs).
    /// Shape varies by model. May be empty for models using default positions.
    std::optional<ov::Tensor> position_ids;

    /// @brief Optional token type IDs (used by some models like Phi-4-MM).
    /// Shape: [1, sequence_length].
    std::optional<ov::Tensor> token_type_ids;

    /// @brief Whether the inputs contain any vision content.
    bool has_vision_content() const {
        return !inputs_embeds.get_shape().empty();
    }
};

} // namespace ov::genai
```

### 2. VLMProcessor — The public processor class
```cpp
// openvino/genai/visual_language/processor.hpp

#pragma once
#include <filesystem>
#include <string>
#include <vector>
#include "openvino/runtime/tensor.hpp"
#include "openvino/genai/tokenizer.hpp"
#include "openvino/genai/visual_language/vlm_inputs.hpp"
#include "openvino/genai/visual_language/perf_metrics.hpp"
#include "openvino/genai/common_types.hpp"

namespace ov::genai {

/// @brief A processor for Visual Language Models that handles
/// vision encoding, text tokenization, embedding merging, and
/// chat template application.
///
/// Analogous to HuggingFace's AutoProcessor — combines an image
/// processor (VisionEncoder) and a tokenizer/embedder into a
/// single preprocessing pipeline.
///
/// Usage:
///   auto processor = VLMProcessor("path/to/models", "GPU");
///   auto inputs = processor.prepare("Describe this image", {image_tensor});
///   auto result = llm.generate(inputs, generation_config);
class OPENVINO_GENAI_EXPORTS VLMProcessor {
public:
    /// @brief Construct a processor from a directory containing
    /// vision encoder, text embeddings model, tokenizer, and configs.
    /// @param models_path Directory with openvino_vision_embeddings_model.xml,
    ///        openvino_text_embeddings_model.xml, tokenizer files, and config.json.
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

    ~VLMProcessor();

    /// @brief Prepare inputs for VLM generation from a text prompt
    /// and optional images/videos.
    ///
    /// Performs the full preprocessing pipeline:
    /// 1. Applies chat template to the prompt (if enabled).
    /// 2. Encodes images/videos through the vision encoder.
    /// 3. Tokenizes the text and computes text embeddings.
    /// 4. Merges vision features into the text embedding sequence
    ///    at <image>/<video> placeholder positions.
    /// 5. Computes attention mask and position IDs.
    ///
    /// For using image and video tags in prompt, see:
    /// https://openvinotoolkit.github.io/openvino.genai/docs/use-cases/image-processing/
    ///
    /// @param prompt Text prompt, optionally containing image/video tags.
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
    /// Useful for manual template control without full prepare().
    ///
    /// @param prompt The raw user message.
    /// @param system_message Optional system prompt.
    /// @param add_generation_prompt If true, append assistant turn-start tokens.
    /// @return Formatted prompt string with special tokens applied.
    std::string apply_chat_template(
        const std::string& prompt,
        const std::string& system_message = "",
        bool add_generation_prompt = true
    );

    /// @brief Extract vision embeddings without merging with text.
    /// Useful for embedding-only workflows (retrieval, caching, analysis)
    /// without invoking the LLM.
    ///
    /// Analogous to HuggingFace's model.get_image_features().
    ///
    /// @param images RGB image tensors.
    /// @return Vector of embedding tensors, one per image.
    ///         Each tensor shape: [num_patches, hidden_size].
    std::vector<ov::Tensor> get_vision_embeddings(
        const std::vector<ov::Tensor>& images
    );

    /// @brief Extract video embeddings without merging with text.
    /// @param videos Video frame tensors.
    /// @return Vector of embedding tensors, one per video.
    std::vector<ov::Tensor> get_video_embeddings(
        const std::vector<ov::Tensor>& videos
    );

    /// @brief Get the underlying tokenizer.
    Tokenizer get_tokenizer() const;

    /// @brief Enable or disable automatic chat template application in prepare().
    /// @param apply If false, prepare() will not wrap the prompt in a chat template.
    void set_apply_chat_template(bool apply);

    /// @brief Override the default chat template.
    /// @param new_template Jinja2-style template string.
    void set_chat_template(const std::string& new_template);

private:
    class Impl;
    std::unique_ptr<Impl> m_impl;
};

} // namespace ov::genai
```

### 3. Redesigned VLMPipeline — LLM-only
```cpp
// openvino/genai/visual_language/pipeline.hpp (redesigned)

#pragma once
#include <filesystem>
#include <string>
#include "openvino/genai/generation_config.hpp"
#include "openvino/genai/streamer_base.hpp"
#include "openvino/genai/tokenizer.hpp"
#include "openvino/genai/visual_language/vlm_inputs.hpp"
#include "openvino/genai/visual_language/perf_metrics.hpp"
#include "openvino/genai/common_types.hpp"

namespace ov::genai {

class OPENVINO_GENAI_EXPORTS VLMDecodedResults : public DecodedResults {
public:
    VLMPerfMetrics perf_metrics;
};

/// @brief A Visual Language Model pipeline that performs text generation
/// given pre-processed VLMInputs from a VLMProcessor.
///
/// This class owns only the LLM (language model). Vision encoding and
/// embedding preparation are handled by VLMProcessor.
///
/// Usage:
///   auto processor = VLMProcessor("path/to/models", "GPU");
///   auto llm = VLMPipeline("path/to/llm_model", "GPU");
///   auto inputs = processor.prepare("Describe this image", {image_tensor});
///   auto result = llm.generate(inputs, generation_config);
class OPENVINO_GENAI_EXPORTS VLMPipeline {
public:
    /// @brief Construct the LLM pipeline from a model directory.
    /// @param models_path Path to directory containing the language model IR.
    /// @param device Inference device.
    /// @param properties Device configuration properties.
    VLMPipeline(
        const std::filesystem::path& models_path,
        const std::string& device,
        const ov::AnyMap& properties = {}
    );

    /// @brief Construct from a pre-loaded language model.
    /// @param model_str Language model IR as string.
    /// @param weights_tensor Model weights.
    /// @param tokenizer Pre-initialized tokenizer (for detokenization of output).
    /// @param device Inference device.
    /// @param properties Device configuration properties.
    VLMPipeline(
        const std::string& model_str,
        const ov::Tensor& weights_tensor,
        const Tokenizer& tokenizer,
        const std::string& device,
        const ov::AnyMap& properties = {}
    );

    ~VLMPipeline();

    /// @brief Generate text from pre-processed VLMInputs.
    /// @param inputs Structured inputs from VLMProcessor::prepare().
    /// @param generation_config Text generation parameters.
    /// @param streamer Optional streamer for token-by-token output.
    /// @return Generated text(s) with scores and performance metrics.
    VLMDecodedResults generate(
        const VLMInputs& inputs,
        const GenerationConfig& generation_config,
        const StreamerVariant& streamer = std::monostate{}
    );

    /// @brief Generate with config as property map (for Python bindings convenience).
    VLMDecodedResults generate(
        const VLMInputs& inputs,
        const ov::AnyMap& config_map
    );

    /// @brief Variadic property overload.
    template <typename... Properties>
    util::EnableIfAllStringAny<VLMDecodedResults, Properties...> generate(
        const VLMInputs& inputs,
        Properties&&... properties
    ) {
        return generate(inputs, AnyMap{std::forward<Properties>(properties)...});
    }

    // ---- Legacy convenience overloads (delegate to internal processor) ----
    // These create a temporary VLMProcessor internally for backward compatibility.
    // New code should prefer the VLMProcessor + generate(VLMInputs) pattern.

    /// @deprecated Use VLMProcessor::prepare() + generate(VLMInputs) instead.
    VLMDecodedResults generate(
        const std::string& prompt,
        const std::vector<ov::Tensor>& images,
        const GenerationConfig& generation_config,
        const StreamerVariant& streamer = std::monostate{}
    );

    /// @deprecated Use VLMProcessor::prepare() + generate(VLMInputs) instead.
    VLMDecodedResults generate(
        const std::string& prompt,
        const ov::AnyMap& config_map
    );

    /// @brief Get the tokenizer (for output detokenization).
    Tokenizer get_tokenizer() const;

    GenerationConfig get_generation_config() const;
    void set_generation_config(const GenerationConfig& config);

private:
    class Impl;
    std::unique_ptr<Impl> m_impl;
};

} // namespace ov::genai
```

### 4. ContinuousBatchingPipeline — VLM-aware add_request()
```cpp
// In continuous_batching_pipeline.hpp — additions to existing class

// New add_request overload accepting VLMInputs:

/// @brief Add a VLM request with pre-processed inputs from VLMProcessor.
/// @param request_id Unique request identifier.
/// @param inputs Pre-processed VLMInputs containing merged embeddings.
/// @param sampling_params Generation configuration.
/// @return Handle to monitor and read generation results.
GenerationHandle add_request(
    uint64_t request_id,
    const VLMInputs& inputs,
    const ov::genai::GenerationConfig& sampling_params
);

// New generate overload for batch VLM:

/// @brief Batch generate from pre-processed VLMInputs.
std::vector<VLMDecodedResults> generate(
    const std::vector<VLMInputs>& inputs,
    const std::vector<GenerationConfig>& sampling_params,
    const StreamerVariant& streamer = std::monostate{}
);
```

## Complete usage examples
### Example 1: VLMPipeline — basic single image
```cpp
// Separate construction: processor owns vision models, pipeline owns LLM
auto processor = ov::genai::VLMProcessor("path/to/models", "GPU");
auto llm = ov::genai::VLMPipeline("path/to/models", "GPU");

// Prepare: encodes images, tokenizes text, merges embeddings
auto inputs = processor.prepare("Describe this image", {image_tensor});

// Generate: runs LLM on pre-merged embeddings
ov::genai::GenerationConfig config;
config.max_new_tokens = 256;
auto result = llm.generate(inputs, config);
std::cout << result.texts.at(0) << std::endl;
```

### Example 2: Multiple images + manual chat template
```cpp
auto processor = ov::genai::VLMProcessor("path/to/models", "GPU");
auto llm = ov::genai::VLMPipeline("path/to/models", "GPU");

// Disable automatic chat template — apply manually
processor.set_apply_chat_template(false);
std::string formatted = processor.apply_chat_template(
    "Compare <ov_genai_image_0> and <ov_genai_image_1>",
    /*system_message=*/"You are a helpful assistant."
);

auto inputs = processor.prepare(formatted, {image1, image2});
auto result = llm.generate(inputs, config);
```

### Example 3: Embedding extraction (no LLM)
```cpp
auto processor = ov::genai::VLMProcessor("path/to/models", "GPU");

// Extract vision embeddings only — no LLM needed
auto embeddings = processor.get_vision_embeddings({image1, image2});
// embeddings[0].get_shape() → {num_patches, hidden_size}

// Use for retrieval, similarity, caching, etc.
```

### Example 4: ContinuousBatchingPipeline — concurrent VLM requests
```cpp
auto processor = ov::genai::VLMProcessor("path/to/models", "GPU");
auto llm = ov::genai::ContinuousBatchingPipeline("path/to/models", scheduler_config, "GPU");

// Prepare multiple requests
auto inputs1 = processor.prepare("What is in this photo?", {photo1});
auto inputs2 = processor.prepare("Describe this diagram", {diagram});

// Submit to continuous batching
auto handle1 = llm.add_request(1, inputs1, config);
auto handle2 = llm.add_request(2, inputs2, config);

// Process until done
while (llm.has_non_finished_requests()) {
    llm.step();
}

auto results1 = handle1->read_all();
auto results2 = handle2->read_all();
```

### Example 5: Inspecting / modifying embeddings between processor and LLM
```cpp
auto processor = ov::genai::VLMProcessor("path/to/models", "GPU");
auto llm = ov::genai::VLMPipeline("path/to/models", "GPU");

auto inputs = processor.prepare("Describe this", {image_tensor});

// User can inspect or transform embeddings before generation
std::cout << "Embedding sequence length: " 
          << inputs.inputs_embeds.get_shape()[1] << std::endl;

// For example, truncate if too long
// ... modify inputs.inputs_embeds, inputs.attention_mask ...

auto result = llm.generate(inputs, config);
```


