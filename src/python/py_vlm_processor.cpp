// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <filesystem>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl/filesystem.h>

#include "openvino/genai/visual_language/vlm_inputs.hpp"
#include "openvino/genai/visual_language/processor.hpp"
#include "tokenizer/tokenizers_path.hpp"
#include "py_utils.hpp"

namespace py = pybind11;
namespace pyutils = ov::genai::pybind::utils;

void init_vlm_processor(py::module_& m) {
    auto vlm_inputs_docstring = R"(
        Structured inputs for VLM generation, produced by VLMProcessor.prepare().
        Contains merged text+vision embeddings, attention mask, and optional
        model-specific tensors (position IDs, token type IDs).

        :param inputs_embeds: Merged embeddings of text tokens and projected vision features.
            Shape: [1, sequence_length, hidden_size].
        :type inputs_embeds: ov.Tensor

        :param attention_mask: Attention mask for the merged sequence.
            Shape: [1, sequence_length]. Values: 1 = attend, 0 = ignore.
        :type attention_mask: ov.Tensor

        :param position_ids: Position IDs for models requiring explicit positioning.
            May be None for models that derive positions from attention_mask.
        :type position_ids: ov.Tensor or None

        :param token_type_ids: Token type IDs for models that distinguish token types.
            May be None.
        :type token_type_ids: ov.Tensor or None

        :param rope_delta: Rope delta value, used by some models (e.g., Qwen2-VL).
        :type rope_delta: int or None
    )";

    py::class_<ov::genai::VLMInputs>(m, "VLMInputs", vlm_inputs_docstring)
        .def(py::init<>())
        .def_readwrite("inputs_embeds", &ov::genai::VLMInputs::inputs_embeds)
        .def_readwrite("attention_mask", &ov::genai::VLMInputs::attention_mask)
        .def_readwrite("position_ids", &ov::genai::VLMInputs::position_ids)
        .def_readwrite("token_type_ids", &ov::genai::VLMInputs::token_type_ids)
        .def_readwrite("rope_delta", &ov::genai::VLMInputs::rope_delta)
        .def("has_vision_content", &ov::genai::VLMInputs::has_vision_content,
            "Check whether vision content was encoded into these inputs.")
        .def("__repr__", [](const ov::genai::VLMInputs& self) {
            std::ostringstream ss;
            ss << "VLMInputs(";
            if (self.inputs_embeds && self.inputs_embeds.get_size() > 0) {
                auto shape = self.inputs_embeds.get_shape();
                ss << "inputs_embeds=[";
                for (size_t i = 0; i < shape.size(); ++i) {
                    if (i > 0) ss << ", ";
                    ss << shape[i];
                }
                ss << "]";
            } else {
                ss << "inputs_embeds=None";
            }
            ss << ")";
            return ss.str();
        });

    auto vlm_processor_docstring = R"(
        A processor for Visual Language Models that handles vision encoding,
        text tokenization, embedding merging, and chat template application.

        Separates preprocessing (vision + text -> merged embeddings) from
        language model generation. Produces VLMInputs that can be passed to
        VLMPipeline.generate() or ContinuousBatchingPipeline.add_request().

        Analogous to HuggingFace's AutoProcessor.

        Example:
            processor = openvino_genai.VLMProcessor("path/to/models", "GPU")
            inputs = processor.prepare("Describe this image", [image_tensor])
            result = llm.generate(inputs, generation_config)
    )";

    py::class_<ov::genai::VLMProcessor>(m, "VLMProcessor", vlm_processor_docstring)
        .def(py::init([](
            const std::filesystem::path& models_path,
            const std::string& device,
            const py::kwargs& kwargs
        ) {
            ScopedVar env_manager(pyutils::ov_tokenizers_module_path());
            return std::make_unique<ov::genai::VLMProcessor>(
                models_path, device, pyutils::kwargs_to_any_map(kwargs));
        }),
        py::arg("models_path"), "folder with exported model files",
        py::arg("device"), "device on which inference will be done",
        R"(
            VLMProcessor constructor.
            models_path (os.PathLike): Path to the folder with exported model files.
            device (str): Device to run the model on (e.g., CPU, GPU).
            kwargs: Device properties
        )")

        .def("prepare",
            [](ov::genai::VLMProcessor& self,
               const std::string& prompt,
               const std::vector<ov::Tensor>& images,
               const std::vector<ov::Tensor>& videos
            ) -> ov::genai::VLMInputs {
                ov::genai::VLMInputs result;
                {
                    py::gil_scoped_release rel;
                    result = self.prepare(prompt, images, videos);
                }
                return result;
            },
            py::arg("prompt"),
            py::arg("images") = std::vector<ov::Tensor>{},
            py::arg("videos") = std::vector<ov::Tensor>{},
            R"(
                Prepare inputs for VLM generation.
                Encodes images/videos, tokenizes text, merges vision features
                into text embedding sequence, and computes attention mask and position IDs.

                :param prompt: Text prompt, optionally containing image/video tags.
                :type prompt: str
                :param images: RGB image tensors with [NHWC] or [HWC] layout.
                :type images: list[ov.Tensor]
                :param videos: Video frame tensors with [NHWC] layout.
                :type videos: list[ov.Tensor]
                :return: VLMInputs ready for generate() or add_request().
                :rtype: VLMInputs
            )")

        .def("apply_chat_template",
            &ov::genai::VLMProcessor::apply_chat_template,
            py::arg("prompt"),
            py::arg("system_message") = "",
            py::arg("add_generation_prompt") = true,
            R"(
                Apply the model's chat template to a prompt string.

                :param prompt: The raw user message.
                :type prompt: str
                :param system_message: Optional system prompt.
                :type system_message: str
                :param add_generation_prompt: If True, append assistant turn-start tokens.
                :type add_generation_prompt: bool
                :return: Formatted prompt string.
                :rtype: str
            )")

        .def("get_vision_embeddings",
            [](ov::genai::VLMProcessor& self,
               const std::vector<ov::Tensor>& images
            ) -> std::vector<ov::Tensor> {
                std::vector<ov::Tensor> result;
                {
                    py::gil_scoped_release rel;
                    result = self.get_vision_embeddings(images);
                }
                return result;
            },
            py::arg("images"),
            R"(
                Extract projected vision embeddings without merging with text.
                Useful for embedding-only workflows (retrieval, caching, similarity).

                :param images: RGB image tensors.
                :type images: list[ov.Tensor]
                :return: Vector of embedding tensors, one per image.
                :rtype: list[ov.Tensor]
            )")

        .def("get_video_embeddings",
            [](ov::genai::VLMProcessor& self,
               const std::vector<ov::Tensor>& videos
            ) -> std::vector<ov::Tensor> {
                std::vector<ov::Tensor> result;
                {
                    py::gil_scoped_release rel;
                    result = self.get_video_embeddings(videos);
                }
                return result;
            },
            py::arg("videos"),
            R"(
                Extract projected video embeddings without merging with text.

                :param videos: Video frame tensors.
                :type videos: list[ov.Tensor]
                :return: Vector of embedding tensors, one per video.
                :rtype: list[ov.Tensor]
            )")

        .def("get_tokenizer", &ov::genai::VLMProcessor::get_tokenizer)
        .def("set_apply_chat_template", &ov::genai::VLMProcessor::set_apply_chat_template,
            py::arg("apply"),
            R"(
                Enable or disable automatic chat template application in prepare().

                :param apply: If False, prepare() will not wrap the prompt in a chat template.
                :type apply: bool
            )")
        .def("set_chat_template", &ov::genai::VLMProcessor::set_chat_template,
            py::arg("new_template"),
            R"(
                Override the default chat template with a custom one.

                :param new_template: Jinja2-style template string.
                :type new_template: str
            )");
}
