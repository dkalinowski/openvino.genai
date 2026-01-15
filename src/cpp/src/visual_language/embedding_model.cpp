// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <fstream>
#include <memory>

#include "openvino/genai/visual_language/embeddings_model.hpp"
#include "openvino/runtime/core.hpp"
#include "openvino/core/preprocess/pre_post_process.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/constant.hpp"

#include "utils.hpp"

#include "embedding_model_impl.hpp"
#include "vlm_config.hpp"

namespace {

// TODO: Do the same for VisionEncoder as well, public API should be able to handle concurrent usage of VisionEncoder
std::unique_ptr<ov::genai::CircularBufferQueue<ov::genai::EmbeddingsRequest>> init(ov::CompiledModel& compiled) {
    auto embeddings_requests_queue = std::make_unique<ov::genai::CircularBufferQueue<ov::genai::EmbeddingsRequest>>(
        compiled.get_property(ov::optimal_number_of_infer_requests),
        [&compiled]() -> ov::genai::EmbeddingsRequest {
            ov::genai::EmbeddingsRequest req;
            req.ireq = compiled.create_infer_request();
            req.cpu_tensor = req.ireq.get_output_tensor();
            ov::RemoteContext context;
            try {
                context = compiled.get_context();
            } catch (const ov::Exception&) {
                req.remote_tensor = req.cpu_tensor;
                return req;
            }
            req.remote_tensor = context.create_tensor(ov::element::f32, req.cpu_tensor.get_shape());
            return req;
        });
    return embeddings_requests_queue;
}

}  // namespace

namespace ov {
namespace genai {

EmbeddingsModelImpl::EmbeddingsModelImpl(const std::filesystem::path& model_dir,
                                 const float scale_emb,
                                 const std::string& device,
                                 const ov::AnyMap& properties) {
    ov::Core core = utils::singleton_core();
    std::shared_ptr<ov::Model> m_model = core.read_model(model_dir / "openvino_text_embeddings_model.xml", {}, properties);
    // apply embedding postprocessing step by merging them into the model
    merge_postprocess(m_model, scale_emb);

    // measure time
    auto start_time = std::chrono::steady_clock::now();
    ov::CompiledModel compiled_model = core.compile_model(m_model, device, properties);
    auto end_time = std::chrono::steady_clock::now();
    auto duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();
    std::cout << "Compilation time for text embeddings model on " << device << ": " << duration_ms << " ms" << std::endl;
    ov::genai::utils::print_compiled_model_properties(compiled_model, "text embeddings model");
    std::cout << "Embeddings shape info:" << std::endl;
    for (const auto& input : compiled_model.inputs()) {
        // consider the shape might have dynamic dimensions
        std::cout << " Input: " << input.get_any_name() << " shape: ";
        for (const auto& dim : input.get_partial_shape()) {
            std::cout << dim << " ";
        }
        std::cout << std::endl;
    }
    for (const auto& output : compiled_model.outputs()) {
        std::cout << " Output: " << output.get_any_name() << " shape: ";
        for (const auto& dim : output.get_partial_shape()) {
            std::cout << dim << " ";
        }
        std::cout << std::endl;
    }
    m_embeddings_requests_queue = init(compiled_model);
}

EmbeddingsModelImpl::EmbeddingsModelImpl(const std::string& model,
                                 const ov::Tensor& weights,
                                 const float scale_emb,
                                 const std::string& device,
                                 const ov::AnyMap& properties) {
    ov::Core core = utils::singleton_core();
    std::shared_ptr<ov::Model> m_model = core.read_model(model, weights);
    // apply embedding postprocessing step by merging them into the model
    merge_postprocess(m_model, scale_emb);

    ov::CompiledModel compiled_model = core.compile_model(m_model, device, properties);
    m_embeddings_requests_queue = init(compiled_model);
}

std::unique_ptr<CircularBufferQueue<EmbeddingsRequest>>& EmbeddingsModelImpl::get_request_queue() {
    return this->m_embeddings_requests_queue;
}

ov::Tensor EmbeddingsModelImpl::infer(EmbeddingsRequest& req, const ov::Tensor& input_idx, bool return_remote_tensor) {
    OPENVINO_ASSERT(req.ireq, "Text embeddings decoder model must be compiled first. Cannot infer non-compiled model");
    // measure time
    auto start_time = std::chrono::steady_clock::now();
    req.ireq.set_input_tensor(input_idx);
    if (return_remote_tensor) {
        req.ireq.set_output_tensor(req.remote_tensor);
    } else {
        req.ireq.set_output_tensor(req.cpu_tensor);
    }
    req.ireq.infer();
    auto end_time = std::chrono::steady_clock::now();
    auto duration_us = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count();
    static int infer_count = 0;
    infer_count++;
    std::cout << "Inference time for text embeddings model on infer #" << infer_count << ": " << (float)duration_us / (float)1000 << " ms" << std::endl;
    return req.ireq.get_output_tensor();
}

void EmbeddingsModelImpl::merge_postprocess(std::shared_ptr<ov::Model> model, float scale_emb) const {
    ov::preprocess::PrePostProcessor ppp(model);

    ppp.output().postprocess().custom([scale_emb](const ov::Output<ov::Node>& node) {
        auto constant = std::make_shared<ov::op::v0::Constant>(ov::element::f32, ov::Shape{1}, scale_emb);
        return std::make_shared<ov::op::v1::Multiply>(node, constant);
    });

    ppp.build();
}

// ======================== EmbeddingsModel (public API) ========================

/// @brief The implementation class that wraps EmbeddingsModelImpl for the public API. Allows for concurrent usage of the model.
class EmbeddingsModel::EmbeddingsModelImplWrapper {
public:
    EmbeddingsModelImplWrapper(
        const std::filesystem::path& model_dir,
        const std::string& device,
        const ov::AnyMap& properties)
    {
        // Read the VLM config to get scale_emb value
        VLMConfig vlm_config = utils::from_config_json_if_exists<VLMConfig>(model_dir, "config.json");
        m_impl = ov::genai::EmbeddingsModelImpl::create(model_dir, vlm_config.scale_emb, device, properties);
    }

    EmbeddingsModelImplWrapper(
        const std::string& model,
        const ov::Tensor& weights,
        const std::string& device,
        const ov::AnyMap& properties)
    {
        // Default scale_emb value of 1.0 when config is not available
        m_impl = ov::genai::EmbeddingsModelImpl::create(model, weights, 1.0f, device, properties);
    }

    ov::Tensor infer(const ov::Tensor& input_ids) {
        auto& queue = m_impl->get_request_queue();
        CircularBufferQueueElementGuard<EmbeddingsRequest> req_guard(queue.get());
        return m_impl->infer(req_guard.get(), input_ids, false);
    }
    
    ov::genai::EmbeddingsModelImpl::Ptr get_internal_impl() const {
        return m_impl;
    }

private:
    ov::genai::EmbeddingsModelImpl::Ptr m_impl;
};

EmbeddingsModel::EmbeddingsModel(
    const std::filesystem::path& model_dir,
    const std::string& device,
    const ov::AnyMap& properties)
    : m_pimpl(std::make_unique<EmbeddingsModelImplWrapper>(model_dir, device, properties)) {}

EmbeddingsModel::EmbeddingsModel(
    const std::string& model,
    const ov::Tensor& weights,
    const std::string& device,
    const ov::AnyMap& properties)
    : m_pimpl(std::make_unique<EmbeddingsModelImplWrapper>(model, weights, device, properties)) {}

EmbeddingsModel::~EmbeddingsModel() = default;

EmbeddingsModel::EmbeddingsModel(EmbeddingsModel&& other) noexcept = default;

EmbeddingsModel& EmbeddingsModel::operator=(EmbeddingsModel&& other) noexcept = default;

ov::Tensor EmbeddingsModel::infer(const ov::Tensor& input_ids) {
    return m_pimpl->infer(input_ids);
}

std::shared_ptr<EmbeddingsModelImpl> EmbeddingsModel::get_internal_impl() const {
    return m_pimpl->get_internal_impl();
}

} // namespace genai
} // namespace ov