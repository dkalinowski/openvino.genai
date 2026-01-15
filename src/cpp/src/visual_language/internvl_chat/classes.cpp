// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "visual_language/internvl_chat/classes.hpp"

#include "visual_language/clip.hpp"

#include "utils.hpp"

namespace ov::genai {

namespace {

std::string NATIVE_TAG = "<image>";

struct SplitImageShape {
    size_t batch_size;
    size_t height;
    size_t width;
};

// Function to determine how to split the image based on its aspect ratio
// Each VLM model class should have similar implementation in order to deduce static shape for Vision Encoder
// Used during reshape in case Vision Encoder is on NPU device
SplitImageShape deduce_model_static_shape_from_resolution(
    int orig_height,
    int orig_width,
    int image_size,
    int min_num = 1,
    int max_num = 12,
    bool use_thumbnail = true) {
    float aspect_ratio = static_cast<float>(orig_width) / orig_height;

    std::vector<std::pair<int, int>> target_ratios;
    for (int n = min_num; n <= max_num; ++n) {
        for (int i = 1; i <= n; ++i) {
            for (int j = 1; j <= n; ++j) {
                if (i * j <= max_num && i * j >= min_num) {
                    target_ratios.emplace_back(i, j);
                }
            }
        }
    }
    std::sort(target_ratios.begin(), target_ratios.end(),
        [](const auto& a, const auto& b) { return a.first * a.second < b.first * b.second; });

    auto find_closest_aspect_ratio = [&](float ar, const std::vector<std::pair<int, int>>& ratios) {
        float best_ratio_diff = std::numeric_limits<float>::max();
        std::pair<int, int> best_ratio = {1, 1};
        int area = orig_width * orig_height;

        for (const auto& ratio : ratios) {
            float target_ar = static_cast<float>(ratio.first) / ratio.second;
            float ratio_diff = std::abs(ar - target_ar);
            if (ratio_diff < best_ratio_diff) {
                best_ratio_diff = ratio_diff;
                best_ratio = ratio;
            } else if (ratio_diff == best_ratio_diff && area > 0.5 * image_size * image_size * ratio.first * ratio.second) {
                best_ratio = ratio;
            }
        }
        return best_ratio;
    };

    auto target_aspect_ratio = find_closest_aspect_ratio(aspect_ratio, target_ratios);

    int blocks = target_aspect_ratio.first * target_aspect_ratio.second;
    size_t batch_size = blocks;
    
    if (use_thumbnail && blocks != 1) {
        batch_size += 1;
    }

    return {batch_size, static_cast<size_t>(image_size), static_cast<size_t>(image_size)};
}

std::vector<clip_image_u8> split_image_internvl(
    const clip_image_u8& image,
    int image_size,
    int min_num = 1,
    int max_num = 12,
    bool use_thumbnail = true) {
    int orig_width = image.nx;
    int orig_height = image.ny;
    float aspect_ratio = static_cast<float>(orig_width) / orig_height;

    std::vector<std::pair<int, int>> target_ratios;
    for (int n = min_num; n <= max_num; ++n) {
        for (int i = 1; i <= n; ++i) {
            for (int j = 1; j <= n; ++j) {
                if (i * j <= max_num && i * j >= min_num) {
                    target_ratios.emplace_back(i, j);
                }
            }
        }
    }
    std::sort(target_ratios.begin(), target_ratios.end(),
        [](const auto& a, const auto& b) { return a.first * a.second < b.first * b.second; });

    auto find_closest_aspect_ratio = [&](float ar, const std::vector<std::pair<int, int>>& ratios) {
        float best_ratio_diff = std::numeric_limits<float>::max();
        std::pair<int, int> best_ratio = {1, 1};
        int area = orig_width * orig_height;

        for (const auto& ratio : ratios) {
            float target_ar = static_cast<float>(ratio.first) / ratio.second;
            float ratio_diff = std::abs(ar - target_ar);
            if (ratio_diff < best_ratio_diff) {
                best_ratio_diff = ratio_diff;
                best_ratio = ratio;
            } else if (ratio_diff == best_ratio_diff && area > 0.5 * image_size * image_size * ratio.first * ratio.second) {
                best_ratio = ratio;
            }
        }
        return best_ratio;
    };

    auto target_aspect_ratio = find_closest_aspect_ratio(aspect_ratio, target_ratios);

    int target_width = image_size * target_aspect_ratio.first;
    int target_height = image_size * target_aspect_ratio.second;
    int blocks = target_aspect_ratio.first * target_aspect_ratio.second;

    clip_image_u8 resized_img;
    bicubic_resize(image, resized_img, target_width, target_height);

    std::vector<clip_image_u8> processed_images;
    for (int i = 0; i < blocks; ++i) {
        int x = (i % (target_width / image_size)) * image_size;
        int y = (i / (target_width / image_size)) * image_size;

        clip_image_u8 split_img;
        split_img.nx = image_size;
        split_img.ny = image_size;
        split_img.buf.resize(3 * image_size * image_size);

        for (int dy = 0; dy < image_size; ++dy) {
            for (int dx = 0; dx < image_size; ++dx) {
                for (int c = 0; c < 3; ++c) {
                    int src_idx = ((y + dy) * target_width + (x + dx)) * 3 + c;
                    int dst_idx = (dy * image_size + dx) * 3 + c;
                    split_img.buf[dst_idx] = resized_img.buf[src_idx];
                }
            }
        }

        processed_images.push_back(std::move(split_img));
    }

    if (use_thumbnail && processed_images.size() != 1) {
        clip_image_u8 thumbnail_img;
        bicubic_resize(image, thumbnail_img, image_size, image_size);
        processed_images.push_back(std::move(thumbnail_img));
    }

    return processed_images;
}

ov::Tensor get_pixel_values_internvl(const ov::Tensor& image, const ProcessorConfig& config) {
    clip_image_u8 input_image = tensor_to_clip_image_u8(image);

    const size_t image_size = config.size_shortest_edge;

    clip_ctx ctx;
    ctx.image_size = image_size;
    std::copy(config.image_mean.begin(), config.image_mean.end(), ctx.image_mean);
    std::copy(config.image_std.begin(), config.image_std.end(), ctx.image_std);

    std::vector<clip_image_u8> splitted_images = split_image_internvl(input_image, image_size);

    std::vector<clip_image_f32> processed_images;
    processed_images.reserve(splitted_images.size());
    for (const auto& image : splitted_images) {
        processed_images.push_back(clip_image_preprocess(ctx, image));
    }

    size_t batch_size = processed_images.size();
    size_t channels = 3;
    size_t height = processed_images[0].ny;
    size_t width = processed_images[0].nx;

    std::cout << "Processed " << batch_size << " images for InternVL Chat vision encoder." << std::endl;
    std::cout << "Height: " << height << ", Width: " << width << std::endl;

    ov::Tensor output_tensor(ov::element::f32, {batch_size, channels, height, width});
    float* output_data = output_tensor.data<float>();

    for (size_t i = 0; i < batch_size; ++i) {
        const auto& img = processed_images[i];
        std::copy(img.buf.begin(), img.buf.end(), output_data + i * channels * height * width);
    }
    return output_tensor;
}

} // namespace

VisionEncoderInternVLChat::VisionEncoderInternVLChat(
    const std::filesystem::path& model_dir,
    const std::string& device,
    const ov::AnyMap properties) {
    m_processor_config = utils::from_config_json_if_exists<ProcessorConfig>(model_dir, "preprocessor_config.json");
    auto model = utils::singleton_core().read_model(model_dir / "openvino_vision_embeddings_model.xml");

    // In case the device is NPU, we need to reshape to static shape
    // In final version the shapes could be taken from properties or from preprocessor_config?
    if (device.find("NPU") != std::string::npos) {
        std::cout << "Proceeding with reshape..." << std::endl;
        auto start_time = std::chrono::steady_clock::now();
        this->reshape(model);  // Each class should have its own reshape implementation
        auto end_time = std::chrono::steady_clock::now();
        auto duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();
        std::cout << "Reshape time for InternVL Chat vision encoder on NPU: " << duration_ms << " ms" << std::endl;
    }

    // measure time
    auto start_time = std::chrono::steady_clock::now();
    auto compiled_model = utils::singleton_core().compile_model(model, device, properties);
    auto end_time = std::chrono::steady_clock::now();
    auto duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();
    std::cout << "Compilation time for InternVL Chat vision encoder on " << device << ": " << duration_ms << " ms" << std::endl;

    ov::genai::utils::print_compiled_model_properties(compiled_model, "VLM vision embeddings model");

    // print input shapes
    std::cout << "Vision shape info:" << std::endl;
    for (const auto& input : compiled_model.inputs()) {
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
    m_ireq_queue_vision_encoder = std::make_unique<CircularBufferQueue<ov::InferRequest>>(
        compiled_model.get_property(ov::optimal_number_of_infer_requests),
        [&compiled_model]() -> ov::InferRequest {
            return compiled_model.create_infer_request();
        });
}

VisionEncoderInternVLChat::VisionEncoderInternVLChat(
    const ModelsMap& models_map,
    const std::filesystem::path& config_dir_path,
    const std::string& device,
    const ov::AnyMap device_config) {
    const auto& vision_encoder_model = utils::get_model_weights_pair(models_map, "vision_embeddings").first;
    const auto& vision_encoder_weights = utils::get_model_weights_pair(models_map, "vision_embeddings").second;
    // Note: This path doesn't load the model separately, so reshape is not applicable here
    // The model is compiled directly from the models_map
    auto compiled_model = utils::singleton_core().compile_model(vision_encoder_model, vision_encoder_weights, device, device_config);
    ov::genai::utils::print_compiled_model_properties(compiled_model, "VLM vision embeddings model");
    m_ireq_queue_vision_encoder = std::make_unique<CircularBufferQueue<ov::InferRequest>>(
        compiled_model.get_property(ov::optimal_number_of_infer_requests),
        [&compiled_model]() -> ov::InferRequest {
            return compiled_model.create_infer_request();
        });
    m_processor_config = utils::from_config_json_if_exists<ProcessorConfig>(config_dir_path, "preprocessor_config.json");
}

EncodedImage VisionEncoderInternVLChat::encode(const ov::Tensor& image, const ov::AnyMap& config_map) {
    CircularBufferQueueElementGuard<ov::InferRequest> infer_request_guard(this->m_ireq_queue_vision_encoder.get());
    ov::InferRequest& encoder = infer_request_guard.get();
    // Measurements for POC purposes, ignore all newly added timing code
    auto start_time = std::chrono::steady_clock::now();
    ProcessorConfig config = utils::from_any_map(config_map, m_processor_config);

    ov::Tensor pixel_values = get_pixel_values_internvl(image, config);
    auto end_time = std::chrono::steady_clock::now();
    auto duration_us = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count();
    std::cout << "Preprocessing time for InternVL Chat vision encoder: " << (float)duration_us / (float)1000 << " ms" << std::endl;

    start_time = std::chrono::steady_clock::now();
    encoder.set_tensor("pixel_values", pixel_values);
    encoder.infer();
    end_time = std::chrono::steady_clock::now();
    duration_us = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count();
    std::cout << "Inference time for InternVL Chat vision encoder: " << (float)duration_us / (float)1000 << " ms" << std::endl;

    start_time = std::chrono::steady_clock::now();
    const ov::Tensor& infer_output = encoder.get_output_tensor();
    ov::Tensor image_features(infer_output.get_element_type(), infer_output.get_shape());
    std::memcpy(image_features.data(), infer_output.data(), infer_output.get_byte_size());
    end_time = std::chrono::steady_clock::now();
    duration_us = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count();
    std::cout << "Postprocessing time for InternVL Chat vision encoder: " << (float)duration_us / (float)1000 << " ms" << std::endl;

    ImageSize resized_source_size{config.crop_size_height / config.patch_size, config.crop_size_width / config.patch_size};

    return {std::move(image_features), resized_source_size};
}

void VisionEncoderInternVLChat::reshape(std::shared_ptr<ov::Model> model) {
    std::cout << "Reshaping InternVL Chat vision encoder model..." << std::endl;

    // For now the reshape is always to 224x336, but in final version it could be taken from properties or preprocessor_config
    auto shape_info = deduce_model_static_shape_from_resolution(
        /*orig_height=*/224,
        /*orig_width=*/336,
        /*image_size=*/m_processor_config.size_shortest_edge,
        /*min_num=*/1,
        /*max_num=*/12,
        /*use_thumbnail=*/true);
    ov::PartialShape input_shape = ov::PartialShape{shape_info.batch_size, 3, shape_info.height, shape_info.width};
    model->reshape({{model->input().get_any_name(), input_shape}});
}

namespace {

ov::Tensor merge_text_and_image_embeddings_internvl(
    const ov::Tensor& input_ids,
    const ov::Tensor& text_embeds,
    const std::vector<ov::Tensor>& image_embeds,
    int64_t image_context_token_id) {
    auto text_embeds_shape = text_embeds.get_shape();
    size_t batch_size = text_embeds_shape.at(0);
    size_t seq_len = text_embeds_shape.at(1);
    size_t embed_dim = text_embeds_shape.at(2);

    ov::Tensor merged_embeds(text_embeds.get_element_type(), text_embeds_shape);

    const float* text_embeds_data = text_embeds.data<float>();
    const int64_t* input_ids_data = input_ids.data<int64_t>();
    float* merged_embeds_data = merged_embeds.data<float>();

    size_t flattened_size = batch_size * seq_len;
    std::vector<bool> image_context_tokens_mask(flattened_size, false);
    size_t image_context_tokens_count = 0;

    for (size_t i = 0; i < flattened_size; ++i) {
        if (input_ids_data[i] == image_context_token_id) {
            image_context_tokens_mask[i] = true;
            ++image_context_tokens_count;
        }
    }

    OPENVINO_ASSERT(image_context_tokens_count > 0, "input_ids does not contain image context token ids");

    size_t image_idx = 0;
    size_t image_context_token_idx = 0;
    for (size_t i = 0; i < batch_size; ++i) {
        for (size_t j = 0; j < seq_len; ++j) {
            size_t flat_idx = i * seq_len + j;
            size_t offset = flat_idx * embed_dim;

            if (image_context_tokens_mask[flat_idx]) {
                const ov::Tensor& single_image_embeds = image_embeds[image_idx];
                const size_t num_all_image_tokens = single_image_embeds.get_shape().at(0) * single_image_embeds.get_shape().at(1); // num_patches * num_image_tokens
                const float* image_embeds_data = single_image_embeds.data<float>();
                std::copy_n(image_embeds_data + image_context_token_idx * embed_dim,
                            embed_dim,
                            merged_embeds_data + offset);
                
                ++image_context_token_idx;

                if (image_context_token_idx == num_all_image_tokens) {
                    ++image_idx;
                    image_context_token_idx = 0;
                }
            } else {
                std::copy_n(text_embeds_data + offset, embed_dim, merged_embeds_data + offset);
            }
        }
    }

    return merged_embeds;
}

} // namespace

InputsEmbedderInternVLChat::InputsEmbedderInternVLChat(
    const VLMConfig& vlm_config,
    const std::filesystem::path& model_dir,
    const std::string& device,
    const ov::AnyMap device_config) :
    IInputsEmbedder(vlm_config, model_dir, device, device_config) { }

InputsEmbedderInternVLChat::InputsEmbedderInternVLChat(
    const VLMConfig& vlm_config,
    const ModelsMap& models_map,
    const Tokenizer& tokenizer,
    const std::filesystem::path& config_dir_path,
    const std::string& device,
    const ov::AnyMap device_config) :
    IInputsEmbedder(vlm_config, models_map, tokenizer, config_dir_path, device, device_config) { }

InputsEmbedderInternVLChat::InputsEmbedderInternVLChat(
    const VLMConfig& vlm_config,
    const Tokenizer& tokenizer,
    VisionEncoderImpl::Ptr vision_encoder_impl,
    EmbeddingsModelImpl::Ptr embeddings_model_impl) :
    IInputsEmbedder(vlm_config, tokenizer, vision_encoder_impl, embeddings_model_impl) { }


NormalizedPrompt InputsEmbedderInternVLChat::normalize_prompt(const std::string& prompt, size_t base_id, const std::vector<EncodedImage>& images) const {
    auto [unified_prompt, images_sequence] = normalize(prompt, NATIVE_TAG, NATIVE_TAG + '\n', base_id, images.size());
    
    std::string image_start_token = m_vlm_config.image_start_token;
    std::string image_context_token = m_vlm_config.image_context_token;
    std::string image_end_token = m_vlm_config.image_end_token;
    std::vector<ov::Tensor> image_embeds;
    image_embeds.reserve(images_sequence.size());
    size_t searched_pos = 0;
    for (size_t new_image_id : images_sequence) {
        image_embeds.push_back(images.at(new_image_id - base_id).resized_source);

        const size_t num_patches = image_embeds.back().get_shape().at(0);
        const size_t num_image_tokens = image_embeds.back().get_shape().at(1);
        
        std::string expanded_tag{image_start_token};
        for (size_t idx = 0; idx < num_patches * num_image_tokens; ++idx) {
            expanded_tag += image_context_token;
        }
        expanded_tag += image_end_token;
        OPENVINO_ASSERT(searched_pos < unified_prompt.length());
        searched_pos = unified_prompt.find(NATIVE_TAG, searched_pos);
        OPENVINO_ASSERT(searched_pos != std::string::npos);
        unified_prompt.replace(searched_pos, NATIVE_TAG.length(), expanded_tag);
        searched_pos += expanded_tag.length();
    }

    return {std::move(unified_prompt), std::move(images_sequence), {}};
}

ov::Tensor InputsEmbedderInternVLChat::get_inputs_embeds(const std::string& unified_prompt, const std::vector<ov::genai::EncodedImage>& images, ov::genai::VLMPerfMetrics& metrics, bool recalculate_merged_embeddings, const std::vector<size_t>& images_sequence) {
    std::vector<ov::Tensor> image_embeds;
    image_embeds.reserve(images_sequence.size());
    size_t searched_pos = 0;
    for (size_t new_image_id : images_sequence) {
        image_embeds.push_back(images.at(new_image_id).resized_source);
    }
    std::string image_context_token = m_vlm_config.image_context_token;

    ov::Tensor input_ids = get_encoded_input_ids(unified_prompt, metrics);
    CircularBufferQueueElementGuard<EmbeddingsRequest> embeddings_request_guard(m_embedding->get_request_queue().get());
    EmbeddingsRequest& req = embeddings_request_guard.get();
    ov::Tensor text_embeds = m_embedding->infer(req, input_ids);

    if (images.empty()) {
        ov::Tensor inputs_embeds(text_embeds.get_element_type(), text_embeds.get_shape());
        std::memcpy(inputs_embeds.data(), text_embeds.data(), text_embeds.get_byte_size());
        return inputs_embeds;
    }
    auto start_tokenizer_time = std::chrono::steady_clock::now();
    ov::Tensor encoded_image_context_token = m_tokenizer.encode(image_context_token, ov::genai::add_special_tokens(false)).input_ids;
    auto end_tokenizer_time = std::chrono::steady_clock::now();
    OPENVINO_ASSERT(metrics.raw_metrics.tokenization_durations.size() > 0);
    metrics.raw_metrics.tokenization_durations[metrics.raw_metrics.tokenization_durations.size() - 1] += ov::genai::MicroSeconds(PerfMetrics::get_microsec(end_tokenizer_time - start_tokenizer_time));
    int64_t image_context_token_id = encoded_image_context_token.data<int64_t>()[encoded_image_context_token.get_size() - 1];
    return merge_text_and_image_embeddings_internvl(input_ids, text_embeds, image_embeds, image_context_token_id);
}

} // namespace ov::genai
