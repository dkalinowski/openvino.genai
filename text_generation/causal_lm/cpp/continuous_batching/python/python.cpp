// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "pybind11/pybind11.h"
#include <pybind11/stl.h>

#include "continuous_batching_pipeline.hpp"

namespace py = pybind11;

PYBIND11_MODULE(py_continuous_batching, m) {
    py::class_<GenerationResult>(m, "GenerationResult")
        .def_readonly("m_request_id", &GenerationResult::m_request_id)
        .def_readonly("m_generation_ids", &GenerationResult::m_generation_ids)
        .def_readonly("m_scores", &GenerationResult::m_scores);

    py::enum_<StopCriteria>(m, "StopCriteria")
        .value("EARLY", StopCriteria::EARLY)
        .value("HEURISTIC", StopCriteria::HEURISTIC)
        .value("NEVER", StopCriteria::NEVER)
        .export_values();

    py::class_<GenerationConfig>(m, "GenerationConfig")
        .def(py::init<>())
        .def_readwrite("max_new_tokens", &GenerationConfig::max_new_tokens)
        .def_readwrite("max_length", &GenerationConfig::max_length)
        .def_readwrite("ignore_eos", &GenerationConfig::ignore_eos)
        .def_readwrite("num_groups", &GenerationConfig::num_groups)
        .def_readwrite("group_size", &GenerationConfig::group_size)
        .def_readwrite("diversity_penalty", &GenerationConfig::diversity_penalty)
        .def_readwrite("stop_criteria", &GenerationConfig::stop_criteria)
        .def_readwrite("num_return_sequences", &GenerationConfig::num_return_sequences)
        .def_readwrite("repetition_penalty", &GenerationConfig::repetition_penalty)
        .def_readwrite("length_penalty", &GenerationConfig::length_penalty)
        .def_readwrite("no_repeat_ngram_size", &GenerationConfig::no_repeat_ngram_size)
        .def_readwrite("temperature", &GenerationConfig::temperature)
        .def_readwrite("top_k", &GenerationConfig::top_k)
        .def_readwrite("top_p", &GenerationConfig::top_p)
        .def_readwrite("do_sample", &GenerationConfig::do_sample);

    py::class_<SchedulerConfig>(m, "SchedulerConfig")
        .def(py::init<>())
        .def_readwrite("max_num_batched_tokens", &SchedulerConfig::max_num_batched_tokens)
        .def_readwrite("num_kv_blocks", &SchedulerConfig::num_kv_blocks)
        .def_readwrite("block_size", &SchedulerConfig::block_size)
        .def_readwrite("dynamic_split_fuse", &SchedulerConfig::dynamic_split_fuse)
        .def_readwrite("max_num_seqs", &SchedulerConfig::max_num_seqs)
        .def_readwrite("max_paddings", &SchedulerConfig::max_paddings);

    py::class_<ContinuousBatchingPipeline>(m, "ContinuousBatchingPipeline")
        .def(py::init<const std::string &, const SchedulerConfig&>())
        .def("get_tokenizer", &ContinuousBatchingPipeline::get_tokenizer)
        .def("get_config", &ContinuousBatchingPipeline::get_config)
        .def("add_request", &ContinuousBatchingPipeline::add_request)
        .def("step", &ContinuousBatchingPipeline::step)
        .def("has_running_requests", &ContinuousBatchingPipeline::has_running_requests)
        .def("generate", &ContinuousBatchingPipeline::generate);

    py::class_<Tokenizer, std::shared_ptr<Tokenizer>>(m, "Tokenizer")
        .def(py::init<const std::string&>())
        .def("encode", &Tokenizer::encode)
        .def("decode", &Tokenizer::decode)
        .def("get_eos_token_id", &Tokenizer::get_eos_token_id);
}