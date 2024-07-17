
// Copyright (C) 2023-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <mutex>
#include <algorithm>
#include <atomic>
#include <queue>
#include <future>
#include "openvino/runtime/core.hpp"

#include "tokenizer.hpp"
#include "timer.hpp"

// From OVMS
template <typename T>
class Queue {
    int front_idx;
    std::atomic<int> back_idx;
    std::vector<int> values;
    std::queue<std::promise<int>> promises;
    std::vector<T> data;
    std::mutex front_mut;
    std::mutex queue_mutex;
public:
    Queue(size_t length, const std::function<T()>& create_fn) :
        values(length),
        front_idx{0},
        back_idx{0} {
        std::iota(values.begin(), values.end(), 0);
        data.reserve(length);
        for (size_t i = 0; i < length; i++) {
            data.emplace_back(std::move(create_fn()));
        }
    }

    T& get(int value) {
        return data[value];
    }

    std::future<int> get_idle() {
        int value;
        std::promise<int> idle_promise;
        std::future<int> idle_future = idle_promise.get_future();
        std::unique_lock<std::mutex> lk(front_mut);
        if (values[front_idx] < 0) {
            std::unique_lock<std::mutex> queueLock(queue_mutex);
            promises.push(std::move(idle_promise));
        } else {
            value = values[front_idx];
            values[front_idx] = -1;
            front_idx = (front_idx + 1) % values.size();
            lk.unlock();
            idle_promise.set_value(value);
        }
        return idle_future;
    }
    
    void return_to(int value) {
        std::unique_lock<std::mutex> lk(queue_mutex);
        if (promises.size()) {
            std::promise<int> promise = std::move(promises.front());
            promises.pop();
            lk.unlock();
            promise.set_value(value);
            return;
        }
        int old_back = back_idx.load();
        while (!back_idx.compare_exchange_weak(
            old_back,
            (old_back + 1) % values.size(),
            std::memory_order_relaxed)) {
        }
        values[old_back] = value;
    }
};

template <typename T>
class QueueAccessGuard {
    Queue<T>& queue;
    int value;
public:
    QueueAccessGuard(Queue<T>& queue) : queue(queue) {
        value = queue.get_idle().get();
    }

    T& get() {
        return queue.get(value);
    }

    ~QueueAccessGuard() {
        queue.return_to(value);
    }
};

class Tokenizer::Impl {
    const size_t TOKENIZER_BATCH_SIZE = 1;
    ov::CompiledModel m_tokenizer;
    ov::CompiledModel m_detokenizer;
    std::size_t m_eos_token_id;

    std::unique_ptr<Queue<ov::InferRequest>> m_ireq_queue_tokenizer;
    std::unique_ptr<Queue<ov::InferRequest>> m_ireq_queue_detokenizer;

public:
    explicit Impl(const std::string& models_path)
    {
        ov::Core core;
        core.add_extension(OPENVINO_TOKENIZERS_PATH);  // OPENVINO_TOKENIZERS_PATH is defined in CMakeLists.txt

        std::shared_ptr<ov::Model> tokenizer_model = core.read_model(models_path + "/openvino_tokenizer.xml");
        const ov::AnyMap& rt_info = tokenizer_model->get_rt_info();
        OPENVINO_ASSERT(rt_info.find("eos_token_id") != rt_info.end(), "Failed to detect \"eos_token_id\" in openvino_tokenizer.xml runtime information");
        m_eos_token_id = rt_info.at("eos_token_id").as<int64_t>();

        const size_t INFER_REQUEST_QUEUE_SIZE = 256;

        // tokenizer and detokenizer work on CPU only
        std::map<std::string, ov::Any> pluginConfig;
        pluginConfig["PERFORMANCE_HINT"] = "LATENCY";

        m_tokenizer = core.compile_model(
            tokenizer_model, "CPU", pluginConfig);
        m_ireq_queue_tokenizer = std::make_unique<Queue<ov::InferRequest>>(
            INFER_REQUEST_QUEUE_SIZE,
            [this]() -> ov::InferRequest {
                return std::move(this->m_tokenizer.create_infer_request());
            });

        m_detokenizer = core.compile_model(
            models_path + "/openvino_detokenizer.xml", "CPU", pluginConfig);
        m_ireq_queue_detokenizer = std::make_unique<Queue<ov::InferRequest>>(
            INFER_REQUEST_QUEUE_SIZE,
            [this]() -> ov::InferRequest {
                return std::move(this->m_detokenizer.create_infer_request());
            });
    }

    ov::Tensor encode(std::string prompt) {
        static ManualTimer timer("tokenize encode");
        timer.start();
        QueueAccessGuard<ov::InferRequest> guard(*m_ireq_queue_tokenizer);
        auto& ireq = guard.get();
        ireq.set_input_tensor(ov::Tensor{ov::element::string, {TOKENIZER_BATCH_SIZE}, &prompt});
        ireq.infer();
        ov::Tensor tmp_tensor = ireq.get_tensor("input_ids");
        ov::Tensor output_tensor(tmp_tensor.get_element_type(), tmp_tensor.get_shape());
        tmp_tensor.copy_to(output_tensor);
        timer.end();
        return std::move(output_tensor);
    }

    std::string decode(std::vector<int64_t> tokens) {
        static ManualTimer timer("tokenize decode");
        timer.start();
        QueueAccessGuard<ov::InferRequest> guard(*m_ireq_queue_detokenizer);
        auto& ireq = guard.get();
        ireq.set_input_tensor(ov::Tensor{ov::element::i64, {TOKENIZER_BATCH_SIZE, tokens.size()}, tokens.data()});
        ireq.infer();
        std::string out = ireq.get_output_tensor().data<std::string>()[0];
        timer.end();
        return std::move(out);
    }

    size_t get_eos_token_id() const {
        return m_eos_token_id;
    }
};

Tokenizer::Tokenizer(const std::string& models_path) {
    m_impl = std::make_shared<Impl>(models_path);
}

ov::Tensor Tokenizer::encode(std::string prompt) {
    return m_impl->encode(std::move(prompt));
}

std::string Tokenizer::decode(std::vector<int64_t> tokens) {
    return m_impl->decode(std::move(tokens));
}

size_t Tokenizer::get_eos_token_id() const {
    return m_impl->get_eos_token_id();
}
