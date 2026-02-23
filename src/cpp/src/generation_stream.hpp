// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <mutex>
#include <atomic>
#include "openvino/genai/continuous_batching_pipeline.hpp"
#include "openvino/genai/generation_handle.hpp"
#include "synchronized_queue.hpp"

namespace ov::genai {
class GenerationStream {
    std::mutex m_mutex;
    GenerationStatus m_status = GenerationStatus::RUNNING;
    SynchronizedQueue<GenerationOutputs> m_output_queue;

    std::condition_variable m_prefill_cv;
    bool m_prefill_finished = false;

public:
    using Ptr = std::shared_ptr<GenerationStream>;

    // Don't use directly
    GenerationStream() = default;

    static GenerationStream::Ptr create() {
        return std::make_shared<GenerationStream>();
    }

    void push(GenerationOutputs outputs) {
        m_output_queue.push(std::move(outputs));
    }

    GenerationOutputs read() {
        return m_output_queue.pull();
    }

    bool can_read() {
        return !m_output_queue.empty();
    }

    void set_generation_status(GenerationStatus status) {
        {
            std::lock_guard<std::mutex> lock(m_mutex);
            m_status = status;
        }
        m_prefill_cv.notify_all();
    }

    GenerationStatus get_status() {
        std::lock_guard<std::mutex> lock(m_mutex);
        return m_status;
    }

    void stop() {
        {
            std::lock_guard<std::mutex> lock(m_mutex);
            m_status = GenerationStatus::STOP;
        }
        m_prefill_cv.notify_all();
    }

    void cancel() {
        {
            std::lock_guard<std::mutex> lock(m_mutex);
            m_status = GenerationStatus::CANCEL;
        }
        m_prefill_cv.notify_all();
    }

    /// @brief Called by the pipeline when prefill (prompt processing) completes for this request.
    void set_prefill_finished() {
        {
            std::lock_guard<std::mutex> lock(m_mutex);
            m_prefill_finished = true;
        }
        m_prefill_cv.notify_all();
    }

    /// @brief Blocks until prefill completes or generation is stopped/cancelled.
    /// @return true when prefill finished normally, false if generation was stopped/cancelled first.
    bool wait_for_prefill() {
        std::unique_lock<std::mutex> lock(m_mutex);
        m_prefill_cv.wait(lock, [this] {
            return m_prefill_finished
                || m_status == GenerationStatus::CANCEL
                || m_status == GenerationStatus::STOP
                || m_status == GenerationStatus::FINISHED;
        });
        return m_prefill_finished;
    }
};
}
