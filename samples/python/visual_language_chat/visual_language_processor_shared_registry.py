#!/usr/bin/env python3
# Copyright (C) 2024-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Sample showing the benefit of sharing a VLMProcessor across pipelines.

A VLMProcessor owns the vision encoder, text embedder and VisionRegistry.
Pipelines built from the same processor reuse those components, so:
  * construction is cheaper than building an independent VLMPipeline;
  * vision encoding is done once inside ``processor.embed()`` and the
    resulting ``Embeddings`` can be fed to multiple pipelines — each
    subsequent ``pipe.generate(inputs, ...)`` call skips image encoding.

Usage:
    python visual_language_processor_shared_registry.py <model_dir> <image_file> [device]
"""

import argparse
import time
from difflib import unified_diff
from pathlib import Path

import numpy as np
import openvino_genai
from PIL import Image
from openvino import Tensor


PROMPT = "Describe the image in one short sentence."
MAX_NEW_TOKENS = 10


def read_image(path: str) -> Tensor:
    pic = Image.open(path).convert("RGB")
    return Tensor(np.array(pic))


def read_images(path: str) -> list[Tensor]:
    entry = Path(path)
    if entry.is_dir():
        return [read_image(str(file)) for file in sorted(entry.iterdir())]
    return [read_image(path)]


def measure(label: str, fn):
    start = time.perf_counter()
    result = fn()
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    print(f"[{elapsed_ms:8.1f} ms] {label}")
    return result, elapsed_ms


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model_dir")
    parser.add_argument("image_dir")
    parser.add_argument("device", nargs="?", default="CPU")
    args = parser.parse_args()

    rgbs = read_images(args.image_dir)

    config = openvino_genai.GenerationConfig()
    config.max_new_tokens = MAX_NEW_TOKENS
    config.do_sample = False  # greedy, deterministic — outputs must match

    print("=== construction ===")
    # Processor owns the vision encoder + text embedder + VisionRegistry.
    processor, _ = measure(
        "VLMProcessor(model_dir, device)",
        lambda: openvino_genai.VLMProcessor(args.model_dir, args.device),
    )
    # Two pipelines built from the same processor reuse its components.
    pipe_a, _ = measure(
        "VLMPipeline(model_dir, processor, device)  (shared #A)",
        lambda: openvino_genai.VLMPipeline(args.model_dir, processor, args.device),
    )
    pipe_b, _ = measure(
        "VLMPipeline(model_dir, processor, device)  (shared #B)",
        lambda: openvino_genai.VLMPipeline(args.model_dir, processor, args.device),
    )
    # Independent pipeline — loads its own vision encoder + embedder.
    pipe_standalone, _ = measure(
        "VLMPipeline(model_dir, device)              (standalone)",
        lambda: openvino_genai.VLMPipeline(args.model_dir, args.device),
    )
    # Two more pipelines sharing the same processor, used via the regular
    # generate(prompt, images) string API (no pre-computed Embeddings).
    pipe_c, _ = measure(
        "VLMPipeline(model_dir, processor, device)  (shared #C, string API)",
        lambda: openvino_genai.VLMPipeline(args.model_dir, processor, args.device),
    )
    pipe_d, _ = measure(
        "VLMPipeline(model_dir, processor, device)  (shared #D, string API)",
        lambda: openvino_genai.VLMPipeline(args.model_dir, processor, args.device),
    )

    # First generate() on a pipeline pays a one-time LM warmup cost.
    # Warm every pipeline up with a synthetic image (different from the real
    # input) so the measurements below do not benefit from any incidental
    # per-image caching.
    warmup_config = openvino_genai.GenerationConfig()
    warmup_config.max_new_tokens = 1
    warmup_image = Tensor(np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8))
    for pipe in (pipe_standalone, pipe_a, pipe_b, pipe_c, pipe_d):
        pipe.generate(PROMPT, images=[warmup_image], generation_config=warmup_config)

    print("\n=== standalone pipeline: re-encodes the image on every generate() ===")
    # Each call to generate(prompt, images) runs the vision encoder internally —
    # there is no caching on the non-chat string-prompt path.
    standalone_res_1, standalone_1 = measure(
        "standalone.generate(prompt, images)  #1",
        lambda: pipe_standalone.generate(PROMPT, images=rgbs, generation_config=config),
    )
    standalone_res_2, standalone_2 = measure(
        "standalone.generate(prompt, images)  #2",
        lambda: pipe_standalone.generate(PROMPT, images=rgbs, generation_config=config),
    )

    print("\n=== shared processor: encode once, reuse the Embeddings across pipelines ===")
    # processor.embed() runs the vision encoder and returns Embeddings.
    # The same Embeddings object is fed to multiple pipelines, so the vision
    # encoder is NOT invoked again inside pipe.generate(inputs, ...).
    inputs, embed_ms = measure(
        "processor.embed(prompt, images)         (one-time)",
        lambda: processor.embed(PROMPT, images=rgbs),
    )
    shared_a_res, shared_a = measure(
        "pipe_A.generate(inputs)                 (reuses embeddings)",
        lambda: pipe_a.generate(inputs, config),
    )
    shared_b_res, shared_b = measure(
        "pipe_B.generate(inputs)                 (reuses embeddings)",
        lambda: pipe_b.generate(inputs, config),
    )

    print("\n=== shared processor + string API: does sharing alone speed up generate()? ===")
    # pipe_C and pipe_D share the processor's InputsEmbedder, but the string
    # overload re-runs the vision encoder on every call (non-chat path does
    # not consult VisionRegistry today). So sharing the processor does NOT
    # help generate(prompt, images) — it only helps construction time and
    # the Embeddings-based API above.
    shared_c_res, shared_string_c = measure(
        "pipe_C.generate(prompt, images)         (shares processor, string API)",
        lambda: pipe_c.generate(PROMPT, images=rgbs, generation_config=config),
    )
    shared_d_res, shared_string_d = measure(
        "pipe_D.generate(prompt, images)         (shares processor, string API)",
        lambda: pipe_d.generate(PROMPT, images=rgbs, generation_config=config),
    )

    print("\n=== summary (total time to answer the same question twice) ===")
    standalone_total = standalone_1 + standalone_2
    shared_total = embed_ms + shared_a + shared_b
    shared_string_total = shared_string_c + shared_string_d
    print(f"  standalone                     (2x encode+generate): {standalone_total:8.1f} ms")
    print(f"  shared processor, string API   (2x encode+generate): {shared_string_total:8.1f} ms")
    print(f"  shared processor, Embeddings   (1x embed + 2x gen):  {shared_total:8.1f} ms")
    speedup = standalone_total / shared_total if shared_total > 0 else float("nan")
    print(f"  Embeddings path vs standalone: {speedup:.2f}x faster")

    print(f"\n  inputs_embeds shape: {inputs.inputs_embeds.shape}")

    print("\n=== output equality check (greedy decoding → outputs must match) ===")
    outputs = {
        "standalone #1":            standalone_res_1.texts[0],
        "standalone #2":            standalone_res_2.texts[0],
        "pipe_A (embeddings)":      shared_a_res.texts[0],
        "pipe_B (embeddings)":      shared_b_res.texts[0],
        "pipe_C (shared, string)":  shared_c_res.texts[0],
        "pipe_D (shared, string)":  shared_d_res.texts[0],
    }
    for label, text in outputs.items():
        print(f"  {label:28s}: {text!r}")

    reference_label, reference_text = next(iter(outputs.items()))
    all_equal = True
    for label, text in outputs.items():
        if text == reference_text:
            continue
        all_equal = False
        print(f"\n  DIFF: {reference_label}  vs  {label}")
        diff = unified_diff(
            reference_text.splitlines(keepends=True) or [""],
            text.splitlines(keepends=True) or [""],
            fromfile=reference_label,
            tofile=label,
        )
        for line in diff:
            print(f"    {line.rstrip()}")
    print(f"\n  all outputs equal: {all_equal}")


if "__main__" == __name__:
    main()
