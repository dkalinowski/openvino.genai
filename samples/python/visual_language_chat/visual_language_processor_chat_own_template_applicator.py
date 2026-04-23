#!/usr/bin/env python3
# Copyright (C) 2024-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Sample showing how to apply the chat template manually.

Usage:
    python visual_language_processor_chat_own_template_applicator.py <model_dir> <image_file> [device]

Unlike ``visual_language_processor_chat.py`` (which delegates chat-template
application to ``VLMProcessor.embed(ChatHistory, ...)``), this sample formats
the prompt on the user side and then calls the string overload
``VLMProcessor.embed(prompt, images)``. Useful when the caller wants full
control over the prompt (custom template, extra system instructions, tool
definitions, etc.).
"""

import argparse
from pathlib import Path

import numpy as np
import openvino_genai
from PIL import Image
from openvino import Tensor


def streamer(subword: str) -> bool:
    print(subword, end="", flush=True)


def read_image(path: str) -> Tensor:
    pic = Image.open(path).convert("RGB")
    return Tensor(np.array(pic))


def read_images(path: str) -> list[Tensor]:
    entry = Path(path)
    if entry.is_dir():
        return [read_image(str(file)) for file in sorted(entry.iterdir())]
    return [read_image(path)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model_dir", help="Path to the model directory")
    parser.add_argument("image_dir", help="Image file or directory with images")
    parser.add_argument("device", nargs="?", default="CPU", help="Inference device (default: CPU)")
    args = parser.parse_args()

    rgbs = read_images(args.image_dir)

    # Step 1: construct the processor and pipeline.
    processor = openvino_genai.VLMProcessor(args.model_dir, args.device)
    pipe = openvino_genai.VLMPipeline(args.model_dir, processor, args.device)

    config = openvino_genai.GenerationConfig()
    config.max_new_tokens = 100

    prompt = input("question:\n")

    # Step 2: apply the chat template manually using the processor's tokenizer.
    # This gives the caller full control — the string could just as well be
    # assembled by hand or produced by an external templating engine.
    tokenizer = processor.get_tokenizer()
    history = openvino_genai.ChatHistory()
    history.append({"role": "user", "content": prompt})
    formatted_prompt = tokenizer.apply_chat_template(history, True)
    print(f"\n[formatted prompt]\n{formatted_prompt}\n[/formatted prompt]\n")

    # Step 3: use the string overload of embed() — the processor does NOT
    # apply a template, the pre-formatted string is used as-is.
    inputs = processor.embed(formatted_prompt, images=rgbs)
    print(f"[inputs_embeds shape: {inputs.inputs_embeds.shape}]\n")

    # Step 4: run generation on the pre-computed embeddings.
    pipe.generate(inputs, config, streamer)
    print()


if "__main__" == __name__:
    main()
