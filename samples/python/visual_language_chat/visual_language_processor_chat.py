#!/usr/bin/env python3
# Copyright (C) 2024-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Sample demonstrating the VLMProcessor + VLMPipeline split.

Usage:
    python visual_language_processor_chat.py <model_dir> <image_file> [device]

The processor handles vision encoding, text tokenization, and embedding merging
and returns an ``Embeddings`` object. The pipeline, constructed from the same
processor, owns only the language model and runs generation on the pre-computed
embeddings.
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

    # Step 1: construct the processor. It owns the vision encoder,
    # text embeddings model, and tokenizer.
    processor = openvino_genai.VLMProcessor(args.model_dir, args.device)

    # Step 2: construct the pipeline from the processor. It owns only
    # the language model and reuses the processor's internals.
    pipe = openvino_genai.VLMPipeline(args.model_dir, processor, args.device)

    config = openvino_genai.GenerationConfig()
    config.max_new_tokens = 100

    prompt = input("question:\n")

    # Apply the chat template explicitly — embed() operates on a raw prompt.
    tokenizer = processor.get_tokenizer()
    history = openvino_genai.ChatHistory()
    history.append({"role": "user", "content": prompt})
    formatted_prompt = tokenizer.apply_chat_template(history, True)

    # Step 3: run the processor to produce Embeddings.
    inputs = processor.embed(formatted_prompt, images=rgbs)
    print(f"\n[inputs_embeds shape: {inputs.inputs_embeds.shape}]\n")

    # Step 4: feed Embeddings to the pipeline.
    pipe.generate(inputs, config, streamer)
    print()


if "__main__" == __name__:
    main()
