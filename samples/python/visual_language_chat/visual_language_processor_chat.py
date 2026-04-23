#!/usr/bin/env python3
# Copyright (C) 2024-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Sample demonstrating multi-turn chat with the VLMProcessor + VLMPipeline split.

Usage:
    python visual_language_processor_chat.py <model_dir> <image_file> [device]

The caller owns the ChatHistory. On every turn, the full history is passed
to ``processor.embed()`` which applies the chat template, runs the vision
encoder, and merges embeddings. The pipeline fully resets its KV cache on
each ``generate()`` call, so no ``start_chat()`` / ``finish_chat()`` is
needed on this path.
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

    processor = openvino_genai.VLMProcessor(args.model_dir, args.device)
    pipe = openvino_genai.VLMPipeline(args.model_dir, processor, args.device)

    config = openvino_genai.GenerationConfig()
    config.max_new_tokens = 100

    history = openvino_genai.ChatHistory()
    first_turn = True

    while True:
        try:
            prompt = input("\nquestion:\n")
        except EOFError:
            break
        if not prompt:
            break

        history.append({"role": "user", "content": prompt})

        # The pipeline keeps its KV cache across calls and reuses vision
        # tokens already prefilled on turn 1. Supply images only on the
        # turn where they are first referenced; subsequent text-only
        # turns call processor.embed(history) with no images argument.
        turn_images = rgbs if not first_turn else []
        inputs = processor.embed(history, images=turn_images)
        first_turn = False

        print()
        result = pipe.generate(inputs, config, streamer)
        print()

        history.append({"role": "assistant", "content": result.texts[0]})


if "__main__" == __name__:
    main()

