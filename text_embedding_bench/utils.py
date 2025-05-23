# -*- coding: utf-8 -*-
"""
Utility functions for embedding benchmark.
"""

import os
from faker import Faker

fake_en = Faker()


def generate_fake_text(token_count: int) -> str:
    """
    Generate English text with slightly fewer than the specified number of tokens.
    To avoid provider tokenization differences, a 10% buffer is reserved (actual tokens = int(token_count * 0.9)).
    Uses tiktoken for accurate tokenization. Raises ImportError if tiktoken is not installed.
    """
    import tiktoken

    enc = tiktoken.get_encoding("cl100k_base")
    buffered_token_count = max(1, int(token_count * 0.9))
    words = []
    # Generate words until buffered token count is reached
    while True:
        words.append(fake_en.word())
        text = " ".join(words)
        tokens = len(enc.encode(text))
        if tokens >= buffered_token_count:
            break
    return text


def ensure_results_dir(base_dir: str = None) -> str:
    """Ensure the results directory exists and return its path."""
    if base_dir is None:
        base_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(base_dir, "results")
    os.makedirs(results_dir, exist_ok=True)
    return results_dir
