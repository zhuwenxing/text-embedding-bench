# -*- coding: utf-8 -*-
"""
Main entry for embedding benchmark tool.
"""

from text_embedding_bench.runner import EmbeddingBenchRunner


def main():
    runner = EmbeddingBenchRunner()
    runner.run()
    runner.save_and_report()


if __name__ == "__main__":
    main()
