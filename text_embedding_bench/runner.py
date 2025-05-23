# -*- coding: utf-8 -*-
"""
Embedding benchmark runner for Milvus text embedding function.
"""

import time
import os  # For file and directory operations
import pandas as pd
from .providers import PROVIDERS_MODELS
from .utils import generate_fake_text, ensure_results_dir
from pymilvus import (
    Collection,
    DataType,
    FieldSchema,
    CollectionSchema,
    utility,
    Function,
    FunctionType,
    connections,
)
from loguru import logger


class EmbeddingBenchRunner:
    def __init__(self):
        self.results = []
        self.batch_size_list = [1, 10]
        self.connections = connections.connect(uri="http://localhost:19530")

    def run(self):
        """Run benchmark for all providers and models."""
        providers_models = PROVIDERS_MODELS.copy()

        token_variations = [
            {"name": "256_tokens", "text": generate_fake_text(256), "tokens": 256},
            {"name": "512_tokens", "text": generate_fake_text(512), "tokens": 512},
            {"name": "1024_tokens", "text": generate_fake_text(1024), "tokens": 1024},
            {"name": "2048_tokens", "text": generate_fake_text(2048), "tokens": 2048},
            {"name": "4096_tokens", "text": generate_fake_text(4096), "tokens": 4096},
            {"name": "8192_tokens", "text": generate_fake_text(8192), "tokens": 8192},
        ]
        for provider, models in providers_models.items():
            for model in models:
                model_name = model["name"]
                dim = model["dim"]
                schema = CollectionSchema(
                    [
                        FieldSchema("id", DataType.INT64, is_primary=True),
                        FieldSchema("text", DataType.VARCHAR, max_length=65535),
                        FieldSchema("embedding", DataType.FLOAT_VECTOR, dim=dim),
                    ]
                )
                params = {"provider": provider, "model_name": model_name}
                if "params" in model:
                    params.update(model["params"])
                text_embedding_function = Function(
                    name=f"{provider}_{model_name.replace('/', '_')}_func",
                    function_type=FunctionType.TEXTEMBEDDING,
                    input_field_names=["text"],
                    output_field_names=["embedding"],
                    params=params,
                )
                schema.add_function(text_embedding_function)
                model_name_safe = (
                    model_name.replace("/", "_")
                    .replace(".", "_")
                    .replace(":", "_")
                    .replace("-", "_")
                )
                collection_name = f"test_text_embedding_perf_{provider}_{model_name_safe}_{int(time.time())}"
                try:
                    collection = Collection(collection_name, schema)
                    res = collection.describe()
                    logger.info(f"Collection {collection_name} created: {res}")
                    for token_var in token_variations:
                        test_text = token_var["text"]
                        token_count = token_var["tokens"]
                        token_name = token_var["name"]
                        for batch_size in self.batch_size_list:
                            # Run 10 times for each batch_size to reduce result variance
                            for i in range(10):
                                try:
                                    # Generate batch_size rows of data, id is unique, text is the same
                                    data = [
                                        {"id": j, "text": test_text}
                                        for j in range(batch_size)
                                    ]
                                    start_time = time.time()
                                    collection.insert(data)
                                    latency = time.time() - start_time
                                    self.results.append(
                                        {
                                            "provider": provider,
                                            "model": model_name,
                                            "token_count": token_count,
                                            "token_name": token_name,
                                            "latency": latency,
                                            "tokens_per_second": token_count / latency,
                                            "batch_size": batch_size,  # record batch size
                                            "status": "success",
                                        }
                                    )
                                    logger.info(
                                        f"{provider} - {model_name} - {token_name} ({token_count} tokens, batch_size={batch_size}) [Run {i + 1}/10]: {latency:.3f}s"
                                    )
                                except Exception as e:
                                    logger.error(
                                        f"Error testing {provider} - {model_name} with {token_count} tokens, batch_size={batch_size} (Run {i + 1}/10): {str(e)}"
                                    )
                                    self.results.append(
                                        {
                                            "provider": provider,
                                            "model": model_name,
                                            "token_count": token_count,
                                            "token_name": token_name,
                                            "latency": None,
                                            "tokens_per_second": None,
                                            "batch_size": batch_size,  # record batch size
                                            "status": f"error: {str(e)}",
                                        }
                                    )
                                    # If the first run fails, stop further attempts for this config, most of the time it is due to token limit
                                    if i == 0:
                                        break

                except Exception as e:
                    logger.error(
                        f"Error setting up {provider} - {model_name}: {str(e)}"
                    )
                    self.results.append(
                        {
                            "provider": provider,
                            "model": model_name,
                            "token_count": "N/A",
                            "token_name": "N/A",
                            "latency": None,
                            "tokens_per_second": None,
                            "test_type": "setup",
                            "status": f"setup error: {str(e)}",
                        }
                    )
                finally:
                    try:
                        utility.drop_collection(collection_name)
                    except Exception:
                        pass

    def save_and_report(self):
        """Save results to CSV and print summary tables."""
        df = pd.DataFrame(self.results)
        if df.empty:
            logger.warning("No successful tests completed.")
            return
        # Use the current working directory as the root for results_dir
        results_dir = ensure_results_dir(os.getcwd())
        from datetime import datetime

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_filename = os.path.join(
            results_dir, f"embedding_performance_{timestamp}.csv"
        )
        df.to_csv(csv_filename, index=False)
        logger.info(f"Detailed results saved to: {csv_filename}")
        # Print summary by model
        import numpy as np  # Add numpy for percentile calculation

        # Add std, median, p95, p99 to the summary
        model_summary = (
            df[df["status"] == "success"]
            .groupby(["provider", "model", "token_count", "token_name", "batch_size"])[
                "latency"
            ]
            .agg(
                [
                    "mean",
                    "std",
                    "median",
                    "min",
                    "max",
                    (lambda x: np.percentile(x, 95)),
                    (lambda x: np.percentile(x, 99)),
                ]
            )
            .rename(columns={"<lambda_0>": "p95", "<lambda_1>": "p99"})
        )
        logger.info(
            "\nPerformance Summary by Model (with std/median/p95/p99):\n"
            + str(model_summary)
        )
