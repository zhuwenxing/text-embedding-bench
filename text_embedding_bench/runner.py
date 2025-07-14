# -*- coding: utf-8 -*-
"""
Embedding benchmark runner for Milvus text embedding function.
"""

import time
import os  # For file and directory operations
import pandas as pd
import threading
import concurrent.futures
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
        self.search_top_k_list = [1, 10]
        self.qps_test_duration = 30  # QPS test duration in seconds (reduced for adaptive testing)
        self.qps_max_concurrent = 100  # Maximum concurrent level to test
        self.qps_improvement_threshold = 0.05  # 5% improvement threshold for adaptive testing
        self.qps_test_method = "exhaustive"  # "adaptive" or "exhaustive"
        self.qps_concurrent_step = 1  # Step size for exhaustive method
        self.connections = connections.connect(uri="http://10.104.13.2:19530")

    def run(self):
        """Run benchmark for all providers and models."""
        providers_models = PROVIDERS_MODELS.copy()

        token_variations = [
            # {"name": "256_tokens", "text": generate_fake_text(256), "tokens": 256},
            {"name": "512_tokens", "text": generate_fake_text(512), "tokens": 512},
            # {"name": "1024_tokens", "text": generate_fake_text(1024), "tokens": 1024},
            # {"name": "2048_tokens", "text": generate_fake_text(2048), "tokens": 2048},
            # {"name": "4096_tokens", "text": generate_fake_text(4096), "tokens": 4096},
            # {"name": "8192_tokens", "text": generate_fake_text(8192), "tokens": 8192},
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
                                            "operation": "insert",
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
                                            "operation": "insert",
                                            "status": f"error: {str(e)}",
                                        }
                                    )
                                    # If the first run fails, stop further attempts for this config, most of the time it is due to token limit
                                    if i == 0:
                                        break

                    # After insert testing, perform search testing
                    self._run_search_tests(collection, provider, model_name, token_variations)
                    
                    # After latency testing, perform QPS testing
                    self._run_qps_tests(collection, provider, model_name, token_variations)

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
                            "operation": "setup",
                            "status": f"setup error: {str(e)}",
                        }
                    )
                finally:
                    try:
                        utility.drop_collection(collection_name)
                    except Exception:
                        pass

    def _run_search_tests(self, collection, provider, model_name, token_variations):
        """Run search performance tests on the collection."""
        logger.info(f"Starting search tests for {provider} - {model_name}")
        
        # First, insert some test data for searching
        try:
            # Insert a small dataset for search testing (100 documents with different token lengths)
            search_data = []
            for i, token_var in enumerate(token_variations):
                # Insert 20 documents for each token variation
                for j in range(20):
                    search_data.append({
                        "id": i * 20 + j + 100000,  # Use high IDs to avoid conflicts
                        "text": token_var["text"]
                    })
            
            collection.insert(search_data)
            collection.flush()  # Ensure data is persisted
            
            # Create index for better search performance
            index_params = {"index_type": "FLAT", "metric_type": "COSINE"}
            collection.create_index("embedding", index_params)
            collection.load()  # Load collection into memory
            
            # Now run search tests
            for token_var in token_variations:
                query_text = token_var["text"]
                token_count = token_var["tokens"]
                token_name = token_var["name"]
                
                for top_k in self.search_top_k_list:
                    # Run 5 times for each top_k to reduce result variance
                    for i in range(5):
                        try:
                            start_time = time.time()
                            _ = collection.search(
                                data=[query_text],
                                anns_field="embedding",
                                param={"metric_type": "COSINE"},
                                limit=top_k,
                                output_fields=["id"]
                            )
                            latency = time.time() - start_time
                            
                            self.results.append({
                                "provider": provider,
                                "model": model_name,
                                "token_count": token_count,
                                "token_name": token_name,
                                "latency": latency,
                                "tokens_per_second": token_count / latency,
                                "batch_size": 1,  # Search is always single query
                                "top_k": top_k,
                                "operation": "search",
                                "status": "success",
                            })
                            
                            logger.info(
                                f"{provider} - {model_name} - SEARCH {token_name} ({token_count} tokens, top_k={top_k}) [Run {i + 1}/5]: {latency:.3f}s"
                            )
                            
                        except Exception as e:
                            logger.error(
                                f"Error in search test {provider} - {model_name} with {token_count} tokens, top_k={top_k} (Run {i + 1}/5): {str(e)}"
                            )
                            self.results.append({
                                "provider": provider,
                                "model": model_name,
                                "token_count": token_count,
                                "token_name": token_name,
                                "latency": None,
                                "tokens_per_second": None,
                                "batch_size": 1,
                                "top_k": top_k,
                                "operation": "search",
                                "status": f"error: {str(e)}",
                            })
                            # If first search fails, skip remaining iterations
                            if i == 0:
                                break
            
        except Exception as e:
            logger.error(f"Error setting up search test data for {provider} - {model_name}: {str(e)}")
            # Record search setup error
            self.results.append({
                "provider": provider,
                "model": model_name,
                "token_count": "N/A",
                "token_name": "N/A", 
                "latency": None,
                "tokens_per_second": None,
                "batch_size": 1,
                "top_k": "N/A",
                "operation": "search_setup",
                "status": f"search setup error: {str(e)}",
            })

    def _run_qps_tests(self, collection, provider, model_name, token_variations):
        """Run QPS performance tests on the collection."""
        logger.info(f"Starting QPS tests for {provider} - {model_name}")
        
        try:
            # Ensure search data is available and collection is loaded
            collection.flush()
            collection.load()
            
            # Test QPS for different token lengths - use available variations
            qps_token_variations = token_variations  # Use all available token variations
            
            for token_var in qps_token_variations:
                token_count = token_var["tokens"]
                token_name = token_var["name"]
                test_text = token_var["text"]
                
                # Test INSERT QPS
                self._run_insert_qps_test(collection, provider, model_name, token_count, token_name, test_text)
                
                # Test SEARCH QPS
                self._run_search_qps_test(collection, provider, model_name, token_count, token_name, test_text)
                
        except Exception as e:
            logger.error(f"Error in QPS testing setup for {provider} - {model_name}: {str(e)}")
            self.results.append({
                "provider": provider,
                "model": model_name,
                "token_count": "N/A",
                "token_name": "N/A",
                "latency": None,
                "tokens_per_second": None,
                "qps": None,
                "concurrent_level": "N/A",
                "operation": "qps_setup",
                "status": f"qps setup error: {str(e)}",
            })

    def _run_insert_qps_test(self, collection, provider, model_name, token_count, token_name, test_text):
        """Run INSERT QPS test with adaptive or exhaustive concurrency level finding."""
        if self.qps_test_method == "exhaustive":
            max_qps, optimal_concurrent = self._find_optimal_qps_exhaustive(
                lambda concurrent: self._execute_insert_qps_test(collection, test_text, concurrent),
                "INSERT", provider, model_name, token_name
            )
        else:
            max_qps, optimal_concurrent = self._find_optimal_qps(
                lambda concurrent: self._execute_insert_qps_test(collection, test_text, concurrent),
                "INSERT", provider, model_name, token_name
            )
        
        # Record the optimal result
        if max_qps is not None:
            self.results.append({
                "provider": provider,
                "model": model_name,
                "token_count": token_count,
                "token_name": token_name,
                "latency": None,
                "tokens_per_second": None,
                "qps": max_qps,
                "concurrent_level": optimal_concurrent,
                "operation": "insert_qps_max",
                "status": "success",
            })
            logger.info(f"INSERT Max QPS: {max_qps:.2f} req/s (optimal_concurrent={optimal_concurrent})")
        else:
            self.results.append({
                "provider": provider,
                "model": model_name,
                "token_count": token_count,
                "token_name": token_name,
                "latency": None,
                "tokens_per_second": None,
                "qps": None,
                "concurrent_level": None,
                "operation": "insert_qps_max",
                "status": "failed to find optimal QPS",
            })

    def _execute_insert_qps_test(self, collection, test_text, concurrent_level):
        """Execute a single INSERT QPS test with specified concurrency."""
        successful_requests = 0
        failed_requests = 0
        start_time = time.time()
        end_time = start_time + self.qps_test_duration
        
        def insert_worker():
            nonlocal successful_requests, failed_requests
            while time.time() < end_time:
                try:
                    # Generate unique ID for each request
                    data = [{
                        "id": int(time.time() * 1000000) % 1000000000 + threading.get_ident(),
                        "text": test_text
                    }]
                    collection.insert(data)
                    successful_requests += 1
                except Exception:
                    failed_requests += 1
                    
        # Run concurrent workers
        with concurrent.futures.ThreadPoolExecutor(max_workers=concurrent_level) as executor:
            futures = [executor.submit(insert_worker) for _ in range(concurrent_level)]
            concurrent.futures.wait(futures, timeout=self.qps_test_duration + 5)
        
        actual_duration = time.time() - start_time
        qps = successful_requests / actual_duration if actual_duration > 0 else 0
        
        return qps, successful_requests, failed_requests, actual_duration

    def _run_search_qps_test(self, collection, provider, model_name, token_count, token_name, test_text):
        """Run SEARCH QPS test with adaptive or exhaustive concurrency level finding."""
        if self.qps_test_method == "exhaustive":
            max_qps, optimal_concurrent = self._find_optimal_qps_exhaustive(
                lambda concurrent: self._execute_search_qps_test(collection, test_text, concurrent),
                "SEARCH", provider, model_name, token_name
            )
        else:
            max_qps, optimal_concurrent = self._find_optimal_qps(
                lambda concurrent: self._execute_search_qps_test(collection, test_text, concurrent),
                "SEARCH", provider, model_name, token_name
            )
        
        # Record the optimal result
        if max_qps is not None:
            self.results.append({
                "provider": provider,
                "model": model_name,
                "token_count": token_count,
                "token_name": token_name,
                "latency": None,
                "tokens_per_second": None,
                "qps": max_qps,
                "concurrent_level": optimal_concurrent,
                "operation": "search_qps_max",
                "status": "success",
            })
            logger.info(f"SEARCH Max QPS: {max_qps:.2f} req/s (optimal_concurrent={optimal_concurrent})")
        else:
            self.results.append({
                "provider": provider,
                "model": model_name,
                "token_count": token_count,
                "token_name": token_name,
                "latency": None,
                "tokens_per_second": None,
                "qps": None,
                "concurrent_level": None,
                "operation": "search_qps_max",
                "status": "failed to find optimal QPS",
            })

    def _execute_search_qps_test(self, collection, test_text, concurrent_level):
        """Execute a single SEARCH QPS test with specified concurrency."""
        successful_requests = 0
        failed_requests = 0
        start_time = time.time()
        end_time = start_time + self.qps_test_duration
        
        def search_worker():
            nonlocal successful_requests, failed_requests
            while time.time() < end_time:
                try:
                    _ = collection.search(
                        data=[test_text],
                        anns_field="embedding",
                        param={"metric_type": "COSINE"},
                        limit=10,  # Use fixed top_k=10 for QPS testing
                        output_fields=["id"]
                    )
                    successful_requests += 1
                except Exception:
                    failed_requests += 1
                    
        # Run concurrent workers
        with concurrent.futures.ThreadPoolExecutor(max_workers=concurrent_level) as executor:
            futures = [executor.submit(search_worker) for _ in range(concurrent_level)]
            concurrent.futures.wait(futures, timeout=self.qps_test_duration + 5)
        
        actual_duration = time.time() - start_time
        qps = successful_requests / actual_duration if actual_duration > 0 else 0
        
        return qps, successful_requests, failed_requests, actual_duration

    def _find_optimal_qps(self, test_function, operation_type, provider, model_name, token_name):
        """Find optimal QPS using adaptive algorithm with gradient ascent and binary search."""
        logger.info(f"Finding optimal QPS for {operation_type} {provider} - {model_name} - {token_name}")
        
        # Phase 1: Gradient ascent to find approximate maximum
        current_concurrent = 1
        best_qps = 0
        best_concurrent = 1
        qps_history = []
        
        while current_concurrent <= self.qps_max_concurrent:
            try:
                logger.info(f"Testing {operation_type} QPS at concurrent_level={current_concurrent}")
                qps, successful, failed, duration = test_function(current_concurrent)
                qps_history.append((current_concurrent, qps))
                
                logger.info(f"  Result: {qps:.2f} QPS (successful={successful}, failed={failed})")
                
                # Update best if this is better
                if qps > best_qps:
                    best_qps = qps
                    best_concurrent = current_concurrent
                
                # Check if we should continue (significant improvement)
                if len(qps_history) >= 2:
                    prev_qps = qps_history[-2][1]
                    improvement = (qps - prev_qps) / prev_qps if prev_qps > 0 else 0
                    
                    if improvement < self.qps_improvement_threshold:
                        logger.info(f"  Improvement {improvement:.3f} < threshold {self.qps_improvement_threshold}, stopping gradient ascent")
                        break
                
                # Adaptive step size: smaller steps as we approach optimum
                if len(qps_history) >= 3:
                    # If QPS is decreasing, use smaller steps
                    recent_qps = [x[1] for x in qps_history[-3:]]
                    if recent_qps[-1] < recent_qps[-2]:
                        current_concurrent += max(1, current_concurrent // 4)
                    else:
                        current_concurrent = min(current_concurrent * 2, current_concurrent + 10)
                else:
                    current_concurrent = min(current_concurrent * 2, current_concurrent + 5)
                    
            except Exception as e:
                logger.error(f"Error in QPS test at concurrent_level={current_concurrent}: {str(e)}")
                break
        
        if not qps_history:
            logger.error("No successful QPS tests completed")
            return None, None
            
        # Phase 2: Binary search refinement around the best found concurrent level
        logger.info(f"Phase 1 complete. Best: {best_qps:.2f} QPS at concurrent={best_concurrent}")
        
        # Define search range around the best point
        search_range = max(5, best_concurrent // 4)
        left = max(1, best_concurrent - search_range)
        right = min(self.qps_max_concurrent, best_concurrent + search_range)
        
        # Binary search for finer optimization
        binary_best_qps = best_qps
        binary_best_concurrent = best_concurrent
        
        logger.info(f"Phase 2: Binary search in range [{left}, {right}]")
        
        for iteration in range(5):  # Limit binary search iterations
            if right - left <= 2:
                break
                
            mid = (left + right) // 2
            try:
                logger.info(f"  Binary search iteration {iteration + 1}: testing concurrent_level={mid}")
                qps, successful, failed, duration = test_function(mid)
                logger.info(f"    Result: {qps:.2f} QPS (successful={successful}, failed={failed})")
                
                if qps > binary_best_qps:
                    binary_best_qps = qps
                    binary_best_concurrent = mid
                    left = mid  # Search in upper half
                else:
                    right = mid  # Search in lower half
                    
            except Exception as e:
                logger.error(f"Error in binary search at concurrent_level={mid}: {str(e)}")
                right = mid
        
        # Phase 3: Stability verification
        logger.info(f"Phase 3: Verifying stability of optimal concurrent_level={binary_best_concurrent}")
        try:
            qps, successful, failed, duration = test_function(binary_best_concurrent)
            logger.info(f"  Verification: {qps:.2f} QPS (successful={successful}, failed={failed})")
            
            # Accept the result if it's within reasonable range of our best
            if qps >= binary_best_qps * 0.9:  # Allow 10% variance
                final_qps = qps
                final_concurrent = binary_best_concurrent
            else:
                logger.warning(f"Verification QPS {qps:.2f} significantly lower than expected {binary_best_qps:.2f}")
                final_qps = binary_best_qps
                final_concurrent = binary_best_concurrent
                
        except Exception as e:
            logger.error(f"Error in stability verification: {str(e)}")
            final_qps = binary_best_qps
            final_concurrent = binary_best_concurrent
        
        logger.info(f"Final optimal QPS: {final_qps:.2f} at concurrent_level={final_concurrent}")
        return final_qps, final_concurrent

    def _find_optimal_qps_exhaustive(self, test_function, operation_type, provider, model_name, token_name):
        """Find optimal QPS by testing all concurrent levels incrementally."""
        logger.info(f"Finding optimal QPS using exhaustive method for {operation_type} {provider} - {model_name} - {token_name}")
        
        best_qps = 0
        best_concurrent = 1
        all_results = []
        consecutive_failures = 0
        
        # Test all concurrent levels from 1 to max
        current_concurrent = 1
        while current_concurrent <= self.qps_max_concurrent:
            try:
                logger.info(f"Testing {operation_type} QPS at concurrent_level={current_concurrent}")
                qps, successful, failed, duration = test_function(current_concurrent)
                
                logger.info(f"  Result: {qps:.2f} QPS (successful={successful}, failed={failed})")
                
                all_results.append({
                    'concurrent': current_concurrent,
                    'qps': qps,
                    'successful': successful,
                    'failed': failed
                })
                
                # Update best if this is better
                if qps > best_qps:
                    best_qps = qps
                    best_concurrent = current_concurrent
                
                # Reset failure counter on success
                consecutive_failures = 0
                
                # If QPS drops significantly (e.g., 20%) from best, we can optionally stop early
                if best_qps > 0 and qps < best_qps * 0.8 and current_concurrent > best_concurrent + 10:
                    logger.info(f"  QPS dropped significantly from best ({best_qps:.2f}), stopping early")
                    break
                    
            except Exception as e:
                logger.error(f"Error in QPS test at concurrent_level={current_concurrent}: {str(e)}")
                consecutive_failures += 1
                
                # Stop if we have too many consecutive failures
                if consecutive_failures >= 3:
                    logger.error("Too many consecutive failures, stopping exhaustive search")
                    break
            
            # Increment by step size
            current_concurrent += self.qps_concurrent_step
        
        if not all_results:
            logger.error("No successful QPS tests completed")
            return None, None
        
        # Log all results for analysis
        logger.info("\nExhaustive search complete. All results:")
        for result in all_results:
            logger.info(f"  Concurrent={result['concurrent']}: {result['qps']:.2f} QPS")
        
        logger.info(f"\nBest QPS: {best_qps:.2f} at concurrent_level={best_concurrent}")
        
        return best_qps, best_concurrent

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
        # Print summary by operation type
        import numpy as np  # Add numpy for percentile calculation

        # Summary for insert operations
        insert_df = df[(df["status"] == "success") & (df["operation"] == "insert")]
        if not insert_df.empty:
            insert_summary = (
                insert_df
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
                "\nINSERT Performance Summary (with std/median/p95/p99):\n"
                + str(insert_summary)
            )

        # Summary for search operations
        search_df = df[(df["status"] == "success") & (df["operation"] == "search")]
        if not search_df.empty:
            search_summary = (
                search_df
                .groupby(["provider", "model", "token_count", "token_name", "top_k"])[
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
                "\nSEARCH Performance Summary (with std/median/p95/p99):\n"
                + str(search_summary)
            )

        # Summary for INSERT QPS operations
        insert_qps_df = df[(df["status"] == "success") & (df["operation"] == "insert_qps_max")]
        if not insert_qps_df.empty:
            insert_qps_summary = (
                insert_qps_df
                .groupby(["provider", "model", "token_count", "token_name", "concurrent_level"])[
                    "qps"
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
                "\nINSERT QPS Performance Summary (with std/median/p95/p99):\n"
                + str(insert_qps_summary)
            )

        # Summary for SEARCH QPS operations
        search_qps_df = df[(df["status"] == "success") & (df["operation"] == "search_qps_max")]
        if not search_qps_df.empty:
            search_qps_summary = (
                search_qps_df
                .groupby(["provider", "model", "token_count", "token_name", "concurrent_level"])[
                    "qps"
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
                "\nSEARCH QPS Performance Summary (with std/median/p95/p99):\n"
                + str(search_qps_summary)
            )
