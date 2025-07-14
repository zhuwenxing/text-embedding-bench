# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

This is a text embedding benchmarking tool that evaluates latency performance across multiple embedding providers and models. It uses Milvus vector database with text embedding functions to test various providers like OpenAI, Cohere, VoyageAI, AWS Bedrock, Google Vertex AI, and others.

## Setup and Environment

### Dependencies
- Python 3.10+ required
- Uses `uv` package manager (recommended)
- Milvus server running on `localhost:19530`

### Installation Commands
```bash
# Create Python 3.10 virtual environment
uv venv -p 3.10
source .venv/bin/activate

# Install dependencies
uv pip install -r requirements.txt --index-strategy unsafe-best-match
```

### Configuration
- API keys: Copy `deployment/.env_example` to `deployment/.env` and fill in provider API keys
- GCP credentials: Copy `deployment/credentials_example.json` to `deployment/credentials.json` if using Google services
- Milvus connection: Ensure Milvus server is running on `localhost:19530`

## Running the Benchmark

### Basic Usage
```bash
# Run complete benchmark suite
python main.py
```

### Docker Deployment
```bash
# From deployment/ directory
docker compose up -d --wait
```

## Code Architecture

### Core Components

1. **`main.py`** - Entry point that initializes and runs the benchmark
2. **`text_embedding_bench/runner.py`** - `EmbeddingBenchRunner` class containing main benchmarking logic
3. **`text_embedding_bench/providers.py`** - Configuration of all embedding providers and models
4. **`text_embedding_bench/utils.py`** - Utility functions for text generation and file operations

### Key Architecture Patterns

- **Provider-Model Configuration**: All embedding providers and models are centrally defined in `PROVIDERS_MODELS` dictionary in `providers.py`
- **Dynamic Collection Creation**: Each test creates a temporary Milvus collection with provider-specific text embedding functions
- **Comprehensive Operation Testing**: Benchmarks INSERT, SEARCH, and adaptive MAX QPS operations for complete performance analysis
- **Multi-dimensional Testing**: Tests run across multiple token lengths (256, 512, 1024, 2048, 4096, 8192), batch sizes (1, 10 for insert), top-k values (1, 10, 100 for search)
- **Statistical Analysis**: Each configuration runs multiple times (10 for insert, 5 for search) to calculate mean, std, median, P95, P99 latency metrics
- **Adaptive QPS Testing**: Uses 3-phase algorithm to find maximum sustainable QPS:
  - Phase 1: Gradient ascent with adaptive step sizes to quickly find optimal range
  - Phase 2: Binary search refinement for precise optimization
  - Phase 3: Stability verification of optimal concurrent level

### Milvus Integration

- Uses Milvus `Function` with `FunctionType.TEXTEMBEDDING` to leverage built-in embedding providers
- Creates collections with schema: `id` (INT64), `text` (VARCHAR), `embedding` (FLOAT_VECTOR)
- Automatically handles provider authentication via environment variables
- Collections are automatically dropped after testing

### Error Handling

- Robust exception handling at both collection setup and individual test levels
- Failed tests are logged and recorded in results with error status
- If first iteration fails, subsequent iterations for that configuration are skipped
- Results include both successful and failed test attempts

### Results Management

- Results saved as timestamped CSV files in `results/` directory with tracking for insert, search, and adaptive max QPS operations
- Comprehensive statistical analysis with pandas and numpy, including separate summaries for latency and optimal QPS performance
- Real-time logging of test progress and errors using loguru
- CSV output includes operation type, batch_size (for insert), top_k (for search), optimal concurrent_level and max QPS
- Adaptive QPS testing automatically finds the true maximum throughput for each provider/model combination

## Adding New Providers

To add a new embedding provider:

1. Add provider configuration to `PROVIDERS_MODELS` in `providers.py`
2. Ensure corresponding environment variables are added to `.env_example`
3. Provider must be supported by Milvus text embedding functions

Example provider configuration:
```python
"new_provider": [
    {
        "name": "model-name",
        "dim": 768,  # embedding dimension
        "params": {"custom_param": "value"}
    }
]
```

## Environment Variables

All provider API keys use `MILVUSAI_` prefix:
- `MILVUSAI_OPENAI_API_KEY`
- `MILVUSAI_COHERE_API_KEY`
- `MILVUSAI_VOYAGEAI_API_KEY`
- etc.

Additional configuration:
- `BEDROCK_REGION`, `GCP_PROJECT_ID`, `GCP_LOCATION`
- `TEI_ENDPOINT` for self-hosted TEI instances
```