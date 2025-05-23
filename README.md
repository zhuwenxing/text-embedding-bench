# text-embedding-bench

## Overview

**text-embedding-bench** is a benchmarking tool dedicated to evaluating the latency of multiple text embedding (vector) services. It enables you to compare the response speed of various embedding providers and models in a unified and automated way.

---

## Features

- Automatic benchmarking of multiple embedding providers and models
- Support for various token lengths and batch sizes
- Automatic fake text generation for consistent benchmarking
- Robust exception handling and error logging
- Results saved as CSV with detailed statistical analysis (mean, std, median, P95, P99, etc.)
- Easy provider/model customization
- Docker and Docker Compose support

---

## Installation

### 1. Clone the Repository

```shell
git clone https://github.com/your-org/text-embedding-bench.git
cd text-embedding-bench
```

### 2. Configure API Keys

Edit `deployment/.env_example` and fill in your API keys for the embedding services you want to use. For example:

```env
OPENAI_API_KEY=your_openai_key
AZURE_API_KEY=your_azure_key
COHERE_API_KEY=your_cohere_key
# ...add others as needed
```

Rename the file to `.env`:

```shell
mv deployment/.env_example deployment/.env
```

### 3. Configure GCP Credentials (Optional)

If you plan to use Google Embedding:

```shell
cp deployment/credentials_example.json deployment/credentials.json
```

Fill in your GCP service account credentials as needed. Skip this step if not using Google services.

### 4. Install uv (Recommended)

[uv](https://github.com/astral-sh/uv) is a fast Python package and virtual environment manager.

Install uv (choose one):

- Using pip (recommended):

    ```shell
    pip install uv
    ```

- Or see the [uv official documentation](https://github.com/astral-sh/uv)

### 5. Create and Activate Python 3.10 Virtual Environment

```shell
uv venv -p 3.10
source .venv/bin/activate
```

### 6. Install Dependencies

```shell
uv pip install -r requirements.txt --index-strategy unsafe-best-match
```

---

## Quick Start

After installation and configuration, run the benchmark:

```shell
python main.py
```

---

## Usage

- Make sure `.env` and `credentials.json` (if present) are in the `deployment` directory along with `docker-compose.yml`.
- For Docker Compose deployment, see `deployment/README.md` for details.
- For provider/model configuration, see below.

---

## Dependencies & Environment

- Python 3.10+
- uv (recommended for environment and dependency management)
- All dependencies listed in `requirements.txt`

---

## Configuration

- API keys: `deployment/.env`
- GCP credentials: `deployment/credentials.json` (if needed)
- Provider/model settings: `text_embedding_bench/providers.py`

---

## How It Works

The benchmarking process is managed by the `EmbeddingBenchRunner` class. The main workflow is:

- **Automatic provider/model traversal:** Reads all embedding providers and models defined in `providers.py` and benchmarks each in turn.
- **Multiple token lengths and batch sizes:** For each model, automatically generates English text of various lengths (e.g., 256/512/1024/2048/4096/8192 tokens) and tests with different batch sizes (e.g., 1, 10) for comprehensive evaluation.
- **Data generation:** Uses `generate_fake_text` in `utils.py` to create text with a specified number of tokens, ensuring consistency across models.
- **Robust execution and error handling:** Each test is executed multiple times, exceptions are caught and logged, and invalid tests (such as those exceeding token limits) are skipped automatically.
- **Result saving and statistics:** All results are saved as CSV files. Detailed statistics (mean, std, median, P95, P99, etc.) are computed and output for easy comparison.

For implementation details, see `text_embedding_bench/runner.py`. You can customize the benchmarking process by modifying this file.

---

## Providers & Models Customization

All available providers and model configurations are defined in `text_embedding_bench/providers.py`. You can add, remove, or modify providers and models as needed.

---

## Contributing

Contributions are welcome! Please open an issue or submit a pull request. For major changes, please open an issue first to discuss what you would like to change.

---

## License

This project is licensed under the MIT License.

---

## Contact

For questions or suggestions, please open an issue on GitHub.
