## Usage

### 1. Prepare API Keys

- Open `.env_example` and fill in the API keys for each text embedding provider you want to use.
  - For example:
    ```env
    OPENAI_API_KEY=your_openai_key
    AZURE_API_KEY=your_azure_key
    COHERE_API_KEY=your_cohere_key
    # ... add other providers as needed
    ```
- After editing, rename the file:
  ```shell
  mv .env_example .env
  ```

### 2. Prepare GCP Credentials (if using Google Embedding)

- Copy `credentials_example.json` to `credentials.json`:
  ```shell
  cp credentials_example.json credentials.json
  ```
- Fill in your GCP service account credentials in `credentials.json`.

### 3. Start the Service with Docker Compose

- Use the following command to start all services:
  ```shell
  docker compose up -d --wait
  ```

---

> **Note:**
> - Make sure `.env` and `credentials.json` are in the same directory as `docker-compose.yml`.
> - If you use a provider that does not require GCP, you can skip the credentials step.
> - For more details, refer to the documentation of each embedding provider.

---
