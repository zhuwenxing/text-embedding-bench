# -*- coding: utf-8 -*-
"""
Provider and model configuration for embedding benchmark.
"""

import dotenv
import os
dotenv.load_dotenv(dotenv_path="deployment/.env")

PROVIDERS_MODELS = {
    # # OpenAI Embedding Models
    # "openai": [
    #     {"name": "text-embedding-ada-002", "dim": 1536, "params": {}},
    #     {"name": "text-embedding-3-small", "dim": 1536, "params": {}},
    #     {"name": "text-embedding-3-large", "dim": 3072, "params": {}},
    # ],
    # # AWS Bedrock Embedding Models
    # "bedrock": [
    #     {
    #         "name": "amazon.titan-embed-text-v2:0",
    #         "dim": 1024,
    #         "params": {"region": os.getenv("BEDROCK_REGION")},
    #     },
    # ],
    # # Google Vertex AI Embedding Models
    # "vertexai": [
    #     {
    #         "name": "text-embedding-005",
    #         "dim": 768,
    #         "params": {"projectid": os.getenv("GCP_PROJECT_ID"), "location": os.getenv("GCP_LOCATION")},
    #     },
    #     {
    #         "name": "text-multilingual-embedding-002",
    #         "dim": 768,
    #         "params": {"projectid": os.getenv("GCP_PROJECT_ID"), "location": os.getenv("GCP_LOCATION")},
    #     },
    # ],
    # # VoyageAI Embedding Models
    # "voyageai": [
    #     {"name": "voyage-3-large", "dim": 1024, "params": {}},
    #     {"name": "voyage-3", "dim": 1024, "params": {}},
    #     {"name": "voyage-3-lite", "dim": 512, "params": {}},
    # ],
    # # Cohere Embedding Models
    # "cohere": [
    #     {"name": "embed-english-v3.0", "dim": 1024, "params": {}},
    #     {"name": "embed-multilingual-v3.0", "dim": 1024, "params": {}},
    # ],
    # # Aliyun Dashscope Embedding Models
    # "dashscope": [
    #     {"name": "text-embedding-v1", "dim": 1536, "params": {}},
    #     {"name": "text-embedding-v2", "dim": 1536, "params": {}},
    #     {"name": "text-embedding-v3", "dim": 1024, "params": {}},
    # ],
    # # Siliconflow Embedding Models
    # "siliconflow": [
    #     {"name": "BAAI/bge-large-zh-v1.5", "dim": 1024, "params": {}},
    #     {"name": "BAAI/bge-large-en-v1.5", "dim": 1024, "params": {}},
    #     {"name": "netease-youdao/bce-embedding-base_v1", "dim": 768, "params": {}},
    #     {"name": "BAAI/bge-m3", "dim": 1024, "params": {}},
    #     {"name": "Pro/BAAI/bge-m3", "dim": 1024, "params": {}},
    # ],
    # TEI (self hoste)
    "tei": [
        {
            "name": "BAAI/bge-base-en-v1.5",
            "dim": 768,
            "params": {"provider": "TEI", "endpoint": "http://10.100.36.193:80"},
        }
    ],
}
