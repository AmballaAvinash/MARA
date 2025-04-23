import os
import weaviate
from typing import List, Dict, Any, Optional

def get_weaviate_client():
    """Initialize Weaviate client with proper configuration"""
    # Check if we should use embedded mode
    try:
        # For newer versions of weaviate-client
        from weaviate.embedded import EmbeddedOptions
        
        # Create client with embedded options
        client = weaviate.Client(
            embedded_options=EmbeddedOptions(
                persistence_data_path="./data",
                additional_env_vars={
                    "ENABLE_MODULES": "text2vec-openai",
                }
            )
        )
    except (ImportError, TypeError):
        # Fall back to standard client initialization
        # For cloud-hosted Weaviate or different client versions
        client = weaviate.Client(
            url="http://localhost:8080",  # Default local Weaviate address
        )
        print("Using standard Weaviate client. Make sure your Weaviate instance is running.")
    
    # Set up schema if it doesn't exist
    if not client.schema.exists("ResearchPaper"):
        schema = {
            "classes": [
                {
                    "class": "ResearchPaper",
                    "description": "A research paper with embeddings",
                    "vectorizer": "text2vec-openai",
                    "properties": [
                        {
                            "name": "title",
                            "description": "Title of the paper",
                            "dataType": ["text"],
                        },
                        {
                            "name": "abstract",
                            "description": "Abstract of the paper",
                            "dataType": ["text"],
                        },
                        {
                            "name": "authors",
                            "description": "Authors of the paper",
                            "dataType": ["text[]"],
                        },
                        {
                            "name": "paper_id",
                            "description": "Unique identifier for the paper",
                            "dataType": ["string"],
                        },
                        {
                            "name": "url",
                            "description": "URL to access the paper",
                            "dataType": ["string"],
                        },
                        {
                            "name": "published_date",
                            "description": "Publication date",
                            "dataType": ["date"],
                        },
                        {
                            "name": "source",
                            "description": "Source of the paper (Arxiv)",
                            "dataType": ["string"],
                        },
                        {
                            "name": "full_text",
                            "description": "Full text of the paper if available",
                            "dataType": ["text"],
                        }
                    ],
                }
            ]
        }
        client.schema.create(schema)
        
    return client

# Create and export the client
client = get_weaviate_client()