# Initialize Weaviate client (embedded for simplicity)
import os
from typing import List, Dict, Any, Optional
from datetime import datetime
import time
import json
import requests
import weaviate
from weaviate.embedded import EmbeddedOptions
from bs4 import BeautifulSoup
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.output_parsers import JsonOutputParser
from langchain_openai import ChatOpenAI
from langchain_deepseek import ChatDeepSeek
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.tools.render import render_text_description
from langchain_core.tools import Tool
from langchain_core.pydantic_v1 import BaseModel, Field
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolExecutor, ToolInvocation


client = weaviate.Client(
    embedded_options=EmbeddedOptions(
        persistence_data_path="./data",
        additional_env_vars={
            "ENABLE_MODULES": "text2vec-openai",
        }
    )
)

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