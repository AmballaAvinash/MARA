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

from react_style import create_react_agent
from DB import client


# Set up embeddings
embeddings = OpenAIEmbeddings()




class QueryDocumentsInput(BaseModel):
    query: str = Field(description="Query to search for documents")
    limit: int = Field(default=5, description="Maximum number of documents to retrieve")

def query_documents(input_data: QueryDocumentsInput) -> List[Dict]:
    """Query documents from Weaviate"""
    query = input_data.query
    limit = input_data.limit
    
    result = (
        client.query
        .get("ResearchPaper", [
            "title", "abstract", "authors", "paper_id", 
            "url", "published_date", "source", "full_text"
        ])
        .with_near_text({"concepts": [query]})
        .with_limit(limit)
        .do()
    )
    
    if not result["data"]["Get"]["ResearchPaper"]:
        return []
    
    return result["data"]["Get"]["ResearchPaper"]




# Create tools
tools = [
    Tool(
        name="query_documents",
        description="Query documents from the vector database",
        func=query_documents,
        args_schema=QueryDocumentsInput
    )
]





# Initialize LLMs
# OpenAI o1 is designed for high compute at test time with slower but more thoughtful responses
openai_o1 = ChatOpenAI(
    model="gpt-4o-2024-05",  # Replace with actual o1 model name when available
    temperature=0,
    model_kwargs={"high_compute": True}  # Placeholder for high compute setting
)

# DeepSeek R1 with high test time compute
deepseek_r1 = ChatDeepSeek(
    model="deepseek-r1-online",  # Adjust as needed
    temperature=0,
    model_kwargs={"compute_setting": "high"}  # Placeholder for high compute setting
)



# Define RAG Agent prompt
rag_agent_prompt = ChatPromptTemplate.from_messages([
    SystemMessage(content="""You are a Research RAG Agent using a ReAct style approach. Your job is to answer research questions by iteratively:
1. Reasoning about what information you need
2. Taking actions to retrieve that information
3. Using the retrieved information to formulate answers

Follow these steps:
1. Analyze the query to understand what information is needed
2. Formulate a search query to retrieve relevant documents
3. Evaluate if the documents answer the question adequately
4. If not, refine your search and try again
5. Once you have sufficient information, synthesize a comprehensive answer

Use the tools available to you and think step by step."""),
    MessagesPlaceholder(variable_name="messages"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])


rag_agent_openai = create_react_agent(openai_o1, rag_agent_prompt,tools)
rag_agent_deepseek = create_react_agent(deepseek_r1, rag_agent_prompt,tools)