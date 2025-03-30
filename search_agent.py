
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


# Define Arxiv API tool
class ArxivSearchInput(BaseModel):
    query: str = Field(description="Search query for Arxiv")
    limit: int = Field(default=5, description="Maximum number of papers to retrieve")

def search_arxiv(input_data: ArxivSearchInput) -> List[Dict]:
    """Search for papers using the Arxiv API"""
    query = input_data.query
    limit = input_data.limit
    
    search = arxiv.Search(
        query=query,
        max_results=limit,
        sort_by=arxiv.SortCriterion.Relevance
    )
    
    papers = []
    for result in search.results():
        paper = {
            "title": result.title,
            "abstract": result.summary,
            "authors": [author.name for author in result.authors],
            "paper_id": result.entry_id.split('/')[-1],
            "url": result.pdf_url,
            "published_date": result.published.strftime("%Y-%m-%d"),
            "source": "Arxiv"
        }
        papers.append(paper)
    
    return papers



# Define web scraping tool for custom sites
class WebScrapeInput(BaseModel):
    url: str = Field(description="URL of the research paper to scrape")

def scrape_paper(input_data: WebScrapeInput) -> Dict:
    """Scrape a research paper from a given URL"""
    url = input_data.url
    
    response = requests.get(url)
    if response.status_code != 200:
        return {"error": f"Failed to fetch data: {response.status_code}"}
    
    soup = BeautifulSoup(response.content, 'html.parser')
    
    # This is a simplified scraper - in practice, you would need custom logic per site
    title = soup.find('title').text if soup.find('title') else ""
    
    # Very basic extraction, would need to be customized based on the site structure
    abstract = ""
    abstract_elem = soup.find('div', {'class': 'abstract'}) or soup.find('section', {'id': 'abstract'})
    if abstract_elem:
        abstract = abstract_elem.text
    
    return {
        "title": title,
        "abstract": abstract,
        "authors": [],  # Would need custom extraction
        "paper_id": url.split('/')[-1],
        "url": url,
        "published_date": datetime.now().strftime("%Y-%m-%d"),  # Default to current date if not found
        "source": "Web Scrape"
    }



# Define database interaction tools
class StoreDocumentInput(BaseModel):
    title: str = Field(description="Title of the paper")
    abstract: str = Field(description="Abstract of the paper")
    authors: List[str] = Field(description="Authors of the paper")
    paper_id: str = Field(description="Unique identifier for the paper")
    url: str = Field(description="URL to access the paper")
    published_date: str = Field(description="Publication date (YYYY-MM-DD)")
    source: str = Field(description="Source of the paper")
    full_text: Optional[str] = Field(default="", description="Full text of the paper if available")

def store_document(input_data: StoreDocumentInput) -> Dict:
    """Store a document in Weaviate"""
    properties = {
        "title": input_data.title,
        "abstract": input_data.abstract,
        "authors": input_data.authors,
        "paper_id": input_data.paper_id,
        "url": input_data.url,
        "published_date": input_data.published_date,
        "source": input_data.source,
        "full_text": input_data.full_text
    }
    
    # Check if document with same paper_id already exists
    result = (
        client.query
        .get("ResearchPaper", ["paper_id"])
        .with_where({
            "path": ["paper_id"],
            "operator": "Equal",
            "valueString": input_data.paper_id
        })
        .do()
    )
    
    if result["data"]["Get"]["ResearchPaper"]:
        return {"status": "Document already exists", "paper_id": input_data.paper_id}
    
    try:
        client.data_object.create(
            properties,
            "ResearchPaper"
        )
        return {"status": "success", "message": f"Stored {input_data.title} in the database"}
    except Exception as e:
        return {"status": "error", "message": str(e)}
    

# Create tools
tools = [
   
    Tool(
        name="search_arxiv",
        description="Search for research papers using Arxiv API",
        func=search_arxiv,
        args_schema=ArxivSearchInput
    ),
    Tool(
        name="scrape_paper",
        description="Scrape a research paper from a given URL",
        func=scrape_paper,
        args_schema=WebScrapeInput
    ),
    Tool(
        name="store_document",
        description="Store a document in the vector database",
        func=store_document,
        args_schema=StoreDocumentInput
    ),
    Tool(
        name="query_documents",
        description="Query documents from the vector database",
        func=query_documents,
        args_schema=QueryDocumentsInput
    )
]

tool_executor = ToolExecutor(tools)




# Define Search Agent prompt
search_agent_prompt = ChatPromptTemplate.from_messages([
    SystemMessage(content="""You are a Research Search Agent. Your job is to find relevant research papers based on the user's query.
    
If the query can be answered by papers in our database, use those. If not, search external sources like Arxiv.
Process each found paper and store it with proper embeddings in our database.

Follow these steps:
1. Analyze the query to understand what research papers are needed
2. Check if papers in our database can answer the query
3. If not, search external sources for relevant papers
4. Process and store new papers in the database
5. Return a summary of what you found

Use the tools available to you and think step by step."""),
    MessagesPlaceholder(variable_name="messages"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])
