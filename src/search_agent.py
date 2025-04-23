import os
import json
import arxiv
import requests
from typing import List, Dict, Any, Optional
from datetime import datetime
from bs4 import BeautifulSoup
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, PromptTemplate
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.tools import Tool
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_react_agent
# Import database client
from db import client

# Define Arxiv API tool
class ArxivSearchInput(BaseModel):
    query: str = Field(description="Search query for Arxiv")
    limit: int = Field(default=5, description="Maximum number of papers to retrieve")

def search_arxiv(query: str) -> List[Dict]:
    """Search for papers using the Arxiv API"""
    limit = 5
    
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

def scrape_paper(url: str) -> Dict:
    """Scrape a research paper from a given URL"""
    
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

# Define tool to check if document exists
class CheckDocumentInput(BaseModel):
    paper_id: str = Field(description="Unique identifier for the paper")

def check_document_exists(paper_id: int) -> Dict:
    """Check if a document already exists in Weaviate"""
    
    result = (
        client.query
        .get("ResearchPaper", ["paper_id"])
        .with_where({
            "path": ["paper_id"],
            "operator": "Equal",
            "valueString": paper_id
        })
        .do()
    )
    
    exists = len(result["data"]["Get"]["ResearchPaper"]) > 0
    return {"exists": exists, "paper_id": paper_id}

# Define database interaction tools

# need to store the embedding properly
class StoreDocumentInput(BaseModel):
    title: str = Field(description="Title of the paper")
    abstract: Optional[str] = Field( default="", description="Abstract of the paper")
    authors: Optional[List[str]]  = Field( default=[""], description="Authors of the paper")
    paper_id: Optional[str] = Field( default="", description="Unique identifier for the paper")
    url: Optional[str] = Field( default="", description="URL to access the paper")
    published_date: Optional[str] = Field( default="", description="Publication date (YYYY-MM-DD)")
    source: Optional[str] = Field( default="", description="Source of the paper")
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
    
    # Check if document already exists
    check_result = check_document_exists(CheckDocumentInput(paper_id=input_data.paper_id))
    if check_result["exists"]:
        return {"status": "Document already exists", "paper_id": input_data.paper_id}
    
    try:
        client.data_object.create(
            properties,
            "ResearchPaper",
            vector_config={"vectorizer": "text2vec-openai"}
        )
        return {"status": "success", "message": f"Stored {input_data.title} in the database"}
    except Exception as e:
        return {"status": "error", "message": str(e)}

# Create search tools list
search_tools = [
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
        name="check_document_exists",
        description="Check if a document exists in the database",
        func=check_document_exists,
        args_schema=CheckDocumentInput
    ),
    Tool(
        name="store_document",
        description="Store a document in the vector database",
        func=store_document,
        args_schema=StoreDocumentInput
    ),
]

# Define Search Agent prompt
# search_agent_prompt = ChatPromptTemplate.from_messages([
#     SystemMessage(content="""You are a Research Search Agent. Your job is to find relevant research papers based on the user's query.
    
# Follow these steps:
# 1. Analyze the query to understand what research papers are needed
# 2. Search external sources like Arxiv for relevant papers using the search_arxiv tool
# 3. For each found paper, check if it exists in our database using check_document_exists
# 4. If not, store it in the database using store_document
# 5. Return a summary of what you found and stored

# Think step by step and be thorough in your search. Your goal is to build a comprehensive database
# of papers related to the query.
# """),
#     MessagesPlaceholder(variable_name="messages"),
#     MessagesPlaceholder(variable_name="agent_scratchpad"),
# ])


search_agent_prompt = PromptTemplate.from_template(
    """You are a Research Search Agent. Your job is to find relevant research papers based on the user's query.

Follow these steps:
1. Analyze the query {input} to understand what research papers are needed
2. Search external sources like Arxiv for relevant papers using the search_arxiv tool
3. For each found paper, check if it exists in our database using check_document_exists
4. If not, store it in the database using store_document. When using store_documents query you should provide
       "title": The title of the paper,
        "abstract": The abstract of the paper,
        "authors": The authors of the paper,
        "paper_id": The paper id of the paper
        "url": The url of the paper
        "published_date": The published date of the paper,
        "source": The source of the paper
        "full_text": The full text of the paper

5. Return a summary of what you found and stored

You have access to the following tools: {tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original question

Begin!

Question: {input}
Thought:{agent_scratchpad}"""
)

def create_search_agent(model="gpt-4o"):
    """Create and return the search agent with appropriate LLM"""

    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0
    )
    
    # Create the agent using the langgraph prebuilt create_react_agent
    agent = create_react_agent(llm, search_tools, prompt=search_agent_prompt)

    # Create with proper configuration
    return AgentExecutor(
        agent=agent,
        tools=search_tools,
        handle_parsing_errors=True,
        max_iterations=5,  # Prevent infinite loops
        verbose=True  # For debugging
    )

# Default search agent using GPT-4o
search_agent = create_search_agent()