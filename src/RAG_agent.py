import os
from typing import List, Dict, Any, Optional
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, PromptTemplate
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.tools import Tool
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_deepseek import ChatDeepSeek
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.agents import AgentExecutor, create_react_agent

# Import database client
from db import client

# Set up embeddings
embeddings = OpenAIEmbeddings()

# Define document query tool
class QueryDocumentsInput(BaseModel):
    query: str = Field(description="Query to search for documents")
    limit: int = Field(default=5, description="Maximum number of documents to retrieve")

def query_documents(query: str) -> List[Dict]:
    """Query documents from Weaviate"""
    limit = 5
    
    result = (
        client.query
        .get("ResearchPaper", [
            "title", "abstract", "authors", "paper_id", 
            "url", "published_date", "source", "full_text"
        ])
        .with_near_text({"concepts": [query]})  # cosine similarity search
        .with_limit(limit)
        .do()
    )
    
    if not result["data"]["Get"]["ResearchPaper"]:
        return {"status": "no_results", "message": "No documents found for this query"}
    
    return result["data"]["Get"]["ResearchPaper"]

# Tool to refine search query
class RefineQueryInput(BaseModel):
    original_query: str = Field(description="Original query query")
    context: Optional[str] = Field(
        default="", 
        description="Additional context to refine the qury"
    )

def refine_query(original_query: str, context: str = "") -> Dict:
    """Refine a search query based on context"""
    # This would typically use an LLM to refine the query
    # For simplicity, we'll just add the context to the query
    refined_query = f"{original_query} {context}".strip()
    return {"refined_query": refined_query}

# Create RAG tools
rag_tools = [
    Tool(
        name="query_documents",
        description="Query documents from the vector database based on semantic similarity",
        func=query_documents,
        args_schema=QueryDocumentsInput
    ),
    Tool(
        name="refine_query",
        description="Refine a search query based on context for better results",
        func=refine_query,
        args_schema=RefineQueryInput
    )
]

# Define RAG Agent prompt
# rag_agent_prompt = ChatPromptTemplate.from_messages([
#     SystemMessage(content="""You are a Research RAG Agent using a ReAct approach. Your job is to answer research questions by iteratively:
# 1. Reasoning about what information you need
# 2. Taking actions to retrieve that information
# 3. Using the retrieved information to formulate comprehensive answers

# Follow these steps:
# 1. Analyze the query to understand what information is needed
# 2. Formulate a search query to retrieve relevant documents using the query_documents tool
# 3. Evaluate if the documents answer the question adequately
# 4. If not, refine your search using the refine_query tool and try again
# 5. Once you have sufficient information, synthesize a comprehensive answer

# Remember to:
# - Cite the papers you're using in your answer (title, authors, year)
# - Synthesize information across multiple papers when applicable
# - Acknowledge limitations or gaps in the available research
# - Think step by step and be thorough in your analysis

# Always aim to provide accurate, well-researched answers based on the available documents.
# """),
#     MessagesPlaceholder(variable_name="messages"),
#     MessagesPlaceholder(variable_name="agent_scratchpad"),
# ])


rag_agent_prompt = PromptTemplate.from_template(
    """You are a Research RAG Agent using a ReAct approach. Your job is to answer research questions by iteratively:
1. Reasoning about what information you need
2. Taking actions to retrieve that information
3. Using the retrieved information to formulate comprehensive answers

Follow these steps:
1. Analyze the query {input} to understand what information is needed
2. Formulate a search query to retrieve relevant documents using the query_documents tool
3. Evaluate if the documents answer the question adequately
4. If not, refine your search using the refine_query tool and try again. When using the refine_query tool, you MUST provide:
            - original_query: The current search query
            - context: Why you want to refine it and what to add
5. Once you have sufficient information, synthesize a comprehensive answer with reasoning

Remember to:
- Cite the papers you're using in your answer (title, authors, year)
- Synthesize information across multiple papers when applicable
- Acknowledge limitations or gaps in the available research
- Think step by step and be thorough in your analysis

Available tools: {tools}

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

def create_rag_agent(model_name="openai"):
    """Create and return RAG agent with appropriate LLM"""

    if model_name.lower() == "deepseek":
        llm = ChatDeepSeek(
            model="deepseek-chat",
            temperature=0,
            # model_kwargs={"compute_setting": "high"}
        )
    else:
        # Default to GPT-4o
        llm =  ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0,
             # model_kwargs={"high_compute": True}
           )

    # Create the agent using the langgraph prebuilt create_react_agent
    agent = create_react_agent(llm, rag_tools, prompt=rag_agent_prompt)

    # Create with proper configuration
    return AgentExecutor(
        agent=agent,
        tools=rag_tools,
        handle_parsing_errors=True,
        max_iterations=5,  # Prevent infinite loops
        verbose=True  # For debugging
    )

# Create RAG agents with different models
rag_agent_openai = create_rag_agent("openai")
rag_agent_deepseek = create_rag_agent("deepseek")