import os
import json
from typing import List, Dict, Any
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, PromptTemplate
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.output_parsers import JsonOutputParser
from langchain_openai import ChatOpenAI
from langchain_core.tools import Tool
from pydantic import BaseModel, Field
# from langgraph.prebuilt import create_react_agent
from langchain.agents import AgentExecutor, create_react_agent


# Import necessary tools
from db import client

# Define tool to check database status
class CheckDatabaseStatusInput(BaseModel):
    query: str = Field(description="Query to check availability of documents")

def check_database_status(query: str) -> Dict:
    """Check if there are relevant documents in the database for a query"""
    # query = input_data.query
    
    # Use the query to find relevant documents
    result = (
        client.query
        .get("ResearchPaper", ["title"])
        .with_near_text({"concepts": [query]})
        .with_limit(3)
        .do()
    )
    
    print(result)

    if result["data"]["Get"]["ResearchPaper"] is None:
         return {
            "status": "unavailable",
            "document_count": 0,
            "query": query
        }
    
    doc_count = len(result["data"]["Get"]["ResearchPaper"])
    
    print(doc_count)

    return {
        "status": "available" if doc_count > 0 else "unavailable",
        "document_count": doc_count,
        "query": query
    }

# Manager tools
manager_tools = [
    Tool(
        name="check_database_status",
        description="Check if relevant documents are available in the database",
        func=check_database_status,
        args_schema=CheckDatabaseStatusInput
    )
]

# Define Manager Agent prompt
# manager_agent_prompt = ChatPromptTemplate.from_messages([
#     SystemMessage(content="""You are a Manager Agent overseeing a research system with two specialized agents:

# 1. Search Agent: Finds and adds new research papers to the database
# 2. RAG Agent: Answers questions based on papers in the database

# Your job is to:
# 1. Analyze the user's query
# 2. Check if there are relevant documents in the database using the 'check_database_status' tool
# 3. Decide which agent should handle the query:
#    - If documents are available, use the RAG Agent
#    - If documents are unavailable or insufficient, use the Search Agent first, then the RAG Agent
# 4. Return your decision as a JSON object

# Be decisive and efficient in your routing decisions. Do NOT keep reasoning endlessly. You should call the tool first.

# """),
#     MessagesPlaceholder(variable_name="messages"),
#     # MessagesPlaceholder(variable_name="agent_scratchpad"),
# ])


# Replace your current prompt with this corrected version
manager_agent_prompt = PromptTemplate.from_template(
    """You are a Manager Agent overseeing a research system with two specialized agents:

1. Search Agent: Finds and adds new research papers to the database
2. RAG Agent: Answers questions based on papers in the database

Your job is to:
1. Analyze the user's query: {input}
2. Check if there are relevant documents in the database using the check_database_status tool
3. Decide which agent should handle the query:
   - If documents are available, use the RAG Agent
   - If documents are unavailable or insufficient, use the Search Agent first, then the RAG Agent

You have access to the following tools: {tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original question. Must include either a RAG Agent or Search Agent in the Final answer. 

Begin!

Question: {input}
Thought:{agent_scratchpad}"""
)

def create_manager_agent():
    """Create and return the manager agent"""
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0
    )
    
    # Create the agent using the langgraph prebuilt create_react_agent
    agent = create_react_agent(llm, manager_tools, prompt= manager_agent_prompt)

    # Create with proper configuration
    return AgentExecutor(
        agent=agent,
        tools=manager_tools,
        handle_parsing_errors=True,
        max_iterations=5,  # Prevent infinite loops
        verbose=True  # For debugging
    )


# Create the manager agent
manager_agent = create_manager_agent()