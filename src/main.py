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

# Configure environment - you should set these in your environment
# os.environ["OPENAI_API_KEY"] = "your-openai-api-key"
# os.environ["DEEPSEEK_API_KEY"] = "your-deepseek-api-key"

from react_style import create_react_agent


# manager agent
openai_o1 = ChatOpenAI(
    model="gpt-4o-2024-05",  # Replace with actual o1 model name when available
    temperature=0,
    model_kwargs={"high_compute": True}  # Placeholder for high compute setting
)


# Define Manager Agent prompt
manager_agent_prompt = ChatPromptTemplate.from_messages([
    SystemMessage(content="""You are a Manager Agent overseeing a collaborative research system. Your job is to:
1. Analyze the user's query
2. Decide which agent should handle it:
   - Search Agent: For finding and adding new research papers
   - RAG Agent: For answering questions based on papers in the database
3. Coordinate their activities
4. Synthesize their outputs into a final answer for the user

Use the tools available to you and think step by step.  
                  
    You must provide your answer in JSON format with the following structure:
    {{
    "agent": "AGENT_NAME",
    "reasoning": "Your step-by-step reasoning for selecting this agent",
    "confidence": 0.95  // Value between 0.0 and 1.0 indicating your confidence in this decision
    }}
                  
                  """),
    MessagesPlaceholder(variable_name="messages"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
    
])

tools = []

manager_agent = create_react_agent(openai_o1, manager_agent_prompt, tools)


# Create a judge for evaluation (using GPT-4o)
judge_llm = ChatOpenAI(model="gpt-4o", temperature=0)


# Define evaluation prompt
evaluation_prompt = ChatPromptTemplate.from_messages([
    SystemMessage(content="""You are an expert evaluator tasked with judging the quality of responses to research questions. 
    
Score each response on a scale of 1-10 based on:
1. Accuracy: How factually correct is the information?
2. Relevance: How well does it address the specific question asked?
3. Comprehensiveness: How complete is the coverage of relevant aspects?
4. Evidence: How well supported are the claims with references?
5. Clarity: How clearly is the information presented?

Provide your overall score and brief justification."""),
        HumanMessage(content="""
            Question: {query}

            Response to evaluate: {response}

            Please evaluate this response.
            """)
            ])


def evaluate_response(query, response):
    """Evaluate a response using the judge LLM"""
    evaluation = judge_llm.invoke(
        evaluation_prompt.format(
            query=query,
            response=response
        )
    )
    return evaluation.content



# Manager agent function
def process_query(query, model="openai"):
    """Process a research query through the agent system"""
    # Start with the manager agent
    manager_result = manager_agent.invoke({"messages": [HumanMessage(content=f"Research query: {query}")]})
    manager_response = manager_result["messages"][-1].content
    
    # Check if the manager determined we need to search for new papers
    if "search_agent" in manager_response.lower():
        search_result = search_agent.invoke({"messages": [HumanMessage(content=f"Find papers for: {query}")]})
        search_response = search_result["messages"][-1].content
        print(f"Search Agent: {search_response}")
    
    # Use the appropriate RAG agent based on model parameter
    if model.lower() == "openai":
        rag_agent = rag_agent_openai
        model_name = "OpenAI o1"
    else:
        rag_agent = rag_agent_deepseek
        model_name = "DeepSeek R1"
    
    # Get the answer from the RAG agent
    rag_result = rag_agent.invoke({"messages": [HumanMessage(content=query)]})
    rag_response = rag_result["messages"][-1].content
    
    # Evaluate the response
    evaluation = evaluate_response(query, rag_response)
    
    return {
        "query": query,
        "model": model_name,
        "response": rag_response,
        "evaluation": evaluation
    }

# Compare both models
def compare_models(query):
    """Compare OpenAI o1 and DeepSeek R1 on the same query"""
    print(f"Processing query with OpenAI o1...")
    openai_result = process_query(query, "openai")
    
    print(f"Processing query with DeepSeek R1...")
    deepseek_result = process_query(query, "deepseek")
    
    return {
        "query": query,
        "openai_result": openai_result,
        "deepseek_result": deepseek_result
    }

# Example usage
if __name__ == "__main__":
    query = "What are the latest developments in large language model reasoning capabilities?"
    results = compare_models(query)
    
    print("\n=== Results ===")
    print(f"Query: {results['query']}")
    print("\nOpenAI o1 Response:")
    print(results['openai_result']['response'])
    print("\nOpenAI o1 Evaluation:")
    print(results['openai_result']['evaluation'])
    print("\nDeepSeek R1 Response:")
    print(results['deepseek_result']['response'])
    print("\nDeepSeek R1 Evaluation:")
    print(results['deepseek_result']['evaluation'])